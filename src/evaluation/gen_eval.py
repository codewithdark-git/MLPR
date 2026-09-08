"""Multi-hop generation and answer scoring for evaluation (v2).

Why v2: in the 50-epoch runs BOTH memorization and generalization
exact-match stayed near zero (7B A_mem 6.25%, 0.5B 0.42%, A_gen 0%
everywhere) even though the same models clearly trained. Diagnosis found a
scoring artifact strong enough to explain the uniform zeros:

  1. `model.generate()` returns PROMPT + continuation; decoding the whole
     tensor and regex-extracting from it mixes the question into the
     prediction whenever the model does not emit a literal "Answer:" prefix.
  2. With no EOS after the answer, greedy generation continues into repeated
     training-format text ("... Answer: X Question ... Answer: Y"), which the
     old non-greedy regex swallowed into the prediction.
  3. Exact match alone gives no partial credit and no diagnostics.

v2 therefore:
  - decodes ONLY the newly generated tokens,
  - truncates the generation at the first newline / repeated-question marker,
  - scores every sample with THREE metrics (exact match, contains match,
    token-level F1) so the paper can separate "cannot recall" from "recalls
    but formats differently",
  - can dump a W&B Table of sample generations for eyeballing
    (log_sample_table) so future runs are debuggable from the dashboard.
"""

import re
from typing import List, Dict, Tuple, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


# --------------------------------------------------------------------- metrics

def _normalize(text: str) -> str:
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_exact_match(prediction: str, target: str) -> bool:
    """Exact match after normalization (lowercase, whitespace, punctuation)."""
    return _normalize(prediction) == _normalize(target)


def compute_contains_match(prediction: str, target: str) -> bool:
    """Partial credit: the normalized target appears inside the prediction.

    Distinguishes 'recalled the entity but with extra words/format' from a
    genuine recall failure -- critical for interpreting the uniform-zero
    A_gen scores in the first 50-epoch runs.
    """
    pred, tgt = _normalize(prediction), _normalize(target)
    return bool(tgt) and tgt in pred


def compute_token_f1(prediction: str, target: str) -> float:
    """Token-level F1 (SQuAD-style) between prediction and target."""
    pred_tokens = _normalize(prediction).split()
    tgt_tokens = _normalize(target).split()
    if not pred_tokens or not tgt_tokens:
        return 0.0
    common = set(pred_tokens) & set(tgt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(tgt_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Mean EM / contains / token-F1 over a results list from the evaluators.

    Also aggregates the validity-aware variants (em_any / contains_any)
    when the records carry them (v3: multi-answer queries scored against
    the full valid-answer set instead of one arbitrary label).
    """
    if not results:
        return {"exact_match": 0.0, "contains_match": 0.0, "token_f1": 0.0,
                "exact_match_any": 0.0, "contains_match_any": 0.0}
    n = len(results)
    return {
        "exact_match": sum(float(r.get("em", 0.0)) for r in results) / n,
        "contains_match": sum(float(r.get("contains", 0.0)) for r in results) / n,
        "token_f1": sum(float(r.get("token_f1", 0.0)) for r in results) / n,
        "exact_match_any": sum(float(r.get("em_any", r.get("em", 0.0)))
                               for r in results) / n,
        "contains_match_any": sum(float(r.get("contains_any",
                                               r.get("contains", 0.0)))
                                  for r in results) / n,
    }


def log_sample_table(name: str, results: List[Dict], max_rows: int = 16) -> None:
    """Log a bounded W&B Table of sample generations (prompt/gen/target/scores).

    Failure-safe: never raises. This is what makes a 0% score diagnosable
    directly from the W&B dashboard instead of requiring a volume fetch.
    """
    try:
        import wandb
        if wandb.run is None or not results:
            return
        rows = []
        for r in results[:max_rows]:
            rows.append([
                str(r.get("prompt", ""))[:220],
                str(r.get("generation", ""))[:220],
                str(r.get("target", "")),
                str(r.get("prediction", "")),
                bool(r.get("em", False)),
                bool(r.get("contains", False)),
                round(float(r.get("token_f1", 0.0)), 3),
            ])
        table = wandb.Table(
            columns=["prompt", "generation", "target", "prediction",
                     "exact_match", "contains_match", "token_f1"],
            data=rows,
        )
        wandb.log({name: table})
    except Exception:
        pass


# ------------------------------------------------------------------ extraction

def _extract_targets(batch_data):
    """
    Extract ground-truth targets from a batch, tolerating different schemas.

    The synthetic dataset uses `label_text`; older/sample datasets may use
    `entity` or `target`. Works for both dict-of-lists (HF Dataset slice)
    and list-of-dicts batch formats.
    """
    candidate_keys = ("label_text", "entity", "target")
    if isinstance(batch_data, dict):
        for key in candidate_keys:
            if key in batch_data:
                return batch_data[key]
        raise KeyError(
            f"No target field found in batch; tried {candidate_keys}. "
            f"Available fields: {list(batch_data.keys())}"
        )
    for key in candidate_keys:
        if batch_data and all(key in item for item in batch_data):
            return [item[key] for item in batch_data]
    raise KeyError(f"No target field found in batch items; tried {candidate_keys}")


def _extract_valid_answers(batch_data):
    """
    Extract per-item valid-answer lists from a batch (v3 scoring).

    Multi-answer queries (e.g. "name one audit required for a product issued
    by {inst}" -- 10 valid audits) were scored against ONE arbitrary label in
    v1/v2, which capped exact match far below true knowledge. The v2 dataset
    generator stores the full valid set in a `valid_answers` field; scoring
    ANY of them correct (em_any / contains_any) separates "cannot recall"
    from "recalled a different valid answer".

    Returns a list aligned with the batch items; missing fields yield None
    (caller falls back to single-target scoring).
    """
    if isinstance(batch_data, dict):
        vals = batch_data.get("valid_answers")
        if vals is None:
            n = len(batch_data.get("text", [])) \
                or len(batch_data.get("label_text", [])) or 0
            return [None] * n
        out = []
        for v in vals:
            if v is None:
                out.append(None)
            elif isinstance(v, (list, tuple)):
                out.append([str(x) for x in v])
            else:
                out.append([str(v)])
        return out
    out = []
    for item in batch_data:
        v = item.get("valid_answers") if isinstance(item, dict) else None
        if v is None:
            out.append(None)
        elif isinstance(v, (list, tuple)):
            out.append([str(x) for x in v])
        else:
            out.append([str(v)])
    return out


def _strip_answer_suffix(text: str) -> str:
    """
    Remove the trailing " Answer: ..." segment from a memorization prompt.

    D_mem texts are stored as "question Answer: label" so the CE loss can
    learn p(answer | question). The memorization evaluation must feed the
    question ONLY: keeping the answer in the prompt would leak the target,
    saturate A_mem at ~1.0 immediately, and fire the lambda gate spuriously.
    """
    idx = text.rfind(" Answer:")
    if idx != -1:
        return text[:idx].rstrip()
    return text


_REPEAT_MARKERS = ("\nQuestion", "\nquestion", "\nQ:", "\n\n", "\nAnswer")


def _clean_generation(text: str) -> str:
    """Truncate a decoded continuation at the first spillover marker.

    Without an EOS after the answer, greedy decoding keeps generating
    training-format text; everything after the first newline boundary is
    spillover, not the answer.
    """
    for marker in _REPEAT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def extract_answer_from_generation(generation: str) -> str:
    """
    Extract the answer from a (continuation-only) generated response.

    Handles formats like:
    - "Answer: FinancialAuthority"
    - "The answer is FinancialAuthority"
    - Just the entity name

    Only the FIRST line is considered by default: without an EOS, greedy
    decoding spills over into hallucinated follow-up questions ("...
    Answer: X \\n Question 2 ... Answer: Y"), and a global regex would
    return the wrong (later) answer.
    """
    first_line = generation.split("\n", 1)[0].strip()

    match = re.search(r"[Aa]nswer:\s*(.+)", first_line)
    if match:
        return match.group(1).strip().rstrip(".")
    match = re.search(r"[Tt]he answer is\s*(.+)", first_line)
    if match:
        return match.group(1).strip().rstrip(".")
    if first_line:
        return first_line.rstrip(".")

    # empty first line (e.g. leading newline) -> scan the whole text
    match = re.search(r"[Aa]nswer:\s*(.+)", generation)
    if match:
        return match.group(1).strip().rstrip(".")
    return generation.strip()


@torch.no_grad()
def _generate_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
) -> List[str]:
    """Batch-generate and decode ONLY the new tokens (left padding, safe restore)."""
    prev_padding = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    # Greedy decoding: clear sampling params from the generation config so
    # transformers stops warning about temperature/top_p/top_k being set while
    # do_sample=False (they are restored afterwards).
    gen_cfg = getattr(model, "generation_config", None)
    saved_gen = None
    if gen_cfg is not None:
        saved_gen = (gen_cfg.do_sample, gen_cfg.temperature,
                     gen_cfg.top_p, gen_cfg.top_k)
        gen_cfg.do_sample = False
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = prev_padding
        if gen_cfg is not None and saved_gen is not None:
            (gen_cfg.do_sample, gen_cfg.temperature,
             gen_cfg.top_p, gen_cfg.top_k) = saved_gen
    return [_clean_generation(t) for t in texts]


def _score_row(generated: str, target: str) -> Dict:
    prediction = extract_answer_from_generation(generated)
    return {
        "generation": generated,
        "prediction": prediction,
        "em": compute_exact_match(prediction, target),
        "contains": compute_contains_match(prediction, target),
        "token_f1": compute_token_f1(prediction, target),
    }


def _score_row_any(generated: str, target: str,
                   valid_answers: Optional[List[str]]) -> Dict:
    """Score against the full valid-answer set when available (v3).

    Adds em_any / contains_any on top of the v2 single-target metrics. When
    no valid list exists the validity-aware scores mirror the single-target
    ones, so downstream aggregation is uniform across datasets.
    """
    row = _score_row(generated, target)
    prediction = row["prediction"]
    if not valid_answers:
        row["em_any"] = row["em"]
        row["contains_any"] = row["contains"]
        row["token_f1_any"] = row["token_f1"]
        return row
    # Best partial credit across the valid set
    f1s = [compute_token_f1(prediction, v) for v in valid_answers]
    row["em_any"] = any(compute_exact_match(prediction, v) for v in valid_answers)
    row["contains_any"] = any(compute_contains_match(prediction, v)
                              for v in valid_answers)
    row["token_f1_any"] = max(f1s) if f1s else row["token_f1"]
    # also record the single best-matching valid target for diagnostics
    best = max(valid_answers, key=lambda v: compute_token_f1(prediction, v)) \
        if valid_answers else target
    row["best_valid_target"] = best
    return row


# ------------------------------------------------------------------ evaluators

@torch.no_grad()
def evaluate_memorization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: str = "cuda",
    max_new_tokens: int = 50,
    batch_size: int = 4,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate memorization accuracy on single-hop QA pairs.

    Returns (exact-match accuracy, per-sample results). Each result row
    carries prompt/generation/prediction plus em/contains/token_f1 scores.
    """
    model.eval()
    results = []
    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating Memorization"):
        batch_end = min(i + batch_size, total)
        batch_data = dataset[i:batch_end]

        if isinstance(batch_data, dict):
            texts = batch_data["text"]
        else:
            texts = [item["text"] for item in batch_data]
        targets = _extract_targets(batch_data)
        valid_lists = _extract_valid_answers(batch_data)

        # Feed the question only (strip the " Answer: <label>" training suffix)
        prompts = [_strip_answer_suffix(t) for t in texts]
        generations = _generate_continuations(
            model, tokenizer, prompts, device, max_new_tokens)

        for j, (generated, target) in enumerate(zip(generations, targets)):
            row = {"text": texts[j] if isinstance(texts, list) else texts,
                   "prompt": prompts[j], "target": target}
            row.update(_score_row_any(generated, target, valid_lists[j]))
            # Headline accuracy is validity-aware when valid sets exist
            correct += int(row.get("em_any", row["em"]))
            results.append(row)

    accuracy = correct / max(total, 1)
    for row in results:
        row.setdefault("em_any", row["em"])
        row.setdefault("contains_any", row["contains"])
        row.setdefault("token_f1_any", row["token_f1"])
    return accuracy, results


@torch.no_grad()
def evaluate_generalization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: str = "cuda",
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate generalization accuracy on multi-hop QA pairs.

    Compositional reasoning on queries that require chaining or intersection
    of memorized facts. Scoring identical to evaluate_memorization (three
    metrics per sample) so the two curves are directly comparable.
    """
    model.eval()
    results = []
    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating Generalization"):
        batch_end = min(i + batch_size, total)
        batch_data = dataset[i:batch_end]

        if isinstance(batch_data, dict):
            texts = batch_data["text"]
            relations = batch_data.get("relations", ["unknown"] * len(batch_data["text"]))
        else:
            texts = [item["text"] for item in batch_data]
            relations = [item.get("relations", ["unknown"]) for item in batch_data]
        targets = _extract_targets(batch_data)
        valid_lists = _extract_valid_answers(batch_data)

        # Gen texts are bare questions -- nothing to strip
        generations = _generate_continuations(
            model, tokenizer, texts, device, max_new_tokens)

        for j, (generated, target) in enumerate(zip(generations, targets)):
            row = {"text": texts[j] if isinstance(texts, list) else texts,
                   "prompt": texts[j], "target": target}
            row.update(_score_row_any(generated, target, valid_lists[j]))
            rel = relations[j] if isinstance(relations, list) else relations
            row["relations"] = rel
            row["type"] = "multi-hop"
            # Headline accuracy is validity-aware when valid sets exist
            correct += int(row.get("em_any", row["em"]))
            results.append(row)

    accuracy = correct / max(total, 1)
    for row in results:
        row.setdefault("em_any", row["em"])
        row.setdefault("contains_any", row["contains"])
        row.setdefault("token_f1_any", row["token_f1"])
    return accuracy, results


@torch.no_grad()
def evaluate_memorization_likelihood(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    answer_prefix: str = " Answer:",
) -> Tuple[float, List[Dict]]:
    """Likelihood-based memorization accuracy (teacher-forced greedy match).

    WHY THIS EXISTS (v2 gate fix): free-generation exact match conflates
    *knowing* a fact with *saying* it in the trained surface format. In the
    50-epoch runs generation EM plateaued at 0-12% vs tau_0=0.9, so the
    lambda gate never opened and the probe loss never activated. The
    teacher-forced variant asks the weaker, cleaner question: when the model
    is primed with "question Answer:" and FORCED to continue, does its greedy
    argmax reproduce the memorized answer tokens? This is the standard
    recall-style probe used in closed-book KB evaluation (ParaRel-style),
    costs ONE forward pass per batch (no autoregressive loop), and is the
    saturation signal the gate should have been using.

    Per-sample score ("em"): ALL gold answer-token positions match the greedy
    argmax (strict, comparable to EM). Each row also records "token_acc"
    (fraction of matching answer tokens) and "prediction" (the argmax answer
    decoded back to text, for W&B sample tables).
    """
    model.eval()
    results = []
    correct = 0
    n_total = len(dataset)
    if num_samples is not None:
        n_total = min(n_total, int(num_samples))

    for i in tqdm(range(0, n_total, batch_size),
                  desc="Likelihood Memorization", leave=False):
        batch_end = min(i + batch_size, n_total)
        batch_data = dataset[i:batch_end]

        if isinstance(batch_data, dict):
            texts = batch_data["text"]
        else:
            texts = [item["text"] for item in batch_data]
        targets = _extract_targets(batch_data)
        valid_lists = _extract_valid_answers(batch_data)
        prompts = [_strip_answer_suffix(t) for t in texts]

        # Full sequence = prompt + " Answer:" + target; the answer token span
        # is the suffix after the un-padded prompt length. The boundary is
        # clean for BPE tokenizers because ":" and " word" never merge.
        full_texts = [p + answer_prefix + " " + str(t)
                      for p, t in zip(prompts, targets)]

        # Batched forward pass with padding.
        enc_full = tokenizer(full_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(device)
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        # use_cache=False: this is a single teacher-forced forward pass -- the
        # KV cache is dead weight here and its tuple form triggered the
        # 'past_key_values as a tuple' deprecation warning in the smoke log.
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]

        # Per-sample prompt/answer lengths from UN-padded encodings.
        prompt_lens = [len(x) for x in tokenizer(prompts, padding=False,
                                                 truncation=True,
                                                 max_length=max_length)["input_ids"]]
        answer_lens = [len(f) - p for f, p in
                       zip(tokenizer(full_texts, padding=False,
                                     truncation=True,
                                     max_length=max_length)["input_ids"],
                           prompt_lens)]

        for j in range(input_ids.size(0)):
            p_len = prompt_lens[j]
            a_len = answer_lens[j]
            if a_len <= 0 or p_len + a_len > input_ids.size(1):
                continue
            # gold answer token ids and greedy predictions at those positions
            gold = input_ids[j, p_len:p_len + a_len]
            preds = logits[j, p_len - 1:p_len + a_len - 1, :].argmax(dim=-1)
            tok_hits = (preds == gold)
            tok_acc = float(tok_hits.float().mean().item())
            strict = bool(tok_hits.all().item())
            pred_text = tokenizer.decode(preds.tolist(),
                                         skip_special_tokens=True)
            row = {
                "text": texts[j] if isinstance(texts, list) else texts,
                "prompt": prompts[j],
                "target": targets[j],
                "prediction": pred_text,
                "em": strict,           # strict per-sample teacher-forced EM
                "contains": strict,     # likelihood eval has no spillover
                "token_f1": tok_acc,    # token-level partial credit
                "token_acc": tok_acc,
            }
            # v3: validity-aware scoring for multi-answer items -- the argmax
            # answer may name a DIFFERENT valid entity than the trained label.
            v_list = valid_lists[j] if j < len(valid_lists) else None
            if v_list:
                row["em_any"] = any(
                    compute_exact_match(pred_text, v) for v in v_list)
                row["contains_any"] = any(
                    compute_contains_match(pred_text, v) for v in v_list)
                row["token_f1_any"] = max(
                    compute_token_f1(pred_text, v) for v in v_list)
            else:
                row["em_any"] = strict
                row["contains_any"] = strict
                row["token_f1_any"] = tok_acc
            # likelihood headline counts validity-aware hits
            correct += int(row["em_any"])
            results.append(row)

    accuracy = correct / max(len(results), 1)
    return accuracy, results


def compute_probe_decodability(
    probe_logits: torch.Tensor,
    probe_labels: torch.Tensor,
) -> float:
    """
    Compute probe decodability accuracy.

    Args:
        probe_logits: Logits from the probe head
        probe_labels: Ground truth entity class IDs

    Returns:
        Probe classification accuracy
    """
    predictions = probe_logits.argmax(dim=-1)
    accuracy = (predictions == probe_labels).float().mean().item()
    return accuracy


def run_full_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mem_dataset,
    gen_dataset,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run full evaluation including both memorization and generalization.
    """
    print("\n" + "="*50)
    print("Running Full Evaluation")
    print("="*50)

    a_mem, mem_results = evaluate_memorization(model, tokenizer, mem_dataset, device)
    print(f"\nMemorization Accuracy (A_mem): {a_mem:.4f}")

    a_gen, gen_results = evaluate_generalization(model, tokenizer, gen_dataset, device)
    print(f"Generalization Accuracy (A_gen): {a_gen:.4f}")

    gap = a_mem - a_gen

    print(f"\nKnowing-Using Gap: {gap:.4f}")
    print("="*50 + "\n")

    return {
        "a_mem": a_mem,
        "a_gen": a_gen,
        "gap": gap,
        "mem_metrics": aggregate_metrics(mem_results),
        "gen_metrics": aggregate_metrics(gen_results),
        "mem_details": mem_results,
        "gen_details": gen_results,
    }
