"""Layerwise probe decodability evaluation (paper figure 4 data source).

Answers the question the generation-based metrics cannot: IS the entity
information linearly present in the mid-layer residual stream at all?

A_probe(dataset) = accuracy of the (trained) linear probe head reading the
hidden state at layer l*, anchored at the SAME token the training loss reads
-- the head-entity anchor token tau_i (the last token of the entity span,
resolved from entity_char_start/end via offset mapping exactly like the
training collator). Measured separately on D_mem (memorized inputs) and
D_gen (held-out multi-hop inputs):

  - A_probe_mem(t)  high  + A_probe_gen(t) low  -> info stored but not used
    (the Knowing-Using Gap, in representation space).
  - A_probe_mem(t)  low                         -> info not even decodable:
    memorization lives elsewhere (or extraction, not knowledge, failed).

ANCHOR CONSISTENCY (v5 fix): the probe is TRAINED at the entity anchor
(entity_pos from the collator) but was previously EVALUATED at the last
prompt token -- the resulting gate signal (0.047) disagreed with the
trainer's own probe train accuracy (0.75-1.0) on identical data by 16x.
Both now read the same position; rows without anchor offsets fall back to
the final prompt token.

This is a single teacher-forced forward pass per batch -- orders of magnitude
cheaper than generation-based A_mem, so it is safe to run every epoch.
"""

from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .gen_eval import _strip_answer_suffix


def _resolve_class_id(row: Dict, entity2id: Optional[Dict[str, int]]) -> Optional[int]:
    """Resolve the probe class id for a raw dataset row, or None to skip."""
    label = row.get("label_text") or row.get("entity") or row.get("target")
    if entity2id and label in entity2id:
        return int(entity2id[label])
    for key in ("probe_label_id", "entity_class_id"):
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def _resolve_prompt(text: str) -> str:
    """Prompts for probe reads: question only (never leak ' Answer: X')."""
    if " Answer:" in text:
        return _strip_answer_suffix(text)
    return text


def _anchor_index_from_offsets(offset_mapping, char_start, char_end) -> Optional[int]:
    """Token index of the entity anchor, matching the training convention.

    The training pipeline (dataset._char_to_token_idx, driven by
    entity_char_end) selects the first token whose character span contains
    char_end. Mirroring that exactly keeps train/eval anchors identical.
    Returns None when offsets are unavailable or the span cannot be found
    (caller falls back to the final prompt token).
    """
    if not offset_mapping:
        return None
    for token_idx, (cs, ce) in enumerate(offset_mapping):
        if cs == 0 and ce == 0 and token_idx > 0:
            continue  # special tokens
        if cs <= char_end <= ce:
            return token_idx
    # char_end beyond every span (truncation): use the last real token
    return len(offset_mapping) - 1


@torch.no_grad()
def evaluate_probe_accuracy(
    model,
    probe_head,
    tokenizer,
    dataset,
    l_star: int,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    entity2id: Optional[Dict[str, int]] = None,
) -> Tuple[float, int]:
    """
    Linear-probe accuracy at layer l*, anchored at the entity anchor token
    (the same position the training probe loss reads).

    Args:
        model: causal LM (any PeftModel/base model)
        probe_head: LinearProbe reading hidden states -> entity logits
        tokenizer: HF tokenizer
        dataset: raw dataset rows with "text" + label fields (+ optional
            entity_char_start / entity_char_end anchor offsets)
        l_star: hidden-state layer to probe (0 = embeddings, L = last)
        num_samples: optional cap (takes the FIRST n rows; dataset order is
            stable across epochs, so the curve is comparable over time)
        entity2id: label_text -> class-id mapping; falls back to the row's
            probe_label_id / entity_class_id when absent

    Returns:
        (accuracy, n_evaluated)
    """
    if probe_head is None or dataset is None or len(dataset) == 0:
        return 0.0, 0

    model.eval()
    probe_head.eval()

    n_total = len(dataset)
    if num_samples is not None:
        n_total = min(n_total, int(num_samples))

    prompts: List[str] = []
    labels: List[int] = []
    anchors: List[Optional[Tuple[int, int]]] = []  # (char_start, char_end)
    for i in range(n_total):
        row = dataset[i]
        cid = _resolve_class_id(row, entity2id)
        if cid is None:
            continue
        prompts.append(_resolve_prompt(row["text"]))
        labels.append(cid)
        cs = row.get("entity_char_start")
        ce = row.get("entity_char_end")
        anchors.append((int(cs), int(ce)) if cs is not None and ce is not None else None)

    if not prompts:
        return 0.0, 0

    prev_padding = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"  # anchor rows end at the sequence tail
    correct, evaluated = 0, 0
    try:
        for i in tqdm(range(0, len(prompts), batch_size),
                      desc="Probe Decodability", leave=False):
            batch_prompts = prompts[i:i + batch_size]
            batch_anchors = anchors[i:i + batch_size]
            batch_labels = torch.tensor(labels[i:i + batch_size], device=device)
            enc = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            # return_tensors="pt" makes offset_mapping a (B, L, 2) tensor;
            # the anchor resolver works on plain lists (and `not tensor`
            # raises for multi-element tensors -- the ar06b lesson).
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            inputs = {k: v.to(device) for k, v in enc.items()}
            seq_lens = inputs["attention_mask"].sum(dim=1)  # unpadded length
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states
            if hidden is None or len(hidden) <= l_star:
                continue
            h_layer = hidden[l_star]
            # Per-row anchor position in PADDED coordinates (left padding:
            # token j of the unpadded row sits at seq_len - unpadded_len + j)
            idxs = []
            for b, anchor in enumerate(batch_anchors):
                unpadded = int(seq_lens[b].item())
                pad_off = inputs["input_ids"].shape[1] - unpadded
                a_idx = None
                if anchor is not None:
                    a_unpadded = _anchor_index_from_offsets(
                        offsets[b], anchor[0], anchor[1])
                    if a_unpadded is not None:
                        a_idx = min(a_unpadded, unpadded - 1) + pad_off
                if a_idx is None:
                    a_idx = inputs["input_ids"].shape[1] - 1  # final token
                idxs.append(a_idx)
            idx_t = torch.tensor(idxs, device=device)
            h_anchor = h_layer[
                torch.arange(h_layer.size(0), device=device), idx_t, :]
            logits = probe_head(h_anchor)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch_labels).sum().item())
            evaluated += int(batch_labels.numel())
    finally:
        tokenizer.padding_side = prev_padding

    return (correct / evaluated if evaluated else 0.0), evaluated
