"""Layerwise probe decodability evaluation (paper figure 4 data source).

Answers the question the generation-based metrics cannot: IS the entity
information linearly present in the mid-layer residual stream at all?

A_probe(dataset) = accuracy of the (trained) linear probe head reading the
hidden state at layer l*, anchored at the last prompt token, over the
candidate entity set. Measured separately on D_mem (memorized inputs) and
D_gen (held-out multi-hop inputs):

  - A_probe_mem(t)  high  + A_probe_gen(t) low  -> info stored but not used
    (the Knowing-Using Gap, in representation space).
  - A_probe_mem(t)  low                         -> info not even decodable:
    memorization lives elsewhere (or extraction, not knowledge, failed).

This is a single teacher-forced forward pass per batch -- orders of magnitude
cheaper than generation-based A_mem, so it is safe to run every epoch.
"""

from typing import Dict, Optional, Tuple

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
    Linear-probe accuracy at layer l* anchored at the final prompt token.

    Args:
        model: causal LM (any PeftModel/base model)
        probe_head: LinearProbe reading hidden states -> entity logits
        tokenizer: HF tokenizer
        dataset: raw dataset rows with "text" + label fields
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

    prompts, labels = [], []
    for i in range(n_total):
        row = dataset[i]
        cid = _resolve_class_id(row, entity2id)
        if cid is None:
            continue
        prompts.append(_resolve_prompt(row["text"]))
        labels.append(cid)

    if not prompts:
        return 0.0, 0

    prev_padding = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"  # anchor = last position for every row
    correct, evaluated = 0, 0
    try:
        for i in tqdm(range(0, len(prompts), batch_size),
                      desc="Probe Decodability", leave=False):
            batch_prompts = prompts[i:i + batch_size]
            batch_labels = torch.tensor(labels[i:i + batch_size], device=device)
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states
            if hidden is None or len(hidden) <= l_star:
                continue
            # left padding -> the final position is the last prompt token
            h_anchor = hidden[l_star][:, -1, :]
            logits = probe_head(h_anchor)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch_labels).sum().item())
            evaluated += int(batch_labels.numel())
    finally:
        tokenizer.padding_side = prev_padding

    return (correct / evaluated if evaluated else 0.0), evaluated
