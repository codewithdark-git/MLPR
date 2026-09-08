"""Memorization gate callback: feeds A_mem(t) into the LambdaScheduler.

Implements the gating signal specified in idea.md:

    lambda(t) = lambda_0 * min(1, max(0, A_mem(t) - tau_0) / delta_0)

A_mem(t) is the memorization accuracy on D_mem, measured with generation-based
exact-match evaluation. Without this callback the LambdaScheduler stays at
lambda = 0 forever and the probe loss never activates (the run silently
degenerates into plain SFT).

To keep the gate cheap, A_mem is estimated on a random subsample of D_mem at
each epoch boundary (default 200 samples, exact-match noise ~ +/-2%).
"""

import random
from typing import Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ..evaluation.gen_eval import (
    evaluate_memorization,
    evaluate_memorization_likelihood,
    evaluate_generalization,
    aggregate_metrics,
    log_sample_table,
)
from ..evaluation.probe_eval import evaluate_probe_accuracy


class MemorizationGateCallback(TrainerCallback):
    """
    At each epoch end: estimate A_mem on D_mem subsample, then update the
    LambdaScheduler so the composite loss L_CE + lambda(t) * L_probe uses the
    correct gate value for the next epoch.
    """

    def __init__(
        self,
        lambda_scheduler,
        mem_dataset,
        tokenizer,
        device: str = "cuda",
        num_samples: int = 200,
        batch_size: int = 16,
        max_new_tokens: int = 24,
        seed: int = 42,
        gen_dataset=None,
        probe_head=None,
        entity2id=None,
        probe_num_samples: int = 200,
        gen_num_samples: int = None,
    ):
        """
        Args:
            lambda_scheduler: LambdaScheduler instance to update
            mem_dataset: Raw memorization dataset (with "text" / "label_text")
            tokenizer: Tokenizer for generation evaluation
            device: Device for evaluation
            num_samples: Subsample size for the A_mem estimate
            batch_size: Generation batch size during gate evaluation
            max_new_tokens: Max new tokens when generating answers
            seed: RNG seed for reproducible subsampling
            gen_dataset: Optional raw generalization dataset (with "text").
                When provided, exact-match A_gen(t) is measured at each epoch
                end too, giving the full per-epoch curve A_mem(t), A_gen(t),
                gap(t) for analysis/plots. None keeps the gate cheap (A_mem only).
            probe_head: Optional LinearProbe. When provided, per-epoch probe
                decodability A_probe_mem(t) / A_probe_gen(t) is measured with
                a single teacher-forced forward pass (cheap) -- the paper's
                representation-space evidence for storage vs. use.
            entity2id: label_text -> probe class id mapping (from MLPDataset).
            probe_num_samples: Subsample cap for the probe accuracy estimate.
            gen_num_samples: Subsample cap for the per-epoch A_gen generation
                pass. None (default) evaluates the FULL D_gen (paper runs);
                autoresearch mode sets a small cap (e.g. 64) so the per-epoch
                gate cost stays inside the fixed training budget.
        """
        self.lambda_scheduler = lambda_scheduler
        self.mem_dataset = mem_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.rng = random.Random(seed)
        self.gen_dataset = gen_dataset
        self.gen_num_samples = gen_num_samples
        self.probe_head = probe_head
        self.entity2id = entity2id
        self.probe_num_samples = probe_num_samples
        self.l_star = None  # injected by main.py (probe layer)
        self.history = []

    def _subsample(self):
        total = len(self.mem_dataset)
        n = min(self.num_samples, total)
        indices = sorted(self.rng.sample(range(total), n))
        return self.mem_dataset.select(indices)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is None or self.mem_dataset is None:
            return

        try:
            subset = self._subsample()

            # ---- v2: likelihood (teacher-forced) A_mem first -- cheap (one
            # forward pass per batch) and it is the "knowing" signal that
            # makes the gate reachable. A failure here must not kill the
            # generation-based measurement below.
            a_mem_lf = None
            lf_results = None
            try:
                a_mem_lf, lf_results = evaluate_memorization_likelihood(
                    model,
                    self.tokenizer,
                    subset,
                    device=self.device,
                    batch_size=max(self.batch_size, 16),
                )
            except Exception as lf_err:
                print(f"[MEM GATE] likelihood A_mem eval failed at epoch "
                      f"{state.epoch}: {lf_err}")

            try:
                a_mem, mem_results = evaluate_memorization(
                    model,
                    self.tokenizer,
                    subset,
                    device=self.device,
                    max_new_tokens=self.max_new_tokens,
                    batch_size=self.batch_size,
                )
            except Exception as first_err:
                # Transient failures (e.g. CUDA OOM during generation) were
                # observed to skip whole epochs of the gate curve; retry once
                # with a much smaller generation batch before giving up.
                small_bs = max(1, self.batch_size // 4)
                print(f"[MEM GATE] A_mem eval failed ({first_err}); "
                      f"retrying once with batch_size={small_bs}")
                a_mem, mem_results = evaluate_memorization(
                    model,
                    self.tokenizer,
                    subset,
                    device=self.device,
                    max_new_tokens=self.max_new_tokens,
                    batch_size=small_bs,
                )
            # Per-epoch probe decodability (representation-space): a cheap
            # teacher-forced forward pass, no generation involved. v5: this
            # runs BEFORE the scheduler update so gate_metric="probe" can
            # use it as the saturation signal (representation knows vs LM
            # head says -- the probe decodes the entity long before the
            # output level can emit the exact surface string).
            probe_acc_mem = probe_acc_gen = None
            if self.probe_head is not None and self.l_star is not None:
                try:
                    probe_acc_mem, _ = evaluate_probe_accuracy(
                        model, self.probe_head, self.tokenizer, subset,
                        l_star=self.l_star, device=self.device,
                        batch_size=self.batch_size,
                        num_samples=self.probe_num_samples,
                        entity2id=self.entity2id,
                    )
                    if self.gen_dataset is not None:
                        probe_acc_gen, _ = evaluate_probe_accuracy(
                            model, self.probe_head, self.tokenizer, self.gen_dataset,
                            l_star=self.l_star, device=self.device,
                            batch_size=self.batch_size,
                            num_samples=self.probe_num_samples,
                            entity2id=self.entity2id,
                        )
                except Exception as e:
                    print(f"[MEM GATE] probe decodability failed at epoch {state.epoch}: {e}")

            self.lambda_scheduler.update_a_mem(
                a_mem, a_mem_lf, probe_acc=probe_acc_mem)
            current_lambda = self.lambda_scheduler.get_lambda()

            # Optional per-epoch A_gen(t): the exact-match generalization
            # curve. Runs AFTER the scheduler update so a failure here can
            # never affect the gate signal; it only costs an extra
            # generation pass.
            a_gen = None
            gen_results = None
            if self.gen_dataset is not None:
                try:
                    gen_subset = self.gen_dataset
                    if (self.gen_num_samples and hasattr(gen_subset, "select")
                            and len(gen_subset) > self.gen_num_samples):
                        gen_subset = gen_subset.select(
                            range(self.gen_num_samples))
                    a_gen, gen_results = evaluate_generalization(
                        model,
                        self.tokenizer,
                        gen_subset,
                        device=self.device,
                        max_new_tokens=self.max_new_tokens,  # answers are short entities; the 100-token default tripled gate cost
                        batch_size=self.batch_size,
                    )
                except Exception as e:
                    print(f"[MEM GATE] A_gen evaluation failed at epoch {state.epoch}: {e}")

            entry = {
                "epoch": state.epoch,
                "a_mem": a_mem,
                "lambda": current_lambda,
            }
            if a_mem_lf is not None:
                entry["a_mem_lf"] = a_mem_lf
                entry["lf_token_acc"] = round(
                    sum(float(r.get("token_acc", 0.0)) for r in (lf_results or []))
                    / max(len(lf_results or []), 1), 6)
            for key, val in aggregate_metrics(mem_results).items():
                entry[f"mem_{key}"] = round(val, 6)
            if a_gen is not None:
                entry["a_gen"] = a_gen
                entry["gap"] = a_mem - a_gen
                for key, val in aggregate_metrics(gen_results or []).items():
                    entry[f"gen_{key}"] = round(val, 6)
            if probe_acc_mem is not None:
                entry["probe_acc_mem"] = probe_acc_mem
            if probe_acc_gen is not None:
                entry["probe_acc_gen"] = probe_acc_gen
            self.history.append(entry)

            log_line = (
                f"[MEM GATE] epoch {state.epoch:.0f}: A_mem={a_mem:.4f}"
                + (f" A_mem_lf={a_mem_lf:.4f}" if a_mem_lf is not None else "")
                + f" -> lambda(t)={current_lambda:.4f} "
                f"(metric={self.lambda_scheduler.gate_metric}, "
                f"saturated={self.lambda_scheduler.is_memorization_saturated()})"
            )
            # Diagnostic partial-credit metrics: a strict-EM of 0.0 is only
            # interpretable next to the teacher-forced token accuracy and the
            # contains/F1 scores (they live in gate_history.json, but they must
            # be visible in the console log too).
            if entry.get("lf_token_acc") is not None:
                log_line += f" | lf_tok_acc={entry['lf_token_acc']:.3f}"
            if entry.get("mem_contains_match") is not None:
                log_line += (f" | mem_contains={entry['mem_contains_match']:.3f}"
                             f" mem_f1={entry.get('mem_token_f1', 0.0):.3f}")
            if entry.get("gen_contains_match") is not None:
                log_line += (f" | gen_contains={entry['gen_contains_match']:.3f}"
                             f" gen_f1={entry.get('gen_token_f1', 0.0):.3f}")
            if a_gen is not None:
                log_line += f" | A_gen={a_gen:.4f} gap={a_mem - a_gen:.4f}"
            if probe_acc_mem is not None:
                log_line += f" | probe_mem={probe_acc_mem:.3f}"
            if probe_acc_gen is not None:
                log_line += f" probe_gen={probe_acc_gen:.3f}"
            print(log_line)

            # Mirror the gate state to W&B when available
            try:
                import wandb

                if wandb.run is not None:
                    gate_metrics = {
                        "gate/a_mem": a_mem,
                        "gate/lambda": current_lambda,
                        "gate/memorization_saturated": float(
                            self.lambda_scheduler.is_memorization_saturated()
                        ),
                        "gate/epoch": state.epoch,
                        "gate/gate_metric": self.lambda_scheduler.gate_metric,
                        "gate/a_mem_effective": self.lambda_scheduler.current_a_mem,
                    }
                    if a_mem_lf is not None:
                        gate_metrics["gate/a_mem_lf"] = a_mem_lf
                    if entry.get("lf_token_acc") is not None:
                        gate_metrics["gate/lf_token_acc"] = entry["lf_token_acc"]
                    if a_gen is not None:
                        gate_metrics["gate/a_gen"] = a_gen
                        gate_metrics["gate/gap"] = a_mem - a_gen
                    for key in ("mem_contains_match", "mem_token_f1"):
                        if key in entry:
                            gate_metrics[f"gate/{key}"] = entry[key]
                    for key in ("gen_contains_match", "gen_token_f1"):
                        if key in entry:
                            gate_metrics[f"gate/{key}"] = entry[key]
                    if probe_acc_mem is not None:
                        gate_metrics["gate/probe_acc_mem"] = probe_acc_mem
                    if probe_acc_gen is not None:
                        gate_metrics["gate/probe_acc_gen"] = probe_acc_gen
                        gate_metrics["gate/probe_gap"] = probe_acc_mem - probe_acc_gen
                    wandb.log(gate_metrics)
                    # Sample tables make a 0% score diagnosable from the dashboard
                    log_sample_table("gate/samples_mem", mem_results or [])
                    if lf_results:
                        log_sample_table("gate/samples_mem_lf", lf_results)
                    if gen_results:
                        log_sample_table("gate/samples_gen", gen_results)
            except Exception:
                pass
        except Exception as e:
            print(f"[MEM GATE] evaluation failed at epoch {state.epoch}: {e}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"gate/eval_failed": 1.0, "gate/epoch": state.epoch})
            except Exception:
                pass
