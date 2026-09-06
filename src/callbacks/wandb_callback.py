"""Weights & Biases callback for matrix logging and checkpoint locking (v3).

Why v3: the matrix-collection job used to exceed its time limit. Root causes
were (a) per-parameter `.detach().cpu().float().numpy()` transfers -- hundreds
of device syncs per epoch on a 7B LoRA stack -- and (b) W&B histograms built
from MILLIONS of raw floats. v3 fixes both and upgrades the science:

Cheap (every epoch, GPU-side, ONE host sync total):
  - per-layer LoRA A and B L2 norms (fixed keys `lora/A_norm/<module>`)
  - run aggregates (mean/max over A and B)
  - EFFECTIVE UPDATE ANALYSIS of dW = (alpha/r) * B @ A per module:
      spectral norm, stable rank, effective rank (entropy of the singular
      spectrum), and nuclear norm. Computed WITHOUT materializing dW:
      the nonzero singular values of B@A are the square roots of the
      eigenvalues of the r x r matrix  M^{1/2} B^T B M^{1/2},  M = A A^T
      (r = 16 -> a 16x16 eigh per module; negligible next to a train step).
      Fixed keys `lora/dW_spectral/<module>` etc. -- this is the paper's
      "effective LoRA update" figure 2 data, logged where the training runs.
  - probe head: W/bias norms, W spectral norm + effective rank, and the
    per-epoch probe GRADIENT norm (hard evidence the probe is training
    after the probe-warm-up fix).

Heavy (every `matrix_log_every_n_epochs` epochs, subsampled + time-budgeted):
  - histograms of A/B values (subsampled to <=20k values -- building a W&B
    histogram from full populations was the timeout culprit)
  - probe W heatmap, largest-B heatmap
  - checkpoint artifact locking

Nothing in this callback may ever kill a training run: every hook is
failure-safe, and heavy sections respect a soft wall-clock budget.
"""

import os
import time
from typing import Optional

import torch  # noqa: F401  -- FIX: used by _dW_singular_stats / svdvals / stack;
#               the missing import made every per-epoch LoRA logging call die
#               with "name 'torch' is not defined" (caught + printed, so the
#               run survived but ALL lora/* and probe spectral metrics vanished
#               from W&B -- exactly the telemetry the paper figures need).
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

try:
    import wandb
    import numpy as np
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# a W&B histogram gets at most this many values (full populations timed out)
_HIST_MAX_VALUES = 20_000
# soft wall-clock budget for the heavy sections, per epoch-end call
_HEAVY_TIME_BUDGET_S = 90.0


def _subsample(arr, max_values: int = _HIST_MAX_VALUES):
    """Flat subsample of an array for histogram logging."""
    flat = arr.reshape(-1)
    if flat.size <= max_values:
        return flat
    idx = np.random.default_rng(0).choice(flat.size, size=max_values, replace=False)
    return flat[idx]


def _dW_singular_stats(B, A, scale: float):
    """Spectral stats of dW = scale * B @ A without materializing dW.

    B: (out, r), A: (r, in). Returns (spectral_norm, stable_rank,
    effective_rank, nuclear_norm). The nonzero eigenvalues of B M B^T
    (M = A A^T, r x r) equal those of M^{1/2} B^T B M^{1/2}; a symmetric
    r x r eigh therefore yields the full nonzero singular spectrum of dW.
    """
    r = A.shape[0]
    Af = A.detach().float()
    Bf = B.detach().float()
    M = Af @ Af.transpose(0, 1)                    # (r, r) PSD
    # M^{1/2} via eigh (PSD -> clip negatives from fp noise)
    m_vals, m_vecs = torch.linalg.eigh(M)
    m_half = (m_vecs * torch.sqrt(m_vals.clamp(min=0.0))) @ m_vecs.transpose(0, 1)
    G = m_half @ (Bf.transpose(0, 1) @ Bf) @ m_half  # (r, r) PSD
    G = 0.5 * (G + G.transpose(0, 1))                # kill fp asymmetry
    eigvals = torch.linalg.eigvalsh(G)
    eigvals = eigvals.clamp(min=0.0)
    sing = scale * torch.sqrt(eigvals)               # singular values of dW

    spec = float(sing.max().item()) if sing.numel() else 0.0
    nuclear = float(sing.sum().item())
    energy = float((sing ** 2).sum().item())
    stable = energy / (spec ** 2) if spec > 0 else 0.0
    # effective rank: exp(entropy) of the energy distribution (Roy & Vetterli)
    p = (sing ** 2)
    p = p / p.sum().clamp(min=1e-12)
    p_nz = p[p > 1e-12]
    ent = float(-(p_nz * torch.log(p_nz)).sum().item()) if p_nz.numel() else 0.0
    erank = float(torch.exp(torch.tensor(ent)).item()) if p_nz.numel() else 0.0
    return spec, stable, erank, nuclear


class WnBMatrixLockCallback(TrainerCallback):
    """GPU-fast matrix/scanline logging + checkpoint artifact locking."""

    def __init__(
        self,
        log_probe_matrix: bool = True,
        log_lora_matrices: bool = True,
        lock_checkpoints: bool = True,
        l_star: int = 14,
        matrix_log_every_n_epochs: int = 5,
    ):
        self.log_probe_matrix = log_probe_matrix
        self.log_lora_matrices = log_lora_matrices
        self.lock_checkpoints = lock_checkpoints
        self.l_star = l_star
        self.matrix_log_every_n_epochs = max(1, matrix_log_every_n_epochs)
        self.logged_artifacts = []
        self._lora_scale = None  # alpha / r, resolved lazily

    def _is_wandb_initialized(self) -> bool:
        return WANDB_AVAILABLE and wandb.run is not None

    # ------------------------------------------------------------------ events

    def on_epoch_end(self, args, state, control, model=None, **kwargs) -> None:
        if not self._is_wandb_initialized() or model is None:
            return

        ep_no = int(round(state.epoch))
        heavy = (ep_no % self.matrix_log_every_n_epochs == 0)
        if heavy:
            self._heavy_t0 = time.time()

        if self.log_probe_matrix:
            self._log_probe_matrix(model, ep_no, heavy)
        if self.log_lora_matrices:
            self._log_lora_matrices(model, ep_no, heavy)
        if self.lock_checkpoints and heavy:
            self._lock_checkpoint(ep_no, args.output_dir)

    # ------------------------------------------------------------------ probe

    def _get_probe_head(self, model):
        probe_head = getattr(model, "probe_head", None)
        if probe_head is None and hasattr(model, "module"):
            probe_head = getattr(model.module, "probe_head", None)
        return probe_head

    def _log_probe_matrix(self, model, ep_no: int, heavy: bool) -> None:
        try:
            probe_head = self._get_probe_head(model)
            if probe_head is None:
                if ep_no <= 1:
                    print("[W&B] Probe head not found, skipping probe matrix logging")
                return

            W = probe_head.weight.detach()
            metrics = {
                "probe/W_norm_l2": float(W.float().norm().item()),
                "probe/W_std": float(W.float().std().item()),
                "probe/W_mean_abs": float(W.float().abs().mean().item()),
            }
            # spectral stats of the (small) probe matrix itself
            try:
                sv = torch.linalg.svdvals(W.detach().float())
                metrics["probe/W_spectral_norm"] = float(sv.max().item())
                p = sv ** 2
                p = p / p.sum().clamp(min=1e-12)
                p_nz = p[p > 1e-12]
                if p_nz.numel():
                    ent = float(-(p_nz * torch.log(p_nz)).sum().item())
                    metrics["probe/W_effective_rank"] = float(torch.exp(torch.tensor(ent)).item())
            except Exception:
                pass

            # per-epoch gradient norm: hard evidence the probe is training
            # (was exactly 0 in the 50-epoch runs while lambda stayed 0)
            grad_sq = 0.0
            for p in probe_head.parameters():
                if p.grad is not None:
                    grad_sq += float(p.grad.float().norm().item() ** 2)
            metrics["probe/grad_norm_l2"] = float(grad_sq ** 0.5)

            bias = getattr(probe_head, "bias", None)
            if bias is not None:
                metrics["probe/b_norm_l2"] = float(bias.detach().float().norm().item())
                metrics["probe/b_std"] = float(bias.detach().float().std().item())

            if heavy and time.time() - getattr(self, "_heavy_t0", time.time()) < _HEAVY_TIME_BUDGET_S:
                W_np = W.detach().cpu().float().numpy()  # single small transfer
                metrics["probe/W_histogram"] = wandb.Histogram(_subsample(W_np))
                if bias is not None:
                    metrics["probe/b_histogram"] = wandb.Histogram(
                        _subsample(bias.detach().cpu().float().numpy()))
                metrics["probe/W_heatmap"] = self._heatmap_image(
                    W_np, f"probe W_p @ l*={self.l_star} (epoch {ep_no})", max_rows=160)

            wandb.log(metrics)
            if heavy:
                print(f"[W&B] probe matrix panels logged (epoch {ep_no})")
        except Exception as e:
            print(f"[W&B] Error logging probe matrix: {e}")

    # ------------------------------------------------------------------ lora

    def _resolve_lora_scale(self, model) -> float:
        """alpha / r from the peft config (fallback 1.0 -> relative stats)."""
        if self._lora_scale is not None:
            return self._lora_scale
        try:
            cfg = getattr(model, "peft_config", None)
            if isinstance(cfg, dict) and "default" in cfg:
                c = cfg["default"]
                self._lora_scale = float(c.lora_alpha) / float(c.r)
            else:
                self._lora_scale = 1.0
        except Exception:
            self._lora_scale = 1.0
        return self._lora_scale

    def _log_lora_matrices(self, model, ep_no: int, heavy: bool) -> None:
        """Per-epoch GPU-side norms + dW effective-update stats; heavy extras."""
        try:
            from peft import PeftModel

            if not isinstance(model, PeftModel):
                if ep_no <= 1:
                    print("[W&B] Model is not a PeftModel, skipping LoRA matrix logging")
                return

            scale = self._resolve_lora_scale(model)

            a_norms, b_norms = {}, {}       # simple name -> GPU scalar tensor
            a_by_mod, b_by_mod = {}, {}     # module key -> (name, tensor)
            for name, param in model.named_parameters():
                if not param.requires_grad or "lora_" not in name:
                    continue
                simple = (name.replace("base_model.model.", "")
                              .replace(".default.", "."))
                key = simple.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
                nrm = param.detach().float().norm()
                if "lora_A" in name:
                    a_norms[simple] = nrm
                    a_by_mod[key] = (simple, param)
                elif "lora_B" in name:
                    b_norms[simple] = nrm
                    b_by_mod[key] = (simple, param)

            if not a_norms and not b_norms:
                return

            # ONE host sync for every norm in the pass
            names = list(a_norms) + list(b_norms)
            vals = torch.stack([a_norms[n] for n in list(a_norms)]
                               + [b_norms[n] for n in list(b_norms)]).cpu().tolist()
            a_vals = dict(zip(list(a_norms), vals[:len(a_norms)]))
            b_vals = dict(zip(list(b_norms), vals[len(a_norms):]))

            metrics = {}
            if a_vals:
                metrics["lora/A_norm_mean"] = float(np.mean(list(a_vals.values())))
                metrics["lora/A_norm_max"] = float(np.max(list(a_vals.values())))
            if b_vals:
                metrics["lora/B_norm_mean"] = float(np.mean(list(b_vals.values())))
                metrics["lora/B_norm_max"] = float(np.max(list(b_vals.values())))
            for n, v in a_vals.items():
                metrics[f"lora/A_norm/{n}"] = v
            for n, v in b_vals.items():
                metrics[f"lora/B_norm/{n}"] = v

            # effective dW = (alpha/r) B A update statistics per module
            dW_spec, dW_stable, dW_erank, dW_nuc = {}, {}, {}, {}
            for key, (_, b_param) in b_by_mod.items():
                if key not in a_by_mod:
                    continue
                try:
                    a_param = a_by_mod[key][1]
                    spec, stable, erank, nuclear = _dW_singular_stats(
                        b_param, a_param, scale)
                    short = _short_module(key)
                    dW_spec[short] = spec
                    dW_stable[short] = stable
                    dW_erank[short] = erank
                    dW_nuc[short] = nuclear
                except Exception:
                    continue
            for d, prefix in ((dW_spec, "lora/dW_spectral"),
                              (dW_stable, "lora/dW_stable_rank"),
                              (dW_erank, "lora/dW_effective_rank"),
                              (dW_nuc, "lora/dW_nuclear")):
                for k, v in d.items():
                    metrics[f"{prefix}/{k}"] = v
            if dW_spec:
                metrics["lora/dW_spectral_mean"] = float(np.mean(list(dW_spec.values())))
                metrics["lora/dW_spectral_max"] = float(np.max(list(dW_spec.values())))
                metrics["lora/dW_erank_mean"] = float(np.mean(list(dW_erank.values())))

            if heavy and time.time() - getattr(self, "_heavy_t0", time.time()) < _HEAVY_TIME_BUDGET_S:
                # subsampled histograms + biggest-B heatmap (bounded transfers)
                b_chunks = []
                for key, (_, b_param) in list(b_by_mod.items()):
                    b_chunks.append(b_param.detach().float().reshape(-1).cpu().numpy())
                    if sum(c.size for c in b_chunks) > 4_000_000:
                        break
                if b_chunks:
                    metrics["lora/B_histogram_all"] = wandb.Histogram(
                        _subsample(np.concatenate(b_chunks)))
                if a_by_mod:
                    a_first = next(iter(a_by_mod.values()))[1].detach().float().cpu().numpy()
                    metrics["lora/A_histogram_sample"] = wandb.Histogram(_subsample(a_first))
                if b_vals:
                    biggest = max(b_vals, key=b_vals.get)
                    for name, param in model.named_parameters():
                        simple = (name.replace("base_model.model.", "")
                                      .replace(".default.", "."))
                        if simple == biggest:
                            metrics["lora/B_heatmap_sample"] = self._heatmap_image(
                                param.detach().cpu().float().numpy(),
                                f"{biggest} (epoch {ep_no})", max_rows=160)
                            break

            wandb.log(metrics)
            if heavy:
                print(f"[W&B] LoRA matrix panels logged (epoch {ep_no}, "
                      f"{len(b_vals)} B + {len(a_vals)} A + {len(dW_spec)} dW)")
        except Exception as e:
            print(f"[W&B] Error logging LoRA matrices: {e}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _heatmap_image(matrix, title: str, max_rows: int = 160):
        """Render a matrix as a W&B image (downsampled) without bloating logs."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = matrix
            if m.shape[0] > max_rows:
                idx = np.linspace(0, m.shape[0] - 1, max_rows, dtype=int)
                m = m[idx]
            if m.shape[1] > max_rows:
                idx = np.linspace(0, m.shape[1] - 1, max_rows, dtype=int)
                m = m[:, idx]

            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im = ax.imshow(m, aspect="auto", cmap="viridis")
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            image = wandb.Image(fig)
            plt.close(fig)
            return image
        except Exception as e:
            print(f"[W&B] heatmap skipped: {e}")
            return None

    def _lock_checkpoint(self, ep_no: int, output_dir: str) -> None:
        """Lock the current checkpoint files as a W&B artifact (every N epochs)."""
        try:
            artifact_name = f"mlpr_model_epoch_{ep_no:03d}"
            artifact = wandb.Artifact(artifact_name, type="model")

            if os.path.exists(output_dir):
                for file in sorted(os.listdir(output_dir)):
                    if file.endswith((".bin", ".safetensors", ".json", ".pt")):
                        artifact.add_file(os.path.join(output_dir, file))

            wandb.log_artifact(artifact)
            self.logged_artifacts.append(artifact_name)
            print(f"[W&B] Locked checkpoint artifact: {artifact_name}")
        except Exception as e:
            print(f"[W&B] Error locking checkpoint: {e}")

    def on_train_end(self, args, state, control, model=None, **kwargs) -> None:
        if not self._is_wandb_initialized():
            return
        if self.lock_checkpoints and model is not None:
            try:
                final_artifact = wandb.Artifact("mlpr_model_final", type="model")
                if os.path.exists(args.output_dir):
                    final_artifact.add_dir(args.output_dir)
                wandb.log_artifact(final_artifact)
                print("[W&B] Final model artifact locked")
            except Exception as e:
                print(f"[W&B] Error creating final artifact: {e}")


def _short_module(key: str) -> str:
    """'...model.layers.21.mlp.gate_proj' -> 'layers.21.mlp.gate_proj'."""
    idx = key.find("layers.")
    return key[idx:] if idx != -1 else key
