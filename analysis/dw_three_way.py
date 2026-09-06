"""Three-way checkpoint ΔW analysis: baseline(early) vs best-validation vs final.

For every run directory on the MLPR outputs volume this script:
  1. inventories checkpoint-* dirs (adapter_model.safetensors + per-checkpoint
     trainer_state.json for eval_loss),
  2. selects three checkpoints: EARLIEST (baseline proxy), BEST (min eval_loss),
     FINAL (max step),
  3. for each selected checkpoint computes per-module effective update stats for
     dW = (alpha/r) * B @ A WITHOUT materializing dW (rank-r Gram trick:
     sigma_i(dW) = sqrt(eig(B^T B @ A A^T)), all r x r problems), and
  4. computes post-best drift per module: cosine(ΔW_best, ΔW_final), the
     Frobenius norm of the update increment, and relative growth.

Output: /vol/outputs/_collection/dw_three_way.json
  { "generated_at", "runs": { vol_dir: {
        n_checkpoints, checkpoints: [...],
        selected: {early/best/final: {dir, step, epoch, eval_loss}},
        dw_stats: {early/best/final: {module_key: {fro, spectral, nuclear,
                    stable_rank, effective_rank, top_sv}}},
        drift: {module_key: {cos_best_final, delta_norm_best_final,
                             rel_growth_best_final}} } } }

Usage:
  modal run MLPR/analysis/dw_three_way.py
"""

import json
import os
import re
import time
from pathlib import Path

import modal

app = modal.App("mlpr-dw-three-way")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "safetensors")
)
outputs_vol = modal.Volume.from_name("mlpr-outputs", create_if_missing=True)

CKPT_RE = re.compile(r"checkpoint-(\d+)$")


@app.function(image=image, volumes={"/vol/outputs": outputs_vol}, timeout=45 * 60)
def analyze():
    t_start = time.time()
    os.chdir("/vol/outputs")
    import numpy as np
    from safetensors.numpy import load_file as load_st

    def _gram_sv(B, A, alpha_over_r):
        """Singular values of (alpha/r)*B@A via the r x r Gram matrix."""
        # B: [out, r], A: [r, in] -> G = (B^T B)(A A^T) : r x r
        G = (B.T @ B) @ (A @ A.T)
        ev = np.linalg.eigvals(G).real
        ev = np.clip(ev, 0.0, None)
        sv = np.sqrt(ev)
        sv = np.sort(sv)[::-1]
        return sv * alpha_over_r

    def _dw_stats(tensors, r, alpha, layer_re):
        stats = {}
        groups = {}
        for k, v in tensors.items():
            m = layer_re.match(k)
            if not m:
                continue
            layer, mod, side = m.group(1), m.group(2), m.group(3)
            groups.setdefault((layer, mod), {})[side] = v.astype("float64")
        scale = float(alpha) / float(r)
        for (layer, mod), sides in sorted(groups.items()):
            if "A" not in sides or "B" not in sides:
                continue
            A, B = sides["A"], sides["B"]
            rr = A.shape[0] if A.shape[0] != r else r  # defensive: per-module r
            sv = _gram_sv(B, A, scale)
            fro = float(np.sqrt((sv ** 2).sum()))
            spec = float(sv[0]) if sv.size else 0.0
            nuc = float(sv.sum())
            p = sv / sv.sum() if sv.sum() > 0 else sv
            p = p[p > 0]
            eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
            stats[f"layers.{layer}.{mod}"] = {
                "fro": fro,
                "spectral": spec,
                "nuclear": nuc,
                "stable_rank": float(fro ** 2 / spec ** 2) if spec > 0 else 0.0,
                "effective_rank": eff,
                "top_sv": [round(float(x), 6) for x in sv[:16]],
            }
        return stats

    def _inner(B1, A1, B2, A2, scale):
        """<dW1, dW2> = trace(A1^T B1^T B2 A2) = trace((B1^T B2)(A2 A1^T))"""
        G = (B1.T @ B2) @ (A2 @ A1.T)
        return float(np.trace(G)) * scale * scale

    def _load_groups(tensors, layer_re):
        groups = {}
        for k, v in tensors.items():
            m = layer_re.match(k)
            if not m:
                continue
            groups.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = v.astype("float64")
        return groups

    LORA_KEY_RE = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\."
        r"(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.lora_([AB])\.weight"
    )
    out_runs = {}

    run_dirs = sorted(p for p in Path("/vol/outputs").iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    for run_dir in run_dirs:
        ckpts = []
        for cdir in sorted(run_dir.glob("checkpoint-*")):
            m = CKPT_RE.search(cdir.name)
            if not m or not (cdir / "adapter_model.safetensors").exists():
                continue
            step = int(m.group(1))
            info = {"dir": cdir.name, "step": step, "epoch": None, "eval_loss": None}
            ts = cdir / "trainer_state.json"
            if ts.exists():
                try:
                    st = json.loads(ts.read_text())
                    info["epoch"] = st.get("epoch")
                    # eval_loss closest to (<=) this checkpoint's step
                    best = None
                    for h in st.get("log_history", []):
                        if "eval_loss" in h and h.get("step", 0) <= step + 1:
                            best = h["eval_loss"]
                    info["eval_loss"] = best
                except Exception:
                    pass
            ckpts.append((step, cdir, info))
        if not ckpts:
            continue
        ckpts.sort(key=lambda x: x[0])
        print(f"[{run_dir.name}] {len(ckpts)} checkpoints "
              f"({ckpts[0][0]}..{ckpts[-1][0]})")

        # also merge root trainer_state eval curve for reference
        root_eval = []
        root_ts = run_dir / "trainer_state.json"
        if root_ts.exists():
            try:
                for h in json.loads(root_ts.read_text()).get("log_history", []):
                    if "eval_loss" in h:
                        root_eval.append({"step": h.get("step"),
                                          "epoch": h.get("epoch"),
                                          "eval_loss": h["eval_loss"]})
            except Exception:
                pass

        # ---- three-way selection --------------------------------------
        early_step, early_dir, early_info = ckpts[0]
        fin_step, fin_dir, fin_info = ckpts[-1]
        evald = [(s, d, i) for s, d, i in ckpts if i["eval_loss"] is not None]
        best_sel = None
        if evald:
            b = min(evald, key=lambda x: x[2]["eval_loss"])
            best_sel = {"dir": b[1].name, "step": b[0], "epoch": b[2]["epoch"],
                        "eval_loss": b[2]["eval_loss"]}
        selected = {
            "early": {"dir": early_dir.name, "step": early_step,
                      "epoch": early_info["epoch"], "eval_loss": early_info["eval_loss"]},
            "best": best_sel,
            "final": {"dir": fin_dir.name, "step": fin_step,
                      "epoch": fin_info["epoch"], "eval_loss": fin_info["eval_loss"]},
        }

        def _load_ckpt(cdir):
            cfg_p = cdir / "adapter_config.json"
            r, alpha = 8, 16.0
            if cfg_p.exists():
                try:
                    cfg = json.loads(cfg_p.read_text())
                    r = int(cfg.get("r", r))
                    alpha = float(cfg.get("lora_alpha", alpha))
                except Exception:
                    pass
            tensors = load_st(str(cdir / "adapter_model.safetensors"))
            return tensors, r, alpha

        dw_stats = {}
        tensors_e, r_e, a_e = _load_ckpt(early_dir)
        dw_stats["early"] = _dw_stats(tensors_e, r_e, a_e, LORA_KEY_RE)
        tensors_f, r_f, a_f = _load_ckpt(fin_dir)
        dw_stats["final"] = _dw_stats(tensors_f, r_f, a_f, LORA_KEY_RE)
        tensors_b = None
        if best_sel is not None:
            bdir = run_dir / best_sel["dir"]
            tensors_b, r_b, a_b = _load_ckpt(bdir)
            dw_stats["best"] = _dw_stats(tensors_b, r_b, a_b, LORA_KEY_RE)

        # ---- post-best drift (early used if no best-eval ckpt) ---------
        drift = {}
        ref_tensors, ref_r, ref_a = ((tensors_b, r_b, a_b) if tensors_b is not None
                                     else (tensors_e, r_e, a_e))
        ref_key = "best" if tensors_b is not None else "early"
        ge = _load_groups(ref_tensors, LORA_KEY_RE)
        gf = _load_groups(tensors_f, LORA_KEY_RE)
        scale_r = float(ref_a) / float(ref_r)
        scale_f = float(a_f) / float(r_f)
        for key, sides_e in sorted(ge.items()):
            sides_f = gf.get(key)
            if not sides_f or "A" not in sides_e or "B" not in sides_e:
                continue
            Ae, Be = sides_e["A"], sides_e["B"]
            Af, Bf = sides_f["A"], sides_f["B"]
            fro_e = float(np.sqrt(((scale_r * Be @ Ae) ** 2).sum()))
            fro_f = float(np.sqrt(((scale_f * Bf @ Af) ** 2).sum()))
            ip = _inner(Be, Ae, Bf, Af, scale_r * scale_f)
            cos = ip / (fro_e * fro_f) if fro_e > 0 and fro_f > 0 else None
            dnorm = float(np.sqrt(max(fro_e ** 2 + fro_f ** 2 - 2 * ip, 0.0)))
            drift[f"layers.{key[0]}.{key[1]}"] = {
                "ref": ref_key,
                "cos_ref_final": round(cos, 6) if cos is not None else None,
                "delta_norm_ref_final": round(dnorm, 6),
                "rel_growth_ref_final": round((fro_f - fro_e) / fro_e, 6) if fro_e > 0 else None,
                "fro_ref": round(fro_e, 6),
                "fro_final": round(fro_f, 6),
            }

        out_runs[run_dir.name] = {
            "n_checkpoints": len(ckpts),
            "checkpoints": [i for _, _, i in ckpts],
            "root_eval_curve": root_eval,
            "selected": selected,
            "dw_stats": dw_stats,
            "drift": drift,
        }
        print(f"[{run_dir.name}] done: early={selected['early']['dir']} "
              f"best={selected['best']['dir'] if selected['best'] else None} "
              f"final={selected['final']['dir']} modules={len(drift)}")

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "runs": out_runs}
    out_path = Path("/vol/outputs/_collection/dw_three_way.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    outputs_vol.commit()
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, "
          f"{time.time() - t_start:.0f}s)")


@app.local_entrypoint()
def main():
    analyze.remote()
