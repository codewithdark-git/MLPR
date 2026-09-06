"""Publication figures for the MLPR paper: gates, probe, and LoRA update dynamics.

Turns a W&B collection JSON (legacy scripts/modal_wandb_collect.py format OR the
v2 analysis/wb_collect.py format) into the paper's diagnostic figures:

  fig1_validation_reversal.png  train vs eval loss; the early eval minimum and
                                the late rise annotated per run.
  fig2_lora_growth_heatmap.png  layer x module heatmap of LoRA-B norm growth
                                AFTER the validation minimum (co-occurrence of
                                the upper-layer MLP gate/up growth with the
                                generalization reversal).
  fig3_gate_timeline.png        per-epoch overlay: losses, A_mem/A_gen/gap,
                                lambda, LR multiplier, controller actions --
                                makes a controller mismatch instantly visible.
  fig4_probe_health.png         probe decodability (A_probe mem vs gen) when
                                present, plus probe W norm / grad norm history
                                (frozen-probe evidence).
  fig5_update_spectrum.png      module-family LoRA-B norm trajectories across
                                checkpoints + final layerwise profile.

Usage:
  python analysis/make_figures.py --data wandb_full_collection.json --out figs/
  python analysis/make_figures.py --data wandb_collection.json \
      --trainer-state half50_trainer_state.json \
      --adaptive-history adaptest_adaptive_history.json \
      --gate-history adaptest_gate_history.json --out figs/

Every figure degrades gracefully: if the data for it is absent, it is skipped
with a printed reason instead of crashing.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------- style consts
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})
# Okabe-Ito colorblind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#111111"}
MODULE_ORDER = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj"]
MODULE_COLORS = {"mlp.gate_proj": C["red"], "mlp.up_proj": C["orange"],
                 "mlp.down_proj": C["yellow"], "self_attn.q_proj": C["blue"],
                 "self_attn.k_proj": C["sky"], "self_attn.v_proj": C["purple"],
                 "self_attn.o_proj": C["green"]}


# ------------------------------------------------------------------ data load

def load_json(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [skip] data file not found: {p}")
        return None
    return json.loads(p.read_text())


def _clean_row(row):
    """Drop None/NaN cells -- legacy wandb history rows carry the full schema
    with None for keys logged at other steps."""
    clean = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        clean[k] = v
    return clean


def normalize_run(entry):
    """Unify legacy (runs[].history) and v2 (history_epoch/history_step) formats."""
    out = {
        "name": entry.get("name", "?"),
        "state": entry.get("state", "?"),
        "is_matrix_collection": entry.get("is_matrix_collection", False),
        "summary": entry.get("summary", {}) or {},
        "hist_step": [_clean_row(r) for r in (entry.get("history_step") or [])],
        "hist_epoch": [_clean_row(r) for r in (entry.get("history_epoch") or [])],
        "gate_history": entry.get("gate_history") or [],
        "adaptive_history": entry.get("adaptive_history") or [],
        "trainer_state": entry.get("trainer_state") or {},
    }
    if "history" in entry:  # legacy single-list format
        gate, adap, step = [], [], []
        for raw in entry["history"]:
            if not isinstance(raw, dict):
                continue
            row = _clean_row(raw)
            if not row:
                continue
            keys = set(row.keys())
            # NOTE: legacy wandb rows can carry key groups from several log
            # calls at once -- a row may belong to multiple lists.
            if any(str(k).startswith("gate/") for k in keys):
                gate.append(row)
            if any(str(k).startswith("adaptive/") for k in keys):
                adap.append(row)
            if any(k in keys for k in ("loss", "train/loss", "eval_loss", "eval/loss")):
                step.append(row)
        out["gate_history"] = out["gate_history"] or gate
        out["adaptive_history"] = out["adaptive_history"] or adap
        out["hist_step"] = out["hist_step"] or step
        out["hist_epoch"] = out["hist_epoch"] or [
            _clean_row(row) for row in entry["history"]
            if isinstance(row, dict) and any(
                str(k).startswith(("lora/", "probe/", "final_", "summary/"))
                for k in row.keys())]
    return out


def load_collection(args):
    data = load_json(args.data)
    if data is None:
        sys.exit("No collection JSON -- pass --data /path/to/wandb_full_collection.json")
    runs = {rid: normalize_run(e) for rid, e in (data.get("runs") or {}).items()}
    # attach extra loose files to the run they best match (legacy workflow)
    if args.trainer_state:
        ts = load_json(args.trainer_state)
        if ts:
            best = _pick_run(runs, prefer_contains=("0.5b-50ep", "0.5b"),
                             exclude_contains=("smoke", "adapt"))
            if best:
                runs[best]["trainer_state"] = ts
                print(f"  attached trainer_state -> {best}")
    if args.gate_history or args.adaptive_history:
        gh = load_json(args.gate_history) or []
        ah = load_json(args.adaptive_history) or []
        name = args.pseudo_name or "adapt-test (0.5B, 6 epochs)"
        runs["pseudo-adapt-test"] = {
            "name": name, "state": "attached", "is_matrix_collection": False,
            "summary": {}, "hist_step": [], "hist_epoch": [],
            "gate_history": gh if isinstance(gh, list) else [],
            "adaptive_history": ah if isinstance(ah, list) else [],
            "trainer_state": {},
        }
        print(f"  attached gate/adaptive history as pseudo-run '{name}'")
    return {"raw": data, "runs": runs,
            "matrix_stats": data.get("matrix_stats") or {}}


def _pick_run(runs, prefer_contains=(), exclude_contains=(),
              min_gate_points=0, min_adaptive_points=0):
    """Choose the run matching `prefer_contains` with the richest telemetry."""

    def richness(r):
        ev = sum(1 for row in r["hist_step"] if "eval_loss" in row or "eval/loss" in row)
        return (len(r["adaptive_history"]) >= min_adaptive_points,
                len(r["gate_history"]) >= min_gate_points,
                ev, len(r["gate_history"]), len(r["hist_step"]))

    cands = [rid for rid, r in runs.items()
             if not r["is_matrix_collection"]
             and len(r["gate_history"]) >= min_gate_points
             and len(r["adaptive_history"]) >= min_adaptive_points]
    for pat in prefer_contains:
        matches = [rid for rid in cands
                   if pat in runs[rid]["name"].lower() or pat in rid.lower()]
        if matches:
            cands = matches
            break
    for pat in exclude_contains:
        if len(cands) > 1:
            filtered = [rid for rid in cands
                        if pat not in runs[rid]["name"].lower()]
            if filtered:
                cands = filtered
    if not cands:
        return None
    return max(cands, key=lambda rid: richness(runs[rid]))


# ------------------------------------------------------------------- curves

def _clean_xy(pairs):
    xs, ys = [], []
    for x, y in pairs:
        if x is None or y is None:
            continue
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def loss_curves(run, x_unit="step"):
    """(x_train, y_train, x_eval, y_eval). x_unit: 'step' or 'epoch' (HF logs
    train/epoch on every logging/eval row -- the only reliable epoch source)."""
    tr, ev = [], []
    for row in run["hist_step"]:
        if x_unit == "epoch":
            x = row.get("train/epoch", row.get("epoch", row.get("_step")))
        else:
            x = row.get("_step")
            if x is None:
                x = row.get("epoch")
        if x is None:
            continue
        for tr_key in ("train/loss", "loss"):
            if tr_key in row and "eval_loss" not in row and "eval/loss" not in row:
                tr.append((x, row[tr_key]))
        for ev_key in ("eval/loss", "eval_loss"):
            if ev_key in row:
                ev.append((x, row[ev_key]))
    # trainer_state fallback (its keys are the raw HF Trainer ones); load
    # each series INDEPENDENTLY so a W&B train curve still gets the file's eval
    # curve (the legacy W&B pull missed eval/loss entirely)
    ts = run["trainer_state"].get("log_history", []) if run["trainer_state"] else []
    if not tr and ts:
        for row in ts:
            if "loss" in row and "eval_loss" not in row:
                x = row.get("epoch") if x_unit == "epoch" else row.get("step")
                tr.append((x, row.get("loss")))
    if not ev and ts:
        for row in ts:
            if "eval_loss" in row:
                x = row.get("epoch") if x_unit == "epoch" else row.get("step")
                ev.append((x, row.get("eval_loss")))
    return (*_clean_xy(tr), *_clean_xy(ev))


def eval_min(losses):
    xtr, ytr, xev, yev = losses
    if len(xev) == 0:
        return None
    i = int(np.argmin(yev))
    return {"step": xev[i], "loss": yev[i], "final": yev[-1],
            "rise_pct": (yev[-1] / yev[i] - 1) * 100 if yev[i] else 0.0}


def gate_curve(run):
    """Per-epoch dict list: epoch, a_mem, a_gen, gap, lambda, probe_acc_*, metrics."""
    # normalize both file-style entries and W&B gate/* rows into one schema
    keymap = {"a_mem": ("a_mem", "gate/a_mem"), "a_gen": ("a_gen", "gate/a_gen"),
              "gap": ("gap", "gate/gap"), "lambda": ("lambda", "gate/lambda"),
              "probe_acc_mem": ("probe_acc_mem", "gate/probe_acc_mem"),
              "probe_acc_gen": ("probe_acc_gen", "gate/probe_acc_gen"),
              "mem_contains": ("mem_contains_match", "gate/mem_contains_match"),
              "mem_f1": ("mem_token_f1", "gate/mem_token_f1"),
              "gen_contains": ("gen_contains_match", "gate/gen_contains_match"),
              "gen_f1": ("gen_token_f1", "gate/gen_token_f1")}
    rows = run["gate_history"]
    if rows:
        out = []
        for i, e in enumerate(rows):
            if not isinstance(e, dict):
                continue
            entry = {"epoch": e.get("epoch", e.get("gate/epoch"))}
            for dst, sources in keymap.items():
                entry[dst] = next((e.get(s) for s in sources if e.get(s) is not None), None)
            # keep only rows with at least one real measurement
            if any(v is not None for k, v in entry.items() if k != "epoch"):
                entry["epoch"] = entry["epoch"] if entry["epoch"] is not None else i + 1
                out.append(entry)
        return out
    # W&B history fallback (gate/* keys) -- only rows that ACTUALLY carry a
    # gate/ metric, and only slots with at least one real measurement
    merged = {}
    for row in run["hist_epoch"] + run["hist_step"]:
        if not any(str(k).startswith("gate/") for k in row):
            continue
        ep = row.get("gate/epoch", row.get("epoch", row.get("_step")))
        if ep is None:
            continue
        slot = merged.setdefault(int(round(float(ep))), {})
        for dst, sources in keymap.items():
            val = next((row.get(s) for s in sources if row.get(s) is not None), None)
            if val is not None:
                slot[dst] = val
        slot.setdefault("epoch", ep)
    return [merged[k] for k in sorted(merged)
            if any(v is not None for kk, v in merged[k].items() if kk != "epoch")]


def adaptive_curve(run):
    rows = run["adaptive_history"]
    if rows:
        return rows
    merged = {}
    for row in run["hist_epoch"] + run["hist_step"]:
        if not any(str(k).startswith("adaptive/") for k in row):
            continue
        ep = row.get("adaptive/epoch", row.get("epoch", row.get("_step")))
        if ep is None:
            continue
        slot = merged.setdefault(int(round(float(ep))), {})
        for src, dst in (("adaptive/mean_train_loss", "mean_train_loss"),
                         ("adaptive/eval_loss", "eval_loss"),
                         ("adaptive/lr_mult", "lr_mult"),
                         ("adaptive/action", "action"),
                         ("adaptive/a_mem", "a_mem"),
                         ("adaptive/lambda", "lambda"),
                         ("adaptive/boosts", "boosts"),
                         ("adaptive/decays", "decays")):
            if src in row and row[src] is not None:
                slot[dst] = row[src]
        slot.setdefault("epoch", ep)
    return [merged[k] for k in sorted(merged)
            if any(v is not None for kk, v in merged[k].items() if kk != "epoch")]


def _step_to_epoch_fn(run):
    """Build (gs_list, epoch_list) from rows that carry train/global_step plus
    an epoch field (gate/epoch or adaptive/epoch) for interpolation."""
    gs, ep = [], []
    for row in run["hist_epoch"] + run["hist_step"]:
        g = row.get("train/global_step")
        for ekey in ("gate/epoch", "adaptive/epoch", "train/epoch"):
            if ekey in row and g is not None:
                gs.append(float(g))
                ep.append(float(row[ekey]))
                break
    if len(gs) < 2:
        return None
    pairs = sorted(zip(gs, ep))
    gs = [p[0] for p in pairs]
    ep = [p[1] for p in pairs]
    # collapse duplicate gs (keep last)
    dg, de = {}, {}
    for a, b in pairs:
        dg[a] = a
        de[a] = b
    gs = sorted(dg)
    ep = [de[a] for a in gs]
    return gs, ep


def probe_series(run, step2ep=None):
    """(epochs, W_norm, grad_norm, acc_mem, acc_gen, b_norm) per-epoch telemetry."""
    wn, gn, am, ag, bn, ep = [], [], [], [], [], []
    for row in run["hist_epoch"] + run["hist_step"]:
        w = row.get("probe/W_norm_l2")
        g = row.get("probe/grad_norm_l2")
        b = row.get("probe/b_norm_l2")
        a_m = row.get("gate/probe_acc_mem")
        a_g = row.get("gate/probe_acc_gen")
        gs = row.get("train/global_step")
        if step2ep and gs is not None:
            x = float(np.interp(float(gs), step2ep[0], step2ep[1]))
        else:
            x = row.get("gate/epoch", row.get("epoch", row.get("_step")))
        if w is None and g is None and a_m is None and a_g is None and b is None:
            continue
        ep.append(x); wn.append(w); gn.append(g); am.append(a_m); ag.append(a_g)
        bn.append(b)
    return (*_clean_xy(zip(ep, wn)), *_clean_xy(zip(ep, gn)),
            *_clean_xy(zip(ep, am)), *_clean_xy(zip(ep, ag)),
            *_clean_xy(zip(ep, bn)))


# -------------------------------------------------------------- matrix stats

_MAT_RE = re.compile(r"layers\.(\d+)\.(mlp|self_attn)\.(gate_proj|up_proj|down_proj|"
                     r"q_proj|k_proj|v_proj|o_proj)\.lora_B\.weight")


def module_of(key):
    m = _MAT_RE.search(key)
    return f"{m.group(2)}.{m.group(3)}" if m else None


def layer_of(key):
    m = _MAT_RE.search(key)
    return int(m.group(1)) if m else None


def bnorm_panel(matrix_stats, run_dir):
    """{checkpoint_step: {(layer, module): l2}} for one run directory."""
    panel = {}
    for rel, mats in matrix_stats.items():
        if not rel.startswith(run_dir + "/") or "adapter_model" not in rel:
            continue
        m = re.search(r"checkpoint-(\d+)", rel)
        step = int(m.group(1)) if m else 10**9  # top-level adapter = final
        cells = {}
        for key, st in mats.items():
            lay, mod = layer_of(key), module_of(key)
            if lay is not None and mod and "lora_B" in key:
                cells[(lay, mod)] = st["l2"]
        if cells:
            panel[step] = cells
    return dict(sorted(panel.items()))


def nearest_step(panel: dict, target: float):
    steps = [s for s in panel if s != 10**9]
    if not steps:
        return None
    if target is None:
        return min(steps)
    return min(steps, key=lambda s: abs(s - target))


# -------------------------------------------------------------------- fig 1

def fig1_validation_reversal(runs, out):
    cands = []
    for rid, run in runs.items():
        if run["is_matrix_collection"]:
            continue
        losses = loss_curves(run)
        xtr, ytr, xev, yev = losses
        # need REAL curves (>=4 clean points) to earn a panel
        if len(xtr) + len(xev) < 4:
            continue
        cands.append((rid, run, losses))
    if not cands:
        print("fig1: no runs with loss curves -- skipped")
        return
    cands.sort(key=lambda x: len(x[2][0]) + len(x[2][3]), reverse=True)
    show = cands[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(4.4 * len(show), 3.4),
                             constrained_layout=True, squeeze=False)
    for ax, (rid, run, losses) in zip(axes[0], show):
        xtr, ytr, xev, yev = losses
        if len(xtr):
            ax2 = ax.twinx()
            ax2.plot(xtr, ytr, color=C["sky"], lw=0.9, alpha=0.85, label="train loss (right)")
            ax2.set_ylabel("train loss", color=C["sky"])
            ax2.tick_params(axis="y", colors=C["sky"])
            ax2.spines["right"].set_visible(True)
            ax2.grid(False)
        if len(xev):
            ax.plot(xev, yev, color=C["blue"], lw=1.8, marker="o", ms=3.5,
                    label="eval loss (D_gen)")
            m = eval_min(losses)
            ax.scatter([m["step"]], [m["loss"]], zorder=5, color=C["red"], s=42,
                       marker="v", label=f"eval min {m['loss']:.3f}")
            ax.annotate(f"min @{int(m['step'])}\n+{m['rise_pct']:.1f}% to "
                        f"{m['final']:.3f}",
                        xy=(m["step"], m["loss"]),
                        xytext=(0.35, 0.14), textcoords="axes fraction",
                        fontsize=8.5, color=C["red"],
                        arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.9))
        ax.set_title(f"{run['name']}\n({rid}, {run['state']})", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("eval loss (D_gen)", color=C["blue"])
        ax.tick_params(axis="y", colors=C["blue"])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = (ax2.get_legend_handles_labels() if len(xtr) else ([], []))
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    fig.suptitle("Eval loss bottoms early, then rises while train loss keeps falling",
                 fontsize=10.5)
    fig.savefig(out / "fig1_validation_reversal.png")
    plt.close(fig)
    print("fig1: saved")


# -------------------------------------------------------------------- fig 2

def fig2_lora_growth_heatmap(runs, matrix_stats, out):
    """Layer x module LoRA-B norm growth AFTER the validation minimum --
    one panel per model scale when both (0.5B and 7B) have checkpoints."""
    dirs = sorted({rel.split("/")[0] for rel in matrix_stats})
    cands = []
    for rid, run in runs.items():
        for d in dirs:
            if not _dir_matches_run(d, run):
                continue
            m = eval_min(loss_curves(run))
            n_ck = sum(1 for rel in matrix_stats
                       if rel.startswith(d + "/") and "adapter_model" in rel)
            if n_ck < 2:
                continue
            score = n_ck + (50 if m else 0) + (20 if "smoke" not in d else 0)
            cands.append((score, d, rid, run, m))
    cands.sort(key=lambda x: x[0], reverse=True)
    # keep at most one dir per model scale, prefer non-smoke, max 2 panels
    chosen, seen_scale = [], set()
    for score, d, rid, run, m in cands:
        scale = "7b" if "7b" in d.lower() else "0.5b" if "0.5b" in d.lower() else "?"
        if "smoke" in d.lower() or "adapt-test" in d.lower():
            continue
        if scale in seen_scale:
            continue
        seen_scale.add(scale)
        chosen.append((d, rid, run, m))
        if len(chosen) == 2:
            break
    if not chosen:  # legacy fallback: largest checkpoint dir
        counts = {d: sum(1 for rel in matrix_stats if rel.startswith(d + "/")
                         and "adapter_model" in rel) for d in dirs}
        if not counts or max(counts.values()) < 2:
            print("fig2: fewer than 2 checkpoint snapshots -- skipped")
            return
        d = max(counts, key=counts.get)
        chosen = [(d, None, None, None)]
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.6 * len(chosen), 7.2),
                             constrained_layout=True, squeeze=False)
    for ax, (run_dir, rid, run, m) in zip(axes[0], chosen):
        panel = bnorm_panel(matrix_stats, run_dir)
        final_step = 10**9 if 10**9 in panel else max(panel)
        if m is not None and m.get("step") is not None:
            base_step = nearest_step(panel, m["step"])
        else:
            base_step = nearest_step(panel, None)
        if base_step is None or base_step == final_step:
            print(f"fig2: no earlier checkpoint to diff against ({run_dir}) -- skipped")
            continue
        layers = sorted({l for s in panel for (l, _) in panel[s]})
        growth = np.full((len(layers), len(MODULE_ORDER)), np.nan)
        for i, lay in enumerate(layers):
            for j, mod in enumerate(MODULE_ORDER):
                f = panel[final_step].get((lay, mod))
                b = panel[base_step].get((lay, mod))
                if f is not None and b is not None:
                    growth[i, j] = f - b
        im = ax.imshow(growth, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels([mo.split(".")[-1] for mo in MODULE_ORDER],
                           rotation=40, ha="right")
        ax.set_ylabel("transformer layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel("module")
        ax.grid(False)
        base_lbl = ("final" if base_step == 10**9 or base_step == max(panel)
                    else f"ckpt-{base_step}")
        vmin_ep = f" (eval min @ step {int(m['step'])})" if m else ""
        ax.set_title(f"||B||$_2$ growth after val minimum{vmin_elide(vmin_ep)}\n"
                     f"{run_dir}  [{base_lbl} -> final]", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75, label="delta ||B||$_2$")
        flat = np.nan_to_num(growth, nan=-1e9)
        idxs = np.dstack(np.unravel_index(np.argsort(flat.ravel())[::-1],
                                          flat.shape))[0][:8]
        for i, j in idxs:
            if flat[i, j] <= -1e8:
                continue
            ax.text(j, i, f"{growth[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold",
                    color="black" if growth[i, j] > np.nanmax(growth) * 0.5 else "white")
    fig.savefig(out / "fig2_lora_growth_heatmap.png")
    plt.close(fig)
    print("fig2: saved (" + "; ".join(c[0] for c in chosen) + ")")


def vmin_elide(s):
    return s


def _dir_matches_run(dir_name, run):
    d = dir_name.lower()
    n = run["name"].lower()
    if "adaptive" in d and "adaptive" not in n:
        return False
    if "0.5b" in d and "0.5b" not in n and "0.5b" not in d.replace("qwen2.5", ""):
        return "0.5b" in n
    return ("0.5b" in d and "0.5b" in n) or ("7b" in d and "7b" in n) or \
           ("smoke" in d and "smoke" in n) or ("adapt-test" in d and "adapt" in n)


# -------------------------------------------------------------------- fig 3

def _epoch_of(entry, fallback):
    """Robust per-epoch x value: prefer explicit epoch fields, else fallback."""
    for key in ("epoch", "gate/epoch", "adaptive/epoch"):
        v = entry.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float(fallback)


def fig3_gate_timeline(runs, out):
    # purpose: make controller behaviour visible -> render one figure per model
    # scale (top-2 runs by controller richness)
    cands = [(rid, r) for rid, r in runs.items()
             if gate_curve(r) and len(gate_curve(r)) >= 2]
    if not cands:
        print("fig3: no gate curves -- skipped")
        return
    cands.sort(key=lambda x: (100 if adaptive_curve(x[1]) else 0,
                              len(gate_curve(x[1]))), reverse=True)
    # keep one run per model scale when identifiable from the name
    chosen, seen = [], set()
    for rid, run in cands:
        n = run["name"].lower()
        scale = "7b" if "7b" in n else "0.5b" if "0.5b" in n else "?"
        if scale in seen:
            continue
        seen.add(scale)
        chosen.append((rid, run, scale))
        if len(chosen) == 2:
            break
    for rid, run, scale in chosen:
        _fig3_one(run, rid, scale, out)


def _fig3_one(run, rid, scale, out):
    gate = gate_curve(run)
    adap = adaptive_curve(run)
    if not gate:
        return
    adap_by_ep = {int(round(float(a.get("epoch", 0)))): a for a in adap if a.get("epoch") is not None}

    eps = [_epoch_of(g, i + 1) for i, g in enumerate(gate)]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(7.4, 7.6), sharex=True, constrained_layout=True)

    # panel 1: losses + eval minimum marker (EPOCH units everywhere)
    xtr, ytr, xev, yev = loss_curves(run, x_unit="epoch")
    if len(xev):
        ax1.plot(xev, yev, color=C["blue"], lw=1.6, marker="o", ms=3, label="eval loss (D_gen)")
    if len(xtr):
        ax1.plot(xtr, ytr, color=C["sky"], lw=0.8, alpha=0.7, label="train loss")
    m = eval_min((xtr, ytr, xev, yev))
    if m:
        ax1.scatter([m["step"]], [m["loss"]], zorder=5, color=C["red"], s=42,
                    marker="v", label=f"eval min {m['loss']:.2f} (+{m['rise_pct']:.0f}%)")
        ax1.axvline(m["step"], color=C["red"], lw=0.9, ls=":", alpha=0.7)
    ax1.set_ylabel("loss")
    ax1.set_title(f"Gate + controller timeline ({scale}) -- {run['name']} ({rid})",
                  fontsize=10)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(fontsize=8, loc="upper right")

    # panel 2: gate accuracies + lambda
    am = [g.get("a_mem") for g in gate]
    ag = [g.get("a_gen") for g in gate]
    lm = [g.get("lambda") for g in gate]
    ax2.plot(eps, am, color=C["red"], lw=1.6, marker="s", ms=3, label="A_mem (exact match)")
    if any(v is not None for v in ag):
        ax2.plot(eps, [v if v is not None else np.nan for v in ag], color=C["purple"],
                 lw=1.6, marker="o", ms=3, label="A_gen (exact match)")
    ax2.axhline(0.9, color=C["green"], lw=1.0, ls="--", alpha=0.8)
    ax2.text(eps[-1], 0.915, "$\\tau_0$=0.9 (gate on)", fontsize=7.5,
             color=C["green"], ha="right")
    if any(v is not None and v > 0 for v in lm):
        ax2.plot(eps, [v if v is not None else np.nan for v in lm], color=C["green"],
                 lw=2.0, label="lambda(t) (x $\\lambda_0$ scale)")
        ax2.fill_between(eps, 0, [v or 0 for v in lm], color=C["green"], alpha=0.15)
    ax2.set_ylabel("accuracy / $\\lambda$")
    ax2.set_ylim(-0.03, 1.05)
    ax2.legend(fontsize=8, loc="upper left")

    # panel 3: LR multiplier + controller actions
    steps = sorted(adap_by_ep)
    if steps:
        mult = [adap_by_ep[s].get("lr_mult") for s in steps]
        ax3.plot(steps, mult, color=C["black"], lw=1.6, marker=".", ms=4,
                 label="adaptive LR multiplier")
        for s in steps:
            act = str(adap_by_ep[s].get("action", ""))
            if "boost" in act:
                ax3.scatter([s], [adap_by_ep[s].get("lr_mult")], color=C["red"],
                            marker="^", s=64, zorder=5)
            elif "decay" in act:
                ax3.scatter([s], [adap_by_ep[s].get("lr_mult")], color=C["blue"],
                            marker="v", s=64, zorder=5)
            elif "diagnosis" in act:
                ax3.scatter([s], [adap_by_ep[s].get("lr_mult")], color=C["orange"],
                            marker="D", s=42, zorder=5)
            elif "stop" in act:
                ax3.scatter([s], [adap_by_ep[s].get("lr_mult")], color=C["black"],
                            marker="x", s=70, zorder=5)
        handles = [Line2D([], [], ls="", marker="^", color=C["red"], label="lr boost"),
                   Line2D([], [], ls="", marker="v", color=C["blue"], label="lr decay"),
                   Line2D([], [], ls="", marker="D", color=C["orange"], label="diagnosis"),
                   Line2D([], [], ls="", marker="x", color=C["black"], label="stop")]
        ax3.legend(handles=handles, fontsize=8, loc="upper left", ncols=2)
    else:
        ax3.text(0.5, 0.5, "no adaptive controller in this run",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=9,
                 color="0.35")
    ax3.set_ylabel("LR multiplier")
    ax3.set_xlabel("epoch")
    # keep all three panels on the same epoch span
    xmax = max([max(eps)] + ([float(max(xev)) if len(xev) else 0.0]))
    ax1.set_xlim(-0.02 * xmax, 1.02 * xmax)

    fname = "fig3_gate_timeline.png" if scale == "7b" else \
        f"fig3_gate_timeline_{scale.replace('.', '_')}.png"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"fig3: saved {fname} ({run['name']})")


# -------------------------------------------------------------------- fig 4

def fig4_probe_health(runs, out):
    cands = [(rid, r) for rid, r in runs.items() if not r["is_matrix_collection"]]
    scored = []
    for rid, run in cands:
        series = probe_series(run, step2ep=_step_to_epoch_fn(run))
        xn, wn, xg, gn, xm, am, xgg, ag, xb, bn = series
        if len(xn) + len(xg) + len(xm) + len(xgg) + len(xb) >= 2:
            scored.append((len(xn) + len(xb), rid, run, series))
    if not scored:
        print("fig4: no probe telemetry in this collection (needs the v3 "
              "callback / probe_eval wiring) -- skipped")
        return
    scored.sort(key=lambda x: x[0], reverse=True)
    _, rid, run, (xn, wn, xg, gn, xm, am, xgg, ag, xb, bn) = scored[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
    ax = axes[0]
    if len(xm):
        ax.plot(xm, am, color=C["red"], lw=1.6, marker="s", ms=3.5,
                label="A_probe on D_mem")
    if len(xgg):
        ax.plot(xgg, ag, color=C["purple"], lw=1.6, marker="o", ms=3.5,
                label="A_probe on D_gen")
    ax.set_xlabel("epoch")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Probe decodability at layer l* (representation space)", fontsize=9.5)
    if len(xm) or len(xgg):
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no probe decodability logged\n(needs probe_eval wiring)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="0.35")
    ax = axes[1]
    if len(xn):
        ax.plot(xn, wn, color=C["blue"], lw=1.5, label="||W_probe||$_2$")
    if len(xb):
        ax3 = ax.twinx()
        ax3.plot(xb, bn, color=C["orange"], lw=1.5, ls="--", label="||b_probe||$_2$")
        ax3.set_ylabel("probe bias norm", color=C["orange"])
        ax3.spines["right"].set_visible(True)
        ax3.grid(False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("probe weight norm")
    ax.set_title("Probe training evidence (flat W + zero b = frozen probe)",
                 fontsize=9.5)
    if len(xn):
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Probe health -- {run['name']} ({rid})", fontsize=10.5)
    fig.savefig(out / "fig4_probe_health.png")
    plt.close(fig)
    print(f"fig4: saved ({run['name']})")


# -------------------------------------------------------------------- fig 5

def fig5_update_spectrum(runs, matrix_stats, out):
    dirs = sorted({rel.split("/")[0] for rel in matrix_stats})
    if not dirs:
        print("fig5: no matrix snapshots -- skipped")
        return
    counts = {d: sum(1 for rel in matrix_stats if rel.startswith(d + "/")
                     and "adapter_model" in rel) for d in dirs}
    d = max(counts, key=counts.get)
    panel = bnorm_panel(matrix_stats, d)
    if len(panel) < 2:
        print("fig5: single snapshot only -- skipped")
        return
    steps = [s for s in panel if s != 10**9] + ([10**9] if 10**9 in panel else [])
    xlabels = ["final" if s == 10**9 else str(s) for s in steps]
    x = np.arange(len(steps))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for mod in MODULE_ORDER:
        ys = []
        for s in steps:
            vals = [v for (lay, m), v in panel[s].items() if m == mod]
            ys.append(np.median(vals) if vals else np.nan)
        ax1.plot(x, ys, color=MODULE_COLORS[mod], lw=1.7, marker="o", ms=3, label=mod)
    # readable ticks: subsample when there are many checkpoints
    tick_idx = list(range(len(steps))) if len(steps) <= 8 else \
        list(range(0, len(steps), max(1, len(steps) // 8)))
    if tick_idx[-1] != len(steps) - 1:
        tick_idx.append(len(steps) - 1)
    ax1.set_xticks([x[i] for i in tick_idx],
                   [xlabels[i] for i in tick_idx], rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("median ||B||$_2$ across layers")
    ax1.set_title(f"Module-family LoRA-B growth ({d})", fontsize=9.5)
    ax1.legend(fontsize=7, ncols=2)

    final_step = 10**9 if 10**9 in panel else max(panel)
    layers = sorted({l for (l, _) in panel[final_step]})
    for mod in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        ys = [panel[final_step].get((l, mod), np.nan) for l in layers]
        ax2.plot(layers, ys, color=MODULE_COLORS[mod], lw=1.7, label=mod)
    attn = [np.nanmean([panel[final_step].get((l, m), np.nan)
                        for m in MODULE_ORDER[3:]]) for l in layers]
    ax2.plot(layers, attn, color=C["black"], lw=1.7, ls="--", label="attention (mean)")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("final ||B||$_2$")
    ax2.set_title("Layerwise final LoRA-B profile (upper-layer MLP concentration)",
                  fontsize=9.5)
    ax2.legend(fontsize=7)
    fig.savefig(out / "fig5_update_spectrum.png")
    plt.close(fig)
    print(f"fig5: saved ({d})")


# -------------------------------------------------------------------- fig 6

def fig6_three_way_dw(dw, out):
    """Three-way checkpoint comparison (baseline/early vs best-val vs final)
    on the EFFECTIVE update dW=(alpha/r)BA: module-family Frobenius norms at
    the three checkpoints, plus post-best increment-norm and cosine-drift
    heatmaps for the 7B run."""
    if not dw or not dw.get("runs"):
        print("fig6: no dw_three_way.json -- skipped")
        return
    # one run per scale, preferring adaptive dirs then the LONGEST training
    by_scale = {}
    for d, e in dw["runs"].items():
        if not e.get("dw_stats"):
            continue
        scale = "7b" if "7b" in d.lower() else "0.5b" if "0.5b" in d.lower() else None
        if scale is None:
            continue
        pref = (1 if "adaptive" in d.lower() else 0,
                int(e.get("n_checkpoints") or 0))
        if scale not in by_scale or pref > by_scale[scale][0]:
            by_scale[scale] = (pref, d, e)
    if not by_scale:
        print("fig6: no usable run dirs in dw json -- skipped")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), constrained_layout=True)
    order = ["0.5b", "7b"]
    for col, scale in enumerate(order):
        ax = axes[0][col]
        if scale not in by_scale:
            ax.axis("off")
            continue
        _, d, e = by_scale[scale]
        sel = e.get("selected", {})
        stats = e["dw_stats"]
        phases = [p for p in ("early", "best", "final") if p in stats]
        x = np.arange(len(phases))
        for mod in MODULE_ORDER:
            ys = []
            for p in phases:
                vals = [v["fro"] for k, v in stats[p].items()
                        if k.endswith("." + mod)]
                ys.append(np.median(vals) if vals else np.nan)
            ax.plot(x, ys, color=MODULE_COLORS[mod], lw=1.8, marker="o", ms=4,
                    label=mod.split(".")[-1])
        xlabels = []
        for p in phases:
            s = sel.get(p) or {}
            lbl = f"{p}\n(s{s.get('step', '?')}"
            if s.get("eval_loss") is not None:
                lbl += f", ev {s['eval_loss']:.2f}"
            lbl += ")"
            xlabels.append(lbl)
        ax.set_xticks(x, xlabels, fontsize=7.5)
        ax.set_ylabel("median $\\|\\Delta W\\|_F$ across layers")
        ax.set_title(f"Effective update growth, {scale} ({d})", fontsize=9.5)
        ax.legend(fontsize=6.5, ncols=2)

    # heatmaps from the 7B run (fallback: any run with drift)
    hd, he = None, None
    for scale in ("7b", "0.5b"):
        if scale in by_scale and by_scale[scale][2].get("drift"):
            hd, he = by_scale[scale][1], by_scale[scale][2]
            break
    if he is not None:
        drift = he["drift"]
        layers = sorted({int(k.split(".")[1]) for k in drift})
        inc = np.full((len(layers), len(MODULE_ORDER)), np.nan)
        cos = np.full((len(layers), len(MODULE_ORDER)), np.nan)
        for i, lay in enumerate(layers):
            for j, mod in enumerate(MODULE_ORDER):
                dd = drift.get(f"layers.{lay}.{mod}")
                if dd:
                    inc[i, j] = dd["delta_norm_ref_final"]
                    if dd.get("cos_ref_final") is not None:
                        cos[i, j] = dd["cos_ref_final"]
        ref = (he.get("selected", {}).get("best") or he.get("selected", {}).get("early") or {})
        ref_lbl = f"$\\Delta W$ norm added after {ref.get('dir', 'ref')}"
        ax = axes[1][0]
        im = ax.imshow(inc, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels([m.split(".")[-1] for m in MODULE_ORDER],
                           rotation=40, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_ylabel("layer")
        ax.set_title(f"{ref_lbl} ({hd})", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="$\\|\\Delta W_f - \\Delta W_r\\|_F$")
        ax = axes[1][1]
        im = ax.imshow(cos, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels([m.split(".")[-1] for m in MODULE_ORDER],
                           rotation=40, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_title("cos($\\Delta W_{ref}$, $\\Delta W_{final}$) per module", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    else:
        for ax in axes[1]:
            ax.axis("off")
    fig.suptitle("Three-way checkpoint comparison: early vs best-validation vs final "
                 "(effective LoRA update $\\Delta W=(\\alpha/r)BA$)", fontsize=10.5)
    fig.savefig(out / "fig6_three_way_dw.png")
    plt.close(fig)
    print("fig6: saved")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collection JSON (v2 or legacy)")
    ap.add_argument("--dw", default=None,
                    help="dw_three_way.json (three-way checkpoint dW analysis)")
    ap.add_argument("--out", default="figs", help="output directory")
    ap.add_argument("--trainer-state", default=None)
    ap.add_argument("--gate-history", default=None)
    ap.add_argument("--adaptive-history", default=None)
    ap.add_argument("--pseudo-name", default="adapt-test (0.5B, 6 epochs)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coll = load_collection(args)

    n = len(coll["runs"])
    print(f"loaded {n} runs, {len(coll['matrix_stats'])} matrix snapshots")
    for rid, r in sorted(coll["runs"].items()):
        kind = " [matrix-collection]" if r["is_matrix_collection"] else ""
        print(f"  {rid}: {r['name']} ({r['state']}){kind} "
              f"gate_pts={len(r['gate_history'])} step_pts={len(r['hist_step'])}")

    fig1_validation_reversal(coll["runs"], out)
    fig2_lora_growth_heatmap(coll["runs"], coll["matrix_stats"], out)
    fig3_gate_timeline(coll["runs"], out)
    fig4_probe_health(coll["runs"], out)
    fig5_update_spectrum(coll["runs"], coll["matrix_stats"], out)
    fig6_three_way_dw(load_json(args.dw), out)
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
