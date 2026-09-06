"""Publication figures v2: model-internals localization + gate reachability.

fig7  Knowledge-storage anatomy -- WHERE the fine-tuned facts are written:
      (a,b) layer x module ||dW||_F heatmaps at the best-validation checkpoint
            for both scales; (c) post-best increment per layer (final - best):
            where memorization KEPT being written after generalization peaked;
      (d) module-family allocation of the total update mass.

fig8  Knowing vs saying + gate reachability:
      (a,b) train-CE saturation against held-out loss reversal (the
            Knowing-Using Gap dynamics) for both scales;
      (c) the free-generation A_mem ("saying") signal vs tau_0 -- measured
          unreachable at both scales -- and the v2 likelihood-gate design.

Inputs (produced by wb_collect.py v3.1 and dw_three_way.py):
  download/mlpr_longrun/wandb_full_collection.json
  download/mlpr_longrun/dw_three_way.json

Outputs: download/figs/fig7_knowledge_storage_anatomy.png
         download/figs/fig8_knowing_vs_saying.png
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOWNLOAD = Path("/home/z/my-project/download")
LONGRUN = DOWNLOAD / "mlpr_longrun"
FIGS = DOWNLOAD / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

MODULE_ORDER = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
MODULE_LABEL = {
    "self_attn.q_proj": "attn q", "self_attn.k_proj": "attn k",
    "self_attn.v_proj": "attn v", "self_attn.o_proj": "attn o",
    "mlp.gate_proj": "MLP gate", "mlp.up_proj": "MLP up",
    "mlp.down_proj": "MLP down",
}
FAMILY = {
    "self_attn.q_proj": "attention", "self_attn.k_proj": "attention",
    "self_attn.v_proj": "attention", "self_attn.o_proj": "attention",
    "mlp.gate_proj": "MLP gate", "mlp.up_proj": "MLP up",
    "mlp.down_proj": "MLP down",
}
C_MLP = "#c0392b"
C_ATTN = "#2471a3"
C_UP = "#e67e22"
C_DOWN = "#7d6608"

RUNS = {
    "7b": {"dw": "qwen2.5-7b-50ep", "wb": "264s6iw7", "l_star": 14, "L": 28,
           "label": "Qwen2.5-7B (50 ep)"},
    "05b": {"dw": "qwen2.5-0.5b-50ep-adaptive", "wb": "pll5xox9",
            "l_star": 12, "L": 24, "label": "Qwen2.5-0.5B (adaptive, 34 ep)"},
}


def load(name):
    with open(LONGRUN / name) as f:
        return json.load(f)


def parse_module(key):
    m = re.search(r"layers\.(\d+)\.(.+)$", key)
    return int(m.group(1)), m.group(2)


def eval_curve(run):
    """(epochs, eval_loss) from the per-epoch history rows."""
    xs, ys = [], []
    for row in run.get("history_step", []):
        if "eval/loss" in row and "train/epoch" in row:
            xs.append(float(row["train/epoch"]))
            ys.append(float(row["eval/loss"]))
    return np.array(xs), np.array(ys)


def train_curve(run):
    """(epochs, train_loss) from the checkpoint's trainer log_history."""
    lh = (run.get("trainer_state") or {}).get("log_history", [])
    xs, ys = [], []
    for row in lh:
        if "loss" in row and "epoch" in row:
            xs.append(float(row["epoch"]))
            ys.append(float(row["loss"]))
    return np.array(xs), np.array(ys)


def smooth(x, y, frac=0.08):
    if len(y) < 8:
        return y
    w = max(3, int(len(y) * frac) | 1)
    ker = np.ones(w) / w
    return np.convolve(y, ker, mode="same")


# ------------------------------------------------------------------ fig 7 --
def fig7(dw, coll):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2),
                             constrained_layout=True)

    # ---- (a,b) layer x module ||dW||_F at the best-val checkpoint ---------
    for col, (sk, cfg) in enumerate(RUNS.items()):
        ax = axes[0, col]
        stats = dw["runs"][cfg["dw"]]["dw_stats"]["best"]
        layers = sorted({parse_module(k)[0] for k in stats})
        grid = np.full((len(layers), len(MODULE_ORDER)), np.nan)
        for key, st in stats.items():
            li, mod = parse_module(key)
            if mod in MODULE_ORDER:
                grid[li, MODULE_ORDER.index(mod)] = st["fro"]
        im = ax.imshow(grid, aspect="auto", cmap="magma",
                       interpolation="nearest")
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels([MODULE_LABEL[m] for m in MODULE_ORDER],
                           rotation=35, ha="right")
        yticks = [0, cfg["l_star"], len(layers) - 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{layers[i]}" for i in yticks])
        ax.axhline(cfg["l_star"], color="cyan", lw=1.0, ls="--", alpha=0.85)
        ax.text(0.2, cfg["l_star"] + 0.25, f"$l^*$={cfg['l_star']}",
                color="cyan", fontsize=7.5)
        ax.set_title(f"({chr(97+col)}) {cfg['label']} — "
                     f"$\\|\\Delta W\\|_F$ at best-val ckpt")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Frobenius norm")

    # ---- (c) post-best growth per layer: where memorization kept being
    #      written AFTER generalization peaked --------------------------------
    ax = axes[1, 0]
    for sk, cfg, color in (("7b", RUNS["7b"], C_MLP), ("05b", RUNS["05b"], C_UP)):
        rdw = dw["runs"][cfg["dw"]]
        best, final = rdw["dw_stats"]["best"], rdw["dw_stats"]["final"]
        per_layer_gate, per_layer_attn = defaultdict(list), defaultdict(list)
        for key in best:
            li, mod = parse_module(key)
            inc = final[key]["fro"] - best[key]["fro"]
            (per_layer_attn if FAMILY[mod] == "attention"
             else per_layer_gate)[li].append(inc)
        layers = sorted(per_layer_gate)
        med_gate = [np.median(per_layer_gate[l]) for l in layers]
        med_attn = [np.median(per_layer_attn[l]) for l in layers]
        short = "7B" if sk == "7b" else "0.5B"
        ax.plot(layers, med_gate, color=color, lw=1.8,
                label=f"{short} MLP (gate/up/down med.)")
        ax.plot(layers, med_attn, color=color, lw=1.4, ls="--",
                alpha=0.75, label=f"{short} attention med.")
        ax.axvline(cfg["l_star"], color="gray", lw=0.8, ls=":")
    ax.set_title("(c) Update growth after the best-val checkpoint\n"
                 r"$\|\Delta W\|_F(\mathrm{final}) - \|\Delta W\|_F(\mathrm{best})$"
                 " per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("post-best $\\|\\Delta W\\|_F$ growth")
    ax.legend(ncol=1, loc="upper left")
    ax.text(0.985, 0.30, "upper-layer MLP growth\nco-occurs with the\n"
            "generalization reversal", transform=ax.transAxes,
            ha="right", fontsize=7.5, color="#555555")

    # ---- (d) module-family allocation of the update mass ------------------
    ax = axes[1, 1]
    fams = ["attention", "MLP gate", "MLP up", "MLP down"]
    fam_colors = {"attention": C_ATTN, "MLP gate": C_MLP,
                  "MLP up": C_UP, "MLP down": C_DOWN}
    width = 0.36
    for i, (sk, cfg) in enumerate(RUNS.items()):
        stats = dw["runs"][cfg["dw"]]["dw_stats"]["best"]
        tot = sum(st["fro"] for st in stats.values())
        alloc = {f: 0.0 for f in fams}
        for key, st in stats.items():
            alloc[FAMILY[parse_module(key)[1]]] += st["fro"]
        shares = [100 * alloc[f] / tot for f in fams]
        xs = np.arange(len(fams)) + (i - 0.5) * width
        bars = ax.bar(xs, shares, width, color=[fam_colors[f] for f in fams],
                      alpha=0.55 if i == 0 else 0.9,
                      edgecolor="white", label=cfg["label"])
        for x, s in zip(xs, shares):
            ax.text(x, s + 0.6, f"{s:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylabel("share of total $\\|\\Delta W\\|_F$ (%)")
    ax.set_title("(d) Where the update mass lives (best-val ckpt)")
    ax.legend()
    ax.set_ylim(0, max(ax.get_ylim()[1], 62))

    fig.suptitle("Knowledge-storage anatomy: mid-layer MLP modules carry the "
                 "factual update at both scales", fontsize=11.5, y=1.02)
    out = FIGS / "fig7_knowledge_storage_anatomy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ------------------------------------------------------------------ fig 8 --
def fig8(coll):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9),
                             constrained_layout=True)

    # ---- (a,b) CE saturation vs held-out reversal -------------------------
    for i, (sk, cfg) in enumerate((("7b", RUNS["7b"]), ("05b", RUNS["05b"]))):
        ax = axes[i]
        run = coll["runs"][cfg["wb"]]
        tx, ty = train_curve(run)
        ex, ey = eval_curve(run)
        ax.plot(tx, smooth(tx, ty), color=C_ATTN, lw=1.5, label="train CE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train CE loss", color=C_ATTN)
        ax.tick_params(axis="y", labelcolor=C_ATTN)
        ax2 = ax.twinx()
        ax2.plot(ex, ey, color=C_MLP, lw=1.7, label="held-out loss (D_gen)")
        imin = int(np.argmin(ey))
        ax2.scatter([ex[imin]], [ey[imin]], s=28, color="black", zorder=5)
        ax2.annotate(f"val-min ep{ex[imin]:.0f}",
                     (ex[imin], ey[imin]), textcoords="offset points",
                     xytext=(7, 7), fontsize=7.5)
        ax2.set_ylabel("held-out loss", color=C_MLP)
        ax2.tick_params(axis="y", labelcolor=C_MLP)
        growth = 100 * (ey[-1] / ey[imin] - 1)
        short = "7B" if sk == "7b" else "0.5B"
        ax.set_title(f"({'ab'[i]}) Qwen2.5-{short}: CE down, held-out "
                     f"loss +{growth:.0f}%")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", framealpha=0.9)

    # ---- (c) gate reachability: the "saying" signal never reaches tau_0 ---
    ax = axes[2]
    for sk, cfg, color in (("7b", RUNS["7b"], C_MLP), ("05b", RUNS["05b"], C_UP)):
        run = coll["runs"][cfg["wb"]]
        gh = run.get("gate_history", [])
        eps = [e["epoch"] for e in gh]
        amem = [e["a_mem"] for e in gh]
        agen = [e.get("a_gen") for e in gh]
        short = "7B" if sk == "7b" else "0.5B"
        ax.plot(eps, amem, color=color, lw=1.7,
                label=f"$A_{{mem}}$ (saying) {short}")
        ax.plot(eps, agen, color=color, lw=1.1, ls=":", alpha=0.8,
                label=f"$A_{{gen}}$ {short}")
    ax.axhline(0.9, color="black", lw=1.4)
    ax.text(0.4, 0.915, "v1 $\\tau_0$=0.9 (generation EM): NEVER reached "
            "(max 11% / 1.5%)", fontsize=7.8)
    ax.axhline(0.8, color="#0a7d32", lw=1.4, ls="--")
    ax.text(0.4, 0.725, "v2 $\\tau_0$=0.8 on likelihood $A_{mem}$ (knowing):\n"
            "reachable — teacher-forced recall of memorized facts",
            fontsize=7.8, color="#0a7d32")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (exact match)")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("(c) Gate signal reachability")
    ax.legend(loc="center left", fontsize=7)

    fig.suptitle("Knowing vs saying: the generation-exact-match gate is "
                 "unreachable exactly while CE shows saturated memorization",
                 fontsize=11, y=1.04)
    out = FIGS / "fig8_knowing_vs_saying.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    dw = load("dw_three_way.json")
    coll = load("wandb_full_collection.json")
    fig7(dw, coll)
    fig8(coll)


if __name__ == "__main__":
    main()
