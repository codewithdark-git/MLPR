"""Full W&B + volume data collection for the MLPR paper figures (v3).

Why v3: the v2 collector silently produced EMPTY histories because
  (a) run.history() without pandas returns a list whose rows do not carry the
      HF step keys, and
  (b) scan_history(keys=[~2200 keys]) returns ZERO rows (server-side limit on
      selected columns), while small key scans work fine.
v3 therefore uses CHUNKED scan_history (<= CHUNK keys per query, rows merged
by _step) for everything, adds the missing eval/* prefixes, and skips the
matrix-collection analysis runs for epoch scanning (dedupe).

Usage:
  modal run MLPR/analysis/wb_collect.py

Output: /vol/outputs/_collection/wandb_full_collection.json
  { "generated_at", "entity", "project", "inventory",
    "runs": { run_id: {name, tags, state, created_at, summary, history_epoch,
                       history_step, gate_history, adaptive_history,
                       trainer_state, volume_files} },
    "matrix_stats": { "run_dir/path": {matrix_key: {shape,mean,std,abs_max,l2}} } }
"""

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("mlpr-wb-collect-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("wandb==0.18.5", "safetensors", "numpy", "huggingface_hub", "pyyaml")
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

outputs_vol = modal.Volume.from_name("mlpr-outputs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-api-key")
wandb_entity_secret = modal.Secret.from_name("wandb-entity")

# per-epoch metric prefixes pulled from the summary key inventory
EPOCH_KEYS_PREFIXES = ("gate/", "adaptive/", "probe/", "lora/", "final_",
                       "baseline_", "improvement_", "summary/", "loss_ce",
                       "loss_probe", "eval/", "train/")
STEP_KEYS = ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch",
             "loss_ce", "loss_probe", "probe/train_acc", "lambda",
             "eval/loss", "eval/runtime", "train/loss", "train/epoch",
             "train/learning_rate")
MATRIX_RUN_NAME = "matrix-collection"
PER_RUN_BUDGET_S = 420     # hard cap per run so one huge run cannot eat the job


@app.function(image=image, volumes={"/vol/outputs": outputs_vol},
              secrets=[wandb_secret, wandb_entity_secret], timeout=45 * 60)
def collect(entity: str = None, samples: int = 4000):
    os.chdir("/vol/outputs")
    import wandb

    if not entity:
        api = wandb.Api()
        entity = api.default_entity
    project = f"{entity}/knowing_using_gap"
    print(f"W&B entity: {entity}")
    t0 = time.time()

    # ---------------- 1. volume inventory -----------------------------------
    inventory = {}
    volume_files = {}
    for run_dir in sorted(Path("/vol/outputs").iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        files = {}
        for p in run_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(run_dir))
                files[rel] = round(p.stat().st_size / 1e6, 2)
        inventory[run_dir.name] = {
            "n_files": len(files),
            "n_checkpoints": sum(1 for r in files if r.startswith("checkpoint-")),
            "total_mb": round(sum(files.values()), 1),
        }
        volume_files[run_dir.name] = files
        print(f"  inventory {run_dir.name}: {len(files)} files")
    outputs_vol.commit()

    def _load_json_maybe(run_dir: Path, fname: str):
        p = run_dir / fname
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return None

    # ---------------- 2. W&B histories (full no-keys scan) -------------------
    api = wandb.Api()

    def _full_scan(run):
        """Full no-keys scan_history (probe-verified to include _step and all
        logged keys), locally filtered to the metric families we need.
        NOTE: scan_history(keys=[...]) silently DROPS _step and big key lists
        return zero rows -- that is why v2/v3-chunked approaches failed."""
        keep_exact = set(STEP_KEYS)
        prefixes = EPOCH_KEYS_PREFIXES
        rows = []
        n_raw = 0
        for row in run.scan_history():
            n_raw += 1
            kept = {}
            for k, v in row.items():
                if k.startswith("_"):
                    continue
                if k in keep_exact or str(k).startswith(prefixes) \
                        or str(k).startswith(("lora/A_norm/", "lora/B_norm/", "lora/dW_")):
                    if v is not None and not (isinstance(v, float) and v != v):
                        kept[k] = v
            if kept:
                kept["_step"] = row.get("_step")
                rows.append(kept)
            if time.time() - t_run > PER_RUN_BUDGET_S:
                print(f"  WARN: scan budget hit for {run.id}")
                break
        print(f"    scanned {n_raw} raw rows -> {len(rows)} kept")
        return rows

    runs_data = {}
    for run in api.runs(project, order="-created_at"):
        rid = run.id
        vkey = _volume_key(run)
        entry = {
            "name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "tags": list(run.tags or []),
            "is_matrix_collection": run.name == MATRIX_RUN_NAME,
            "volume_dir": vkey,
            "summary": {},
            "history_epoch": [],
            "history_step": [],
            "volume_files": volume_files.get(vkey, {}),
        }
        try:
            entry["summary"] = {k: v for k, v in run.summary.items()
                                if isinstance(v, (int, float, str, bool))}
        except Exception:
            pass

        t_run = time.time()
        try:
            if run.name == MATRIX_RUN_NAME:
                # dedupe: analysis-only runs -- summary is enough
                print(f"run {rid} ({run.name}): matrix-collection, skipped epoch scan")
            else:
                rows = _full_scan(run)
                # epoch-level view: gate/adaptive/lora/probe/eval metric rows
                entry["history_epoch"] = [
                    r for r in rows
                    if any(str(k).startswith(EPOCH_KEYS_PREFIXES) for k in r)
                ]
                # step-level view: HF trainer loss/LR/grad rows
                step_keys = ("loss", "eval_loss", "eval/loss", "learning_rate",
                             "grad_norm", "loss_ce", "loss_probe", "lambda")
                entry["history_step"] = [
                    r for r in rows if any(str(k) in step_keys for k in r)
                ]
                print(f"run {rid} ({run.name}): epoch rows={len(entry['history_epoch'])} "
                      f"step rows={len(entry['history_step'])} "
                      f"({time.time() - t_run:.0f}s)")
        except Exception as e:
            entry["history_error"] = f"{type(e).__name__}: {e}"
            print(f"  ERROR scanning {rid}: {e}")

        runs_data[rid] = entry

    # ---------------- 3. inline volume JSON artifacts ------------------------
    for run_dir in sorted(Path("/vol/outputs").iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        rid = None
        for r, entry in runs_data.items():
            if entry.get("volume_dir") == run_dir.name:
                rid = r
                break
        for fname, field in (("gate_history.json", "gate_history"),
                             ("adaptive_history.json", "adaptive_history"),
                             ("trainer_state.json", "trainer_state")):
            data = _load_json_maybe(run_dir, fname)
            if data is not None and rid:
                runs_data[rid][field] = data
                print(f"  inlined {fname} from {run_dir.name} -> {rid}")
        # trainer_state.json lives only inside checkpoint-* dirs for completed
        # runs -- the FINAL checkpoint's copy contains the COMPLETE log_history
        # (train loss curve), which W&B history rows lack for these runs.
        if rid and not runs_data[rid].get("trainer_state"):
            try:
                ckpts = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                               key=lambda p: int(p.parent.name.split("-")[-1]))
                if ckpts:
                    st = json.loads(ckpts[-1].read_text())
                    n_loss = sum(1 for h in st.get("log_history", []) if "loss" in h)
                    runs_data[rid]["trainer_state"] = st
                    print(f"  inlined {ckpts[-1].parent.name}/trainer_state.json "
                          f"({n_loss} loss pts) from {run_dir.name} -> {rid}")
            except Exception as e:
                print(f"  WARN: trainer_state pickup failed for {run_dir.name}: {e}")

    # ---------------- 4. matrix stats from adapter files ---------------------
    matrix_stats = _matrix_stats_block()

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entity": entity,
        "project": "knowing_using_gap",
        "inventory": inventory,
        "runs": runs_data,
        "matrix_stats": matrix_stats,
    }
    out_path = Path("/vol/outputs/_collection/wandb_full_collection.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    outputs_vol.commit()
    size_mb = out_path.stat().st_size / 1e6
    print(f"collection saved to volume: {out_path} "
          f"({size_mb:.1f} MB, {time.time() - t0:.0f}s total)")


def _volume_key(run) -> str:
    """Best-effort mapping of a W&B run to its /vol/outputs directory name."""
    name = (run.name or "").lower()
    if "adapt-test" in name:
        return "adapt-test-0.5b"
    if "gatecheck" in name:
        return "gatecheck-0.5b"
    if "1ep-v4" in name:
        return "smoke-qwen2.5-0.5b-v4-1ep"
    if "smoke" in name:
        return "smoke-qwen2.5-0.5b-v3"
    if "0.5b-50ep-adaptive-v4" in name or ("0.5b" in name and "v4" in name):
        return "qwen2.5-0.5b-50ep-adaptive-v4"
    if "7b-50ep-adaptive-v4" in name or ("7b" in name and "v4" in name):
        return "qwen2.5-7b-50ep-adaptive-v4"
    if "0.5b-50ep-adaptive" in name or ("0.5b" in name and "adaptive" in name):
        return "qwen2.5-0.5b-50ep-adaptive"
    if "7b-50ep-adaptive" in name or ("7b" in name and "adaptive" in name):
        return "qwen2.5-7b-50ep"
    if "0.5b-50ep" in name:
        return "qwen2.5-0.5b-50ep"
    if "7b" in name:
        return "qwen2.5-7b"
    if "0.5b" in name:
        return "qwen2.5-0.5b-50ep"
    return ""


def _matrix_stats_block() -> dict:
    """Per-matrix stats (shape/mean/std/abs_max/l2) for every adapter file."""
    import numpy as np
    from safetensors.numpy import load_file as load_st_numpy

    stats = {}
    for src in sorted(Path("/vol/outputs").rglob("adapter_model.safetensors")):
        rel = str(src.relative_to(Path("/vol/outputs")))
        try:
            tensors = load_st_numpy(str(src))
            entry = {}
            for k, v in tensors.items():
                flat = v.reshape(-1).astype("float32")
                entry[k] = {
                    "shape": list(v.shape),
                    "mean": float(flat.mean()),
                    "std": float(flat.std()),
                    "abs_max": float(np.abs(flat).max()),
                    "l2": float(np.linalg.norm(flat)),
                }
            stats[rel] = entry
            print(f"  matrix stats: {rel} ({len(tensors)} tensors)")
        except Exception as e:
            print(f"  WARN: parse failed {rel}: {e}")
    return stats


@app.local_entrypoint()
def main():
    collect.remote()
