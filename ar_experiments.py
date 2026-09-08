"""
MLPR autoresearch runner (Modal H100).

Adaptation of karpathy/autoresearch to the MLPR project:
- ONE experiment = ONE Modal call: fixed wall-clock training budget
  (`ar_time_budget_s`, checked at epoch end) + fast subsampled evals.
- main.py prints a greppable summary block at the end (score = validity-aware
  multi-hop generalization `a_gen_em_any`).
- The agent edits code/config between experiments; this file is the harness
  and is NOT where research ideas live (see autoresearch/program.md).

Usage:
    modal run ar_experiments.py --exp baseline-v1 --dataset dataset
    modal run ar_experiments.py --exp data-v2 --dataset dataset_v2 \
        --overrides '{"tau_0": 0.9, "delta_0": 0.05}'
    modal run ar_experiments.py --exp q1 --detach          # server-side spawn

    # Sequential queue inside one detached call (survives client exit):
    modal run ar_experiments.py --queue exp_a,exp_b,exp_c --dataset dataset_v2
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_DIR = Path(__file__).resolve().parent

app = modal.App("mlpr-autoresearch")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "wandb==0.18.5",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "tqdm",
        "safetensors",
        "huggingface_hub[hf_transfer]",
    )
    .apt_install("git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/vol/hf_cache",   # persistent HF cache across experiments
        }
    )
    .add_local_dir(
        str(REPO_DIR),
        remote_path="/root/MLPR",
        ignore=lambda path: ".git" in Path(path).parts
        or "__pycache__" in Path(path).parts,
    )
)

outputs_vol = modal.Volume.from_name("mlpr-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("mlpr-hf-cache", create_if_missing=True)

# Secrets are OPTIONAL (fresh accounts may not have them): main.py degrades
# gracefully to W&B-disabled mode and skips HF pushes.
try:
    secrets = [
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("wandb-entity"),
        modal.Secret.from_name("hf-token"),
    ]
except Exception:
    secrets = []

# Fixed-budget defaults shared by every experiment (comparability).
AR_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "l_star": 12,                       # 0.5B: 24 layers -> 0.5 * L
    # Epoch ceiling must be REACHABLE inside the budget: warmup_ratio applies
    # to the PLANNED total steps, so an inflated ceiling pushes warmup past
    # the budget wall and the run dies inside warmup (diag-tiny lesson:
    # LR stuck at 3.2e-6 of 2e-5, loss pinned at init). 12 epochs x 63 steps
    # = 756 planned steps; warmup 5% = 38 steps (~0.6 epochs) -> real LR for
    # the whole effective budget.
    "num_train_epochs": 12,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "constant_with_warmup",  # budget dies inside a planned decay tail otherwise
    "ar_mode": True,
    "eval_mem_gen_n": 150,
    "eval_mem_lf_n": 200,
    "eval_gen_n": 150,
    "lifecycle_enabled": False,
    "matrix_log_every_n_epochs": 60,    # effectively off (AR speed)
    "gate_num_samples": 128,            # fast per-epoch gate signal
    "probe_eval_num_samples": 128,
    "gate_gen_num_samples": 64,         # cap per-epoch A_gen generation pass
    "adaptive_enabled": False,          # isolate the gate from the controller
    "learning_rate": 2.0e-5,
    "probe_learning_rate": 1e-3,
    "gate_metric": "max",
    "tau_0": 0.8,
    "delta_0": 0.1,
    "lambda_0": 0.3,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "save_strategy": "no",              # no checkpoint spam in AR mode
    "evaluation_strategy": "epoch",
    "wandb_entity": None,
    "hub_model_id": None,
}


def _run_with_tee(cmd: list, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    with open(log_path, "w") as logf:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
    return proc.wait()


@app.function(image=image, gpu="H100", timeout=45 * 60,
              volumes={"/vol/outputs": outputs_vol, "/vol/hf_cache": hf_cache_vol},
              secrets=secrets)
def run_experiment(exp_name: str, overrides_json: str = "{}",
                   dataset: str = "dataset_v2", budget: int = 360,
                   wandb_project: str = "knowing_using_gap_ar"):
    """Run ONE fixed-budget MLPR experiment; print the AR summary block."""
    import yaml

    os.chdir("/root/MLPR")
    user_overrides = json.loads(overrides_json) if overrides_json else {}
    overrides = dict(AR_DEFAULTS)
    overrides.update(user_overrides)
    if "dataset_name" not in user_overrides:
        overrides["dataset_name"] = f"./{dataset}"
    overrides["ar_time_budget_s"] = int(budget)
    overrides["wandb_run_name"] = f"ar-{exp_name}"

    cfg_path = f"/tmp/mlpr_ar_{exp_name}.yaml"
    with open("configs/qwen2.5_7b.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    entity = os.environ.get("WANDB_ENTITY")
    cmd = [sys.executable, "main.py", "--config", cfg_path,
           "--output_dir", f"/vol/outputs/ar/{exp_name}",
           "--wandb_project", wandb_project]
    if entity:
        cmd += ["--wandb_entity", entity]

    print("=" * 60)
    print(f"AR EXPERIMENT: {exp_name}")
    print(f"  dataset : {dataset}")
    print(f"  budget  : {budget}s training wall clock")
    print(f"  overrides: {json.dumps({k: v for k, v in overrides.items() if k not in AR_DEFAULTS or AR_DEFAULTS.get(k) != v})}")
    print("=" * 60)

    log_path = f"/vol/outputs/ar/{exp_name}/training_log.txt"
    rc = _run_with_tee(cmd, log_path)
    outputs_vol.commit()
    hf_cache_vol.commit()

    if rc != 0:
        raise RuntimeError(f"AR experiment {exp_name} failed (exit {rc})")
    print(f"=== AR experiment {exp_name} finished successfully ===")


@app.function(image=image, gpu="H100", timeout=8 * 60 * 60,
              volumes={"/vol/outputs": outputs_vol, "/vol/hf_cache": hf_cache_vol},
              secrets=secrets)
def run_queue(exp_names: list, override_jsons: list, dataset: str,
              budget: int, wandb_project: str):
    """Server-side sequential queue: one container, many experiments.

    Warm container between experiments (no per-exp cold start); a failing
    experiment is logged and the queue continues.
    NOTE: gpu="H100" is REQUIRED here — a queue without it runs training on
    CPU at ~12s/step (caught by the 2-3 min log check: 744 steps ETA 2.5h).
    """
    results = []
    for i, (name, ov) in enumerate(zip(exp_names, override_jsons)):
        print("#" * 60)
        print(f"AR QUEUE [{i + 1}/{len(exp_names)}]: {name}")
        print("#" * 60)
        ok = _run_experiment_local(name, ov, dataset, budget, wandb_project)
        results.append((name, "ok" if ok else "FAILED"))
    print("AR QUEUE SUMMARY:")
    for name, status in results:
        print(f"  {name}: {status}")
    print(f"AR QUEUE: {len(exp_names)} experiments processed")


def _run_experiment_local(exp_name, overrides_json, dataset, budget,
                          wandb_project):
    """Execute the experiment body inside the queue container."""
    import yaml

    os.chdir("/root/MLPR")
    user_overrides = json.loads(overrides_json) if overrides_json else {}
    overrides = dict(AR_DEFAULTS)
    overrides.update(user_overrides)
    # Queue-level dataset applies UNLESS the experiment overrides it
    # explicitly (per-experiment dataset comparisons like v1 vs v2).
    if "dataset_name" not in user_overrides:
        overrides["dataset_name"] = f"./{dataset}"
    overrides["ar_time_budget_s"] = int(budget)
    overrides["wandb_run_name"] = f"ar-{exp_name}"

    cfg_path = f"/tmp/mlpr_ar_{exp_name}.yaml"
    with open("configs/qwen2.5_7b.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    entity = os.environ.get("WANDB_ENTITY")
    cmd = [sys.executable, "main.py", "--config", cfg_path,
           "--output_dir", f"/vol/outputs/ar/{exp_name}",
           "--wandb_project", wandb_project]
    if entity:
        cmd += ["--wandb_entity", entity]

    log_path = f"/vol/outputs/ar/{exp_name}/training_log.txt"
    rc = _run_with_tee(cmd, log_path)
    outputs_vol.commit()
    hf_cache_vol.commit()
    if rc != 0:
        print(f"!!! AR experiment {exp_name} FAILED (exit {rc}) — continuing queue")
        return False
    print(f"=== AR experiment {exp_name} finished successfully ===")
    return True


@app.local_entrypoint()
def main(exp: str = None, overrides: str = "{}", dataset: str = "dataset_v2",
         budget: int = 360, queue: str = None, override_list: str = None,
         detached: bool = True, wandb_project: str = "knowing_using_gap_ar"):
    """Launch AR experiments.

    IMPORTANT (detached lifetime): pass Modal's own --detach BEFORE the
    script path:  modal run --detach ar_experiments.py --queue a,b,c
    (This entrypoint used to define a `detach` flag which shadowed Modal's
    option when written after the script path -> the app was stopped right
    after the local entrypoint returned and the spawned queue was canceled
    with 0 tasks. Queue launches therefore always spawn server-side; the
    entrypoint-level `--detached` flag only controls single --exp runs.)
    """
    if queue:
        names = [q.strip() for q in queue.split(",") if q.strip()]
        ovs = ([o.strip() for o in override_list.split("||")] if override_list
               else ["{}"] * len(names))
        if len(ovs) == 1 and len(names) > 1:
            ovs = ovs * len(names)
        assert len(ovs) == len(names), "override_list count mismatch"
        call = run_queue.spawn(names, ovs, dataset, budget, wandb_project)
        print(f"Spawned detached AR queue ({' -> '.join(names)}): {call}")
        print("Monitor with: modal app logs <app-id>")
        return
    if exp is None:
        raise SystemExit("Provide --exp <name> or --queue a,b,c")
    if detached:
        call = run_experiment.spawn(exp, overrides, dataset, budget,
                                    wandb_project)
        print(f"Spawned detached AR experiment {exp}: {call}")
        print("Monitor with: modal app logs <app-id>")
        return
    run_experiment.remote(exp, overrides, dataset, budget, wandb_project)
