"""
Modal deployment for MLPR fine-tuning (Qwen2.5 / H100 / bf16 LoRA).

Usage:
    modal run modal_train.py --chain             # RECOMMENDED: detached server-side
                                                 # smoke -> full chain (survives client exit)
    modal run modal_train.py                     # local-driven smoke, then full (blocked on CLI)
    modal run modal_train.py --no-smoke          # full 7B run only
    modal run modal_train.py --no-full           # smoke test only

What this sets up:
- Container image: Python 3.11 + pinned torch/transformers/peft stack
  (+ huggingface-lifecycle from git) with the repo mounted at /root/MLPR
- Persistent Volumes: mlpr-outputs (checkpoints/final model), mlpr-hf-cache (HF weights)
- Secrets: "wandb-api-key" (WANDB_API_KEY), "wandb-entity", "hf-token" (HF_TOKEN)
- GPU: H100 80GB, bf16 LoRA exactly as the paper specifies
- Chained jobs: smoke (Qwen2.5-0.5B, 1 epoch, pipeline validation) -> full
  (Qwen2.5-7B-Instruct, 3 epochs per configs/qwen2.5_7b.yaml)

IMPORTANT: with a plain `modal run`, the smoke->full chaining is driven by the LOCAL
CLI process; if that process is killed, Modal cancels the running inputs even with
--detach. Use --chain instead: it spawns a server-side driver function (via
Function.spawn) that runs both jobs sequentially inside Modal, immune to local
process death. Monitor with `modal app logs <app-id>`.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_DIR = Path(__file__).resolve().parent

app = modal.App("mlpr-finetuning")

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
        "matplotlib>=3.8",  # in-container LoRA growth heatmap -> W&B (was: "heatmap skipped: No module named 'matplotlib'")
        "tqdm",
        "safetensors",
        "huggingface_hub[hf_transfer]",
    )
    .apt_install("git")
    .pip_install("git+https://github.com/codewithdark-git/huggingface-lifecycle.git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        str(REPO_DIR),
        remote_path="/root/MLPR",
        ignore=lambda path: ".git" in Path(path).parts
        or "__pycache__" in Path(path).parts,
    )
)

# Qwen3 architecture support requires transformers >= 4.51 (Qwen2.5 runs on
# the proven 4.44.2 stack above; the 1.7B scale point uses this image).
image_qwen3 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "datasets==2.21.0",
        "wandb==0.18.5",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8",
        "tqdm",
        "safetensors",
        "huggingface_hub[hf_transfer]",
    )
    .apt_install("git")
    .pip_install("git+https://github.com/codewithdark-git/huggingface-lifecycle.git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
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


def _optional_secret(name: str):
    """Return the named Modal secret, or an empty placeholder if absent.

    Fresh accounts may not have wandb-api-key / wandb-entity / hf-token yet.
    main.py degrades gracefully in that case (W&B disabled mode, HF push
    skipped), so training never blocks on credentials.
    """
    try:
        s = modal.Secret.from_name(name)
        s.hydrate()  # client-side existence check (raises if missing)
        return s
    except Exception:
        print(f"[modal_train] Modal secret '{name}' not found on this "
              f"account -- continuing without it")
        return modal.Secret.from_dict({})


wandb_secret = _optional_secret("wandb-api-key")
wandb_entity_secret = _optional_secret("wandb-entity")
hf_secret = _optional_secret("hf-token")

# ------------------------------------------------------------------
# AR-CHAMPION (sep6, 4-arm fixed-budget loop): representation-saturation
# gate. AR evidence: the output-gate (max/0.8) never opens (lambda==0 over
# 12 AR epochs; never in v4's 50 epochs at 0.5B) while the probe-
# decodability gate opens at ~ep9 for tau_0=0.3 and probe-gradient
# injection cuts mean train loss 4.71 -> 2.76 (1.7x faster optimization).
# ------------------------------------------------------------------
AR_CHAMPION_GATE = {
    "gate_metric": "probe",
    "tau_0": 0.3,
    "delta_0": 0.15,
    "lambda_0": 0.3,
    "probe_learning_rate": 1e-3,
}

# Shared resource spec for all training-executing functions
_TRAIN_FN_KWARGS = dict(
    image=image,
    gpu="H100",
    volumes={
        "/vol/outputs": outputs_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[wandb_secret, wandb_entity_secret, hf_secret],
)

_TRAIN_FN_KWARGS_QWEN3 = dict(
    image=image_qwen3,
    gpu="H100",
    volumes={
        "/vol/outputs": outputs_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[wandb_secret, wandb_entity_secret, hf_secret],
)


def _resolve_wandb_entity():
    """Prefer WANDB_ENTITY env, else derive the default entity from the API key."""
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        return entity
    try:
        import wandb

        api = wandb.Api()
        return api.default_entity
    except Exception as e:
        print(f"Could not resolve W&B entity ({e}); wandb will use its default.")
        return None


def _resolve_hf_user():
    """Resolve the HF username from the token so we can build hub_model_id."""
    try:
        from huggingface_hub import whoami

        info = whoami(token=os.environ.get("HF_TOKEN"))
        return info.get("name")
    except Exception as e:
        print(f"Could not resolve HF user ({e}); Hub push will be skipped.")
        return None


def _write_config(base_path: str, out_path: str, overrides: dict) -> None:
    import yaml

    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg


def _upload_telemetry(output_dir: str, hub_model_id: str) -> None:
    """Upload run telemetry to the Hub so the full run saves EVERYTHING.

    Pushes: gate_history.json, adaptive_history.json, the latest
    trainer_state.json, and the complete stdout training log -- alongside
    the model/adapter checkpoints the lifecycle manager already pushes.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    uploads = {
        os.path.join(output_dir, "gate_history.json"): "telemetry/gate_history.json",
        os.path.join(output_dir, "adaptive_history.json"): "telemetry/adaptive_history.json",
        os.path.join(output_dir, "training_log.txt"): "telemetry/training_log.txt",
    }
    # Latest checkpoint's trainer_state.json (per-epoch loss/metric history)
    ckpt_dirs = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if ckpt_dirs:
        ts = ckpt_dirs[-1] / "trainer_state.json"
        if ts.exists():
            uploads[str(ts)] = "telemetry/trainer_state.json"

    for local, remote in uploads.items():
        if not os.path.exists(local):
            print(f"[HF] telemetry skip (missing): {local}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=hub_model_id,
                repo_type="model",
            )
            print(f"[HF] uploaded {remote} -> {hub_model_id}")
        except Exception as e:
            print(f"[HF] WARNING: failed to upload {remote}: {e}")


def _run_with_tee(cmd: list, log_path: str) -> int:
    """Run cmd, streaming stdout live to container logs AND a log file."""
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


def _run_training_impl(job: str = "smoke"):
    """Run one MLPR training job inside a H100 container via main.py."""
    os.chdir("/root/MLPR")

    if job == "smoke3":
        # Alias: the standard 1-epoch smoke, run on the Qwen3 image —
        # cheap validation of the transformers>=4.51 stack before the
        # 1.7B production job rides on it.
        job = "smoke"

    # Hub repo suffix per job type (keeps long runs from overwriting 3-epoch ones)
    hub_suffix = {
        "full": "mlpr-qwen2.5-7b-instruct",
        "half50": "mlpr-qwen2.5-0.5b-instruct-50ep-adaptive",
        "full50": "mlpr-qwen2.5-7b-instruct-50ep-adaptive",
        "half50v2": "mlpr-qwen2.5-0.5b-instruct-50ep-adaptive-v4",
        "full50v2": "mlpr-qwen2.5-7b-instruct-50ep-adaptive-v4",
        "half50v5": "mlpr-qwen2.5-0.5b-instruct-50ep-probegate-v5",
        "half50v5ctrl": "mlpr-qwen2.5-0.5b-instruct-50ep-sft-control-v5",
        "q17b50v5": "mlpr-qwen3-1.7b-50ep-probegate-v5",
    }

    if job == "smoke":
        # 1-EPOCH smoke (user spec: minimal compute on smoke, real compute on
        # training): mirrors the v2 long-run machinery so a green smoke means
        # the 50-epoch code path is validated end-to-end (adaptive controller,
        # matrix logging, gated probe gradient, probe LR fix) in a single pass.
        overrides = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "l_star": 12,  # 0.5B has 24 layers -> 0.5 * L
            "num_train_epochs": 1,
            "lifecycle_enabled": False,  # nothing to push for a smoke run
            # PIPELINE VALIDATION, not science: with <=5 epochs A_mem stays
            # low and could never exceed any real tau_0, so the smoke would
            # exercise ONLY the lambda==0 path (old smoke log: lambda=0.0000
            # everywhere). tau_0 < 0 forces the gate OPEN from epoch 1
            # (lambda = lambda_0 = 0.3), so the smoke actually validates the
            # gated probe gradient into the backbone, lambda>0 logging, and
            # the controller's success path. Do NOT copy these two values to
            # real runs.
            "tau_0": -0.1,
            "delta_0": 0.1,
            "gate_metric": "max",
            "probe_learning_rate": 1e-3,
            # Same eval-aware adaptive machinery as the v2 long runs.
            "adaptive_enabled": True,
            "adaptive_eval_decay_patience": 2,
            "adaptive_eval_boost_block": 2,
            "adaptive_stop_on_exhausted": True,
            # Log LoRA/probe matrices in the single epoch (validates the
            # wandb callback torch-import fix instead of 0x in 1 epoch).
            "matrix_log_every_n_epochs": 1,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-smoke-0.5b-1ep-v4",
        }
        output_dir = "/vol/outputs/smoke-qwen2.5-0.5b-v4-1ep"
        push = False
    elif job == "full":
        overrides = {
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-qwen2.5-7b-3ep",
        }
        output_dir = "/vol/outputs/qwen2.5-7b"
        push = True
    elif job == "half50":
        # LONG RUN (paper): Qwen2.5-0.5B, 50 epochs -- fast 2nd scale for the
        # memorization-saturation / gate-activation dynamics curves.
        overrides = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "l_star": 12,
            "num_train_epochs": 50,
            "lifecycle_push_every_n_epochs": 10,
            "adaptive_enabled": True,
            "matrix_log_every_n_epochs": 5,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-qwen2.5-0.5b-50ep-adaptive",
        }
        output_dir = "/vol/outputs/qwen2.5-0.5b-50ep-adaptive"
        push = True
    elif job == "full50":
        # LONG RUN (paper): Qwen2.5-7B, 50 epochs. At 3 epochs A_mem stayed ~0
        # (below tau_0=0.9) so the lambda gate never fired; 50 epochs + the
        # adaptive LR driver should saturate memorization and exercise the
        # full MLPR mechanism.
        overrides = {
            "num_train_epochs": 50,
            "lifecycle_push_every_n_epochs": 10,
            "adaptive_enabled": True,
            "matrix_log_every_n_epochs": 5,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-qwen2.5-7b-50ep-adaptive",
        }
        output_dir = "/vol/outputs/qwen2.5-7b-50ep"
        push = True
    elif job in ("half50v2", "full50v2"):
        # V2 LONG RUN (paper): the relaunch with ALL diagnosed fixes active:
        #   1. eval-aware adaptive controller (damping instead of boosting
        #      while validation diverges)
        #   2. probe warm-up (probe trains from step 0 on detached states;
        #      backbone still only receives the lambda-gated gradient)
        #   3. REACHABLE GATE: gate_metric="max" lets the lambda gate open on
        #      the teacher-forced likelihood A_mem ("knowing") instead of
        #      requiring free-generation EM ("saying") at tau_0=0.9, which
        #      was diagnosed unreachable (plateaus 0-12% vs 0.9).
        #      tau_0=0.8 / delta_0=0.1: saturation = >=80% of D_mem answers
        #      recalled under forcing, lambda ramps to lambda_0 over 10%.
        is_7b = job == "full50v2"
        overrides = {
            "num_train_epochs": 50,
            "lifecycle_push_every_n_epochs": 10,
            "adaptive_enabled": True,
            "matrix_log_every_n_epochs": 5,
            "gate_metric": "max",
            "tau_0": 0.8,
            "delta_0": 0.1,
            "lambda_0": 0.3,
            "probe_learning_rate": 1e-3,
            "adaptive_eval_decay_patience": 2,
            "adaptive_eval_boost_block": 2,
            # v4 anti-spiral floor (gatecheck-validated on GPU): eval-driven
            # decays clamp at lr_mult=0.5 so the half50v3-style decay spiral
            # (1.0 -> 0.49 -> 0.34 -> 0.24 -> 0.17) can never recur; rising
            # eval loss is the memorization signature the gate waits for.
            "adaptive_eval_decay_floor": 0.5,
            "adaptive_stop_on_exhausted": True,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": (
                "mlpr-qwen2.5-7b-50ep-adaptive-v4" if is_7b
                else "mlpr-qwen2.5-0.5b-50ep-adaptive-v4"
            ),
        }
        if not is_7b:
            overrides.update({
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "l_star": 12,
            })
        output_dir = (
            "/vol/outputs/qwen2.5-7b-50ep-adaptive-v4" if is_7b
            else "/vol/outputs/qwen2.5-0.5b-50ep-adaptive-v4"
        )
        push = True
    elif job in ("half50v5", "half50v5ctrl", "q17b50v5"):
        # V5 LONG RUNS (autoresearch champions at production length):
        # the 7B scale point is REPLACED by Qwen3-1.7B (user directive:
        # 0.5B / 1.7B only). Both scales train on dataset_v2 (valid-answer
        # ground truth, |A|=2000, reverse probing).
        #   half50v5      : 0.5B, PROBE-GATED (representation saturation,
        #                   tau_0=0.3) -- Condition B for McNemar.
        #   half50v5ctrl  : 0.5B, lambda_0=0 PURE SFT -- Condition A for
        #                   McNemar (identical data/seed/budget).
        #   q17b50v5      : Qwen3-1.7B-Base, probe-gated (scale point).
        is_qwen17 = job == "q17b50v5"
        is_ctrl = job == "half50v5ctrl"
        overrides = {
            "num_train_epochs": 50,
            "lifecycle_push_every_n_epochs": 10,
            "adaptive_enabled": True,
            "matrix_log_every_n_epochs": 5,
            "dataset_name": "./dataset_v2",
            "adaptive_eval_decay_patience": 2,
            "adaptive_eval_boost_block": 2,
            "adaptive_eval_decay_floor": 0.5,
            "adaptive_stop_on_exhausted": True,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": (
                "mlpr-qwen3-1.7b-50ep-probegate-v5" if is_qwen17
                else ("mlpr-qwen2.5-0.5b-50ep-sft-control-v5" if is_ctrl
                      else "mlpr-qwen2.5-0.5b-50ep-probegate-v5")
            ),
        }
        if is_ctrl:
            # Condition A: identical everything, but the probe loss is
            # never mixed in (lambda_0=0 -> L = L_ce at every step). The
            # probe still trains detached (warm-up) so representation-space
            # curves remain comparable across conditions.
            overrides.update({
                "lambda_0": 0.0,
                "gate_metric": "probe",
                "tau_0": 99.0,
                "delta_0": 0.15,
                "probe_learning_rate": 1e-3,
            })
        else:
            overrides.update(AR_CHAMPION_GATE)
        if is_qwen17:
            overrides.update({
                "model_name": "Qwen/Qwen3-1.7B-Base",
                "l_star": 14,           # Qwen3-1.7B: 28 layers -> 0.5 * L
            })
        else:
            overrides.update({
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "l_star": 12,
            })
        output_dir = (
            "/vol/outputs/qwen3-1.7b-50ep-probegate-v5" if is_qwen17
            else ("/vol/outputs/qwen2.5-0.5b-50ep-sft-control-v5" if is_ctrl
                  else "/vol/outputs/qwen2.5-0.5b-50ep-probegate-v5")
        )
        push = True
    elif job == "gatecheck":
        # GPU VALIDATION of the eval-decay floor fix (v4): a 5-epoch run with
        # an UNREACHABLE gate (tau_0=0.8: A_mem_lf can't get there in 5 ep) so
        # lambda stays 0 the whole run while eval_loss rises -- the exact
        # regime where the half50v3 chain decay-spiralled
        # (lr_mult 1.0 -> 0.49 -> 0.34 -> 0.24 -> 0.17, breaching
        # min_lr_mult=0.3 and choking memorization). Pass criteria:
        # lr_mult never < 0.5 (eval_decay_floor), eval-driven decays stop at
        # the floor, run completes, gate/probe machinery healthy.
        overrides = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "l_star": 12,
            "num_train_epochs": 5,
            "lifecycle_enabled": False,
            "tau_0": 0.8,          # reachable only via real saturation
            "delta_0": 0.1,
            "gate_metric": "max",
            "lambda_0": 0.3,
            "probe_learning_rate": 1e-3,
            "adaptive_enabled": True,
            "adaptive_eval_decay_patience": 2,
            "adaptive_eval_boost_block": 2,
            "adaptive_eval_decay_floor": 0.5,
            "adaptive_stop_on_exhausted": False,
            "matrix_log_every_n_epochs": 1,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-gatecheck-0.5b-5ep",
        }
        output_dir = "/vol/outputs/gatecheck-0.5b"
        push = False
    elif job == "adapt-test":
        # VALIDATION of the adaptive controller: tight thresholds so the
        # plateau boost + stagnation diagnosis demonstrably fire within a
        # few epochs on the 0.5B model (no push, lifecycle off).
        overrides = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "l_star": 12,
            "num_train_epochs": 6,
            "lifecycle_enabled": False,
            "adaptive_enabled": True,
            "adaptive_window_epochs": 2,
            "adaptive_plateau_min_improvement": 0.05,
            "adaptive_stagnation_epochs": 2,
            "adaptive_diagnose_every": 2,
            "adaptive_diagnose_samples": 8,
            "adaptive_boost_factor": 1.5,
            "adaptive_max_boosts": 2,
            "adaptive_stop_on_exhausted": False,
            "matrix_log_every_n_epochs": 1,
            "wandb_entity": None,
            "hub_model_id": None,
            "wandb_run_name": "mlpr-adapt-test-0.5b",
        }
        output_dir = "/vol/outputs/adapt-test-0.5b"
        push = False
    else:
        raise ValueError(
            f"Unknown job: {job!r} (expected 'smoke', 'full', 'half50', "
            f"'full50', 'half50v2', 'full50v2', 'gatecheck' or 'adapt-test')"
        )

    # Resolve W&B entity from the Modal secret's API key
    wandb_entity = _resolve_wandb_entity()
    if wandb_entity:
        overrides["wandb_entity"] = wandb_entity

    # Resolve HF hub target for pushing runs
    hub_model_id = None
    if push:
        hf_user = _resolve_hf_user()
        if hf_user:
            hub_model_id = f"{hf_user}/{hub_suffix[job]}"
            overrides["hub_model_id"] = hub_model_id

    cfg_path = f"/tmp/mlpr_{job}.yaml"
    _write_config("configs/qwen2.5_7b.yaml", cfg_path, overrides)

    cmd = [
        sys.executable,
        "main.py",
        "--config", cfg_path,
        "--output_dir", output_dir,
        "--wandb_project", "knowing_using_gap",
    ]
    if wandb_entity:
        cmd += ["--wandb_entity", wandb_entity]
    if push and hub_model_id:
        cmd += ["--push_to_hub", "--hub_model_id", hub_model_id]

    print("=" * 60)
    print(f"MLPR {job.upper()} RUN")
    print(f"  model      : {overrides.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')}")
    print(f"  epochs     : {overrides.get('num_train_epochs', 3)}")
    print(f"  output_dir : {output_dir}")
    print(f"  wandb      : entity={wandb_entity} project=knowing_using_gap run={overrides.get('wandb_run_name')}")
    print(f"  hub        : {hub_model_id if hub_model_id else '(no push)'}")
    print("=" * 60)

    training_log = os.path.join(output_dir, "training_log.txt")
    returncode = _run_with_tee(cmd, training_log)

    # Persist anything written during the run
    outputs_vol.commit()
    hf_cache_vol.commit()

    if returncode != 0:
        # Still push the (partial) telemetry so failures are debuggable
        # remotely, then fail the job.
        if push and hub_model_id:
            try:
                _upload_telemetry(output_dir, hub_model_id)
            except Exception as e:
                print(f"[HF] WARNING: telemetry upload after failure failed: {e}")
        outputs_vol.commit()
        raise RuntimeError(f"Training failed for job={job} (exit code {returncode})")

    # Full persistence for push jobs: model/checkpoints (lifecycle) + probe
    # head (main.py) + all telemetry JSONs + the complete training log.
    if push and hub_model_id:
        _upload_telemetry(output_dir, hub_model_id)

    print(f"=== MLPR {job} run finished successfully ===")


@app.function(**_TRAIN_FN_KWARGS, timeout=14 * 60 * 60)  # 14h ceiling per single job (50-epoch runs)
def run_training(job: str = "smoke"):
    """Run a single training job as one Modal input."""
    _run_training_impl(job)


@app.function(**_TRAIN_FN_KWARGS_QWEN3, timeout=14 * 60 * 60)
def run_training_qwen3(job: str = "q17b50v5"):
    """Qwen3-image variant (transformers >= 4.51) of run_training."""
    _run_training_impl(job)


@app.function(**_TRAIN_FN_KWARGS, timeout=20 * 60 * 60)  # 20h ceiling: half50v2 (~3h) + full50v2 (~8-12h) sequential
def chained_driver(jobs: list = None):
    """
    Server-side driver: run the given jobs sequentially in one container.

    Because all jobs execute INSIDE this Modal function, the chain is immune
    to local process death (the CLI that spawned this input can disappear).
    If a job fails, the remaining jobs are aborted.
    """
    if jobs is None:
        jobs = ["smoke", "full"]
    for i, job in enumerate(jobs):
        print("=" * 60)
        print(f"CHAIN DRIVER [{i + 1}/{len(jobs)}]: starting {job.upper()} job")
        print("=" * 60)
        _run_training_impl(job)
    print(f"CHAIN DRIVER: all jobs completed successfully: {jobs}")


@app.function(**_TRAIN_FN_KWARGS_QWEN3, timeout=20 * 60 * 60)
def chained_driver_qwen3(jobs: list = None):
    """chained_driver on the Qwen3 image (for queues containing q17b jobs)."""
    if jobs is None:
        jobs = ["q17b50v5"]
    for i, job in enumerate(jobs):
        print("=" * 60)
        print(f"CHAIN DRIVER(Q3) [{i + 1}/{len(jobs)}]: starting {job.upper()} job")
        print("=" * 60)
        _run_training_impl(job)
    print(f"CHAIN DRIVER(Q3): all jobs completed successfully: {jobs}")


@app.local_entrypoint()
def main(smoke: bool = True, full: bool = True, chain: bool = False,
         long: bool = False, adaptive_test: bool = False, skip_smoke: bool = False,
         jobs: str = ""):
    if jobs:
        # Generic detached chain, e.g. --jobs smoke  |  --jobs half50v2,full50v2
        # Jobs that need the Qwen3 image (transformers >= 4.51) route to the
        # qwen3 driver; everything else stays on the proven 4.44 stack.
        job_list = [j.strip() for j in jobs.split(",") if j.strip()]
        if any(j.startswith("q17") or j == "smoke3" for j in job_list):
            call = chained_driver_qwen3.spawn(job_list)
        else:
            call = chained_driver.spawn(job_list)
        call_id = getattr(call, "object_id", None) or str(call)
        print(f"Spawned detached chained run ({' -> '.join(job_list)}): {call_id}")
        print("Monitor with: modal app logs <app-id>")
        return
    if adaptive_test:
        # Validate the adaptive controller on GPU before the long run:
        # sanity smoke -> tight-threshold adaptive run (boosts + diagnosis).
        jobs = ["smoke", "adapt-test"]
        call = chained_driver.spawn(jobs)
        call_id = getattr(call, "object_id", None) or str(call)
        print(f"Spawned detached ADAPTIVE-TEST chained run ({' -> '.join(jobs)}): {call_id}")
        print("Monitor with: modal app logs <app-id>")
        return
    if long:
        # Paper run: sanity smoke (skippable) -> 0.5B x 50 epochs -> 7B x 50
        # epochs, fire-and-forget server-side execution. v2 jobs carry every
        # diagnosed fix (eval-aware controller, probe warm-up, reachable
        # likelihood gate).
        jobs = ["smoke", "half50v2", "full50v2"] if not skip_smoke else ["half50v2", "full50v2"]
        call = chained_driver.spawn(jobs)
        call_id = getattr(call, "object_id", None) or str(call)
        print(f"Spawned detached LONG chained run ({' -> '.join(jobs)}): {call_id}")
        print("Monitor with: modal app logs <app-id>")
        return
    if chain:
        # Fire-and-forget server-side execution: safe to close the terminal
        # immediately after this returns.
        call = chained_driver.spawn(["smoke", "full"])
        call_id = getattr(call, "object_id", None) or str(call)
        print(f"Spawned detached chained run (smoke -> full): {call_id}")
        print("Monitor with: modal app logs <app-id>")
        return
    if smoke:
        run_training.remote("smoke")
    if full:
        run_training.remote("full")
