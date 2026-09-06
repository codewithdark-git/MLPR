"""One-off diagnostic: how does the W&B API behave for our runs?

Probes run 264s6iw7 (7B adaptive, 50 ep) with different history access
patterns to find out why scan_history(keys=[~2200 keys]) returned nothing.

Usage: modal run MLPR/analysis/wb_probe.py
"""

import modal

app = modal.App("mlpr-wb-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("wandb==0.18.5")
wandb_secret = modal.Secret.from_name("wandb-api-key")
wandb_entity_secret = modal.Secret.from_name("wandb-entity")

RUN_ID = "264s6iw7"


@app.function(image=image, secrets=[wandb_secret, wandb_entity_secret], timeout=15 * 60)
def probe():
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    run = api.run(f"{entity}/knowing_using_gap/{RUN_ID}")
    print("run:", run.name, run.state)

    summ = run.summary
    all_keys = list(summ.keys())
    print(f"summary keys: {len(all_keys)}")
    gate_keys = [k for k in all_keys if str(k).startswith(("gate/", "adaptive/", "lora/"))]
    print("summary gate/adaptive/lora keys:", len(gate_keys), gate_keys[:8])

    # 1. tiny key scan
    n = 0
    last = None
    for row in run.scan_history(keys=["gate/epoch", "gate/a_mem", "gate/lambda",
                                      "gate/a_gen", "gate/gap"]):
        n += 1
        last = row
    print(f"[1] scan 5 gate keys -> rows={n}, last={dict(last) if last else None}")

    # 2. no-keys scan
    n = 0
    first = None
    for row in run.scan_history():
        if first is None:
            first = sorted(row.keys())[:20] if hasattr(row, "keys") else None
        n += 1
        if n >= 5:
            break
    print(f"[2] scan no-keys -> first5 rows={n}, first row keys={first}")

    # 3. history() columns
    h = run.history(samples=200)
    cols = list(h.columns)
    print(f"[3] history(200) shape={h.shape}, n_cols={len(cols)}")
    print("    columns[:40]:", cols[:40])
    interesting = [c for c in cols if c in
                   ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch",
                    "gate/epoch", "gate/a_mem", "gate/lambda")]
    print("    interesting cols present:", interesting)

    # 4. big-key scan (what the collector did)
    big = [f"lora/B_norm/base_model.model.model.layers.{i}.mlp.gate_proj.weight"
           for i in range(28)]
    try:
        n = 0
        for row in run.scan_history(keys=big):
            n += 1
            if n >= 3:
                break
        print(f"[4] scan 28 lora keys -> rows>= {n}")
    except Exception as e:
        print(f"[4] scan 28 lora keys FAILED: {type(e).__name__}: {e}")

    # 5. scan with many keys including the full set
    try:
        many = sorted(set(gate_keys + [k for k in all_keys
                                       if str(k).startswith(("adaptive/", "probe/",
                                                             "lora/", "eval/", "train/"))]))
        print(f"[5] trying scan with {len(many)} keys")
        n = 0
        for row in run.scan_history(keys=many):
            n += 1
            if n >= 60:
                break
        print(f"[5] scan {len(many)} keys -> rows>= {n}")
    except Exception as e:
        print(f"[5] big scan FAILED: {type(e).__name__}: {e}")


@app.local_entrypoint()
def main():
    probe.remote()
