#!/usr/bin/env bash
# Modal token setup for the MLPR training account.
# 2026-08-31: ahsanumar-dev (spend limit exhausted).
# 2026-09-06: ahsanumer267 (current, provided by the user) — active account.
# Keeps the codebase self-contained so any machine can launch jobs with:
#   bash modal_setup.sh && modal run modal_train.py
set -euo pipefail

MODAL_TOKEN_ID="ak-v05YxzRoxaJkqCqag00K47"
MODAL_TOKEN_SECRET="as-VOLfJdPGEW2KnGABNkiPX8"
MODAL_PROFILE="ahsanumer267"

modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"

# Optional Modal secrets (existed on the old ahsanumar-dev account; if absent
# on this account the pipeline still runs — W&B goes to disabled mode and HF
# push is skipped, artifacts land on the Modal volume):
#   wandb-api-key  -> WANDB_API_KEY
#   wandb-entity   -> WANDB_ENTITY
#   hf-token       -> HF_TOKEN
echo "Modal profile active: $(modal profile current)"
echo "Secrets optional on account: wandb-api-key, wandb-entity, hf-token"
