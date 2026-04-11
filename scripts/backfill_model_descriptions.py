"""
backfill_model_descriptions.py — One-time script to add descriptions to existing
MLflow registered model versions that predate the automated description logic.

Usage:
    python scripts/backfill_model_descriptions.py
"""

import os
import sys
from pathlib import Path

import dagshub.auth
dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN", ""))
import dagshub
dagshub.init(repo_owner='smbrownai', repo_name='shopper_intervention', mlflow=True)

import mlflow
from mlflow.tracking import MlflowClient

MODEL_REGISTRY_NAME = "shopper_best_model"
INTERVENTION_THRESHOLD = 0.30

client = MlflowClient()

# --- Registered model description ---
model_description = (
    "Shopper purchase-intent classifier for e-commerce session intervention.\n\n"
    "Predicts P(purchase) for a browsing session using 17 behavioral and temporal features "
    "(page views, bounce rates, session duration, month, visitor type, etc.).\n\n"
    f"Sessions with P(purchase) < intervention_threshold (default {INTERVENTION_THRESHOLD}) "
    "are flagged for promotional intervention.\n\n"
    "Champion/challenger versions are maintained so the API can A/B compare models at runtime. "
    "Primary selection metric: ROC-AUC on a held-out 20% test split.\n\n"
    "Training data: UCI Online Shoppers Intention dataset (12,330 sessions, ~16% purchase rate)."
)
client.update_registered_model(name=MODEL_REGISTRY_NAME, description=model_description)
print(f"✅ Updated registered model description for '{MODEL_REGISTRY_NAME}'")

# --- Per-version descriptions ---
versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
for mv in versions:
    # Fetch the run to pull metrics and tags
    try:
        run = client.get_run(mv.run_id)
        roc_auc = run.data.metrics.get("roc_auc", float("nan"))
        model_type = run.data.params.get("model_type", "unknown")
        threshold = run.data.params.get("intervention_threshold", str(INTERVENTION_THRESHOLD))
        data_hash = run.data.params.get("data_version_hash", "untracked")
    except Exception as e:
        print(f"  ⚠️  Could not fetch run {mv.run_id} for v{mv.version}: {e}")
        roc_auc = float("nan")
        model_type = "unknown"
        threshold = str(INTERVENTION_THRESHOLD)
        data_hash = "untracked"

    # Determine role from aliases
    aliases = mv.aliases if hasattr(mv, "aliases") else []
    if "champion" in aliases:
        role = "CHAMPION"
    elif "challenger" in aliases:
        role = "CHALLENGER"
    else:
        role = "UNALIASED"

    description = (
        f"{role} — {model_type}\n\n"
        f"Test ROC-AUC : {roc_auc:.4f}\n"
        f"Intervention threshold: P(purchase) < {threshold}\n"
        f"Data version hash: {data_hash}\n"
        f"MLflow run: {mv.run_id}\n\n"
        "Description backfilled by scripts/backfill_model_descriptions.py."
    )

    client.update_model_version(name=MODEL_REGISTRY_NAME, version=mv.version, description=description)
    print(f"  ✅ v{mv.version} ({model_type}, {role})")

print("\nDone.")
