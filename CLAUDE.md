# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Train models (logs to MLflow, updates models/best_model_meta.json)
python scripts/train.py

# Run API server locally
uvicorn api.main:app --reload --port 8000

# Run Streamlit dashboard locally
streamlit run ui/app.py

# Install dependencies
pip install -r requirements.txt
```

Environment variables for cloud MLflow tracking (optional for local dev):
```bash
export DAGSHUB_TOKEN=your_token
export MLFLOW_TRACKING_URI=https://dagshub.com/smbrownai/shopper_intervention.mlflow
```

## Architecture

This is an end-to-end ML system for predicting which e-commerce sessions are unlikely to convert, so promotional interventions can be offered.

```
data/online_shoppers_intention.csv  (12,330 sessions, 18 features, ~16% purchase rate)
    ↓
scripts/train.py  →  MLflow experiment tracking  →  models/best_model_meta.json
    ↓
api/main.py (FastAPI)  ←  loaded by  →  ui/app.py (Streamlit)
```

### Key Design Principles

**Shared preprocessing (`scripts/features.py`):** Both `train.py` and `api/main.py` import `build_preprocessor()` from this module. This is the critical link ensuring training and inference use identical feature transformations. When changing preprocessing logic, this is the single source of truth.

**Champion/challenger pattern:** Two models (best and 2nd-best ROC-AUC from each training run) are loaded into memory at API startup. The Streamlit UI can select which to use per-request. Run metadata and MLflow run IDs are persisted in `models/best_model_meta.json`.

**Threshold as business logic:** The intervention threshold (default 0.30) is applied post-prediction and is NOT a model parameter. It is stored in `best_model_meta.json` and adjustable at runtime via `POST /threshold`. Intervene when `P(purchase) < threshold` (single mode) or within a range (range mode).

**Async retraining:** `POST /retrain` writes hyperparameter overrides to a temp JSON file, spawns a subprocess running `train.py` (reads `TRAIN_OVERRIDES_PATH` env var), and Streamlit polls `GET /retrain-status`. After training completes, the API reloads models from MLflow without restarting.

### Data and Models

- **Target:** `Revenue` (bool) — ~84% False (no purchase), ~16% True (purchase)
- **Class imbalance handling:** `class_weight='balanced'` for sklearn models; `scale_pos_weight=5` for XGBoost
- **Primary metric:** ROC-AUC (champion selection)
- **Four models trained per run:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Pipeline structure:** `ColumnTransformer(numeric: impute→scale, categorical: impute→OHE)` + classifier

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check |
| `GET /model-info` | Champion/challenger metadata |
| `POST /predict` | Score single session |
| `POST /predict-batch` | Score up to 25,000 sessions |
| `GET/POST /threshold` | Read or set intervention threshold |
| `POST /retrain` | Trigger async retraining |
| `GET /retrain-status` | Poll training progress |

### Streamlit Dashboard Tabs

1. **Dataset Explorer** — KPIs, purchase rate by month, visitor type breakdown, page value distributions
2. **Score a Session** — Manual form → API call → probability gauge + intervention flag
3. **Batch Scoring** — CSV upload → batch API call → 5 analysis charts + downloadable results
4. **Model Performance** — Champion/challenger metadata, intervention logic explanation
5. **Retrain Model** — Hyperparameter overrides per model, preprocessing options, training status polling

The dashboard reads `API_URL` env var (defaults to the Render deployment URL) for all API calls.

## Deployment

- **API:** Deployed on Render (free tier) via `render.yaml` — runs `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Dashboard:** Deployed on Streamlit Community Cloud
- **Experiment tracking:** DagHub-hosted MLflow at `dagshub.com/smbrownai/shopper_intervention.mlflow`
