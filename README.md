# Online Shopper Intervention System

Predicts sessions **unlikely to convert** so a promotional incentive can be offered in real time.

## Architecture

```
data/  → raw CSV
scripts/train.py  → MLflow experiment: Logistic Regression, Decision Tree, Random Forest
api/main.py       → FastAPI inference server (loads best MLflow model)
ui/app.py         → Streamlit dashboard (EDA + live scoring + intervention flag)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (logs to MLflow)
```bash
python scripts/train.py
```

### 3. Start MLflow UI (optional, to compare runs)
```bash
mlflow ui --port 5050
# open http://localhost:5050
```

### 4. Start FastAPI server
```bash
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 5. Launch Streamlit UI
```bash
streamlit run ui/app.py
```

## Target Variable
`Revenue = False` → session did NOT purchase → candidate for promotion intervention.

The model outputs a **purchase probability**. Sessions below a configurable threshold
(default 0.30) are flagged for intervention.

## Features Used
- Page interaction counts & durations (Administrative, Informational, ProductRelated)
- Bounce/Exit rates, Page values
- Special day proximity, Month, Weekend flag
- Visitor type, OS, Browser, Region, Traffic type
