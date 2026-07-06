# diabetes-mlops

Production-style MLOps stack for predicting **30-day hospital readmission of
diabetic patients** (UCI "Diabetes 130-US hospitals" dataset): a FastAPI
serving layer, LightGBM training pipeline, MLflow model registry, Prometheus/
Grafana monitoring, and a Helm chart deployed to EKS via GitHub Actions.

## Architecture

```
                        ┌──────────────────────────────────────────┐
 raw CSV ──► ml/preprocess ──► ml/train ──► sklearn Pipeline artifact
                        │        (one-hot + LightGBM, one object)  │
                        │              │                           │
                        │              ├─► MLflow registry (sklearn flavor)
                        │              └─► models/latest_model.joblib
                        └──────────────────────────────────────────┘
                                       │
   client ──► FastAPI (app/) ──► ModelLoader fallback chain:
              /predict            1. MLflow registry (Production stage)
              /predict/explain    2. local joblib artifact
              /model/reload       3. DummyModel (dev only)
              /health /ready /metrics
                                       │
              Prometheus ◄── /metrics  └─► Grafana dashboards + alert rules
```

The **train/serve contract** lives in [ml/features.py](ml/features.py): the
model trains on exactly the columns a `PredictionRequest` supplies, and all
preprocessing (one-hot encoding, unseen-category handling) is inside the
persisted Pipeline. `tests/test_integration_ml.py` pins this contract.

## Quickstart (local)

```bash
# 1. Install dependencies (Python 3.11+, Poetry 2.x)
poetry install --with dev,train

# 2. Start the full local platform: API, MLflow, MinIO, Postgres, Prometheus, Grafana
make dev

# 3. Check it
make smoke          # cross-platform smoke test against localhost:8000
```

Services: API `:8000` (docs at `/docs`) · MLflow `:5000` · Grafana `:3000`
(admin/admin) · Prometheus `:9090` · MinIO console `:9001`.

Without a trained model the API serves a **DummyModel** and logs a warning —
handy for development, never for production. `/ready` returns 503 until a
model is loaded.

## Data

The dataset is **not** committed (19 MB). Download it from the
[UCI repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
and place `diabetic_data.csv` in the repo root.

## Train a model

```python
from pathlib import Path
from ml.preprocess import preprocess_diabetes_data
from ml.train import train_diabetes_model

preprocess_diabetes_data(Path("diabetic_data.csv"), Path("data/processed.parquet"))
train_diabetes_model(
    Path("data/processed.parquet"),
    model_output_path=Path("models/latest_model.joblib"),
    use_mlflow=True,   # logs + registers "diabetes-readmission" in MLflow
)
```

Check training stability: `python -m ml.check_consistency data/processed.parquet --trials 3`

## Tests

```bash
make test        # full suite, including integration tests
make test-unit   # fast unit tests only (-m "not integration")
make lint        # ruff --fix + black
```

Unit tests inject the DummyModel directly (see `tests/conftest.py`), so they
run with zero infrastructure and never import MLflow.

## Deployment

GitHub Actions ([.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)):

- **pull request / push to main** → `test` (lint + full suite) and `smoke`
  (builds the production Docker image on the runner, boots it, and runs
  `scripts/smoke_test.py` against it) — full diagnostics with **no AWS needed**
- **tag `v*`** → test → smoke → build/push image to ECR (`sha-<sha>` tag,
  auto-creating the ECR repo if missing) → deploy to `production` on
  `prod-diabetes-eks` → Slack notification
- **"Run workflow" button** (Actions tab) → same full chain on demand,
  deploying to `development` on `conai-cluster`
- Required repo secrets (already configured): `AWS_ACCOUNT_ID`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `SLACK_WEBHOOK`; optional repo
  variables: `MLFLOW_URI`, `JWKS_URL`, `ISSUER`.

The Helm chart creates the `diabetes-secrets` Secret from values by default
(`secrets.create: true`); set it to `false` to manage the secret externally:

```bash
kubectl create secret generic diabetes-secrets \
  --from-literal=mlflow-uri=http://mlflow.mlops.svc.cluster.local:5000 \
  --from-literal=jwks-url=https://your-idp/.well-known/jwks.json \
  --from-literal=issuer=https://your-idp/
```

API auth (`JWKS_URL`/`ISSUER`) is enforced outside `development` on the
`/model/reload` endpoint via PyJWT + JWKS.

## Repo layout

| Path | Purpose |
|---|---|
| `app/` | FastAPI serving layer (schemas, auth, metrics, SHAP explanations) |
| `ml/` | Preprocessing, feature contract, training, batch prediction |
| `tests/` | Unit + integration suite |
| `helm/` | Kubernetes chart (HPA, PDB, probes, secret, ServiceMonitor) |
| `infra/` | Terraform (EKS, RDS, S3, IAM, network) |
| `monitoring/`, `prometheus/`, `grafana/` | Alert rules, scrape config, dashboards |
| `dags/` | Airflow pipeline **design docs** (not deployed — see dags/README.md) |
| `docs/ops/` | One-off operational artifacts kept for reference |
