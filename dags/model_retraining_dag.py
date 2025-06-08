from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
import requests

from monitoring.data_drift import detect_drift
from ml.train import train_diabetes_model

CONFIG_PATH = Path("config/development.yaml")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {"monitoring": {"drift_threshold": 0.1, "drift_method": "kl"}}


def _load_latest_predictions() -> pd.DataFrame:
    path = Path("data/predictions/latest.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_reference_data() -> pd.DataFrame:
    path = Path("data/reference.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def check_model_performance() -> bool:
    """Check for data drift using latest predictions."""
    ref = _load_reference_data()
    cur = _load_latest_predictions()
    if ref.empty or cur.empty:
        return False

    columns = [c for c in ref.columns if c in cur.columns]
    threshold = CONFIG.get("monitoring", {}).get("drift_threshold", 0.1)
    method = CONFIG.get("monitoring", {}).get("drift_method", "kl")
    return detect_drift(ref[columns], cur[columns], method=method, threshold=threshold)

def retrain_model() -> None:
    """Retrain the diabetes model using latest processed data."""
    data_path = Path("data/processed/latest.parquet")
    model_path = Path("models/latest_model.joblib")

    train_diabetes_model(data_path, model_output_path=model_path, use_mlflow=True)

    try:
        requests.post("http://localhost:8000/model/reload", timeout=2)
    except Exception:
        pass

def notify_team(context):
    print("Model retraining completed")

dag = DAG(
    'diabetes_model_retraining',
    default_args={'owner': 'mlops-team'},
    description='Weekly model retraining check',
    schedule_interval='@weekly',
    start_date=datetime(2025, 1, 1)
)

check_task = ShortCircuitOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
    on_success_callback=notify_team
)

check_task >> retrain_task
