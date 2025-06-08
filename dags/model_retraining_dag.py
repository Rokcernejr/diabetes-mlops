from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path

from monitoring.data_drift import detect_drift
from ml.train import train_diabetes_model

def check_model_performance() -> bool:
    """Check for data drift and decide if retraining is needed."""
    config_file = Path("config/development.yaml")
    threshold = 0.1
    method = "kl"
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        threshold = cfg.get("monitoring", {}).get("drift_threshold", threshold)
        method = cfg.get("monitoring", {}).get("drift_method", method)

    ref = pd.read_parquet("data/reference.parquet")
    cur = pd.read_parquet("data/current.parquet")
    columns = [c for c in ref.columns if c in cur.columns and c != "readmitted"]
    return detect_drift(ref, cur, columns, method=method, threshold=threshold)

def retrain_model():
    """Retrain the model using the latest processed data."""
    data_path = Path("data/current.parquet")
    model_path = Path("models/latest_model.joblib")
    train_diabetes_model(data_path, model_path, use_mlflow=False)

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
    dag=dag
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
    on_success_callback=notify_team
)

check_task >> retrain_task
