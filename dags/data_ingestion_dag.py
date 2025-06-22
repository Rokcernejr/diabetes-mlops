from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd

from ml.preprocess import preprocess_diabetes_data

logger = logging.getLogger(__name__)

RAW_SOURCE = Path("data/hospital/raw.csv")
RAW_DATA = Path("data/raw/latest.parquet")


def ingest_hospital_data() -> None:
    """Ingest new hospital data and store it for processing."""
    if not RAW_SOURCE.exists():
        logger.warning("Raw data source %s not found", RAW_SOURCE)
        return

    df = pd.read_csv(RAW_SOURCE)
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DATA, index=False)
    logger.info("Ingested %d rows to %s", len(df), RAW_DATA)

PROCESSED_DATA = Path("data/processed/latest.parquet")


def trigger_preprocessing() -> None:
    """Run preprocessing on the latest ingested data."""
    if not RAW_DATA.exists():
        logger.error("Raw dataset %s is missing", RAW_DATA)
        return

    preprocess_diabetes_data(RAW_DATA, output_path=PROCESSED_DATA)
    logger.info("Preprocessed data saved to %s", PROCESSED_DATA)

dag = DAG(
    'diabetes_data_ingestion',
    default_args={
        'owner': 'mlops-team',
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    },
    description='Daily hospital data ingestion',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False
)

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_hospital_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data', 
    python_callable=trigger_preprocessing,
    dag=dag
)

ingest_task >> preprocess_task
