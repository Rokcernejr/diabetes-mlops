# Keep the server version aligned with the mlflow-skinny client pinned in poetry.lock
FROM ghcr.io/mlflow/mlflow:v3.14.0
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3
