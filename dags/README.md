# Airflow DAGs (design docs)

These DAGs describe the intended scheduled pipelines (data ingestion and
model retraining), but **no Airflow deployment exists in this repo's
infrastructure** — nothing schedules them today.

Treat them as design documentation. If/when scheduling is needed, the
lightest honest option is a Kubernetes CronJob that runs
`python -m ml.check_consistency` (or a retraining entrypoint) on a schedule;
a full Airflow install is only worth it once there are several pipelines.
