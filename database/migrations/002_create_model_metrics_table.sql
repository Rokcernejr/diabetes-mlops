-- database/migrations/002_create_model_metrics_table.sql
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    dataset_size INTEGER
);

CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX idx_model_metrics_version ON model_metrics(model_version);
