-- database/migrations/001_create_predictions_table.sql
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    patient_data JSONB,
    prediction BOOLEAN,
    probability FLOAT,
    model_version VARCHAR(50),
    response_time_ms INTEGER
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);
