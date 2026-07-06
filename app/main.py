import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_token
from app.deps import get_model
from app.logging_config import setup_logging
from app.metrics import MetricsMiddleware, generate_metrics, record_prediction
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ShapResponse,
    TopFeature,
)

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting MLOps API...")

    try:
        app.state.model = get_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        # Continue startup; /ready reports 503 until a model is available
        app.state.model = None

    yield

    logger.info("Shutting down MLOps API...")


app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="Production-ready MLOps API for predicting 30-day hospital readmission",
    version="1.0.0",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    return [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]


_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Browsers reject credentialed requests against a wildcard origin
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)


def _current_model():
    return getattr(app.state, "model", None)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetes Readmission Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness: the process is up (degraded if no model is loaded)"""
    model_status = "healthy" if _current_model() else "unhealthy"

    return HealthResponse(
        status="healthy" if model_status == "healthy" else "degraded",
        model_status=model_status,
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.get("/ready")
async def readiness_check():
    """Readiness: only accept traffic once a model is loaded"""
    if not _current_model():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint"""
    return generate_metrics()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Predict 30-day readmission for a single encounter"""
    model = _current_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        df = request.as_dataframe()

        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0][1])
        model_version = getattr(model, "version", "1.0.0")

        record_prediction(model_version)
        background_tasks.add_task(
            log_prediction,
            request.model_dump(),
            int(prediction),
            probability,
            model_version,
        )

        return PredictionResponse(
            readmit=bool(prediction),
            probability=probability,
            model_version=model_version,
        )

    except Exception:
        logger.exception("Prediction failed")
        # Deliberately generic: error details can leak model internals/paths
        raise HTTPException(status_code=500, detail="Prediction failed") from None


@app.post("/predict/explain", response_model=ShapResponse)
async def explain_prediction(request: PredictionRequest):
    """Get SHAP explanations for a prediction"""
    model = _current_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        from app.shap_utils import explain_prediction as shap_explain
        from app.shap_utils import get_top_features
    except ImportError:
        raise HTTPException(
            status_code=501, detail="SHAP explanations not available"
        ) from None

    try:
        df = request.as_dataframe()
        feature_names, shap_values, base_value = shap_explain(model, df)
        top_features = get_top_features(feature_names, shap_values, n_top=10)

        return ShapResponse(
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            top_features=[
                TopFeature(feature=name, importance=value)
                for name, value in top_features
            ],
        )

    except Exception:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail="Explanation failed") from None


@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    model = _current_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")

    return {
        "model_name": getattr(model, "model_name", "Unknown"),
        "version": getattr(model, "version", "1.0.0"),
        "type": type(model).__name__,
        "features": getattr(model, "features", []),
        "loaded_at": "startup",
    }


@app.post("/model/reload")
async def reload_model(user: dict = Depends(verify_token)):
    """Force-reload the model from MLflow/local artifacts (authenticated)"""
    try:
        new_model = get_model(force=True)
    except Exception:
        logger.exception("Model reload failed")
        raise HTTPException(status_code=500, detail="Model reload failed") from None

    app.state.model = new_model
    logger.info("Model reloaded successfully")

    return {
        "status": "success",
        "message": "Model reloaded",
        "model_version": getattr(new_model, "version", "1.0.0"),
    }


async def log_prediction(
    input_data: dict, prediction: int, probability: float, model_version: str
):
    """Log prediction for monitoring and drift detection"""
    logger.info(
        f"Prediction logged: {prediction} (prob: {probability:.3f}) model: {model_version}"
    )


if __name__ == "__main__":
    import uvicorn

    # Import string (not the app object) — required by uvicorn for reload
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT", "development") == "development",
    )
