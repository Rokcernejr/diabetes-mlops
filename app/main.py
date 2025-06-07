from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from prometheus_client import Counter, Histogram, generate_latest
import time

# Import our schemas and dependencies
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.deps import get_model

from app.logging_config import setup_logging
logger = setup_logging()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total ML predictions made', ['model_version'])
MODEL_ACCURACY = Counter('ml_model_accuracy', 'Current model accuracy score')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting MLOps API...")
    
    # Load model on startup
    try:
        # Use our model loader
        model = get_model()
        app.state.model = model
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Continue startup but mark as unhealthy
        app.state.model = None
    
    yield
    
    logger.info("🛑 Shutting down MLOps API...")

app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="Production-ready MLOps API for predicting 30-day hospital readmission",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Add metrics to all requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetes Readmission Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with model status"""
    model_status = "healthy" if hasattr(app.state, 'model') and app.state.model else "unhealthy"
    
    return HealthResponse(
        status="healthy" if model_status == "healthy" else "degraded",
        model_status=model_status,
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "dev")
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make prediction with better error handling"""
    # Check if model is available
    if not hasattr(app.state, 'model') or not app.state.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert to DataFrame
        df = request.as_dataframe()
        
        # Make prediction
        prediction = app.state.model.predict(df)[0]
        probability = app.state.model.predict_proba(df)[0][1]
        
        # Get model version
        model_version = getattr(app.state.model, 'version', '1.0.0')
        
        # Record prediction metric
        PREDICTION_COUNT.labels(model_version=model_version).inc()
        
        # Log prediction for monitoring (async)
        background_tasks.add_task(
            log_prediction, 
            request.model_dump(), 
            prediction, 
            probability,
            model_version
        )
        
        return PredictionResponse(
            readmit=bool(prediction),
            probability=float(probability),
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/explain")
async def explain_prediction(request: PredictionRequest):
    """Get SHAP explanations for a prediction"""
    if not hasattr(app.state, 'model') or not app.state.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Import SHAP utilities
        from app.shap_utils import explain_prediction, get_top_features
        
        df = request.as_dataframe()
        feature_names, shap_values, base_value = explain_prediction(app.state.model, df)
        
        # Get top features
        top_features = get_top_features(feature_names, shap_values, n_top=10)
        
        return {
            "feature_names": feature_names,
            "shap_values": shap_values,
            "base_value": base_value,
            "top_features": [{"feature": name, "importance": value} for name, value in top_features]
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP explanations not available")
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    if not hasattr(app.state, 'model') or not app.state.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    model = app.state.model
    
    return {
        "model_name": getattr(model, 'model_name', 'Unknown'),
        "version": getattr(model, 'version', '1.0.0'),
        "type": type(model).__name__,
        "features": getattr(model, 'features', []),
        "loaded_at": "startup"  # Could track actual load time
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the model (for development/testing)"""
    try:
        # Reload model
        new_model = get_model()
        app.state.model = new_model
        
        logger.info("🔄 Model reloaded successfully")
        
        return {
            "status": "success",
            "message": "Model reloaded",
            "model_version": getattr(new_model, 'version', '1.0.0')
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

async def log_prediction(input_data: dict, prediction: int, probability: float, model_version: str):
    """Log prediction for monitoring and drift detection"""
    # This would typically go to a metrics store or message queue
    logger.info(f"Prediction logged: {prediction} (prob: {probability:.3f}) model: {model_version}")
    
    # Could add additional logging here:
    # - Send to MLflow
    # - Store in database
    # - Send to monitoring system
    # - Calculate drift metrics

# Additional development endpoints
@app.get("/debug/request-stats")
async def request_stats():
    """Get request statistics (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "total_requests": REQUEST_COUNT._value._value,  # Access internal counter
        "avg_duration": "calculated_from_histogram",  # Would need proper calculation
        "model_predictions": PREDICTION_COUNT._value._value
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development"
    )
