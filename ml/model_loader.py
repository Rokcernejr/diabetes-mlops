"""Simplified model loading with automatic fallback"""
import os
import joblib
import mlflow
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_latest_model():
    """Load model with fallback strategies"""
    
    # Strategy 1: Load from MLflow (production)
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            model = mlflow.pyfunc.load_model("models:/diabetes-readmission/Production")
            logger.info("✅ Loaded model from MLflow Production")
            return model
    except Exception as e:
        logger.warning(f"MLflow model loading failed: {e}")
    
    # Strategy 2: Load from local file (development)
    local_model_path = Path("models/latest_model.joblib")
    if local_model_path.exists():
        model = joblib.load(local_model_path)
        logger.info("✅ Loaded local model")
        return model
    
    # Strategy 3: Load default/dummy model
    logger.warning("⚠️  Loading dummy model - not for production!")
    from ml.dummy_model import DummyModel
    return DummyModel()

class ModelManager:
    """Manages model lifecycle with hot-reloading"""
    
    def __init__(self):
        self.current_model = None
        self.model_version = None
    
    def reload_model(self):
        """Hot-reload model without service interruption"""
        try:
            new_model = load_latest_model()
            self.current_model = new_model
            logger.info("🔄 Model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Model reload failed: {e}")
            return False