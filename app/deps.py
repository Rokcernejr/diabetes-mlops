import os
from typing import Optional
from functools import lru_cache
import mlflow
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@lru_cache()
def get_settings():
    """Get application settings"""
    return {
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "model_name": os.getenv("MODEL_NAME", "diabetes-readmission"),
        "model_stage": os.getenv("MODEL_STAGE", "Production")
    }

class ModelLoader:
    """Singleton model loader with fallback strategies"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load model with fallback strategies"""
        if self._model is not None:
            return self._model
            
        settings = get_settings()
        
        # Strategy 1: Load from MLflow
        try:
            mlflow.set_tracking_uri(settings["mlflow_uri"])
            model_uri = f"models:/{settings['model_name']}/{settings['model_stage']}"
            self._model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ Loaded model from MLflow")
            return self._model
        except Exception as e:
            logger.warning(f"MLflow model loading failed: {e}")
        
        # Strategy 2: Load from local file
        local_model_path = Path("models/latest_model.joblib")
        if local_model_path.exists():
            self._model = joblib.load(local_model_path)
            logger.info("✅ Loaded local model")
            return self._model
        
        # Strategy 3: Dummy model for development
        logger.warning("⚠️ Loading dummy model - not for production!")
        from ml.dummy_model import DummyModel
        self._model = DummyModel()
        return self._model

def get_model():
    """Dependency to get the current model"""
    loader = ModelLoader()
    return loader.load_model()