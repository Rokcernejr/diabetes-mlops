import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


@lru_cache
def get_settings():
    """Get application settings"""
    return {
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "model_name": os.getenv("MODEL_NAME", "diabetes-readmission"),
        "model_stage": os.getenv("MODEL_STAGE", "Production"),
        "model_path": os.getenv("MODEL_PATH", "models/latest_model.joblib"),
    }


class ModelLoader:
    """Singleton model loader with fallback strategies"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, force: bool = False):
        """Return the cached model; re-run the fallback chain when force=True"""
        if self._model is not None and not force:
            return self._model

        type(self)._model = self._load_fresh()
        return self._model

    def _load_fresh(self):
        settings = get_settings()

        # Strategy 1: MLflow registry. The sklearn flavor keeps predict_proba,
        # which the pyfunc flavor does not expose. Imported lazily so unit
        # tests and dummy-model startups never pay the mlflow import chain.
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(settings["mlflow_uri"])
            model_uri = f"models:/{settings['model_name']}/{settings['model_stage']}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded model from MLflow")
            return model
        except Exception as e:
            logger.warning(f"MLflow model loading failed: {e}")

        # Strategy 2: local artifact
        local_model_path = Path(settings["model_path"])
        if local_model_path.exists():
            model = joblib.load(local_model_path)
            logger.info("Loaded local model")
            return model

        # Strategy 3: dummy model for development
        logger.warning("Loading dummy model - not for production!")
        from ml.dummy_model import DummyModel

        return DummyModel()


def get_model(force: bool = False):
    """Dependency to get the current model"""
    return ModelLoader().load_model(force=force)
