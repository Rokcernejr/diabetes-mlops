from functools import lru_cache
import shap
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_explainer(model):
    """Get SHAP explainer for the model (cached)"""
    try:
        # For tree-based models (LightGBM, XGBoost, etc.)
        if hasattr(model, 'predict_proba') and hasattr(model, 'n_features_'):
            return shap.TreeExplainer(model)
        
        # For sklearn models
        elif hasattr(model, 'predict_proba'):
            return shap.Explainer(model.predict_proba)
        
        # Fallback to generic explainer
        else:
            return shap.Explainer(model.predict)
            
    except Exception as e:
        logger.warning(f"Could not create SHAP explainer: {e}")
        return None

def explain_prediction(model, features: pd.DataFrame) -> Tuple[List[str], List[float], float]:
    """
    Generate SHAP explanations for a prediction
    
    Returns:
        feature_names: List of feature names
        shap_values: List of SHAP values for the prediction
        base_value: Base/expected value
    """
    try:
        explainer = get_explainer(model)
        if explainer is None:
            return [], [], 0.0
        
        # Get SHAP values
        shap_values = explainer.shap_values(features)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - take positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Get feature names
        feature_names = features.columns.tolist()
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Return values for single prediction
        return (
            feature_names,
            shap_values[0].tolist() if len(shap_values.shape) > 1 else shap_values.tolist(),
            float(base_value)
        )
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return [], [], 0.0

def get_top_features(feature_names: List[str], shap_values: List[float], n_top: int = 10) -> List[Tuple[str, float]]:
    """Get top N most important features by absolute SHAP value"""
    if not feature_names or not shap_values:
        return []
    
    # Pair features with their absolute SHAP values
    feature_importance = list(zip(feature_names, shap_values))
    
    # Sort by absolute SHAP value
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return feature_importance[:n_top]