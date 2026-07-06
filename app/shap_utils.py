import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Single-entry cache holding (estimator, explainer). Identity-checked against
# the live estimator so a model reload always rebuilds the explainer.
_explainer_cache: tuple = (None, None)


def _split_pipeline(model):
    """Return (preprocessor, estimator) for sklearn Pipelines, (None, model) otherwise."""
    if hasattr(model, "steps") and hasattr(model, "named_steps"):
        return model[:-1], model.steps[-1][1]
    return None, model


def get_explainer(estimator):
    """Get SHAP explainer for the final estimator (cached per estimator)"""
    global _explainer_cache
    cached_estimator, cached_explainer = _explainer_cache
    if cached_estimator is estimator:
        return cached_explainer

    explainer = None
    try:
        # Tree-based models (LightGBM, XGBoost, sklearn ensembles).
        # sklearn exposes the fitted feature count as n_features_in_.
        if hasattr(estimator, "predict_proba") and hasattr(estimator, "n_features_in_"):
            explainer = shap.TreeExplainer(estimator)
    except Exception as e:
        logger.warning(f"Could not create SHAP explainer: {e}")
        explainer = None

    _explainer_cache = (estimator, explainer)
    return explainer


def explain_prediction(
    model, features: pd.DataFrame
) -> tuple[list[str], list[float], float]:
    """
    Generate SHAP explanations for a prediction.

    Accepts either a bare estimator or an sklearn Pipeline whose last step is
    the estimator; for Pipelines the features are transformed through the
    preprocessing steps first.

    Returns:
        feature_names: List of feature names
        shap_values: List of SHAP values for the prediction
        base_value: Base/expected value
    """
    try:
        preprocessor, estimator = _split_pipeline(model)

        if preprocessor is not None:
            X = preprocessor.transform(features)
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"f{i}" for i in range(X.shape[1])]
        else:
            X = features
            feature_names = features.columns.tolist()

        explainer = get_explainer(estimator)
        if explainer is None:
            return [], [], 0.0

        shap_values = explainer.shap_values(X)

        # Normalise across shap output formats: list-per-class,
        # (n, features) or (n, features, classes) arrays.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        base_value = explainer.expected_value
        if isinstance(base_value, (np.ndarray, list)):
            base_value = base_value[-1]

        row = shap_values[0] if shap_values.ndim > 1 else shap_values
        return feature_names, row.tolist(), float(base_value)

    except Exception:
        logger.exception("SHAP explanation failed")
        return [], [], 0.0


def get_top_features(
    feature_names: list[str], shap_values: list[float], n_top: int = 10
) -> list[tuple[str, float]]:
    """Get top N most important features by absolute SHAP value"""
    if not feature_names or not shap_values:
        return []

    feature_importance = list(zip(feature_names, shap_values, strict=False))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return feature_importance[:n_top]
