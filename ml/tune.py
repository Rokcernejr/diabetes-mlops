import optuna
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbosity': -1,
        'random_state': 42
    }
    
    # Create model
    model = lgb.LGBMClassifier(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna
    
    Returns:
        best_params: Best hyperparameters found
    """
    
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params