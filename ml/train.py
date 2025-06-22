import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, List
import warnings

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, target_col: str = 'readmitted') -> None:
    """Validate dataset before training"""
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Dataset contains {missing} missing values")
    
    # Check class distribution
    class_counts = df[target_col].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Target variable needs at least 2 classes")
    
    minority_class_pct = min(class_counts) / class_counts.sum() * 100
    if minority_class_pct < 1:
        raise ValueError(f"Severe class imbalance: minority class is {minority_class_pct:.1f}%")
    
    # Check for constant features
    feature_cols = [col for col in df.columns if col != target_col]
    constant_features = [col for col in feature_cols if df[col].nunique() <= 1]
    if constant_features:
        logger.warning(f"Found {len(constant_features)} constant features: {constant_features}")
    
    logger.info(f"Data validation passed: {df.shape}, target distribution: {class_counts.to_dict()}")

def split_data(df: pd.DataFrame, target_col: str = 'readmitted', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with validation"""
    
    # Validate data first
    validate_data(df, target_col)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Log feature information
    logger.info(f"Features: {len(X.columns)}")
    logger.info(f"Feature types: numeric={len(X.select_dtypes(include=[np.number]).columns)}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Log split information
    logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    """Train LightGBM model with early stopping and validation"""
    
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000  # Will be reduced by early stopping
        }
    
    logger.info(f"Training LightGBM with parameters: {params}")
    
    model = lgb.LGBMClassifier(**params)
    
    # Prepare evaluation sets
    eval_sets = [(X_train, y_train)]
    eval_names = ['train']
    
    if X_val is not None and y_val is not None:
        eval_sets.append((X_val, y_val))
        eval_names.append('validation')
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=eval_sets,
        eval_names=eval_names,
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=0)  # Suppress training logs
        ]
    )
    
    # Log training results
    if hasattr(model, 'best_iteration'):
        logger.info(f"Best iteration: {model.best_iteration}")
        logger.info(f"Best score: {model.best_score_}")
    
    logger.info("LightGBM model trained successfully")
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    
    logger.info("Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Core metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Classification report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Extract metrics safely
    metrics = {
        'auc': auc_score,
        'accuracy': report['accuracy'],
        'precision': report.get('1', {}).get('precision', 0.0),
        'recall': report.get('1', {}).get('recall', 0.0),
        'f1_score': report.get('1', {}).get('f1-score', 0.0),
        'precision_0': report.get('0', {}).get('precision', 0.0),
        'recall_0': report.get('0', {}).get('recall', 0.0),
        'f1_score_0': report.get('0', {}).get('f1-score', 0.0)
    }
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0
    })
    
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    return metrics

def cross_validate_model(X: pd.DataFrame, y: pd.Series, model_params: Dict[str, Any] = None, cv_folds: int = 5) -> Dict[str, float]:
    """Perform cross-validation for more robust evaluation"""
    
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    if model_params is None:
        model_params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
    
    model = lgb.LGBMClassifier(**model_params)
    
    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    cv_metrics = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'cv_auc_min': cv_scores.min(),
        'cv_auc_max': cv_scores.max()
    }
    
    logger.info(f"Cross-validation AUC: {cv_metrics['cv_auc_mean']:.4f} ± {cv_metrics['cv_auc_std']:.4f}")
    
    return cv_metrics

def train_diabetes_model(data_path: Path, model_output_path: Path = None, use_mlflow: bool = True) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """
    Complete model training pipeline with comprehensive evaluation
    
    Returns:
        model: Trained LightGBM model
        metrics: Dictionary of evaluation metrics
    """
    
    logger.info("=" * 60)
    logger.info("STARTING DIABETES READMISSION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load processed data
    logger.info(f"Loading processed data from {data_path}")
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples for training")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Create validation set from training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Training set: {len(X_train_split)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Start MLflow run
    all_metrics = {}
    if use_mlflow:
        mlflow.start_run()
        mlflow.log_param("train_samples", len(X_train_split))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("target_distribution", dict(y_train.value_counts()))
    
    try:
        # Cross-validation for model stability assessment
        cv_metrics = cross_validate_model(X_train, y_train)
        all_metrics.update(cv_metrics)
        
        # Train final model with early stopping
        logger.info("Training final model...")
        model = train_lightgbm_model(X_train_split, y_train_split, X_val, y_val)
        
        # Evaluate model on test set
        test_metrics = evaluate_model(model, X_test, y_test)
        all_metrics.update(test_metrics)
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(10)
            logger.info("Top 10 most important features:")
            for _, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            all_metrics['n_important_features'] = (feature_importance['importance'] > 0).sum()
        
        # Log metrics to MLflow
        if use_mlflow:
            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.lightgbm.log_model(model, "model", registered_model_name="diabetes-readmission")
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance_path = "feature_importance.csv"
                feature_importance.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)
            
            # Log data artifact
            mlflow.log_artifact(str(data_path), "data")
        
        # Save model locally
        if model_output_path:
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model, model_output_path)
            logger.info(f"Model saved to {model_output_path}")
            
            # Save feature names for consistency
            feature_names_path = model_output_path.parent / "feature_names.txt"
            with open(feature_names_path, 'w') as f:
                for feature in X_train.columns:
                    f.write(f"{feature}\n")
            logger.info(f"Feature names saved to {feature_names_path}")
        
        # Final model summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Final model type: {type(model).__name__}")
        logger.info(f"Training samples: {len(X_train_split)}")
        logger.info(f"Test AUC: {all_metrics.get('auc', 0):.4f}")
        logger.info(f"Cross-validation AUC: {all_metrics.get('cv_auc_mean', 0):.4f} ± {all_metrics.get('cv_auc_std', 0):.4f}")
        logger.info(f"Test Accuracy: {all_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Test Precision: {all_metrics.get('precision', 0):.4f}")
        logger.info(f"Test Recall: {all_metrics.get('recall', 0):.4f}")
        logger.info(f"Test F1-Score: {all_metrics.get('f1_score', 0):.4f}")
        
        # Model performance assessment
        auc_score = all_metrics.get('auc', 0)
        if auc_score > 0.8:
            logger.info("🎉 Excellent model performance (AUC > 0.8)")
        elif auc_score > 0.7:
            logger.info("✅ Good model performance (AUC > 0.7)")
        elif auc_score > 0.6:
            logger.info("⚠️ Fair model performance (AUC > 0.6)")
        else:
            logger.warning("❌ Poor model performance (AUC ≤ 0.6) - consider feature engineering")
        
        logger.info("Model training completed successfully")
        return model, all_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        
    finally:
        if use_mlflow:
            mlflow.end_run()

# Additional utility functions for the enhanced train.py

def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, save_path: Path = None):
    """Plot and optionally save feature importance"""
    try:
        import matplotlib.pyplot as plt
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping feature importance plot")
    except Exception as e:
        logger.error(f"Failed to create feature importance plot: {e}")

def generate_model_report(model, metrics: Dict[str, float], feature_names: List[str], output_path: Path = None) -> str:
    """Generate a comprehensive model report"""
    
    report_lines = [
        "=" * 80,
        "DIABETES READMISSION MODEL REPORT",
        "=" * 80,
        f"Model Type: {type(model).__name__}",
        f"Number of Features: {len(feature_names)}",
        f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PERFORMANCE METRICS:",
        "-" * 40,
        f"AUC-ROC: {metrics.get('auc', 0):.4f}",
        f"Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"Precision: {metrics.get('precision', 0):.4f}",
        f"Recall (Sensitivity): {metrics.get('recall', 0):.4f}",
        f"Specificity: {metrics.get('specificity', 0):.4f}",
        f"F1-Score: {metrics.get('f1_score', 0):.4f}",
        "",
        "CROSS-VALIDATION RESULTS:",
        "-" * 40,
        f"CV AUC Mean: {metrics.get('cv_auc_mean', 0):.4f}",
        f"CV AUC Std: {metrics.get('cv_auc_std', 0):.4f}",
        f"CV AUC Range: [{metrics.get('cv_auc_min', 0):.4f}, {metrics.get('cv_auc_max', 0):.4f}]",
        "",
        "CONFUSION MATRIX:",
        "-" * 40,
        f"True Positives: {metrics.get('true_positives', 0)}",
        f"True Negatives: {metrics.get('true_negatives', 0)}",
        f"False Positives: {metrics.get('false_positives', 0)}",
        f"False Negatives: {metrics.get('false_negatives', 0)}",
        "",
    ]
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        report_lines.extend([
            "TOP 10 IMPORTANT FEATURES:",
            "-" * 40,
        ])
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            report_lines.append(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        report_lines.append("")
    
    # Model interpretation
    auc_score = metrics.get('auc', 0)
    report_lines.extend([
        "MODEL INTERPRETATION:",
        "-" * 40,
    ])
    
    if auc_score > 0.8:
        report_lines.append("✅ EXCELLENT: This model shows excellent discriminative ability.")
    elif auc_score > 0.7:
        report_lines.append("✅ GOOD: This model shows good discriminative ability.")
    elif auc_score > 0.6:
        report_lines.append("⚠️ FAIR: This model shows fair discriminative ability.")
    else:
        report_lines.append("❌ POOR: This model shows poor discriminative ability.")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
    ])
    
    if auc_score < 0.7:
        report_lines.extend([
            "• Consider additional feature engineering",
            "• Explore different algorithms (XGBoost, Neural Networks)",
            "• Check for data quality issues",
            "• Consider collecting more training data",
        ])
    else:
        report_lines.extend([
            "• Model is ready for production deployment",
            "• Monitor model performance over time",
            "• Consider A/B testing against current system",
            "• Set up automated retraining pipeline",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Model report saved to {output_path}")
    
    return report_text





# Generalized Version
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, target_col: str = 'readmitted', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("LightGBM model trained successfully")
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'auc': auc_score,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score']
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics

def train_diabetes_model(data_path: Path, model_output_path: Path = None, use_mlflow: bool = True) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    
    
    # Load processed data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples for training")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Start MLflow run
    if use_mlflow:
        mlflow.start_run()
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(X_train.columns))
    
    try:
        # Train model
        model = train_lightgbm_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log to MLflow
        if use_mlflow:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.lightgbm.log_model(model, "model")
            mlflow.log_artifact(str(data_path), "data")
        
        # Save model locally
        if model_output_path:
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_output_path)
            logger.info(f"Model saved to {model_output_path}")
        
        return model, metrics
        
    finally:
        if use_mlflow:
            mlflow.end_run()
"""
