# Add this function to ml/train.py
import logger
from pathlib import Path
from typing import Dict, Any
from ml.train import train_diabetes_model

def test_model_consistency(data_path: Path, n_trials: int = 3) -> Dict[str, float]:
    """Test model training consistency"""
    
    logger.info(f"Testing model consistency with {n_trials} trials")
    
    results = []
    for i in range(n_trials):
        logger.info(f"Training trial {i+1}/{n_trials}")
        
        # Use different random seeds for each trial
        import random
        import numpy as np
        seed = random.randint(1, 1000)
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            model, metrics = train_diabetes_model(
                data_path, 
                model_output_path=None,  # Don't save intermediate models
                use_mlflow=False
            )
            results.append(metrics['auc'])
        except Exception as e:
            logger.error(f"Trial {i+1} failed: {e}")
            continue
    
    if not results:
        raise ValueError("All trials failed")
    
    mean_auc = sum(results) / len(results)
    std_auc = (sum((x - mean_auc)**2 for x in results) / len(results))**0.5
    
    consistency_metrics = {
        'trials': len(results),
        'auc_scores': results,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'is_consistent': std_auc < 0.01  # Less than 1% standard deviation
    }
    
    logger.info(f"Consistency test results: {consistency_metrics}")
    return consistency_metrics
