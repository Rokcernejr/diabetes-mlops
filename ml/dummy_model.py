import numpy as np
import pandas as pd
from typing import Union

class DummyModel:
    """
    Dummy model for development/testing when real model isn't available
    """
    
    def __init__(self):
        self.model_name = "DummyDiabetesModel"
        self.version = "dev-1.0.0"
        self.features = [
            'time_in_hospital', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', 'number_diagnoses'
        ]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make dummy predictions based on simple rules"""
        if isinstance(X, pd.DataFrame):
            # Rule-based prediction for demo
            predictions = []
            for _, row in X.iterrows():
                # Simple logic: higher risk if longer hospital stay + more medications
                risk_score = 0
                
                if 'time_in_hospital' in row:
                    risk_score += row['time_in_hospital'] * 0.1
                
                if 'num_medications' in row:
                    risk_score += row['num_medications'] * 0.05
                
                if 'number_emergency' in row:
                    risk_score += row['number_emergency'] * 0.2
                
                # Convert to binary prediction
                predictions.append(1 if risk_score > 0.6 else 0)
            
            return np.array(predictions)
        else:
            # For numpy arrays, predict based on sum
            return (np.sum(X, axis=1) > 10).astype(int)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return prediction probabilities"""
        if isinstance(X, pd.DataFrame):
            probas = []
            for _, row in X.iterrows():
                # Calculate probability based on features
                prob_positive = 0.1  # base probability
                
                if 'time_in_hospital' in row:
                    prob_positive += min(row['time_in_hospital'] * 0.05, 0.3)
                
                if 'num_medications' in row:
                    prob_positive += min(row['num_medications'] * 0.02, 0.2)
                
                if 'number_emergency' in row:
                    prob_positive += min(row['number_emergency'] * 0.15, 0.3)
                
                # Ensure probability is between 0 and 1
                prob_positive = max(0.05, min(0.95, prob_positive))
                
                probas.append([1 - prob_positive, prob_positive])
            
            return np.array(probas)
        else:
            # For numpy arrays
            n_samples = X.shape[0]
            prob_positive = np.random.uniform(0.1, 0.7, n_samples)
            return np.column_stack([1 - prob_positive, prob_positive])