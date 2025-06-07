from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
from pydantic import validator

class PredictionRequest(BaseModel):
    """Request model for diabetes readmission prediction"""
    race: str = Field(..., example="Caucasian")
    gender: str = Field(..., example="Female") 
    age: str = Field(..., example="[60-70)")
    time_in_hospital: int = Field(..., ge=1, le=14, example=5)
    num_medications: int = Field(..., ge=0, example=15)
    number_outpatient: int = Field(0, ge=0, example=0)
    number_emergency: int = Field(0, ge=0, example=0)
    number_inpatient: int = Field(0, ge=0, example=0)
    number_diagnoses: int = Field(..., ge=1, example=9)
    a1c_result: str = Field("None", example=">7")
    max_glu_serum: str = Field("None", example="None")
    change: str = Field("No", example="Ch")
    diabetesMed: str = Field("Yes", example="Yes")
    
    def as_dataframe(self) -> pd.DataFrame:
        """Convert request to DataFrame for model prediction"""
        return pd.DataFrame([self.model_dump()])
    def validate_hospital_stay(cls, v):
        if not 1 <= v <= 14:
            raise ValueError('time_in_hospital must be between 1 and 14 days')
        return v
    def validate_medications(cls, v):
        if not 0 <= v <= 50:
            raise ValueError('num_medications must be between 0 and 50')
        return v
    def validate_age_format(cls, v):
        valid_ages = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        if v not in valid_ages:
            raise ValueError(f'age must be one of: {valid_ages}')
        return v
    def validate_age_format(cls, v):
        valid_ages = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        if v not in valid_ages:
            raise ValueError(f'age must be one of: {valid_ages}')
        return v
    def validate_gender(cls, v):
        if v not in ['Male', 'Female', 'Unknown']:
            raise ValueError('gender must be Male, Female, or Unknown')
        return v
    
    

class PredictionResponse(BaseModel):
    """Response model for diabetes readmission prediction"""
    readmit: bool = Field(..., description="True if readmission predicted within 30 days")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of readmission")
    model_version: str = Field(..., example="1.0.0")

class ShapResponse(BaseModel):
    """Response model for SHAP explanations"""
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_status: str
    version: str
    environment: str