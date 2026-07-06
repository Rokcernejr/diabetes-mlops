from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for diabetes readmission prediction"""

    race: str = Field(..., json_schema_extra={"example": "Caucasian"})
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    age: str = Field(..., json_schema_extra={"example": "[60-70)"})
    time_in_hospital: int = Field(..., ge=1, le=14, json_schema_extra={"example": 5})
    num_medications: int = Field(..., ge=0, le=50, json_schema_extra={"example": 15})
    number_outpatient: int = Field(0, ge=0, json_schema_extra={"example": 0})
    number_emergency: int = Field(0, ge=0, json_schema_extra={"example": 0})
    number_inpatient: int = Field(0, ge=0, json_schema_extra={"example": 0})
    number_diagnoses: int = Field(..., ge=1, json_schema_extra={"example": 9})
    a1c_result: str = Field("None", json_schema_extra={"example": ">7"})
    max_glu_serum: str = Field("None", json_schema_extra={"example": "None"})
    change: str = Field("No", json_schema_extra={"example": "Ch"})
    diabetesMed: str = Field("Yes", json_schema_extra={"example": "Yes"})

    def as_dataframe(self) -> pd.DataFrame:
        """Convert request to DataFrame for model prediction"""
        return pd.DataFrame([self.model_dump()])

    @field_validator("age")
    @classmethod
    def validate_age_format(cls, v: str) -> str:
        valid_ages = [
            "[0-10)",
            "[10-20)",
            "[20-30)",
            "[30-40)",
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
            "[90-100)",
        ]
        if v not in valid_ages:
            raise ValueError(f"age must be one of: {valid_ages}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ["Male", "Female", "Unknown"]:
            raise ValueError("gender must be Male, Female, or Unknown")
        return v


class PredictionResponse(BaseModel):
    """Response model for diabetes readmission prediction"""

    readmit: bool = Field(
        ..., description="True if readmission predicted within 30 days"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of readmission"
    )
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})


class TopFeature(BaseModel):
    """A single feature contribution in a SHAP explanation"""

    feature: str
    importance: float


class ShapResponse(BaseModel):
    """Response model for SHAP explanations"""

    feature_names: list[str]
    shap_values: list[float]
    base_value: float
    top_features: list[TopFeature] = []


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_status: str
    version: str
    environment: str
