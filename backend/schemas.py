"""
schemas.py
==========
Pydantic request/response models for the Breast Cancer Diagnosis Support API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CellFeatures(BaseModel):
    """
    The 30 numeric features computed from a digitized FNA (fine needle
    aspirate) image of a breast mass, as defined by the WDBC dataset.
    Each of the 10 base measurements is reported as mean, standard error
    (se), and 'worst' (largest / most severe) value.
    """
    mean_radius: float = Field(..., gt=0, description="Mean of distances from center to points on the perimeter")
    mean_texture: float = Field(..., gt=0, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., gt=0)
    mean_area: float = Field(..., gt=0)
    mean_smoothness: float = Field(..., gt=0, description="Local variation in radius lengths")
    mean_compactness: float = Field(..., ge=0)
    mean_concavity: float = Field(..., ge=0, description="Severity of concave portions of the contour")
    mean_concave_points: float = Field(..., ge=0, description="Number of concave portions of the contour")
    mean_symmetry: float = Field(..., gt=0)
    mean_fractal_dimension: float = Field(..., gt=0)

    radius_error: float = Field(..., gt=0)
    texture_error: float = Field(..., gt=0)
    perimeter_error: float = Field(..., gt=0)
    area_error: float = Field(..., gt=0)
    smoothness_error: float = Field(..., gt=0)
    compactness_error: float = Field(..., ge=0)
    concavity_error: float = Field(..., ge=0)
    concave_points_error: float = Field(..., ge=0)
    symmetry_error: float = Field(..., gt=0)
    fractal_dimension_error: float = Field(..., gt=0)

    worst_radius: float = Field(..., gt=0)
    worst_texture: float = Field(..., gt=0)
    worst_perimeter: float = Field(..., gt=0)
    worst_area: float = Field(..., gt=0)
    worst_smoothness: float = Field(..., gt=0)
    worst_compactness: float = Field(..., ge=0)
    worst_concavity: float = Field(..., ge=0)
    worst_concave_points: float = Field(..., ge=0)
    worst_symmetry: float = Field(..., gt=0)
    worst_fractal_dimension: float = Field(..., gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 14.13, "mean_texture": 19.29, "mean_perimeter": 91.97,
                "mean_area": 654.89, "mean_smoothness": 0.096, "mean_compactness": 0.104,
                "mean_concavity": 0.089, "mean_concave_points": 0.048, "mean_symmetry": 0.181,
                "mean_fractal_dimension": 0.063,
                "radius_error": 0.405, "texture_error": 1.217, "perimeter_error": 2.866,
                "area_error": 40.34, "smoothness_error": 0.007, "compactness_error": 0.025,
                "concavity_error": 0.032, "concave_points_error": 0.012, "symmetry_error": 0.020,
                "fractal_dimension_error": 0.0038,
                "worst_radius": 16.27, "worst_texture": 25.68, "worst_perimeter": 107.26,
                "worst_area": 880.58, "worst_smoothness": 0.132, "worst_compactness": 0.254,
                "worst_concavity": 0.272, "worst_concave_points": 0.115, "worst_symmetry": 0.290,
                "worst_fractal_dimension": 0.0839
            }
        }


class PredictionResponse(BaseModel):
    diagnosis: str = Field(..., description="Predicted class: 'malignant' or 'benign'")
    malignant_probability: float = Field(..., description="Model probability that the mass is malignant (0-1)")
    benign_probability: float = Field(..., description="Model probability that the mass is benign (0-1)")
    risk_level: str = Field(..., description="Human-readable triage bucket: Low / Moderate / High")
    confidence: float = Field(..., description="Model confidence in the predicted class (0-1)")
    disclaimer: str = Field(
        default="This tool provides decision support only and does not replace professional "
                "medical diagnosis. All results must be reviewed by a qualified clinician."
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    n_train_samples: int
    n_test_samples: int
    test_metrics: Dict
    malignant_class_recall: float
    feature_order: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
