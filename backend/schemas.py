"""API response models. This file is the API contract shared with the frontend
(mirrored in frontend/src/types.ts)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Research prototype, not a medical device. The output is a model score for a "
    "radiographic finding defined by report-mined dataset labels. It does not diagnose "
    "cancer or any disease and must not be used for clinical decisions."
)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelStatus(BaseModel):
    loaded: bool
    message: str
    task: str
    classes: list[str]
    model_version: str | None = None
    architecture: str | None = None
    score_type: Literal["uncalibrated", "calibrated"] | None = None
    # Test-set metrics copied from a real evaluation run (Phase 6); empty until then.
    metrics: dict[str, float] = Field(default_factory=dict)
    max_upload_mb: float


class ClassScore(BaseModel):
    label: str
    display_name: str
    score: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_display_name: str
    scores: list[ClassScore]
    score_type: Literal["uncalibrated", "calibrated"]
    threshold: float
    model_version: str
    inference_ms: float
    image_width: int
    image_height: int
    disclaimer: str = DISCLAIMER


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
