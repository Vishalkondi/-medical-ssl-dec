"""FastAPI backend for the research prototype.

Run from the project root (virtual environment active):
    uvicorn backend.main:app --reload --port 8000

Environment variables (all optional):
    MODEL_PATH       default artifacts/model.pt
    MODEL_META_PATH  default artifacts/model_meta.json
    MAX_UPLOAD_MB    default 10
    CORS_ORIGINS     comma-separated, default http://localhost:5173
    DEVICE           auto | cpu   (default auto)

Privacy: uploads are processed in memory and never written to disk. Logs
record only status codes, sizes and timings, never filenames or pixels.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    ClassScore, ErrorResponse, HealthResponse, ModelStatus, PredictionResponse,
)
from backend.services.predictor import DISPLAY_NAMES, Predictor
from src.data.preprocessing import ImageValidationError, decode_grayscale, validate_content
from src.utils.config import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_TYPES = {"image/png", "image/jpeg"}
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

CFG = load_config()
TASK_NAME = "Binary classification: " + " vs ".join(CFG["task"]["classes"])

predictor = Predictor(
    model_path=Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "model.pt")),
    meta_path=Path(os.getenv("MODEL_META_PATH", PROJECT_ROOT / "artifacts" / "model_meta.json")),
    device_pref=os.getenv("DEVICE", "auto"),
)


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="CXR SSL-DEC Research API", version="0.1.0", lifespan=lifespan,
              responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                         415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    log.info("request rejected: %s (%d)", exc.code, exc.status)
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return _error(422, "INVALID_REQUEST", "Request is missing an image file field named 'file'.")


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    log.exception("unhandled error")
    return _error(500, "INTERNAL_ERROR", "Unexpected server error. Please try again.")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/model/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    meta = predictor.meta if predictor.loaded else {}
    return ModelStatus(
        loaded=predictor.loaded,
        message=predictor.message,
        task=TASK_NAME,
        classes=CFG["task"]["classes"],
        model_version=meta.get("model_version"),
        architecture=meta.get("architecture"),
        score_type=meta.get("score_type", "uncalibrated") if predictor.loaded else None,
        metrics=meta.get("metrics", {}),
        max_upload_mb=MAX_UPLOAD_MB,
    )


def _looks_like_image(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    # 1. Validate the upload before touching the model, so users get useful
    #    file errors even when no model is loaded.
    if file.content_type not in ALLOWED_TYPES:
        raise APIError(415, "UNSUPPORTED_TYPE",
                       "Only PNG and JPEG images are supported. DICOM is not supported yet.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise APIError(413, "FILE_TOO_LARGE", f"File exceeds the {MAX_UPLOAD_MB:g} MB limit.")
    if not data:
        raise APIError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if not _looks_like_image(data):
        raise APIError(400, "INVALID_IMAGE", "The file content is not a valid PNG or JPEG image.")
    try:
        gray = decode_grayscale(data)
        validate_content(gray)
    except ImageValidationError as exc:
        raise APIError(400, "INVALID_IMAGE", f"Image could not be used: {exc}") from exc

    # 2. Inference.
    if not predictor.loaded:
        raise APIError(503, "MODEL_NOT_LOADED", predictor.message)
    result = predictor.predict(gray)
    log.info("prediction ok: %d bytes, %.1f ms", len(data), result.inference_ms)

    h, w = gray.shape
    return PredictionResponse(
        predicted_label=result.predicted_label,
        predicted_display_name=DISPLAY_NAMES.get(result.predicted_label, result.predicted_label),
        scores=[ClassScore(label=c, display_name=DISPLAY_NAMES.get(c, c), score=s)
                for c, s in result.scores.items()],
        score_type=predictor.meta.get("score_type", "uncalibrated"),
        threshold=result.threshold,
        model_version=predictor.meta["model_version"],
        inference_ms=round(result.inference_ms, 1),
        image_width=w,
        image_height=h,
    )
