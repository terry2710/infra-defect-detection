"""FastAPI service: serves the phase 2 Czech YOLO11n baseline for single-image defect detection.

Routes:
    GET  /health   - liveness/readiness: is the model loaded, what's it serving.
    POST /predict  - upload one road-surface image, get back detected defects.
    GET  /metrics  - Prometheus exposition format, scraped by monitoring/prometheus.yml.

Run directly for local development:
    uvicorn app.main:app --reload --port 8000
Or via Docker (see Dockerfile / README phase 5 section) for anything that isn't "just me, on my
own machine, iterating".
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.inference import DEFAULT_MODEL_PATH, DefectDetector, InvalidImageError, ModelNotLoadedError
from app.logging_config import configure_logging
from app.metrics import (
    DETECTIONS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INFERENCE_DURATION_SECONDS,
    INVALID_IMAGE_TOTAL,
    PREDICT_REQUESTS_WITH_NO_DETECTIONS_TOTAL,
)

logger = configure_logging(level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

detector = DefectDetector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)
_model_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_load_error
    try:
        detector.load()
        logger.info(
            "model loaded",
            extra={"model_path": str(detector.model_path), "conf_threshold": detector.conf_threshold},
        )
    except Exception as exc:
        # Deliberately does not raise: a bad MODEL_PATH should surface as a failing /health check
        # (so an orchestrator's readiness probe catches it) rather than crash the process on boot,
        # which would just restart-loop with a less informative error in the container logs.
        _model_load_error = str(exc)
        logger.error("model failed to load", extra={"error": _model_load_error})
    yield


app = FastAPI(title="infra-defect-detection API", version="phase5", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_s = time.perf_counter() - t0

    route = request.scope.get("route")
    path_template = route.path if route is not None else request.url.path

    HTTP_REQUESTS_TOTAL.labels(request.method, path_template, response.status_code).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path_template).observe(duration_s)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_s * 1000, 2),
        },
    )
    return response


@app.get("/health")
def health():
    body = {
        "status": "ok" if detector.is_loaded else "unhealthy",
        "model_loaded": detector.is_loaded,
        "model_path": str(detector.model_path),
        "conf_threshold": detector.conf_threshold,
    }
    if detector.is_loaded:
        body["classes"] = detector.class_names
    else:
        body["error"] = _model_load_error
    return JSONResponse(body, status_code=200 if detector.is_loaded else 503)


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    request_id = request.state.request_id
    image_bytes = await file.read()

    try:
        result = detector.predict(image_bytes)
    except InvalidImageError as exc:
        INVALID_IMAGE_TOTAL.inc()
        return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=400)
    except ModelNotLoadedError as exc:
        return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=503)

    INFERENCE_DURATION_SECONDS.observe(result.inference_ms / 1000)
    for det in result.detections:
        DETECTIONS_TOTAL.labels(det.class_name).inc()
    if not result.detections:
        PREDICT_REQUESTS_WITH_NO_DETECTIONS_TOTAL.inc()

    logger.info(
        "prediction",
        extra={
            "request_id": request_id,
            "uploaded_filename": file.filename,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "inference_ms": result.inference_ms,
            "num_detections": len(result.detections),
        },
    )

    return {
        "request_id": request_id,
        "filename": file.filename,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "inference_ms": result.inference_ms,
        "num_detections": len(result.detections),
        "detections": [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox_xyxy": det.bbox_xyxy,
            }
            for det in result.detections
        ],
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
