"""Model loading and single-image inference for the serving API.

Deliberately thin: this module owns nothing except "load the weights once, run one image through
them, return plain Python data". FastAPI request/response handling, logging, and metrics all live
in app/main.py so this module stays easy to unit-test or reuse from a batch script.
"""
import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "phase2" / "czech_baseline" / "weights" / "best.pt"
DEFAULT_CONF_THRESHOLD = 0.25
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB - generous for a single road photo, not for arbitrary uploads


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass
class PredictionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before load() has succeeded."""


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes can't be decoded as an image, or exceed the size cap."""


class DefectDetector:
    """Wraps a single Ultralytics YOLO model for repeated single-image inference.

    One instance is created at app startup and reused for every request - re-loading the ~5.5MB
    checkpoint per request would be wasteful and would also hide model-load failures behind the
    first request instead of behind the health check.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self._model: YOLO | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            raise ModelNotLoadedError("call load() before reading class_names")
        return self._model.names

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model weights not found at {self.model_path} - this path is baked into the "
                f"Docker image at build time (see Dockerfile), so this usually means the image "
                f"was built without the runs/phase2/czech_baseline/weights/best.pt file present, "
                f"or MODEL_PATH was overridden to a path that doesn't exist in this container."
            )
        self._model = YOLO(str(self.model_path))

    def predict(self, image_bytes: bytes) -> PredictionResult:
        if self._model is None:
            raise ModelNotLoadedError("model is not loaded - /health should have reported this")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise InvalidImageError(
                f"image is {len(image_bytes)} bytes, exceeding the {MAX_IMAGE_BYTES}-byte cap"
            )
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # force-decode now, inside our try/except, not lazily later during predict()
            image = image.convert("RGB")
        except Exception as exc:
            raise InvalidImageError(f"could not decode uploaded bytes as an image: {exc}") from exc

        t0 = time.perf_counter()
        results = self._model.predict(image, conf=self.conf_threshold, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        boxes = results[0].boxes
        detections = [
            Detection(
                class_id=int(box.cls[0]),
                class_name=self._model.names[int(box.cls[0])],
                confidence=round(float(box.conf[0]), 4),
                bbox_xyxy=[round(float(v), 1) for v in box.xyxy[0]],
            )
            for box in boxes
        ]
        return PredictionResult(
            detections=detections,
            image_width=image.width,
            image_height=image.height,
            inference_ms=round(inference_ms, 2),
        )
