"""Prometheus metrics definitions - the "structured monitoring" half of phase 5.

Kept as one module so /metrics (app/main.py) and the Prometheus scrape config
(monitoring/prometheus.yml) are both working from the same, single list of what's actually
measured, rather than metrics appearing ad hoc wherever someone adds a counter.
"""
from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP requests received, by route and outcome.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request latency, including request parsing and response serialization.",
    ["method", "path"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "inference_duration_seconds",
    "Model forward-pass latency only (model.predict call), excluding image decode/HTTP overhead.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

DETECTIONS_TOTAL = Counter(
    "detections_total",
    "Individual defect detections returned, by class. A proxy for 'is the model still finding "
    "the same mix of defect types it was validated on' - a sustained shift in this distribution "
    "without a corresponding shift in traffic is the kind of thing a dashboard alert should catch.",
    ["class_name"],
)

PREDICT_REQUESTS_WITH_NO_DETECTIONS_TOTAL = Counter(
    "predict_requests_with_no_detections_total",
    "/predict requests that completed successfully but found zero defects above the confidence "
    "threshold. Legitimate on a clean road photo; a sudden spike in this rate against a flat "
    "detections_total is a leading indicator of a silently-broken model (e.g. a bad weights swap) "
    "that a raw error-rate metric would not catch, since these requests still return HTTP 200.",
)

INVALID_IMAGE_TOTAL = Counter(
    "invalid_image_uploads_total",
    "Uploads rejected because they could not be decoded as an image, or exceeded the size cap.",
)
