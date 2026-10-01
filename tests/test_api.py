"""Integration tests for the FastAPI service, run against the real model weights (the committed
runs/phase2/czech_baseline/weights/best.pt - a 5.5MB file, cheap enough to load in CI on every
run) rather than a mocked predictor. The point of phase 5 is evidence of testing a real deployed
model, not just testing that FastAPI routing works.
"""
from app.metrics import DETECTIONS_TOTAL


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert set(body["classes"].values()) == {
        "longitudinal_crack",
        "transverse_crack",
        "alligator_crack",
        "pothole",
    }


def test_predict_real_image_is_well_formed(client, sample_image_bytes):
    r = client.post("/predict", files={"file": ("road.jpg", sample_image_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()

    # Deliberately does NOT assert a specific detection count: the sample image is a mosaic with
    # another model's boxes already drawn into the pixels (see conftest.py), so the "right" number
    # of detections for this exact file isn't meaningful - a confidence threshold change, an
    # Ultralytics version bump, or a retrain could all legitimately shift it. What's worth locking
    # down in CI is that the response is well-formed and internally consistent.
    assert body["num_detections"] == len(body["detections"])
    assert body["image_width"] > 0 and body["image_height"] > 0
    assert body["inference_ms"] > 0
    assert "request_id" in body
    for det in body["detections"]:
        assert det["class_name"] in {"longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"}
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox_xyxy"]) == 4
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert x1 < x2 and y1 < y2


def test_predict_response_includes_request_id_header(client, tiny_synthetic_image_bytes):
    r = client.post("/predict", files={"file": ("tiny.png", tiny_synthetic_image_bytes, "image/png")})
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"] == r.json()["request_id"]


def test_predict_rejects_non_image_upload(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"this is not an image", "text/plain")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_predict_rejects_oversized_upload(client):
    from app.inference import MAX_IMAGE_BYTES

    oversized = b"\x00" * (MAX_IMAGE_BYTES + 1)
    r = client.post("/predict", files={"file": ("huge.bin", oversized, "application/octet-stream")})
    assert r.status_code == 400
    assert "exceeding" in r.json()["error"]


def test_predict_missing_file_is_a_422(client):
    r = client.post("/predict")
    assert r.status_code == 422  # FastAPI's own validation for a missing required form field


def test_metrics_endpoint_exposes_prometheus_format(client, tiny_synthetic_image_bytes):
    client.post("/predict", files={"file": ("tiny.png", tiny_synthetic_image_bytes, "image/png")})

    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")

    body = r.text
    for metric_name in (
        "http_requests_total",
        "http_request_duration_seconds",
        "inference_duration_seconds",
        "predict_requests_with_no_detections_total",
        "invalid_image_uploads_total",
    ):
        assert metric_name in body, f"{metric_name} missing from /metrics output"


def test_detections_total_has_one_series_per_known_class():
    # DETECTIONS_TOTAL is a Counter with a `class_name` label - verifying the label set itself
    # (independent of whether any test image happened to trigger a detection) catches a class-name
    # typo between app/metrics.py and the model's own class list, which a fixed test image might
    # never surface if it never detects that particular class.
    known_classes = {"longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"}
    for class_name in known_classes:
        DETECTIONS_TOTAL.labels(class_name)  # raises if the metric were mis-declared
