import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

# A real, already-committed Czech road photo to exercise the real model with - not a synthetic
# stand-in. It's phase 4's val-batch visualization (a 3x3 mosaic with RT-DETR's own predicted
# boxes burned into the pixels), not a clean single image, because data/processed/ (the clean
# per-image originals) is gitignored and isn't available outside a fresh data-prep run. That's
# fine for this test's purpose: it proves the real CNN weights load and run end-to-end on a real
# JPEG of realistic size, not that the model finds anything in particular - see the comment on
# test_predict_real_image_is_well_formed for why no detection count is asserted.
SAMPLE_IMAGE_PATH = REPO_ROOT / "runs" / "phase4" / "eval" / "czech_rtdetr" / "val_batch0_pred.jpg"


@pytest.fixture(scope="session")
def client():
    # Imported lazily (inside the fixture, not at module scope) so that test collection doesn't
    # pay the ~5s Ultralytics/torch import cost unless a test actually needs the app.
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes() -> bytes:
    return SAMPLE_IMAGE_PATH.read_bytes()


@pytest.fixture
def tiny_synthetic_image_bytes() -> bytes:
    """A trivial in-memory PNG - for tests that only care about the request/response plumbing
    (schema, status codes) and shouldn't depend on a real road photo being present."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()
