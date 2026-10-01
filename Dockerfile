# Phase 5: containerizes the phase 2 Czech YOLO11n baseline behind the FastAPI service in app/.
#
# Two stages: the first installs Python dependencies into a throwaway prefix; the second copies
# only that prefix plus the small set of runtime system libraries opencv-python needs (libGL and
# friends - the classic "ImportError: libGL.so.1" on a slim base image), so the final image never
# carries pip's wheel cache or anything used only to install packages.
#
# requirements-serving.txt installs plain `torch`/`torchvision` from PyPI rather than the
# official CPU-only wheel index (download.pytorch.org/whl/cpu), which would cut a couple of GB off
# this image - that index wasn't reachable from the network this Dockerfile was developed and
# tested against (see scripts/download_rdd2022.py's docstring for an earlier instance of the same
# kind of restricted-egress problem with FigShare). If your build environment can reach
# download.pytorch.org, swap the two lines in requirements-serving.txt for
# `--extra-index-url https://download.pytorch.org/whl/cpu` to get the slimmer CPU build.
#
# Development note on how this was actually verified: the sandbox this was built in blocks ALL
# container registries at the network egress layer (Docker Hub, GHCR, GCR, Quay, MCR every one
# returned a 403 at the proxy, not from the registry itself) - so `docker build` itself could
# never be run here, only app/ and requirements-serving.txt against a real Python venv (pytest,
# 8/8 passing against the real model weights - see tests/). The libgl1/libglib2.0-0 line below is
# verified indirectly: this same sandbox's own Ubuntu base already has those two packages
# installed, and `import cv2` succeeds there - it is not a substitute for an actual `docker build`,
# which is what .github/workflows/ci.yml now does on every push, on a runner with normal internet
# access. Treat that workflow's first green run as this Dockerfile's real first build, the same
# way phases 2-4 treated their first Kaggle run as the real first GPU test of code only
# self-tested locally against synthetic fixtures beforehand.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt


FROM python:3.11-slim

# libgl1 + libglib2.0-0: opencv-python (an ultralytics dependency) dynamically links against
# libGL at import time even though this service never renders anything - without these two, the
# container fails at `import ultralytics` with "ImportError: libGL.so.1: cannot open shared
# object file", not at build time, which makes it a nasty one to debug from a successful `docker
# build` alone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY app/ app/
COPY runs/phase2/czech_baseline/weights/best.pt runs/phase2/czech_baseline/weights/best.pt

ENV MODEL_PATH=/app/runs/phase2/czech_baseline/weights/best.pt \
    CONF_THRESHOLD=0.25 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN mkdir -p /tmp/ultralytics && chown -R appuser:appuser /tmp/ultralytics /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
