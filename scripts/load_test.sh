#!/usr/bin/env bash
# Phase 6 incident drill: sends a steady stream of /predict requests against the locally running
# api container, so an incident's effect (or its fix) shows up as a sustained trend on the
# Grafana dashboard rather than a single blip.
#
# Usage: scripts/load_test.sh [request_count] [delay_seconds]
set -euo pipefail

COUNT="${1:-60}"
DELAY="${2:-2}"
IMAGE="runs/phase3/failures/United_States/rank01_score10_United_States__United_States_000458.jpg"
URL="http://localhost:8000/predict"

if [ ! -f "$IMAGE" ]; then
  echo "Sample image not found at $IMAGE - run this from the repo root." >&2
  exit 1
fi

echo "Sending $COUNT requests to $URL, ${DELAY}s apart..."
for i in $(seq 1 "$COUNT"); do
  num_detections=$(curl -sf -F "file=@${IMAGE}" "$URL" | python3 -c "import json,sys; print(json.load(sys.stdin)['num_detections'])")
  ts=$(date -u +%H:%M:%S)
  echo "[$ts UTC] request $i/$COUNT -> $num_detections detection(s)"
  sleep "$DELAY"
done