"""Phase 5: containerized inference service for the phase 2 Czech YOLO11n baseline.

Serves the CNN model, not the phase 4 RT-DETR one - phase 4's own diagnosis concluded the
transformer bought localization precision at 4.4x the training cost with no mAP@50 gain on this
dataset size, so the CNN is the one this project actually recommends deploying.
"""
