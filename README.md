# infra-defect-detection

Real-world road/pavement defect detection - the flagship CV project built in response to concrete
interview feedback (Sync Tech, Neara) pointing at two gaps: no public evidence of diagnosing messy
real-world data (vs. clean synthetic/benchmark data), and no public evidence of owning a production
ML system's full lifecycle (deployment, monitoring, incident response). Full 12-week roadmap:
see the published roadmap artifact (link in project notes).

## Why RDD2022

[RDD2022](https://github.com/sekilab/RoadDamageDetector) ([paper](https://arxiv.org/abs/2209.08538))
is a 47,420-image, 55,000+-instance road damage dataset spanning six countries (Japan, India, Czech
Republic, Norway, United States, China) captured with different equipment and conditions - a real,
naturally-occurring cross-country distribution shift, not a synthetically injected one. Four damage
classes: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), D40 (pothole).
Images are CC BY-SA 4.0 per the authors' own GitHub README (FigShare's listing for the same data
shows CC BY 4.0 - an unreconciled discrepancy between the two official sources; treat CC BY-SA 4.0,
the more restrictive one, as the operative license until this is clarified).

## Phase 1: data setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_rdd2022.py

python scripts/convert_voc_to_yolo.py

python scripts/eda_report.py
```

`download_rdd2022.py` downloads one combined ~12.35GB zip (all six countries) from FigShare's
official mirror, MD5-verifies it against the published checksum, extracts it, then links each
country's folder into `data/raw/<country>/` so the next steps don't care about the zip's internal
layout. Safe to re-run - skips the download/verify/extract steps that already succeeded. Originally
this pulled seven per-country zips directly from Sekilab's own S3 bucket, but that bucket started
returning 403 Forbidden on every file (2026-09-16) - see the script's docstring for the full story.
Expect it to take a while over a normal home connection; fine to run in the background.

`convert_voc_to_yolo.py` converts PASCAL VOC XML annotations to YOLO format and builds
`data/manifest.csv` (one row per converted image, tagged with its country - this is what makes the
phase-3 cross-country experiment possible) plus a combined `data/dataset.yaml`.

`eda_report.py` quantifies how the six countries actually differ (class mix, resolution,
brightness) - the documented basis for phase 3's train/test country split, not a guess.

**Result:** 23,767 images converted across seven country/capture-method splits (Japan 7,900,
United_States 4,805, India 3,223, Norway 2,914, China_MotorBike 1,934, China_Drone 1,919,
Czech 1,072). Full cross-country profile in `data/eda_report.md`.

## Phase 2: CNN baseline (Czech)

```bash
python scripts/make_country_split.py --country Czech --seed 42

python scripts/train_cnn_baseline.py \
    --data data/splits/Czech/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --seed 42
```

`make_country_split.py` builds a reproducible 70/15/15 train/val/test split for a single country
from `manifest.csv` - writing image-list `.txt` files, a `dataset.yaml`, and `split_config.json`
(seed, fractions, counts) so phase 3 can reference "the same Czech test set" rather than a fresh
random subset.

`train_cnn_baseline.py` trains a YOLO11 model and evaluates it on the held-out TEST split - never
seen during training or checkpoint selection, unlike the `val` split Ultralytics uses internally -
writing `baseline_metrics.json` and `baseline_report.md` with per-class precision/recall/AP, not
just an aggregate mAP.

`extract_convert_per_country.py` is a Kaggle-specific variant of the download+convert step above:
Kaggle's `/kaggle/working` has a 19.5GiB disk quota, so it reads each requested country's nested
zip fully into memory and deletes the 12.35GB combined zip before extracting anything to disk,
instead of keeping both on disk at once the way `download_rdd2022.py` + `convert_voc_to_yolo.py`
do locally. Takes a `--countries` filter so a single-country run (phase 2's Czech-only run) skips
extracting the other six countries.

**Result:** trained on Kaggle (Tesla T4, 100 epochs, 13.4 min) on a 750/160/162 train/val/test
split of Czech's 1,072 images, evaluated on the held-out test split:

| metric | value |
|---|---|
| mAP@50 | 0.3113 |
| mAP@50-95 | 0.1070 |
| mean precision | 0.3650 |
| mean recall | 0.3562 |

Per-class AP50 ranges from 0.45 (`longitudinal_crack`, the best-represented class in the test
split) down to 0.19 (`pothole`, the least-represented) - the class-imbalance pattern phase 3's
cross-country experiment will need to account for. Full per-class breakdown, confusion matrices,
and training curves in `runs/detect/val/baseline_report.md`.

## Project layout

```
scripts/
  download_rdd2022.py            # phase 1: fetch combined zip (FigShare) + MD5 verify + extract + link per-country
  convert_voc_to_yolo.py         # phase 1: VOC XML -> YOLO txt + manifest.csv
  eda_report.py                  # phase 1: cross-country data profile (committed to git)
  extract_convert_per_country.py # phase 1/2: Kaggle-only disk-quota-safe download+extract+convert, --countries filter
  make_country_split.py          # phase 2: reproducible per-country train/val/test split
  train_cnn_baseline.py          # phase 2: YOLO11 train + held-out-test evaluation + report
data/
  zips/                   # gitignored - raw downloads
  raw/                    # gitignored - extracted per-country VOC data
  processed/              # gitignored - converted YOLO images/labels
  manifest.csv            # gitignored (regenerable, ~24k rows)
  eda_summary.csv         # committed - small, human-checkable summary table
  eda_report.md           # committed - the phase-1 deliverable
  eda_figures/            # committed - class/brightness/count charts per country
  splits/<country>/       # committed - train/val/test .txt + dataset.yaml + split_config.json (phase 2+)
runs/
  phase2/<country>_baseline/     # committed - training run output: weights/best.pt, results.png
  detect/val/                    # committed - held-out test eval: baseline_report.md, baseline_metrics.json, confusion matrices
```

## Roadmap status

- [x] Phase 1 - data setup and cross-country profiling
- [x] Phase 2 - CNN baseline (YOLO11, single-country training) - Czech: mAP@50=0.3113
- [ ] Phase 3 - cross-country distribution-shift diagnosis and mitigation
- [ ] Phase 4 - transformer detector (RT-DETR) fine-tune and architecture comparison
- [ ] Phase 5 - Docker + CI/CD deployment with structured monitoring
- [ ] Phase 6 - deliberately induced + resolved production incident, postmortem
- [ ] Phase 7 - Model Card, public release, resume narrative
