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

## Project layout

```
scripts/
  download_rdd2022.py     # phase 1: fetch combined zip (FigShare) + MD5 verify + extract + link per-country
  convert_voc_to_yolo.py  # phase 1: VOC XML -> YOLO txt + manifest.csv
  eda_report.py           # phase 1: cross-country data profile (committed to git)
data/
  zips/                   # gitignored - raw downloads
  raw/                    # gitignored - extracted per-country VOC data
  processed/              # gitignored - converted YOLO images/labels
  manifest.csv            # gitignored (regenerable, ~tens of thousands of rows)
  eda_summary.csv         # committed - small, human-checkable summary table
  eda_report.md           # committed - the phase-1 deliverable
  eda_figures/            # committed - class/brightness/count charts per country
```

## Roadmap status

- [ ] Phase 1 - data setup and cross-country profiling (this README's scope)
- [ ] Phase 2 - CNN baseline (YOLO11, single-country training)
- [ ] Phase 3 - cross-country distribution-shift diagnosis and mitigation
- [ ] Phase 4 - transformer detector (RT-DETR) fine-tune and architecture comparison
- [ ] Phase 5 - Docker + CI/CD deployment with structured monitoring
- [ ] Phase 6 - deliberately induced + resolved production incident, postmortem
- [ ] Phase 7 - Model Card, public release, resume narrative
