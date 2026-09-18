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
Expect it to take a while over a normal home connection; fine to run in the background. On Kaggle,
this (and `extract_convert_per_country.py` below) will additionally check `/kaggle/input/` for a
pre-built Dataset cache before hitting FigShare at all - see the "Kaggle Dataset caching" note under
phase 3.

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
cross-country experiment needed to account for. Full per-class breakdown, confusion matrices,
and training curves in `runs/detect/val/baseline_report.md`.

## Phase 3: cross-country distribution-shift diagnosis

```bash
python scripts/make_cross_country_split.py --seed 42

python scripts/train_cross_country_baseline.py \
    --data data/splits/cross_country/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --seed 42

python scripts/diagnose_failures.py \
    --weights runs/phase3/cross_country_baseline/weights/best.pt \
    --split-dir data/splits/cross_country \
    --top-n 6 \
    --conf 0.25 \
    --iou 0.5
```

Phase 2 established a same-country baseline. Phase 3 asks the harder question: how much worse does
it get on a country the model never trained on, and *why*.

`make_cross_country_split.py` merges three source countries (Japan, India, Czech - keeping phase
2's Czech data in the training mix rather than discarding it) into a 70/15/15 train/val/
in_domain_test split (seed=42, shuffled across countries so no split accidentally ends up all one
country), and writes a separate full-image evaluation list for each of four held-out target
countries (Norway, United_States, China_MotorBike, China_Drone) that never appear in training -
China's motorbike- and drone-mounted captures are kept as two distinct targets rather than merged
into one "China" number, since they're different equipment/altitude conditions.

`train_cross_country_baseline.py` trains one YOLO11 model, then evaluates that *same* model on the
in-domain held-out test split and on each target country through the identical `model.val()` code
path, so every number in the table below is directly comparable.

`diagnose_failures.py` IoU-matches predictions against ground truth per target country, ranks
images by failure severity (missed + spurious + misclassified detections), and saves annotated
GT-vs-prediction images for the worst cases along with per-image brightness - the evidence used to
test (and rule out) the brightness hypothesis below.

**Result:** trained on Kaggle (Tesla T4, 100 epochs, patience=20, 135 min) on 8,536 training images
merged from Japan+India+Czech, evaluated on the in-domain held-out test split and on each target
country:

| domain | n images | mAP@50 | Δ vs in-domain |
|---|---|---|---|
| in_domain (fair baseline) | 1,830 | 0.5198 | - |
| United_States | 4,805 | 0.4316 | -0.088 |
| China_MotorBike | 1,934 | 0.2429 | -0.277 |
| China_Drone | 1,919 | 0.2268 | -0.293 |
| Norway | 2,914 | 0.0686 | -0.451 |

The interesting part isn't the numbers, it's *why*. Cross-referencing against phase 1's
`eda_report.md` rules out the leading hypothesis (brightness - Norway sits mid-pack at 140.3, while
United_States, the target with the *smallest* drop, is the brightest country in the whole dataset)
and finds a more mechanistic cause instead: Norway's collapse tracks its unique image geometry - the
only country in the dataset that isn't square (mean aspect ratio 1.83 vs. 1.00 everywhere else) and
by far the highest object density (3.85/image vs. 1.6-2.4 elsewhere), so resizing to the training
`imgsz=640` distorts it far more than any other country. United_States (smallest drop) is the
closest format match to training - its native 640x640 resolution exactly matches the training
imgsz. The two China splits, whose image format already matches training, show a genuine
content-domain gap instead. Pothole recall is the weakest class in every target country regardless
of which failure mode applies. Full write-up in `runs/phase3/diagnosis_report.md`; per-domain
metrics and confusion matrices in `runs/phase3/eval/`; annotated failure cases in
`runs/phase3/failures/`.

**Kaggle Dataset caching:** every phase that needs a fresh Kaggle kernel was re-downloading the same
~12.35GB zip from scratch, since `/kaggle/working` doesn't persist across kernel sessions. A
one-time kernel (`kaggle/rdd2022_cache_builder.ipynb`) downloads + MD5-verifies the zip once and
turns its output into a private Kaggle Dataset. Along the way this surfaced a real Kaggle quirk
worth documenting: "New Dataset from notebook output" recursively auto-extracts every zip it finds,
including zips nested inside other zips - so the cache Dataset ended up holding a fully-extracted
per-country directory tree rather than a re-downloadable zip. `download_rdd2022.py` and
`extract_convert_per_country.py` both detect this automatically (`find_kaggle_cached_extracted_countries()`)
and, when the Dataset is mounted, convert straight from it - skipping the FigShare download and the
in-memory zip extraction entirely - falling back to the original path unchanged when no cache is
present.

## Phase 4: transformer detector and architecture comparison

```bash
python scripts/train_transformer_baseline.py \
    --data data/splits/Czech/dataset.yaml \
    --model rtdetr-l.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 8 \
    --seed 42
```

Phase 2 established a CNN baseline. Phase 4 changes the model architecture and nothing else, to
answer whether a transformer detector is actually better on this data - the question behind the
"architecture breadth beyond CNN" interview feedback.

`train_transformer_baseline.py` fine-tunes RT-DETR (Ultralytics' real-time DETR) on the *same*
Czech split phase 2 used, evaluates on the same held-out test split, and emits a side-by-side
comparison table against phase 2's committed `baseline_metrics.json`. It deliberately does not
regenerate the split: it expects phase 2's committed split files, so both runs see byte-identical
images. `kaggle/phase4_kaggle_notebook.ipynb` enforces this rather than assuming it - it
regenerates the split with the same seed and refuses to start training unless the result
hash-matches phase 2's committed files, so a silent data change cannot be mistaken for an
architecture effect.

**Result:** trained on Kaggle (Tesla T4, early-stopped at epoch 73 of 100, patience=20, 59.3 min)
on the identical 750/160/162 Czech split, evaluated on the identical held-out test split:

| metric | YOLO11n (CNN, ~2.6M) | RT-DETR-L (transformer, ~32.8M) | Δ |
|---|---|---|---|
| mAP@50 | 0.3113 | 0.3098 | -0.0015 |
| mAP@50-95 | 0.1070 | 0.1195 | +0.0125 |
| mAP@75 | 0.0370 | 0.0632 | **+0.0262 (+70.8%)** |
| mean precision | 0.3650 | 0.3778 | +0.0128 |
| mean recall | 0.3562 | 0.3404 | -0.0158 |
| training time | 13.4 min | 59.3 min | 4.4x |

The headline metric says the two are tied; the strict-IoU metric says they are not. mAP@50 moves
-0.5% while mAP@75 moves +70.8% - the signature of better box regression rather than better object
finding, which is what RT-DETR's NMS-free direct box prediction would predict. An evaluation
reporting only mAP@50 would have recorded this experiment as a wash.

Two per-class findings needed the confusion matrices to reach, not the AP table. `pothole` AP50
halves (0.1905 -> 0.0921), which reads as a detection failure but is not: at a fixed confidence
threshold RT-DETR actually finds *more* potholes than the CNN (0.31 vs 0.19 correctly classified,
0.58 vs 0.77 lost to background) while its pothole precision drops to 0.1138 from 0.2801. It detects
them and cannot rank them - an operationally different problem with a different fix. And the obvious
explanation for a transformer struggling on one class, data scarcity, is contradicted by Czech's own
class distribution: `alligator_crack` is the rarest class (9.6% of objects) and *improved* +14.2%,
while the more common `pothole` (11.4%) regressed. Full analysis in
`runs/phase4/diagnosis_report.md`; metrics and confusion matrices in `runs/phase4/eval/`.

The conclusion this supports is not that transformers are better here. A 12.6x larger model at 4.4x
the training cost bought localization precision, not detection coverage, on a 1,072-image
single-country dataset - which is a reason to keep the CNN for this deployment and revisit the
choice if the data scales.

## Project layout

```
scripts/
  download_rdd2022.py               # phase 1: fetch combined zip (FigShare) + MD5 verify + extract + link per-country; checks Kaggle Dataset cache first
  convert_voc_to_yolo.py            # phase 1: VOC XML -> YOLO txt + manifest.csv
  eda_report.py                     # phase 1: cross-country data profile (committed to git)
  extract_convert_per_country.py    # phase 1/2/3: Kaggle-only disk-quota-safe download+extract+convert, --countries filter, Kaggle Dataset cache fast path
  make_country_split.py             # phase 2: reproducible per-country train/val/test split
  train_cnn_baseline.py             # phase 2: YOLO11 train + held-out-test evaluation + report
  make_cross_country_split.py       # phase 3: merge source countries -> train/val/in_domain_test + per-target-country eval lists
  train_cross_country_baseline.py   # phase 3: train once, evaluate in-domain + every target country, comparison report
  diagnose_failures.py              # phase 3: IoU-matched failure diagnosis, annotated GT-vs-prediction images, brightness
  train_transformer_baseline.py      # phase 4: RT-DETR fine-tune on phase 2's exact split + automatic CNN-vs-transformer comparison table
data/
  zips/                   # gitignored - raw downloads
  raw/                    # gitignored - extracted per-country VOC data
  processed/               # gitignored - converted YOLO images/labels
  manifest.csv            # gitignored (regenerable, ~24k rows)
  eda_summary.csv         # committed - small, human-checkable summary table
  eda_report.md            # committed - the phase-1 deliverable
  eda_figures/             # committed - class/brightness/count charts per country
  splits/<country>/        # committed - train/val/test .txt + dataset.yaml + split_config.json (phase 2)
  splits/cross_country/    # committed - phase 3: source train/val/in_domain_test + per-target-country eval lists + split_config.json
runs/
  phase2/<country>_baseline/     # committed - training run output: weights/best.pt, results.png
  detect/val/                    # committed - held-out test eval: baseline_report.md, baseline_metrics.json, confusion matrices
  phase3/cross_country_baseline/ # committed - phase 3 training run output: weights/best.pt, results.png, results.csv
  phase3/eval/                   # committed - phase 3: per-domain metrics, confusion matrices, cross_country_report.md
  phase3/failures/               # committed - phase 3: annotated worst-case images per target country, failures_report.md
  phase3/diagnosis_report.md     # committed - the phase-3 deliverable: cross-references phase 1 EDA to explain the drops
  phase4/czech_rtdetr/           # committed - phase 4 RT-DETR training run: results.png, results.csv (weights/last.pt gitignored - 66MB)
  phase4/eval/czech_rtdetr/      # committed - phase 4: transformer_report.md (incl. CNN comparison table), transformer_metrics.json, confusion matrices
  phase4/diagnosis_report.md     # committed - the phase-4 deliverable: what the architecture change actually bought, and which explanations the data refutes
kaggle/
  phase2_kaggle_notebook.ipynb      # phase 2 notebook (writes scripts, downloads, trains, evaluates)
  phase3_kaggle_notebook.ipynb      # phase 3 notebook (writes scripts, wired to the Kaggle Dataset cache)
  phase4_kaggle_notebook.ipynb      # phase 4 notebook (regenerates phase 2's split + hash-verifies it matches before training RT-DETR)
  rdd2022_cache_builder.ipynb       # one-time notebook: builds the rdd2022-figshare-zip-cache Kaggle Dataset
```

## Roadmap status

- [x] Phase 1 - data setup and cross-country profiling
- [x] Phase 2 - CNN baseline (YOLO11, single-country training) - Czech: mAP@50=0.3113
- [x] Phase 3 - cross-country distribution-shift diagnosis - in-domain mAP@50=0.520 vs. targets 0.069-0.432, root-caused (not just measured) per target country in `runs/phase3/diagnosis_report.md`
- [x] Phase 4 - transformer detector (RT-DETR) fine-tune and architecture comparison - mAP@50 tied (0.3098 vs 0.3113) but mAP@75 +70.8%, on a byte-identical split; per-class regressions root-caused in `runs/phase4/diagnosis_report.md`
- [ ] Phase 5 - Docker + CI/CD deployment with structured monitoring
- [ ] Phase 6 - deliberately induced + resolved production incident, postmortem
- [ ] Phase 7 - Model Card, public release, resume narrative
