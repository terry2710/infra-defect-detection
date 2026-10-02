# Model Card: infra-defect-detection

**Model type:** Object detection (road/pavement surface defects)
**Architecture:** YOLO11n (CNN, ~2.6M parameters)
**Framework:** Ultralytics YOLO
**Training hardware:** Kaggle, Tesla T4 GPU
**License (code):** MIT
**License (training data):** CC BY-SA 4.0 (see Training Data below)
**Author:** Terry Wang

## Model Details

This model detects four categories of road pavement defects in photographs:
longitudinal cracks, transverse cracks, alligator (fatigue) cracks, and
potholes. It is a YOLO11n model fine-tuned on a single-country subset
(Czech Republic) of the RDD2022 road damage dataset. It is served behind a
FastAPI inference API with Prometheus/Grafana monitoring and a Docker
Compose deployment (see the project README for the full pipeline, phases
1 through 6).

An architecturally larger transformer detector (RT-DETR-L, ~32.8M
parameters) was also trained and evaluated on the identical data and test
split, as a deliberate comparison. It is documented here because the
decision not to deploy it is itself part of the model's story (see
Evaluation Results).

## Intended Use

**Intended for:** a portfolio / technical-demonstration system showing an
end-to-end detection pipeline: data curation, training, cross-domain
failure diagnosis, architecture comparison, and a production-style
deployment with monitoring that can detect its own silent failures.

**Not intended for:** operational road-safety or infrastructure-maintenance
decisions. The model has not been validated against professional pavement
inspection standards, has not been tested on imagery outside the RDD2022
distribution, and its accuracy (mAP@50 ~0.31 on its own test set) is not
sufficient for unsupervised safety-critical use.

## Training Data

Source: [RDD2022](https://github.com/sekilab/RoadDamageDetector)
([paper](https://arxiv.org/abs/2209.08538)), a 47,420-image, 55,000+-instance
road damage dataset spanning six countries (Japan, India, Czech Republic,
Norway, United States, China), collected with different equipment and
conditions per country.

License note: the dataset authors' GitHub README states CC BY-SA 4.0;
FigShare's listing of the same data states CC BY 4.0. This is an
unreconciled discrepancy between the two official sources. This project
treats CC BY-SA 4.0 - the more restrictive of the two - as the operative
license.

This model is trained on the Czech Republic subset only: 1,072 images,
split 750/160/162 into train/val/test (seed 42, reproducible via
`make_country_split.py`). Four classes, matching RDD2022's own taxonomy:

| RDD2022 code | class name | share of test-set objects |
|---|---|---|
| D00 | longitudinal_crack | best-represented |
| D10 | transverse_crack | - |
| D20 | alligator_crack | rarest at 9.6% (full-dataset cross-country split) |
| D40 | pothole | least-represented in Czech test split |

## Training Procedure

YOLO11n, 100 epochs, image size 640, batch size 16, seed 42, trained on a
Kaggle Tesla T4 (13.4 minutes). Checkpoint selected on the `val` split;
final reported metrics are evaluated separately on the `test` split, held
out from both training and checkpoint selection.

## Evaluation Results

### Deployed model (YOLO11n, Czech-only, held-out test set, 162 images)

| metric | value |
|---|---|
| mAP@50 | 0.3113 |
| mAP@50-95 | 0.1070 |
| mean precision | 0.3650 |
| mean recall | 0.3562 |

Per-class AP50 ranges from 0.45 (longitudinal_crack, best-represented
class in the test split) down to 0.19 (pothole, least-represented) -
pothole is consistently the weakest class in every experiment in this
project.

### Architecture comparison (same Czech test set, same split)

| metric | YOLO11n (CNN, ~2.6M) | RT-DETR-L (transformer, ~32.8M) | delta |
|---|---|---|---|
| mAP@50 | 0.3113 | 0.3098 | -0.0015 |
| mAP@50-95 | 0.1070 | 0.1195 | +0.0125 |
| mAP@75 | 0.0370 | 0.0632 | +0.0262 (+70.8%) |
| mean precision | 0.3650 | 0.3778 | +0.0128 |
| mean recall | 0.3562 | 0.3404 | -0.0158 |
| training time | 13.4 min | 59.3 min | 4.4x |

mAP@50 says the two architectures are tied; mAP@75 (strict IoU) says RT-DETR
localizes boxes substantially better. On a 1,072-image single-country
dataset, that gain did not justify a 12.6x larger model and 4.4x the
training cost, so YOLO11n is the model actually deployed.

### Out-of-distribution generalization (separate model, not the deployed one)

A second YOLO11n model, trained on a merged Japan+India+Czech split
(8,536 training images, 70/15/15), was evaluated in-domain and on four
held-out countries never seen in training, to characterize how this model
family degrades under distribution shift:

| domain | n images | mAP@50 | delta vs in-domain |
|---|---|---|---|
| in-domain (held-out) | 1,830 | 0.5198 | - |
| United_States | 4,805 | 0.4316 | -0.088 |
| China_MotorBike | 1,934 | 0.2429 | -0.277 |
| China_Drone | 1,919 | 0.2268 | -0.293 |
| Norway | 2,914 | 0.0686 | -0.451 |

Root cause (diagnosed via failure-case inspection, not assumed): Norway's
collapse tracks its image geometry, not brightness - it is the only
country whose images are not square (mean aspect ratio 1.83 vs. 1.00
elsewhere) and has the highest object density, so resizing to the training
resolution distorts it far more than other countries. United_States (the
smallest drop) has the closest native resolution match to training.

**This experiment used a different training set than the deployed model**
(three countries, not Czech-only), so these numbers characterize the
architecture's sensitivity to distribution shift in general - they are not
a direct measurement of the deployed model's own cross-country behavior,
which has not been separately tested.

## Limitations

- Small, single-country training set for the deployed model (1,072 images)
  - accuracy and generalization should be assumed narrow until evaluated
    on a larger or more diverse sample.
- Pothole is the weakest class across every experiment in this project.
- Demonstrated sensitivity to deployment misconfiguration: an incident
  drill (see `POSTMORTEM.md`) showed that an incorrectly set confidence
  threshold (0.97 instead of the default 0.25) silently drives detections
  to zero while the API continues returning HTTP 200 - a failure mode
  invisible to standard error-rate and latency monitoring, caught only by
  a business-semantic metric built specifically for this scenario.
- Not validated for real-time or safety-critical use.

## Serving Details

Served via a FastAPI inference service (`app/`), containerized with
Docker, with Prometheus metrics and a Grafana dashboard (`monitoring/`).
Confidence threshold is configurable via the `CONF_THRESHOLD` environment
variable (default 0.25) with no startup-time validation of its value -
see Limitations above and `POSTMORTEM.md` for why that matters in
practice. Endpoints: `GET /health`, `POST /predict`, `GET /metrics`.

## Ethical Considerations

Images are of road and pavement surfaces; they do not contain personal
data. Because the training data is CC BY-SA 4.0, redistributing the
trained model weights (as opposed to just the code, which is MIT) should
credit RDD2022 in the spirit of share-alike, even though that obligation
is not fully settled given the authors' own GitHub/FigShare licensing
discrepancy noted above.

## Citation

```
Arya, D., Maeda, H., Ghosh, S.K. et al. RDD2022: A multi-national image
dataset for automatic Road Damage Detection. arXiv:2209.08538 (2022).
https://github.com/sekilab/RoadDamageDetector
```