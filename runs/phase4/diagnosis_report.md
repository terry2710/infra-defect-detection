# Phase 4 diagnosis: RT-DETR (transformer) vs. YOLO11n (CNN)

Companion to `runs/phase4/eval/czech_rtdetr/transformer_report.md`, which reports the numbers.
This file is the analysis: what actually changed when the architecture changed, which of the
obvious explanations survive contact with the evidence, and which do not.

## What was held constant

The comparison is only worth reading if exactly one variable moved. It did. The Czech
`train/val/test.txt` files used by this run are **byte-identical** to the ones phase 2 committed
(verified by `diff`, not just by row counts): the same 750 training images, the same 160 validation
images, the same 162 held-out test images, the same `split_config.json`. Both runs use seed=42,
imgsz=640, 100 epochs with patience=20, and evaluate through the same `model.val(split="test")`
code path. The only differences are the model (`yolo11n.pt` -> `rtdetr-l.pt`) and the batch size
(16 -> 8, forced by RT-DETR-L's memory footprint).

The notebook enforces this rather than trusting it: it regenerates the split and refuses to train
unless the regenerated files hash-match phase 2's committed ones.

## Headline numbers

| metric | YOLO11n (CNN, ~2.6M params) | RT-DETR-L (transformer, ~32.8M params) | Δ | relative |
|---|---|---|---|---|
| mAP@50 | 0.3113 | 0.3098 | -0.0015 | -0.5% |
| mAP@50-95 | 0.1070 | 0.1195 | +0.0125 | +11.7% |
| mAP@75 | 0.0370 | 0.0632 | +0.0262 | **+70.8%** |
| mean precision | 0.3650 | 0.3778 | +0.0128 | +3.5% |
| mean recall | 0.3562 | 0.3404 | -0.0158 | -4.4% |
| training time | 13.4 min | 59.3 min | +45.9 min | **4.4x** |

RT-DETR early-stopped at epoch 73; its best validation mAP@50 (0.4008) was reached at epoch 53.

## Finding 1: the gain is localization quality, not detection coverage

mAP@50 is flat (-0.5%, well inside run-to-run noise) while mAP@75 rises 70.8%. That specific
shape - tied at a loose IoU threshold, far ahead at a strict one - is the signature of better box
regression rather than better object finding. Both models locate roughly the same defects; RT-DETR
draws tighter boxes around them.

This is the expected consequence of the architectural difference rather than a surprise. RT-DETR
regresses boxes directly from object queries under a one-to-one Hungarian matching and needs no NMS,
so there is no anchor-quantisation step and no IoU-threshold-based suppression deciding which of
several overlapping candidates survives. The CNN detector's anchor-and-NMS pipeline is tuned to
produce a box that clears IoU 0.5; nothing in it pushes that box toward IoU 0.75.

Reporting only mAP@50 - the single number most road-damage papers headline - would have shown this
experiment as a wash. It is not a wash; the metric was just too coarse to see it.

## Finding 2: "pothole collapsed" is the wrong conclusion

The per-class table looks unambiguous at first read:

| class | AP50 (CNN) | AP50 (RT-DETR) | relative |
|---|---|---|---|
| longitudinal_crack | 0.4520 | 0.4455 | -1.4% |
| transverse_crack | 0.2788 | 0.3316 | +18.9% |
| alligator_crack | 0.3240 | 0.3700 | +14.2% |
| pothole | 0.1905 | 0.0921 | **-51.6%** |

Read on its own this says RT-DETR stopped finding potholes. The confusion matrices say otherwise.
At the fixed operating point both matrices use (conf >= 0.25), the fraction of true objects assigned
the correct class, and the fraction lost to background, are:

| class | CNN correct | CNN -> background | RT-DETR correct | RT-DETR -> background |
|---|---|---|---|---|
| longitudinal_crack | 0.52 | 0.44 | 0.68 | 0.29 |
| transverse_crack | 0.32 | 0.66 | 0.60 | 0.38 |
| alligator_crack | 0.52 | 0.41 | 0.44 | 0.33 |
| pothole | 0.19 | 0.77 | **0.31** | **0.58** |

At that threshold RT-DETR finds *more* potholes than the CNN does (0.31 vs 0.19) and misses fewer
(0.58 vs 0.77). It misses fewer objects of every class. So the AP50 drop is not a detection failure.

What separates the two metrics is confidence ranking. AP integrates precision across the whole
recall range and rewards a model whose confident predictions are its correct ones; the confusion
matrix fixes one threshold and ignores ranking entirely. RT-DETR's pothole precision at its best-F1
point is 0.1138 against the CNN's 0.2801 - it emits many pothole predictions that are wrong, mixed
in among the right ones at similar confidence. It detects potholes and cannot tell which of its
pothole detections to trust.

That distinction matters operationally. A model that misses potholes needs more data or a better
backbone. A model that finds them but ranks them badly can often be fixed at the threshold or with
calibration, and its raw detections are still usable in a human-review workflow. These call for
different responses, and mAP alone does not distinguish them.

## Finding 3: class frequency does not explain the per-class pattern

The obvious explanation for a transformer doing worse on one class in a 750-image training set is
data scarcity - DETR-family models are known to be data-hungry, and a rare class starves first.
Czech's class distribution refutes it:

| class | share of Czech objects | AP50 change |
|---|---|---|
| longitudinal_crack | 56.7% | -1.4% |
| transverse_crack | 22.3% | +18.9% |
| alligator_crack | 9.6% | +14.2% |
| pothole | 11.4% | -51.6% |

`alligator_crack` is the **rarest** class in Czech (9.6%, roughly 167 of 1,745 objects) and it
improved. `pothole` is more common (11.4%) and it regressed. Frequency and outcome do not line up,
so scarcity is not doing the work here. This is the same trap phase 3's brightness hypothesis fell
into: a plausible mechanism that the dataset's own statistics contradict.

The remaining candidate explanation is object geometry rather than object count. The three crack
classes are extended linear or mesh-like structures occupying a large, elongated image region;
potholes are compact, small, low-contrast blobs. DETR-family detectors are documented to be weaker
on small objects, and a fixed set of object queries has to commit to a small number of predictions
per image, which penalises small objects competing against large obvious ones. This is consistent
with everything measured here but is **not confirmed by this experiment** - confirming it needs the
per-class box-area distribution of the Czech labels, which `data/processed/` is gitignored and would
require re-running the data preparation step to recover. Recorded as the open question, not as a
conclusion.

## Finding 4: alligator_crack errors move from missed to misclassified

`alligator_crack` is the one class where RT-DETR's correct-classification rate falls (0.52 -> 0.44)
even though its miss rate also falls (0.41 -> 0.33). The difference goes to `longitudinal_crack`:
19% of true alligator cracks are predicted as longitudinal cracks by RT-DETR, against 7% for the
CNN. Its AP50 still improves (+14.2%), so this is a labelling error rather than a blindness error.

The reading that fits: alligator cracking is a mesh of individual cracks, and a model with sharper
box regression is more prone to latch onto one linear crack inside the mesh and label it
longitudinal, where a coarser model boxes the whole patch. Better localization has a cost on classes
whose ground-truth boxes are deliberately coarse region annotations.

## The cost side

RT-DETR-L is 12.6x the parameters and 4.4x the training time of YOLO11n, for no mAP@50 gain, a
+11.7% mAP@50-95 gain, and a per-class profile that is better on three classes and worse on one.
On a single Kaggle T4, that is 59.3 minutes against 13.4. Inference cost was not measured here and
would matter more than training cost for any deployment; phase 5 is where that should be quantified.

## What this experiment is evidence of

Not "transformers are better at road damage detection". The defensible claims are narrower and
more useful:

Switching architecture bought localization precision, not detection coverage, and the standard
headline metric (mAP@50) is blind to that - the experiment is an argument for reporting mAP@75 or
mAP@50-95 alongside it. The per-class regression that looks like a detection failure is a confidence
ranking failure, visible only by reading the confusion matrix against the AP table. And on a
1,072-image single-country dataset, a 12.6x larger transformer does not buy a better detector, which
is a reason to prefer the CNN for this deployment and to revisit the choice only if the data scales
or if tight boxes are worth 4.4x the training budget.

## Open question

Whether object size drives the pothole regression (Finding 3). Settling it: recover
`data/processed/Czech/labels/`, compute the box-area distribution per class, and check whether
pothole boxes are systematically smaller than crack boxes by enough to matter. If they are, the
mitigation is a higher-resolution input or a small-object-aware augmentation schedule rather than
more data.
