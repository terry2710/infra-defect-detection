"""Train + evaluate a single-country RT-DETR (transformer) baseline on the SAME split as phase 2's
YOLO11 (CNN) baseline (roadmap: infra-defect-detection, phase 4 - architecture comparison).

WHY THIS EXISTS: phase 2 established a same-country CNN baseline (YOLO11n on Czech, mAP@50=0.3113).
The interview feedback this project responds to (Neara) specifically flagged a lack of evidence of
architecture breadth beyond CNN/face-recognition work - phase 4 closes that gap by fine-tuning a
transformer detector (RT-DETR, Ultralytics' real-time DETR variant) and comparing it against the
CNN baseline with EXACTLY ONE variable changed: the model architecture. Everything else - the
Czech train/val/test split (same 750/160/162 images, same seed=42), the held-out-test evaluation
discipline, the metrics reported - is held constant, because a comparison that also changes the
data isn't a clean architecture comparison (that's why this script does NOT regenerate the split
via make_country_split.py: it expects the exact split files phase 2 already produced and committed
under data/splits/Czech/, so both runs are trained/evaluated on byte-identical images).

Usage:
    python scripts/train_transformer_baseline.py --data data/splits/Czech/dataset.yaml
    python scripts/train_transformer_baseline.py --data data/splits/Czech/dataset.yaml \\
        --compare-against runs/detect/val/baseline_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from ultralytics import RTDETR

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "phase4"
DEFAULT_CNN_METRICS = Path(__file__).resolve().parent.parent / "runs" / "detect" / "val" / "baseline_metrics.json"


def _load_cnn_comparison(path):
    """Load phase 2's CNN baseline_metrics.json for the comparison table, if it's present. Returns
    None (not an error) when the file is missing - the report just skips the comparison section
    rather than failing, since this script is still useful standalone (e.g. re-running with a
    different seed/model) even without phase 2's numbers on hand."""
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to the SAME dataset.yaml phase 2 used (data/splits/Czech/dataset.yaml)")
    parser.add_argument("--model", default="rtdetr-l.pt",
                         help="Ultralytics RT-DETR checkpoint to start from (default: rtdetr-l.pt, "
                              "COCO-pretrained - RT-DETR-L, not the larger -x variant, to keep "
                              "training time controlled on a single Kaggle GPU). Use a bare "
                              "'rtdetr-l.yaml' instead to train from random init with no internet "
                              "download, e.g. if the pretrained-weights download is ever unavailable.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                         help="default 8, not phase 2's 16 - RT-DETR-L (~32M params) has a "
                              "noticeably larger memory footprint than YOLO11n (~2.6M params) at "
                              "the same imgsz, so this halves the batch size to fit Kaggle's T4/P100 "
                              "16GB headroom. Raise it if you confirm your GPU has room.")
    parser.add_argument("--seed", type=int, default=42, help="must match the seed already baked into the split files (42)")
    parser.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs with no val improvement)")
    parser.add_argument("--project", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--name", default=None, help="run name (default: derived from the country in --data)")
    parser.add_argument("--compare-against", default=str(DEFAULT_CNN_METRICS),
                         help="path to phase 2's baseline_metrics.json (CNN/YOLO11n numbers) to build "
                              "a Δ comparison table against - default is where phase 2 actually wrote "
                              "it. Pass an empty string to skip the comparison section entirely.")
    args = parser.parse_args(argv)

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"{data_yaml_path} not found - this script expects phase 2's already-committed split "
              f"(data/splits/Czech/dataset.yaml), not a fresh make_country_split.py run.", file=sys.stderr)
        return 1

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])

    split_config_path = data_yaml_path.parent / "split_config.json"
    split_config = json.loads(split_config_path.read_text()) if split_config_path.exists() else None
    country = (split_config or {}).get("country") or data_yaml_path.parent.name
    run_name = args.name or f"{country.lower()}_rtdetr"

    cnn_metrics = _load_cnn_comparison(args.compare_against) if args.compare_against else None
    if args.compare_against and cnn_metrics is None:
        print(f"NOTE: --compare-against {args.compare_against!r} not found - proceeding without a "
              f"CNN comparison table (this run's own numbers are still written normally).")
    elif cnn_metrics is not None:
        if split_config and cnn_metrics.get("split_config") and cnn_metrics["split_config"] != split_config:
            print(f"WARNING: the CNN comparison file's split_config does not match this run's "
                  f"split_config.json - the two runs may not have used the same train/val/test "
                  f"images, which would invalidate a direct comparison. Proceeding anyway, but check "
                  f"this before trusting the Δ numbers.", file=sys.stderr)

    print(f"Training {args.model} on {country} (data={data_yaml_path}, seed={args.seed}, "
          f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch})...")
    if split_config:
        print(f"  split (from {split_config_path.name}): train={split_config['n_train']} "
              f"val={split_config['n_val']} test={split_config['n_test']} "
              f"(split seed={split_config['seed']}, fractions "
              f"{split_config['train_frac']}/{split_config['val_frac']}/{split_config['test_frac']})")

    model = RTDETR(args.model)
    t0 = time.time()
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=run_name,
        exist_ok=True,
        plots=True,
    )
    train_seconds = time.time() - t0

    # Evaluate on the held-out TEST split, same discipline as phase 2 - never seen during training or
    # checkpoint selection. Explicit project/name (unlike letting Ultralytics default to runs/detect/val)
    # so this doesn't collide with/overwrite phase 2's CNN val run living at that same default path.
    print(f"\nEvaluating best checkpoint on the TEST split (never seen during training)...")
    eval_project = str(DEFAULT_RUNS_DIR / "eval")
    test_results = model.val(data=str(data_yaml_path), split="test", plots=True,
                              project=eval_project, name=run_name, exist_ok=True)
    run_dir = Path(test_results.save_dir)

    box = test_results.box
    per_class = []
    for idx in range(len(class_names)):
        if idx in box.ap_class_index:
            p, r, ap50, ap = box.class_result(list(box.ap_class_index).index(idx))
            per_class.append({"class": class_names[idx], "precision": float(p), "recall": float(r),
                               "ap50": float(ap50), "ap50_95": float(ap)})
        else:
            per_class.append({"class": class_names[idx], "precision": None, "recall": None,
                               "ap50": None, "ap50_95": None, "note": "no test-set instances of this class"})

    metrics = {
        "country": country,
        "model": args.model,
        "architecture": "transformer (RT-DETR)",
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_seconds": round(train_seconds, 1),
        "split_config": split_config,
        "overall": {
            "precision_mean": float(box.mp),
            "recall_mean": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "map75": float(box.map75),
        },
        "per_class": per_class,
        "train_run_dir": str(Path(args.project) / run_name),
        "val_run_dir": str(run_dir),
    }

    metrics_path = run_dir / "transformer_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    report_lines = [
        f"# Phase 4 transformer (RT-DETR) baseline - {country}",
        "",
        f"Generated by `scripts/train_transformer_baseline.py`. Same {country} train/val/test split "
        f"as phase 2's CNN baseline (`data/splits/{country}/`, seed={split_config['seed'] if split_config else args.seed}) - "
        f"only the model architecture changed (YOLO11n -> RT-DETR), so any difference below is "
        f"attributable to architecture, not data.",
        "",
        "## Config (for reproducibility)",
        "",
        f"- model: `{args.model}`",
        f"- seed: {args.seed}",
        f"- epochs: {args.epochs} (patience={args.patience})",
        f"- imgsz: {args.imgsz}, batch: {args.batch}",
        f"- train time: {train_seconds / 60:.1f} min",
    ]
    if split_config:
        report_lines += [
            f"- split: train={split_config['n_train']} val={split_config['n_val']} "
            f"test={split_config['n_test']} out of {split_config['n_total']} total "
            f"(seed={split_config['seed']}, fractions "
            f"{split_config['train_frac']}/{split_config['val_frac']}/{split_config['test_frac']})",
        ]
    report_lines += [
        "",
        "## Overall metrics (on the held-out TEST split)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| mAP@50 | {box.map50:.4f} |",
        f"| mAP@50-95 | {box.map:.4f} |",
        f"| mAP@75 | {box.map75:.4f} |",
        f"| mean precision | {box.mp:.4f} |",
        f"| mean recall | {box.mr:.4f} |",
    ]

    if cnn_metrics is not None:
        cnn_overall = cnn_metrics.get("overall", {})
        report_lines += [
            "",
            "## Architecture comparison: RT-DETR (transformer) vs. YOLO11n (CNN, phase 2)",
            "",
            f"Same {country} split, same seed, same held-out test images - the only thing that "
            f"changed between this run and phase 2's is the model architecture "
            f"(`{cnn_metrics.get('model', 'yolo11n.pt')}` -> `{args.model}`).",
            "",
            "| metric | YOLO11n (CNN) | RT-DETR (transformer) | Δ (transformer - CNN) |",
            "|---|---|---|---|",
        ]
        for key, label in [("map50", "mAP@50"), ("map50_95", "mAP@50-95"), ("map75", "mAP@75"),
                            ("precision_mean", "mean precision"), ("recall_mean", "mean recall")]:
            cnn_v = cnn_overall.get(key)
            rtdetr_v = metrics["overall"][key]
            if cnn_v is None:
                report_lines.append(f"| {label} | - | {rtdetr_v:.4f} | - |")
            else:
                report_lines.append(f"| {label} | {cnn_v:.4f} | {rtdetr_v:.4f} | {rtdetr_v - cnn_v:+.4f} |")
        cnn_train_s = cnn_metrics.get("train_seconds")
        if cnn_train_s:
            report_lines += [
                "",
                f"Training time: YOLO11n {cnn_train_s / 60:.1f} min vs. RT-DETR {train_seconds / 60:.1f} min "
                f"({train_seconds / cnn_train_s:.1f}x).",
            ]
    else:
        report_lines += [
            "",
            "## Architecture comparison",
            "",
            f"No CNN comparison file found at `{args.compare_against}` - run phase 2's "
            f"`train_cnn_baseline.py` first (or pass `--compare-against`) to get a side-by-side table.",
        ]

    report_lines += [
        "",
        "## Per-class precision / recall / AP (on the held-out TEST split)",
        "",
        "| class | precision | recall | AP50 | AP50-95 |",
        "|---|---|---|---|---|",
    ]
    for c in per_class:
        if c["precision"] is None:
            report_lines.append(f"| {c['class']} | - | - | - | - ({c.get('note', '')}) |")
        else:
            report_lines.append(f"| {c['class']} | {c['precision']:.4f} | {c['recall']:.4f} | "
                                 f"{c['ap50']:.4f} | {c['ap50_95']:.4f} |")
    train_run_dir = Path(args.project) / run_name
    report_lines += [
        "",
        "## Artifacts",
        "",
        f"- confusion matrix: `{run_dir / 'confusion_matrix.png'}` (raw counts) and "
        f"`{run_dir / 'confusion_matrix_normalized.png'}`",
        f"- PR / F1 / precision / recall curves: `{run_dir}/Box{{PR,F1,P,R}}_curve.png`",
        f"- training curves (loss/mAP per epoch): `{train_run_dir / 'results.png'}` "
        f"(from the `train` run at `{train_run_dir}`, not this `val` run directory)",
        f"- machine-readable metrics: `{metrics_path}`",
        "",
    ]
    report_path = run_dir / "transformer_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"\nWrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(f"\nOverall: mAP50={box.map50:.4f} mAP50-95={box.map:.4f} "
          f"(precision={box.mp:.4f}, recall={box.mr:.4f})")
    if cnn_metrics is not None:
        print(f"vs. CNN (phase 2): mAP50={cnn_metrics['overall']['map50']:.4f} "
              f"(Δ={box.map50 - cnn_metrics['overall']['map50']:+.4f})")
    if any(c["precision"] is None for c in per_class):
        missing = [c["class"] for c in per_class if c["precision"] is None]
        print(f"\nNOTE: {missing} had no instances in the test split - their AP is undefined, not "
              f"zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
