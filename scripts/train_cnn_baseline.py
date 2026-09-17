"""Train + evaluate a single-country YOLO11 baseline (roadmap: infra-defect-detection, phase 2).

WHY THIS EXISTS: phase 2's job is to establish a "same-country train/test" detection baseline
BEFORE phase 3 asks "how much worse does this get when train and test countries differ" - that
comparison is only meaningful if this baseline is: (1) evaluated on a held-out TEST split the model
never saw during training or checkpoint selection (not the 'val' split Ultralytics uses internally
to pick the best epoch - that would leak), (2) reported as more than one aggregate number (mAP alone
hides whether the model just gives up on one class entirely), and (3) fully reproducible (seed, split
fractions, hyperparameters all recorded next to the numbers, not just in this script's defaults).

Usage:
    python scripts/train_cnn_baseline.py --data data/splits/Czech/dataset.yaml
    python scripts/train_cnn_baseline.py --data data/splits/Czech/dataset.yaml --model yolo11s.pt --epochs 100
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "phase2"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to a dataset.yaml written by make_country_split.py")
    parser.add_argument("--model", default="yolo11n.pt",
                         help="Ultralytics model/checkpoint to start from (default: yolo11n.pt, the "
                              "smallest pretrained YOLO11 - appropriate for a baseline on a small "
                              "single-country dataset and Kaggle's free T4 quota). Use a bare "
                              "'yolo11n.yaml' instead to train from random init with no internet "
                              "download, if the pretrained-weights download is ever unavailable.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="must match (or be recorded alongside) the seed used for the split")
    parser.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs with no val improvement)")
    parser.add_argument("--project", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--name", default=None, help="run name (default: derived from the country in --data)")
    args = parser.parse_args(argv)

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"{data_yaml_path} not found - run make_country_split.py first.", file=sys.stderr)
        return 1

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])

    # Pull the split's own recorded config (seed/fractions/counts) if make_country_split.py wrote one
    # alongside dataset.yaml, so the final report shows the ACTUAL split used, not just this script's
    # own --seed default (which only controls training, not how images were assigned to splits).
    split_config_path = data_yaml_path.parent / "split_config.json"
    split_config = json.loads(split_config_path.read_text()) if split_config_path.exists() else None
    country = (split_config or {}).get("country") or data_yaml_path.parent.name
    run_name = args.name or f"{country.lower()}_baseline"

    print(f"Training {args.model} on {country} (data={data_yaml_path}, seed={args.seed}, "
          f"epochs={args.epochs}, imgsz={args.imgsz})...")
    if split_config:
        print(f"  split (from {split_config_path.name}): train={split_config['n_train']} "
              f"val={split_config['n_val']} test={split_config['n_test']} "
              f"(split seed={split_config['seed']}, fractions "
              f"{split_config['train_frac']}/{split_config['val_frac']}/{split_config['test_frac']})")

    model = YOLO(args.model)
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

    # Evaluate on the held-out TEST split explicitly - Ultralytics' training loop only ever looks at
    # 'val' (for early stopping / best-checkpoint selection), so this is the model's first and only
    # exposure to test images, which is what makes this number usable as phase 3's baseline to beat.
    print(f"\nEvaluating best checkpoint on the TEST split (never seen during training)...")
    test_results = model.val(data=str(data_yaml_path), split="test", plots=True)
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

    metrics_path = run_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    report_lines = [
        f"# Phase 2 CNN baseline - {country}",
        "",
        f"Generated by `scripts/train_cnn_baseline.py`. Same-country train/test baseline - phase 3's "
        f"cross-country experiment reports how much this degrades when train and test countries differ.",
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
    report_path = run_dir / "baseline_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"\nWrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(f"\nOverall: mAP50={box.map50:.4f} mAP50-95={box.map:.4f} "
          f"(precision={box.mp:.4f}, recall={box.mr:.4f})")
    if any(c["precision"] is None for c in per_class):
        missing = [c["class"] for c in per_class if c["precision"] is None]
        print(f"\nNOTE: {missing} had no instances in the test split - their AP is undefined, not "
              f"zero. Look at split_config.json / re-run make_country_split.py with a different seed "
              f"if this class is important and the test set is just too small to contain it by chance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
