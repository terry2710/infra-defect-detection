"""Train ONE YOLO11 model on merged source countries, then evaluate it separately on the in-domain
held-out test split AND every target (cross-domain) country, reporting the gap between them
(roadmap: infra-defect-detection, phase 3 - distribution-shift diagnosis).

WHY THIS EXISTS: phase 2 established a same-country baseline (train and test both Czech). Phase 3's
job is to quantify - not just intuit - how much worse detection gets when the test country wasn't in
training, and whether that gap is uniform or concentrated in specific classes/countries. That needs
exactly one trained model evaluated the SAME way (same code path, same metric definitions) on
multiple held-out sets: (1) the in-domain test split - drawn from the source countries but never
trained on, the "fair" number a cross-domain result should be compared against, since it isolates the
country-shift effect from ordinary train/test variance - and (2) each target country's full image
set, evaluated independently rather than pooled, so a report can say e.g. "Norway drops less than
China_Drone" instead of hiding that behind one averaged cross-domain number.

Usage:
    python scripts/train_cross_country_baseline.py --data data/splits/cross_country/dataset.yaml
    python scripts/train_cross_country_baseline.py --data data/splits/cross_country/dataset.yaml \\
        --model yolo11s.pt --epochs 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "phase3"


def _class_names(data_cfg):
    return data_cfg.get("names", [])


def _summarize_box(box, class_names):
    per_class = []
    for idx in range(len(class_names)):
        if idx in box.ap_class_index:
            p, r, ap50, ap = box.class_result(list(box.ap_class_index).index(idx))
            per_class.append({"class": class_names[idx], "precision": float(p), "recall": float(r),
                               "ap50": float(ap50), "ap50_95": float(ap)})
        else:
            per_class.append({"class": class_names[idx], "precision": None, "recall": None,
                               "ap50": None, "ap50_95": None, "note": "no instances of this class in this eval set"})
    overall = {
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "map75": float(box.map75),
    }
    return overall, per_class


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to the cross-country dataset.yaml written by make_cross_country_split.py")
    parser.add_argument("--model", default="yolo11n.pt",
                         help="Ultralytics model/checkpoint to start from (default: yolo11n.pt - kept "
                              "the same as phase 2's baseline so the two are architecture-comparable; "
                              "phase 4 is where architecture itself becomes the variable).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="must match (or be recorded alongside) the seed used for the split")
    parser.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs with no val improvement)")
    parser.add_argument("--project", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--name", default="cross_country_baseline", help="training run name")
    args = parser.parse_args(argv)

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"{data_yaml_path} not found - run make_cross_country_split.py first.", file=sys.stderr)
        return 1

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    class_names = _class_names(data_cfg)

    split_dir = data_yaml_path.parent
    split_config_path = split_dir / "split_config.json"
    if not split_config_path.exists():
        print(f"{split_config_path} not found - {data_yaml_path} doesn't look like it was written by "
              f"make_cross_country_split.py (no split_config.json alongside it).", file=sys.stderr)
        return 1
    split_config = json.loads(split_config_path.read_text())
    test_countries = split_config["test_countries"]

    missing_eval_yamls = [c for c in test_countries if not (split_dir / f"eval_{c}.yaml").exists()]
    if missing_eval_yamls:
        print(f"Missing eval yaml(s) for {missing_eval_yamls} in {split_dir} - re-run "
              f"make_cross_country_split.py.", file=sys.stderr)
        return 1

    print(f"Training {args.model} on source countries {split_config['train_countries']} "
          f"(data={data_yaml_path}, seed={args.seed}, epochs={args.epochs}, imgsz={args.imgsz})...")
    print(f"  split (from {split_config_path.name}): train={split_config['n_train']} "
          f"val={split_config['n_val']} in_domain_test={split_config['n_in_domain_test']} "
          f"(split seed={split_config['seed']}, fractions "
          f"{split_config['train_frac']}/{split_config['val_frac']}/{split_config['in_domain_test_frac']})")
    print(f"  target countries (never trained on): {test_countries}")

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
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    train_seconds = time.time() - t0
    train_run_dir = Path(args.project) / args.name

    # Evaluate the SAME trained model on the in-domain test split and on each target country, one
    # domain at a time, using the identical model.val(..., split="test") code path for all of them so
    # the numbers are comparable - the only thing that changes between calls is which dataset.yaml
    # (and therefore which images) is passed in.
    eval_project = Path(args.project) / "eval"
    domains = {}  # domain name -> {"overall":..., "per_class":..., "n_images":..., "run_dir":...}

    print(f"\nEvaluating on in-domain test split ({split_config['n_in_domain_test']} images, "
          f"never seen during training)...")
    in_domain_results = model.val(data=str(data_yaml_path), split="test", plots=True,
                                   project=str(eval_project), name="in_domain", exist_ok=True)
    overall, per_class = _summarize_box(in_domain_results.box, class_names)
    domains["in_domain"] = {"overall": overall, "per_class": per_class,
                             "n_images": split_config["n_in_domain_test"],
                             "run_dir": str(in_domain_results.save_dir)}
    print(f"  in_domain: mAP50={overall['map50']:.4f} mAP50-95={overall['map50_95']:.4f}")

    for country in test_countries:
        eval_yaml_path = split_dir / f"eval_{country}.yaml"
        n_images = split_config["target_country_counts"].get(country, "?")
        print(f"\nEvaluating on target country {country} ({n_images} images, never trained on)...")
        results = model.val(data=str(eval_yaml_path), split="test", plots=True,
                             project=str(eval_project), name=country, exist_ok=True)
        overall, per_class = _summarize_box(results.box, class_names)
        domains[country] = {"overall": overall, "per_class": per_class, "n_images": n_images,
                             "run_dir": str(results.save_dir)}
        print(f"  {country}: mAP50={overall['map50']:.4f} mAP50-95={overall['map50_95']:.4f}")

    in_domain_map50 = domains["in_domain"]["overall"]["map50"]

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_seconds": round(train_seconds, 1),
        "split_config": split_config,
        "domains": domains,
        "train_run_dir": str(train_run_dir),
    }
    metrics_path = eval_project / "cross_country_metrics.json"
    eval_project.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    report_lines = [
        "# Phase 3 cross-country baseline",
        "",
        f"Generated by `scripts/train_cross_country_baseline.py`. One model trained on "
        f"{split_config['train_countries']}, evaluated separately on the in-domain held-out test "
        f"split and on each of {test_countries} - none of which the model ever trained on.",
        "",
        "## Config (for reproducibility)",
        "",
        f"- model: `{args.model}`",
        f"- seed: {args.seed}",
        f"- epochs: {args.epochs} (patience={args.patience})",
        f"- imgsz: {args.imgsz}, batch: {args.batch}",
        f"- train time: {train_seconds / 60:.1f} min",
        f"- source (train) countries: {split_config['train_countries']} - "
        f"train={split_config['n_train']} val={split_config['n_val']} "
        f"in_domain_test={split_config['n_in_domain_test']} "
        f"(split seed={split_config['seed']})",
        "",
        "## In-domain vs. cross-domain comparison",
        "",
        "The in-domain row is the fair baseline to compare every target country against - same model,"
        " same held-out discipline, only difference is whether the country was in the training mix.",
        "",
        "| domain | n images | mAP@50 | mAP@50-95 | precision | recall | Δ mAP@50 vs in-domain |",
        "|---|---|---|---|---|---|---|",
    ]
    for domain_name, d in domains.items():
        o = d["overall"]
        delta = "-" if domain_name == "in_domain" else f"{o['map50'] - in_domain_map50:+.4f}"
        report_lines.append(
            f"| {domain_name} | {d['n_images']} | {o['map50']:.4f} | {o['map50_95']:.4f} | "
            f"{o['precision_mean']:.4f} | {o['recall_mean']:.4f} | {delta} |"
        )

    report_lines += ["", "## Per-class breakdown by domain", ""]
    for domain_name, d in domains.items():
        report_lines += [
            f"### {domain_name}",
            "",
            "| class | precision | recall | AP50 | AP50-95 |",
            "|---|---|---|---|---|",
        ]
        for c in d["per_class"]:
            if c["precision"] is None:
                report_lines.append(f"| {c['class']} | - | - | - | - ({c.get('note', '')}) |")
            else:
                report_lines.append(f"| {c['class']} | {c['precision']:.4f} | {c['recall']:.4f} | "
                                     f"{c['ap50']:.4f} | {c['ap50_95']:.4f} |")
        report_lines.append("")

    report_lines += ["## Artifacts", ""]
    for domain_name, d in domains.items():
        run_dir = Path(d["run_dir"])
        report_lines.append(
            f"- **{domain_name}**: confusion matrix `{run_dir / 'confusion_matrix.png'}`, "
            f"PR/F1/P/R curves `{run_dir}/Box{{PR,F1,P,R}}_curve.png`"
        )
    report_lines += [
        f"- training curves (loss/mAP per epoch): `{train_run_dir / 'results.png'}`",
        f"- machine-readable metrics: `{metrics_path}`",
        "",
    ]
    report_path = eval_project / "cross_country_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"\nWrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(f"\nSummary (mAP@50): in_domain={in_domain_map50:.4f}", end="")
    for country in test_countries:
        m = domains[country]["overall"]["map50"]
        print(f", {country}={m:.4f} ({m - in_domain_map50:+.4f})", end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
