"""Sample and visualize concrete misdetections per target country, for phase 3's qualitative
failure-mode diagnosis (roadmap: infra-defect-detection, phase 3 - distribution-shift diagnosis).

WHY THIS EXISTS: `train_cross_country_baseline.py` answers "how much worse" (a number per country).
This script answers "worse HOW" - a diagnostic report needs concrete failure images, not just an
mAP drop, to argue convincingly for WHICH cross-country factor (lighting/weather, camera
resolution/equipment, road-material appearance, or annotation-standard differences between
countries) is actually driving the gap, and to design a TARGETED mitigation instead of a generic one.

For every image in a target country's held-out eval set, this script runs the trained model,
matches its predictions against ground truth by IoU + class, and classifies each ground-truth box as
a true positive, a missed detection (false negative), or a wrong-class match, and each prediction
with no matching ground-truth region as a false positive. It then saves annotated images (ground
truth in blue, predictions in red) for the worst-scoring images per country - the ones most worth
looking at by eye - plus a machine-readable per-image breakdown and a markdown report.

This script does NOT itself conclude which distribution-shift factor is responsible - that's a
judgment call for a human looking at the saved images - it produces the evidence (ranked failure
cases + per-image brightness, as a cheap proxy for lighting differences worth cross-referencing
against data/eda_report.md's per-country brightness numbers from phase 1) that judgment is based on.

Usage:
    python scripts/diagnose_failures.py --weights runs/phase3/cross_country_baseline/weights/best.pt \\
        --split-dir data/splits/cross_country
    python scripts/diagnose_failures.py --weights <best.pt> --split-dir <dir> \\
        --countries Norway China_Drone --top-n 8 --conf 0.25 --iou 0.5
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat
from ultralytics import YOLO

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "phase3" / "failures"

GT_COLOR = (30, 100, 255)     # blue - ground truth
PRED_COLOR = (255, 40, 40)    # red - model prediction


def _label_path_for(image_path: Path) -> Path:
    # Ultralytics convention: last 'images' path component -> 'labels', extension -> .txt.
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _read_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Returns list of (cls, x1, y1, x2, y2) in pixel coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls, cx, cy, w, h = line.split()[:5]
        cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(gt_boxes, pred_boxes, iou_thresh):
    """gt_boxes/pred_boxes: list of (cls, x1, y1, x2, y2); pred_boxes assumed sorted by confidence
    descending. Returns (tp, fn, fp, wrong_class) counts and per-box match info for drawing."""
    gt_matched = [False] * len(gt_boxes)
    tp = fn = fp = wrong_class = 0
    pred_status = []  # "tp" | "wrong_class" | "fp", parallel to pred_boxes
    for p_cls, *p_box in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, (g_cls, *g_box) in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = _iou(p_box, g_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_thresh:
            gt_matched[best_idx] = True
            if gt_boxes[best_idx][0] == p_cls:
                tp += 1
                pred_status.append("tp")
            else:
                wrong_class += 1
                pred_status.append("wrong_class")
        else:
            fp += 1
            pred_status.append("fp")
    fn = sum(1 for m in gt_matched if not m)
    return tp, fn, fp, wrong_class, gt_matched, pred_status


def _draw_annotated(image_path, gt_boxes, pred_boxes, gt_matched, pred_status, class_names, out_path, banner):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (cls, x1, y1, x2, y2), matched in zip(gt_boxes, gt_matched):
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=2)
        label = f"GT:{class_names[cls]}" + ("" if matched else " (MISSED)")
        draw.text((x1 + 2, max(0, y1 - 10)), label, fill=GT_COLOR, font=font)

    for (cls, x1, y1, x2, y2, conf), status in zip(pred_boxes, pred_status):
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=2)
        tag = {"tp": "", "wrong_class": " (WRONG CLASS)", "fp": " (FALSE POS)"}[status]
        draw.text((x1 + 2, y2 + 2), f"pred:{class_names[cls]} {conf:.2f}{tag}", fill=PRED_COLOR, font=font)

    # Banner: pad a strip at the top with the failure summary so the image is self-explanatory
    # without needing to cross-reference the JSON/report.
    banner_h = 18
    banner_img = Image.new("RGB", (img.width, img.height + banner_h), color=(255, 255, 255))
    banner_img.paste(img, (0, banner_h))
    ImageDraw.Draw(banner_img).text((2, 2), banner, fill=(0, 0, 0), font=font)
    banner_img.save(out_path, "JPEG")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="path to a trained .pt checkpoint (e.g. runs/phase3/cross_country_baseline/weights/best.pt)")
    parser.add_argument("--split-dir", required=True, help="dir written by make_cross_country_split.py (has split_config.json + eval_<country>.txt)")
    parser.add_argument("--countries", nargs="+", default=None, help="target countries to diagnose (default: all test_countries in split_config.json)")
    parser.add_argument("--top-n", type=int, default=6, help="number of worst-scoring images to save per country")
    parser.add_argument("--conf", type=float, default=0.25, help="prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching a prediction to a ground-truth box")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"{weights_path} not found - run train_cross_country_baseline.py first.", file=sys.stderr)
        return 1

    split_dir = Path(args.split_dir)
    split_config_path = split_dir / "split_config.json"
    if not split_config_path.exists():
        print(f"{split_config_path} not found - {split_dir} doesn't look like a make_cross_country_split.py output dir.", file=sys.stderr)
        return 1
    split_config = json.loads(split_config_path.read_text())
    countries = args.countries or split_config["test_countries"]

    missing = [c for c in countries if not (split_dir / f"eval_{c}.txt").exists()]
    if missing:
        print(f"No eval_<country>.txt for {missing} in {split_dir}.", file=sys.stderr)
        return 1

    model = YOLO(str(weights_path))
    class_names = model.names if isinstance(model.names, list) else [model.names[i] for i in sorted(model.names)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    country_summaries = {}
    for country in countries:
        image_paths = [Path(p) for p in (split_dir / f"eval_{country}.txt").read_text().splitlines() if p.strip()]
        print(f"\n{country}: diagnosing {len(image_paths)} images (conf>={args.conf}, iou>={args.iou})...")

        per_image = []
        for image_path in image_paths:
            if not image_path.exists():
                print(f"  WARNING: {image_path} not found, skipping", file=sys.stderr)
                continue
            with Image.open(image_path) as im:
                img_w, img_h = im.size
                brightness = ImageStat.Stat(im.convert("L")).mean[0]

            gt_boxes = _read_gt_boxes(_label_path_for(image_path), img_w, img_h)

            result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]
            pred_boxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                pred_boxes.append((int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])))
            pred_boxes.sort(key=lambda b: -b[5])
            pred_boxes_for_match = [(b[0], b[1], b[2], b[3], b[4]) for b in pred_boxes]

            tp, fn, fp, wrong_class, gt_matched, pred_status = _match(gt_boxes, pred_boxes_for_match, args.iou)
            failure_score = fn + fp + wrong_class

            per_image.append({
                "image": str(image_path), "n_gt": len(gt_boxes), "n_pred": len(pred_boxes),
                "tp": tp, "fn": fn, "fp": fp, "wrong_class": wrong_class,
                "failure_score": failure_score, "brightness": round(brightness, 1),
                "_gt_boxes": gt_boxes, "_pred_boxes": pred_boxes,
                "_gt_matched": gt_matched, "_pred_status": pred_status,
            })

        per_image.sort(key=lambda r: -r["failure_score"])
        top = per_image[:args.top_n]

        country_out_dir = out_dir / country
        country_out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for rank, r in enumerate(top, start=1):
            image_path = Path(r["image"])
            banner = (f"{country} | {image_path.name} | TP={r['tp']} FN={r['fn']} FP={r['fp']} "
                      f"wrong_class={r['wrong_class']} | brightness={r['brightness']}")
            out_path = country_out_dir / f"rank{rank:02d}_score{r['failure_score']}_{image_path.stem}.jpg"
            _draw_annotated(image_path, r["_gt_boxes"], r["_pred_boxes"], r["_gt_matched"],
                             r["_pred_status"], class_names, out_path, banner)
            saved.append(str(out_path))
            print(f"  rank{rank}: {image_path.name} (failure_score={r['failure_score']}, "
                  f"TP={r['tp']} FN={r['fn']} FP={r['fp']} wrong_class={r['wrong_class']}) -> {out_path}")

        totals = {
            "n_images": len(per_image),
            "total_gt": sum(r["n_gt"] for r in per_image),
            "total_tp": sum(r["tp"] for r in per_image),
            "total_fn": sum(r["fn"] for r in per_image),
            "total_fp": sum(r["fp"] for r in per_image),
            "total_wrong_class": sum(r["wrong_class"] for r in per_image),
            "mean_failure_score": round(sum(r["failure_score"] for r in per_image) / len(per_image), 2) if per_image else 0.0,
            "mean_brightness": round(sum(r["brightness"] for r in per_image) / len(per_image), 1) if per_image else 0.0,
        }
        country_summaries[country] = {
            "totals": totals,
            "top_images": [{k: v for k, v in r.items() if not k.startswith("_")} | {"annotated_path": p}
                           for r, p in zip(top, saved)],
        }
        print(f"  {country} totals: TP={totals['total_tp']} FN={totals['total_fn']} "
              f"FP={totals['total_fp']} wrong_class={totals['total_wrong_class']} "
              f"(recall~{totals['total_tp']/(totals['total_tp']+totals['total_fn']):.2f})"
              if totals["total_tp"] + totals["total_fn"] > 0 else "")

    summary_path = out_dir / "failures_summary.json"
    summary_path.write_text(json.dumps({"conf": args.conf, "iou": args.iou, "countries": country_summaries}, indent=2) + "\n")

    report_lines = [
        "# Phase 3 failure-mode diagnosis",
        "",
        f"Generated by `scripts/diagnose_failures.py` from `{weights_path}` (conf>={args.conf}, "
        f"iou>={args.iou}). Ground truth boxes are blue, model predictions are red, in every "
        f"annotated image below - annotated images are saved under `{out_dir}/<country>/`.",
        "",
        "## Country severity summary",
        "",
        "| country | images | GT boxes | TP | FN (missed) | FP (extra) | wrong-class | recall | mean brightness |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for country, s in country_summaries.items():
        t = s["totals"]
        recall = t["total_tp"] / (t["total_tp"] + t["total_fn"]) if (t["total_tp"] + t["total_fn"]) > 0 else float("nan")
        report_lines.append(
            f"| {country} | {t['n_images']} | {t['total_gt']} | {t['total_tp']} | {t['total_fn']} | "
            f"{t['total_fp']} | {t['total_wrong_class']} | {recall:.2f} | {t['mean_brightness']} |"
        )

    report_lines += ["", "## Worst-scoring images per country", "",
                      "Cross-reference `mean brightness` here against each country's brightness in "
                      "`data/eda_report.md` (phase 1) - a target country noticeably darker/brighter "
                      "than the source countries is evidence for a lighting-driven gap; if brightness "
                      "is similar but FN/FP is still high, look at the images themselves for "
                      "resolution, road-material, or annotation-style differences instead.", ""]
    for country, s in country_summaries.items():
        report_lines.append(f"### {country}")
        report_lines.append("")
        for img in s["top_images"]:
            report_lines.append(
                f"- `{Path(img['image']).name}` - TP={img['tp']} FN={img['fn']} FP={img['fp']} "
                f"wrong_class={img['wrong_class']} brightness={img['brightness']} -> "
                f"`{img['annotated_path']}`"
            )
        report_lines.append("")
    report_path = out_dir / "failures_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"\nWrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
