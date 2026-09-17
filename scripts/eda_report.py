"""Quantify how the six RDD2022 countries differ (roadmap: infra-defect-detection, phase 1).

This is the deliverable phase 1 is actually for: not "look at some pictures" but a report that puts
numbers on exactly which properties differ across countries, so phase 3's cross-country
generalization experiment has a documented hypothesis to test rather than a vague "the data is
probably different" assumption. Reads data/manifest.csv (written by convert_voc_to_yolo.py) and
the images/labels it points at.

Computes, per country:
  - image count, object count, objects-per-image
  - class distribution (proportion of D00/D10/D20/D40)
  - resolution distribution (most common width x height, and how much it varies)
  - brightness (mean grayscale pixel value, sampled - see --sample-size) as a cheap proxy for
    lighting-condition differences (weather, time of day, camera exposure settings)

Deliberately NOT trying to detect "blur" or "occlusion" quantitatively here - those need either a
trained model's failure cases (that's phase 3, once there's a baseline to probe) or a much more
careful CV pipeline (Laplacian-variance blur detection is noisy on road-texture images specifically,
since sharp asphalt texture can score similarly to blur). Brightness and resolution are the two
properties measurable directly from pixels without a model in the loop, so that's what phase 1
reports; phase 3's failure-case analysis is where blur/occlusion get characterized properly, on the
* subset that actually caused cross-country failures* rather than on the whole dataset speculatively.

Usage:
    python scripts/eda_report.py                    # samples 200 images/country for brightness
    python scripts/eda_report.py --sample-size 500   # slower, more precise brightness estimate
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import numpy as np
    from PIL import Image
    HAVE_IMAGING = True
except ImportError:
    HAVE_IMAGING = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def sample_brightness(image_paths, sample_size, seed=42):
    """Mean/std of grayscale brightness over a random sample of image_paths, resized small first
    purely for speed - brightness is a coarse global statistic, a 64x64 downsample is plenty."""
    if not HAVE_IMAGING:
        return None, None, 0
    rng = random.Random(seed)
    sample = rng.sample(image_paths, min(sample_size, len(image_paths)))
    values = []
    for p in sample:
        try:
            with Image.open(p) as im:
                im = im.convert("L").resize((64, 64))
                values.append(np.asarray(im, dtype="float32").mean())
        except Exception:
            continue
    if not values:
        return None, None, 0
    arr = np.array(values)
    return float(arr.mean()), float(arr.std()), len(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sample-size", type=int, default=200,
                         help="images per country to sample for the brightness statistic (default: %(default)s)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"{manifest_path} not found - run convert_voc_to_yolo.py first.")
        return 1

    df = pd.read_csv(manifest_path)
    df["aspect_ratio"] = df["width"] / df["height"]

    rows = []
    figures_dir = data_dir / "eda_figures"
    figures_dir.mkdir(exist_ok=True)

    class_dist_by_country = {}
    for country, group in df.groupby("country"):
        class_counter = Counter()
        for classes_field in group["classes"].fillna(""):
            for c in str(classes_field).split(";"):
                if c:
                    class_counter[int(c)] += 1
        total_objects = sum(class_counter.values())
        class_dist_by_country[country] = {
            CLASS_NAMES[i]: class_counter.get(i, 0) / total_objects if total_objects else 0.0
            for i in range(len(CLASS_NAMES))
        }

        res_counts = group.groupby(["width", "height"]).size().sort_values(ascending=False)
        top_res = res_counts.index[0] if len(res_counts) else (None, None)
        res_diversity = len(res_counts)

        image_paths = [data_dir / p for p in group["image"]]
        brightness_mean, brightness_std, n_sampled = sample_brightness(image_paths, args.sample_size)

        rows.append({
            "country": country,
            "n_images": len(group),
            "n_objects": int(group["num_objects"].sum()),
            "objects_per_image": group["num_objects"].mean(),
            "top_resolution": f"{top_res[0]}x{top_res[1]}" if top_res[0] else "unknown",
            "distinct_resolutions": res_diversity,
            "mean_aspect_ratio": group["aspect_ratio"].mean(),
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "brightness_n_sampled": n_sampled,
        })

    summary = pd.DataFrame(rows).sort_values("country")
    summary_path = data_dir / "eda_summary.csv"
    summary.to_csv(summary_path, index=False)

    if HAVE_MPL:
        # Class distribution per country - a grouped bar chart is the whole point: if the bars look
        # similar across countries, class balance isn't a big source of cross-country difficulty;
        # if they diverge a lot, that alone could explain part of a cross-country mAP drop before
        # even getting to appearance differences (lighting, resolution).
        fig, ax = plt.subplots(figsize=(10, 5))
        countries = summary["country"].tolist()
        x = range(len(countries))
        width_bar = 0.2
        for i, name in enumerate(CLASS_NAMES):
            values = [class_dist_by_country[c][name] for c in countries]
            ax.bar([xi + i * width_bar for xi in x], values, width=width_bar, label=name)
        ax.set_xticks([xi + 1.5 * width_bar for xi in x])
        ax.set_xticklabels(countries, rotation=30, ha="right")
        ax.set_ylabel("share of annotated objects")
        ax.set_title("Damage class distribution by country")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "class_distribution_by_country.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["country"], summary["brightness_mean"], yerr=summary["brightness_std"])
        ax.set_ylabel("mean grayscale brightness (0-255)")
        ax.set_title(f"Brightness by country (sampled, n<= {args.sample_size}/country)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figures_dir / "brightness_by_country.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["country"], summary["n_images"])
        ax.set_ylabel("images")
        ax.set_title("Image count by country")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(figures_dir / "image_count_by_country.png", dpi=150)
        plt.close(fig)
    else:
        print("matplotlib not installed - skipping figure generation, CSV/markdown still written.")

    report_lines = [
        "# RDD2022 six-country data profile\n",
        "Generated by `scripts/eda_report.py`. This is the phase-1 deliverable: quantifying how the "
        "six countries' data actually differs, as the documented basis for the phase-3 cross-country "
        "generalization experiment - not an assumption.\n",
        "## Summary table\n",
        summary.to_markdown(index=False) if hasattr(summary, "to_markdown") else summary.to_string(index=False),
        "\n\n## Class distribution by country (share of annotated objects)\n",
    ]
    class_dist_df = pd.DataFrame(class_dist_by_country).T[CLASS_NAMES]
    report_lines.append(class_dist_df.to_markdown() if hasattr(class_dist_df, "to_markdown") else class_dist_df.to_string())
    report_lines.append(
        "\n\n## Figures\n\n"
        "- `eda_figures/class_distribution_by_country.png`\n"
        "- `eda_figures/brightness_by_country.png`\n"
        "- `eda_figures/image_count_by_country.png`\n"
    )
    report_path = data_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines))

    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {report_path}")
    if HAVE_MPL:
        print(f"Wrote figures to {figures_dir}")
    print("\nNext: read eda_report.md, note which countries look most different from the rest on "
          "brightness/resolution/class-mix, then design the phase-3 train/test country split around "
          "those specific differences instead of an arbitrary split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
