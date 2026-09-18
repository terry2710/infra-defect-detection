"""Build the phase-3 cross-country split: merge several SOURCE countries into a train/val/
in-domain-test split, and write a separate full-image evaluation list for each TARGET country that
the model never trains on (roadmap: infra-defect-detection, phase 3 - distribution-shift diagnosis).

WHY THIS EXISTS: phase 2 answered "how good is a same-country baseline". Phase 3 asks "how much
worse does this get when the test country wasn't in training, and why" - that comparison needs two
things this script produces: (1) an IN-DOMAIN test split carved out of the source countries the same
way phase 2 did (held out, never trained on, reproducible via a recorded seed) as the "fair" number
to compare against, and (2) one full-image list per target country - no split, since the model never
sees any of a target country's images during training, so every one of its images is fair to
evaluate on.

Design (per the roadmap's phase-3 spec, chosen for a NATURAL - not synthetic - distribution shift):
  - source (train) countries: Japan, India, Czech - largest source-country image counts among the
    non-target countries per data/eda_report.md, plus Czech to keep phase 2's already-trained-on
    country in-domain rather than discarding that data.
  - target (cross-domain) countries: Norway, United_States, China_MotorBike, China_Drone - held out
    entirely from training so each can be reported as its own distribution-shift data point (motorbike-
    mounted vs. drone-mounted capture in China are different enough equipment/altitude conditions that
    lumping them into one "China" number would hide which one drives any gap).

Usage:
    python scripts/make_cross_country_split.py
    python scripts/make_cross_country_split.py --train-countries Japan India Czech \\
        --test-countries Norway United_States China_MotorBike China_Drone --seed 42
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

DEFAULT_TRAIN_COUNTRIES = ["Japan", "India", "Czech"]
DEFAULT_TEST_COUNTRIES = ["Norway", "United_States", "China_MotorBike", "China_Drone"]


def _write_image_list(list_path, rows, data_dir):
    with open(list_path, "w") as f:
        for r in rows:
            # manifest.csv's "image" column is relative to data_dir - resolve to an absolute path so
            # `yolo train`/`yolo val` behave the same regardless of the invoking directory.
            f.write(str((data_dir / r["image"]).resolve()) + "\n")


def _write_dataset_yaml(yaml_path, data_dir, comment, **splits):
    """splits: e.g. train=Path, val=Path, test=Path - only the keys given are written."""
    lines = [
        f"# {comment}\n",
        "# Labels are found by Ultralytics' convention: same path with the LAST 'images' path "
        "component\n# replaced by 'labels' and the extension replaced by .txt.\n",
    ]
    for key, path in splits.items():
        lines.append(f"{key}: {Path(path).resolve()}\n")
    lines.append(f"nc: {len(CLASS_NAMES)}\n")
    lines.append(f"names: {CLASS_NAMES}\n")
    yaml_path.write_text("".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-countries", nargs="+", default=DEFAULT_TRAIN_COUNTRIES,
                         help=f"countries merged into train/val/in-domain-test (default: {DEFAULT_TRAIN_COUNTRIES})")
    parser.add_argument("--test-countries", nargs="+", default=DEFAULT_TEST_COUNTRIES,
                         help=f"countries held out entirely, each getting its own full-image eval "
                              f"list (default: {DEFAULT_TEST_COUNTRIES})")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="fixed shuffle seed - record this alongside any reported metric")
    parser.add_argument("--out-name", default="cross_country", help="subfolder under data/splits/ to write into")
    args = parser.parse_args(argv)

    overlap = set(args.train_countries) & set(args.test_countries)
    if overlap:
        print(f"--train-countries and --test-countries overlap: {sorted(overlap)} - a country must be "
              f"either fully in-domain (trained on) or fully held out, not both.", file=sys.stderr)
        return 1

    if args.train_frac + args.val_frac >= 1.0:
        print(f"--train-frac ({args.train_frac}) + --val-frac ({args.val_frac}) must leave room for a "
              f"non-empty in-domain-test split (currently sums to {args.train_frac + args.val_frac})",
              file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"{manifest_path} not found - run extract_convert_per_country.py first.", file=sys.stderr)
        return 1

    with open(manifest_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    source_rows = [r for r in all_rows if r["country"] in args.train_countries]
    missing_source = set(args.train_countries) - {r["country"] for r in source_rows}
    if missing_source:
        print(f"No rows for train-country(ies) {sorted(missing_source)} in {manifest_path} - check "
              f"spelling, or that extract_convert_per_country.py was run with these countries.",
              file=sys.stderr)
        return 1
    if not source_rows:
        print(f"No rows found for any of --train-countries {args.train_countries} in {manifest_path}.",
              file=sys.stderr)
        return 1

    target_rows_by_country = {c: [r for r in all_rows if r["country"] == c] for c in args.test_countries}
    missing_target = [c for c, rows in target_rows_by_country.items() if not rows]
    if missing_target:
        print(f"No rows for test-country(ies) {missing_target} in {manifest_path} - check spelling, "
              f"or that extract_convert_per_country.py was run with these countries.", file=sys.stderr)
        return 1

    # Sort first so the shuffle is deterministic regardless of manifest.csv's row order, then shuffle
    # ACROSS the merged source countries (not per-country) so train/val/in-domain-test each get a
    # representative mix rather than e.g. all of Japan in train and all of Czech in val by chance.
    source_rows.sort(key=lambda r: (r["country"], r["image"]))
    rng = random.Random(args.seed)
    rng.shuffle(source_rows)

    n = len(source_rows)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train": source_rows[:n_train],
        "val": source_rows[n_train:n_train + n_val],
        "in_domain_test": source_rows[n_train + n_val:],
    }

    split_dir = data_dir / "splits" / args.out_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source countries {args.train_countries}: {n} total images")
    per_split_country_counts = {}
    for split_name, split_rows in splits.items():
        list_path = split_dir / f"{split_name}.txt"
        _write_image_list(list_path, split_rows, data_dir)
        counts = dict(Counter(r["country"] for r in split_rows))
        per_split_country_counts[split_name] = counts
        print(f"  {split_name}: {len(split_rows)} images ({counts}) -> {list_path}")

    if any(len(v) == 0 for v in splits.values()):
        print(f"\nWARNING: at least one source split is empty - {args.train_countries} may not have "
              f"enough images for these fractions.", file=sys.stderr)

    dataset_yaml_path = split_dir / "dataset.yaml"
    _write_dataset_yaml(
        dataset_yaml_path, data_dir,
        comment=f"Auto-generated by make_cross_country_split.py. train_countries={args.train_countries}, seed={args.seed}.",
        train=split_dir / "train.txt",
        val=split_dir / "val.txt",
        test=split_dir / "in_domain_test.txt",
    )
    print(f"  dataset.yaml -> {dataset_yaml_path}")

    # Each target country: one full-image list (no split - the model never trains on any of it, so
    # every image is fair game for evaluation) and its own single-purpose eval yaml, so
    # train_cross_country_baseline.py can call model.val(data=<this yaml>, split="test") the SAME WAY
    # for the in-domain test set and for every target country - one code path, not a special case
    # per country.
    target_counts = {}
    for country, rows in target_rows_by_country.items():
        rows = sorted(rows, key=lambda r: r["image"])  # deterministic order, even though unsplit
        list_path = split_dir / f"eval_{country}.txt"
        _write_image_list(list_path, rows, data_dir)
        target_counts[country] = len(rows)
        eval_yaml_path = split_dir / f"eval_{country}.yaml"
        _write_dataset_yaml(
            eval_yaml_path, data_dir,
            comment=f"Auto-generated by make_cross_country_split.py. Full held-out eval set for "
                    f"target country={country} (never trained on) - use with "
                    f"model.val(data=this, split='test'). train/val below are NOT used for training "
                    f"(this yaml is only ever passed to model.val, never model.train) - they're set "
                    f"to the same list as 'test' only because Ultralytics' check_det_dataset() "
                    f"requires 'train' and 'val' keys to be present in every data yaml it loads.",
            train=list_path,
            val=list_path,
            test=list_path,
        )
        print(f"  target {country}: {len(rows)} images -> {list_path} (+ {eval_yaml_path.name})")

    split_config = {
        "train_countries": args.train_countries,
        "test_countries": args.test_countries,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "in_domain_test_frac": round(1.0 - args.train_frac - args.val_frac, 6),
        "n_source_total": n,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_in_domain_test": len(splits["in_domain_test"]),
        "per_split_country_counts": per_split_country_counts,
        "target_country_counts": target_counts,
    }
    config_path = split_dir / "split_config.json"
    config_path.write_text(json.dumps(split_config, indent=2) + "\n")
    print(f"  split_config.json -> {config_path}")

    print(f"\nSource: {n} images -> train={len(splits['train'])} val={len(splits['val'])} "
          f"in_domain_test={len(splits['in_domain_test'])} (seed={args.seed})")
    print(f"Targets: {target_counts}")
    print(f"\nNext: python scripts/train_cross_country_baseline.py --data {dataset_yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
