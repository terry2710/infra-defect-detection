"""Convert RDD2022's PASCAL-VOC-XML annotations into YOLO format, and build a unified manifest
across all six countries (roadmap: infra-defect-detection, phase 1).

Why this exists rather than just pointing Ultralytics at the raw VOC XML: (1) YOLO training wants
one .txt label per image with normalized [class cx cy w h] rows, not VOC's per-object XML; (2) more
importantly for this project, we need a single manifest that records which COUNTRY every image
came from, because phase 3's whole point is training on a subset of countries and evaluating
cross-country generalization - that experiment is impossible without country labels surviving the
conversion step.

Design choice - discover files by globbing + matching by filename stem, not by assuming a fixed
"images/" + "annotations/xmls/" subfolder layout: RDD2022's own directory-structure reference file
was unreachable when this project was set up (see download_rdd2022.py's docstring), so hardcoding
an assumed layout would risk silently processing zero files if the real layout differs. Globbing
recursively is slower but correct regardless of how each country's zip is actually organized
internally.

Damage classes (from the RDD2022/CRDDC'2022 label map):
    D00 - longitudinal crack
    D10 - transverse crack
    D20 - alligator crack
    D40 - pothole
Any other class name encountered (older RDD releases had more, e.g. D01/D11/D43/D44/D50) is logged
and SKIPPED, not silently merged into the nearest class - if you see a nontrivial skip count for a
country, that's worth a manual look before assuming the data converted cleanly.

Usage:
    python scripts/convert_voc_to_yolo.py                  # converts every country under data/raw/
    python scripts/convert_voc_to_yolo.py --countries Japan Czech
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def find_files(root, suffix):
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def parse_voc_xml(xml_path):
    """Return (image_filename, width, height, [(class_name, xmin, ymin, xmax, ymax), ...]).

    width/height come from the XML's <size> block when present; if absent or zero (seen in some
    messy real-world VOC exports), the caller falls back to opening the image with PIL - this is
    exactly the kind of "annotation doesn't quite match what a clean benchmark would give you"
    messiness the project is supposed to be diagnosing, so we handle it rather than crash on it.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename_el = root.find("filename")
    image_filename = filename_el.text.strip() if filename_el is not None and filename_el.text else xml_path.stem + ".jpg"

    size_el = root.find("size")
    width = height = 0
    if size_el is not None:
        w_el, h_el = size_el.find("width"), size_el.find("height")
        width = int(w_el.text) if w_el is not None and w_el.text else 0
        height = int(h_el.text) if h_el is not None and h_el.text else 0

    objects = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or bnd is None:
            continue
        name = (name_el.text or "").strip()
        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue
        objects.append((name, xmin, ymin, xmax, ymax))

    return image_filename, width, height, objects


def voc_box_to_yolo_line(class_idx, xmin, ymin, xmax, ymax, width, height):
    cx = (xmin + xmax) / 2 / width
    cy = (ymin + ymax) / 2 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_country(country, raw_dir, processed_dir, manifest_rows, stats):
    xml_files = find_files(raw_dir, ".xml")
    jpg_files = find_files(raw_dir, ".jpg") + find_files(raw_dir, ".jpeg") + find_files(raw_dir, ".JPG")
    image_by_stem = {}
    for p in jpg_files:
        image_by_stem.setdefault(p.stem, p)

    if not xml_files:
        print(f"  WARNING: no .xml annotation files found under {raw_dir} - is the zip actually extracted here?")
        return

    out_images = processed_dir / country / "images"
    out_labels = processed_dir / country / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_ok = n_orphan_xml = n_bad_size = n_no_objects = n_skipped_class = 0

    for xml_path in xml_files:
        image_filename, width, height, objects = parse_voc_xml(xml_path)
        stem = Path(image_filename).stem

        image_path = image_by_stem.get(stem) or image_by_stem.get(xml_path.stem)
        if image_path is None:
            n_orphan_xml += 1
            continue

        if width <= 0 or height <= 0:
            if Image is None:
                n_bad_size += 1
                continue
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                n_bad_size += 1
                continue

        yolo_lines = []
        for name, xmin, ymin, xmax, ymax in objects:
            if name not in CLASS_MAP:
                n_skipped_class += 1
                stats["skipped_classes"][name] = stats["skipped_classes"].get(name, 0) + 1
                continue
            xmin, xmax = sorted((max(0, xmin), min(width, xmax)))
            ymin, ymax = sorted((max(0, ymin), min(height, ymax)))
            if xmax <= xmin or ymax <= ymin:
                continue
            yolo_lines.append(voc_box_to_yolo_line(CLASS_MAP[name], xmin, ymin, xmax, ymax, width, height))
            stats["class_counts"][name] = stats["class_counts"].get(name, 0) + 1

        if not yolo_lines:
            n_no_objects += 1
            continue

        dest_stem = f"{country}__{stem}"
        dest_image = out_images / f"{dest_stem}{image_path.suffix.lower()}"
        dest_label = out_labels / f"{dest_stem}.txt"
        if not dest_image.exists():
            dest_image.write_bytes(image_path.read_bytes())
        dest_label.write_text("\n".join(yolo_lines) + "\n")

        manifest_rows.append({
            "country": country,
            "image": str(dest_image.relative_to(processed_dir.parent)),
            "label": str(dest_label.relative_to(processed_dir.parent)),
            "width": width,
            "height": height,
            "num_objects": len(yolo_lines),
            "classes": ";".join(sorted({l.split()[0] for l in yolo_lines})),
        })
        n_ok += 1

    print(f"  {country}: {n_ok} converted, {n_orphan_xml} orphan xml (no matching image), "
          f"{n_bad_size} bad/unreadable size, {n_no_objects} had zero valid objects after class "
          f"filtering, {n_skipped_class} individual boxes skipped for unrecognized class names")
    stats["per_country"][country] = {
        "converted": n_ok, "orphan_xml": n_orphan_xml, "bad_size": n_bad_size,
        "no_objects": n_no_objects, "skipped_class_boxes": n_skipped_class,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--countries", nargs="+", default=None,
                         help="subset of country folder names under data/raw/ to convert (default: all found)")
    args = parser.parse_args(argv)

    if Image is None:
        print("NOTE: Pillow not installed - images with missing/zero <size> in their XML will be "
              "skipped instead of measured. `pip install Pillow` to handle those too.", file=sys.stderr)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    if args.countries:
        countries = args.countries
    else:
        countries = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

    if not countries:
        print(f"No country folders found under {raw_dir} - run download_rdd2022.py first.", file=sys.stderr)
        return 1

    manifest_rows = []
    stats = {"class_counts": {}, "skipped_classes": {}, "per_country": {}}

    print(f"Converting {len(countries)} countries: {countries}")
    for country in countries:
        print(f"\n[{country}]")
        convert_country(country, raw_dir / country, processed_dir, manifest_rows, stats)

    manifest_path = data_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["country", "image", "label", "width", "height", "num_objects", "classes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml = data_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "# Auto-generated by convert_voc_to_yolo.py - combined view across all converted countries.\n"
        "# For the phase-3 cross-country experiments, build per-experiment yaml files that point at\n"
        "# a subset of countries' image folders instead of reusing this combined one directly.\n"
        f"path: {processed_dir}\n"
        "train: */images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    print(f"\nWrote manifest ({len(manifest_rows)} images) to {manifest_path}")
    print(f"Wrote combined dataset.yaml to {dataset_yaml}")
    print("\nClass distribution across all converted countries:")
    for name, idx in CLASS_MAP.items():
        print(f"  {name}: {stats['class_counts'].get(name, 0)}")
    if stats["skipped_classes"]:
        print("\nSkipped (unrecognized) class names encountered - investigate before trusting counts above:")
        for name, count in sorted(stats["skipped_classes"].items(), key=lambda kv: -kv[1]):
            print(f"  {name!r}: {count}")

    print("\nNext: python scripts/eda_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
