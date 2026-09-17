"""Extract + convert RDD2022 ONE COUNTRY AT A TIME, deleting each country's intermediate data as
soon as it's converted (roadmap: infra-defect-detection, phase 1 - Kaggle disk-constrained variant).

WHY THIS EXISTS (2026-09-17, v2->v3->v4): the straightforward approach - download_rdd2022.py's normal
extract_one()/reorganize_countries(), which calls zf.extractall() on the WHOLE combined zip at once
- needs the 12.35GB combined zip AND all the extracted raw VOC data on disk simultaneously, which
blew through a Kaggle notebook's working-directory quota on the first attempt.

v2 tried to fix this by extracting one country's files at a time straight out of the combined zip's
member list - but that assumed the combined zip was a flat tree of images/XML per country. It
ISN'T: FigShare's combined zip (b62bd51d2ffcfaa76c60f234f0cc2bb3, the officially-published MD5) is
actually a ZIP-OF-ZIPS - exactly 7 entries, one per country, each itself a complete nested .zip
(confirmed 2026-09-17 by actually listing the real zip's namelist on Kaggle: `RDD2022/Japan.zip`,
`RDD2022/India.zip`, `RDD2022/Czech.zip`, `RDD2022/Norway.zip`, `RDD2022/United_States.zip`,
`RDD2022/China_MotorBike.zip`, `RDD2022/China_Drone.zip`). v2's matching logic looked for a path
COMPONENT exactly equal to a country alias (e.g. a directory literally named "Japan"), which never
matched because the actual component is "Japan.zip" (a file, not a directory) - so v2 silently
converted 0 images across all 7 countries and wrote an empty manifest.csv.

v3 fixed the matching (match by filename STEM instead of path component) but still copied each
country's nested zip out to a temp file ON DISK before extracting it, while the 12.35GB combined zip
stayed on disk the whole time. That's fine for the small countries, but Norway's nested zip alone is
9.9GB - so by the time v3 reached Norway, disk needed 12.35GB (combined zip, still present, only
deleted at the very end) + already-converted data from the 5 prior countries + a 9.9GB temp copy of
Norway's nested zip, all at once. That blew past Kaggle's 19.5GiB /kaggle/working quota with
`OSError: [Errno 28] No space left on device` mid-copy - not a transient hiccup, this was guaranteed
to happen as soon as processing reached the largest country, regardless of retry.

v4 (this version) fixes the actual disk-budget problem instead of just the matching bug:
  1. Read EVERY country's nested-zip bytes into RAM first (`ZipFile.read()`, not `copyfileobj` to a
     temp file) - Kaggle's RAM (~31GB, most of it free) isn't quota-limited the way /kaggle/working
     is, so holding all 7 countries' compressed bytes (summing to the same ~12.35GB as the combined
     zip) in memory costs nothing against the disk quota.
  2. Only ONCE ALL SEVEN have been read into memory - and the combined zip is no longer needed for
     anything - close it and delete the 12.35GB file from disk, BEFORE extracting a single country to
     disk. This frees the full 19.5GiB quota (minus whatever's already used) for the extraction step,
     regardless of which country happens to be biggest or what order they're processed in.
  3. Per country: extract from an in-memory `io.BytesIO` (no on-disk temp zip at all), convert, delete
     the extracted raw folder, and drop that country's bytes from the in-memory dict - so RAM usage
     shrinks back down as we go instead of holding all 7 for the whole run.

Usage:
    python scripts/extract_convert_per_country.py                  # all countries, then deletes the combined zip
    python scripts/extract_convert_per_country.py --keep-zip       # don't delete the combined zip early or at the end
                                                                    # (only use this if you have >32GB of quota - it
                                                                    # defeats the whole point of the v4 fix above)
    python scripts/extract_convert_per_country.py --data-dir data
"""
import argparse
import csv
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_rdd2022 as dl          # noqa: E402  (FIGSHARE_ZIP_NAME, COUNTRY_ALIASES)
import convert_voc_to_yolo as cv       # noqa: E402  (convert_country, CLASS_MAP, CLASS_NAMES)


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _print_disk_usage(label, working_dir=None):
    result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
    print(f"  [{label}] df -h /:\n" + "\n".join("    " + l for l in result.stdout.splitlines()))
    # df -h / reports the container's overall overlay filesystem, which on Kaggle does NOT move in
    # step with /kaggle/working - the thing actually counted against the 19.5GiB quota is disk usage
    # UNDER /kaggle/working specifically, which `du` reports correctly.
    if working_dir is not None:
        du = subprocess.run(["du", "-sh", str(working_dir)], capture_output=True, text=True)
        print(f"  [{label}] du -sh {working_dir}: {du.stdout.strip() or du.stderr.strip()}")


def match_country_by_stem(entry_name):
    """Which COUNTRY_ALIASES key this combined-zip entry is, by comparing its filename stem (the
    name with the LAST extension stripped, e.g. "RDD2022/China_MotorBike.zip" -> "China_MotorBike")
    against the known aliases, case-insensitively. This is the v3 fix: v2 matched whole path
    COMPONENTS looking for a directory named e.g. "Japan", which never matched because the real
    entries are files named "Japan.zip", not directories named "Japan"."""
    stem_lower = Path(entry_name).stem.lower()
    for country, aliases in dl.COUNTRY_ALIASES.items():
        for alias in aliases:
            if alias.lower() == stem_lower:
                return country
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--keep-zip", action="store_true",
                         help="don't delete the combined zip early (once all countries are read into "
                              "memory) or at the end - default is to delete it early, which is what "
                              "makes the largest country (Norway, ~9.9GB) fit under Kaggle's quota")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    zip_path = data_dir / "zips" / dl.FIGSHARE_ZIP_NAME
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    working_dir = data_dir.resolve().parent  # e.g. /kaggle/working - what the quota actually tracks

    if not zip_path.exists():
        print(f"{zip_path} not found - run download_rdd2022.py --skip-extract first.", file=sys.stderr)
        return 1

    # Clean slate for raw_dir: it's always transient intermediate storage, never a final deliverable,
    # so wipe any leftover partial state from a previous crashed run (e.g. a half-written
    # _nested_Norway.zip from the v3 run that hit the disk-quota error) before starting.
    if raw_dir.exists():
        print(f"Removing leftover {raw_dir} from a previous run (transient data only, safe to wipe)...")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    stats = {"class_counts": {}, "skipped_classes": {}, "per_country": {}}

    print(f"Opening {zip_path} to read its member list (no extraction yet, costs no disk space)...")
    outer_zf = zipfile.ZipFile(zip_path)
    infos = [i for i in outer_zf.infolist() if not i.filename.endswith("/")]
    print(f"  combined zip contains {len(infos)} entries")

    by_country = {}
    unmatched = []
    for info in infos:
        country = match_country_by_stem(info.filename)
        if country:
            by_country[country] = info
        else:
            unmatched.append(info.filename)

    print("\nEntries matched (this is the combined zip's REAL layout - one nested zip per country):")
    for country in dl.COUNTRY_ALIASES:
        if country in by_country:
            info = by_country[country]
            print(f"  {country}: {info.filename} ({_format_bytes(info.file_size)})")
        else:
            print(f"  {country}: NOT FOUND")
    if unmatched:
        print(f"\n  {len(unmatched)} entries matched no known country alias:")
        for n in unmatched:
            print(f"    {n}")

    _print_disk_usage("before reading anything", working_dir)

    # Step 1: read EVERY country's nested-zip bytes into RAM before extracting any of them to disk.
    # This is what lets us delete the 12.35GB combined zip BEFORE the largest country (Norway,
    # 9.9GB) needs to be extracted, instead of only being able to delete it at the very end.
    print(f"\nReading all {len(by_country)} countries' nested-zip bytes into memory (RAM isn't "
          f"quota-limited the way /kaggle/working is - this avoids ever needing the 12.35GB combined "
          f"zip and a country's extracted data on disk at the same time)...")
    country_bytes = {}
    for country, info in by_country.items():
        print(f"  reading {country} ({info.filename}, {_format_bytes(info.file_size)}) into memory...")
        country_bytes[country] = outer_zf.read(info)
    zip_size = zip_path.stat().st_size
    outer_zf.close()

    if not args.keep_zip:
        zip_path.unlink()
        print(f"\nDeleted {zip_path} ({_format_bytes(zip_size)}) early - every country's bytes are "
              f"already in memory, so the combined zip is no longer needed and this frees up disk "
              f"headroom before extracting any country.")
    else:
        print(f"\n--keep-zip set: leaving {zip_path} ({_format_bytes(zip_size)}) on disk (less headroom "
              f"for the extraction step below - only safe with a much larger quota than 19.5GiB).")
    _print_disk_usage("after reading all countries into memory" + ("" if args.keep_zip else " + deleting the combined zip"), working_dir)

    # Step 2: per country, extract from the in-memory bytes (no on-disk temp zip), convert, clean up.
    for country in list(country_bytes.keys()):
        data = country_bytes.pop(country)  # drop from the dict now so RAM shrinks as we go

        country_raw = raw_dir / country
        if country_raw.exists():
            shutil.rmtree(country_raw)
        country_raw.mkdir(parents=True)

        print(f"\n[{country}] extracting {_format_bytes(len(data))} (in memory) into {country_raw}...")
        with zipfile.ZipFile(io.BytesIO(data)) as inner_zf:
            bad = inner_zf.testzip()
            if bad is not None:
                print(f"  WARNING: {country}'s nested zip is corrupt (bad member: {bad}) - skipping {country}")
                shutil.rmtree(country_raw)
                del data
                continue
            inner_zf.extractall(country_raw)
        del data  # free this country's RAM now that it's on disk as extracted files

        print(f"  converting...")
        cv.convert_country(country, country_raw, processed_dir, manifest_rows, stats)

        print(f"  deleting {country_raw} (already converted, no longer needed) to free space for the next country...")
        shutil.rmtree(country_raw)
        _print_disk_usage(f"after {country}", working_dir)

    manifest_path = data_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["country", "image", "label", "width", "height", "num_objects", "classes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml = data_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "# Auto-generated by extract_convert_per_country.py - combined view across all converted countries.\n"
        "# For the phase-3 cross-country experiments, build per-experiment yaml files that point at\n"
        "# a subset of countries' image folders instead of reusing this combined one directly.\n"
        f"path: {processed_dir}\n"
        "train: */images\n"
        f"nc: {len(cv.CLASS_NAMES)}\n"
        f"names: {cv.CLASS_NAMES}\n"
    )

    print(f"\nWrote manifest ({len(manifest_rows)} images) to {manifest_path}")
    print(f"Wrote combined dataset.yaml to {dataset_yaml}")
    print("\nClass distribution across all converted countries:")
    for name in cv.CLASS_MAP:
        print(f"  {name}: {stats['class_counts'].get(name, 0)}")
    if stats["skipped_classes"]:
        print("\nSkipped (unrecognized) class names encountered - investigate before trusting counts above:")
        for name, count in sorted(stats["skipped_classes"].items(), key=lambda kv: -kv[1]):
            print(f"  {name!r}: {count}")

    if not manifest_rows:
        print("\nWARNING: manifest is EMPTY - 0 images were converted. Check the 'Entries matched' listing "
              "above for NOT FOUND countries or unmatched entries before trusting anything downstream.")

    _print_disk_usage("final", working_dir)
    print("\nNext: python scripts/eda_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
