"""Download and extract the RDD2022 road damage dataset (roadmap: infra-defect-detection, phase 1).

RDD2022 covers six countries (Japan, India, Czech Republic, Norway, United States, China) with
47,420 road images and 55,000+ annotated damage instances across four classes (D00 longitudinal
crack, D10 transverse crack, D20 alligator crack, D40 pothole). Images are CC BY-SA 4.0 per the
sekilab/RoadDamageDetector GitHub README (attribute sekilab/RoadDamageDetector + the RDD2022 paper,
arxiv.org/abs/2209.08538, in any README/Model Card that uses this data) - note the FigShare listing
below shows "CC BY 4.0" for the same dataset; this discrepancy between the two official sources is
unresolved, so treat CC BY-SA 4.0 (the more restrictive of the two, and the one stated by the
dataset's own authors on their own repo) as the operative license until/unless clarified.

SOURCE CHANGE (2026-09-16): this originally downloaded seven per-country zips from Sekilab's own S3
bucket (bigdatacup.s3.ap-northeast-1.amazonaws.com/.../Country_Specific_Data_CRDDC2022/...). That
bucket now returns HTTP 403 Forbidden on every file, confirmed independently from two unrelated
networks - the bucket's access policy appears to have changed, not a transient fluke. This version
instead downloads FigShare's official combined mirror of the same dataset (one zip, all six
countries, published by the RDD2022/CRDDC2022 organizers themselves), verified by MD5 checksum
against FigShare's own published hash so a partial/corrupted 12GB+ download is caught rather than
silently producing bad data. If FigShare's link ever breaks too, check
https://github.com/sekilab/RoadDamageDetector for current mirror links before assuming this script
is broken.

Because this is one combined zip (not one zip per country), there's no way to download only a
subset of countries - the whole ~12.35GB has to come down regardless. --countries now only controls
which countries get linked into data/raw/ for the conversion step afterward (convert_voc_to_yolo.py
reads data/raw/<country>/), not what gets downloaded.

The zip's exact internal folder layout was not independently verified before writing this script
(Sekilab's own "Directory_Structure_CRDDC_RDD2022.txt" reference file was unreachable from this
environment - see reorganize_countries() below for how this script copes with that uncertainty at
extraction time instead of assuming a fixed layout).

NOTE (2026-09-17, Kaggle handoff): on the original Mac/home-network run, aria2c's multi-connection
mode was confirmed to get an immediate HTTP 403 from FigShare's CDN regardless of connection count
(1, 4, or 16) - the CDN appears to reject Range/segmented requests outright, not just high
concurrency. So on Kaggle this will (correctly) fall back to the plain single-connection urllib
path every time; that's expected, not a bug - the win here is Kaggle's raw single-connection
bandwidth to this host, not multi-connection parallelism.

NOTE (2026-09-17, Kaggle dataset cache): every phase that needs a fresh Kaggle kernel re-downloads
this same ~12.35GB zip from scratch, since /kaggle/working doesn't persist across kernel sessions -
phase 2 confirmed this the hard way. To stop paying that cost every phase, a one-time "cache
builder" kernel downloads + MD5-verifies the zip once and its output becomes a private Kaggle
Dataset; every later notebook just adds that Dataset as an input.

UPDATE (2026-09-17, same day): the cache Dataset did NOT turn out to contain a re-downloadable zip.
Kaggle's "New Dataset from notebook output" flow recursively auto-extracts every .zip file it finds
in the output - not just the outer combined zip, but the 7 per-country zips nested inside it too -
so the Dataset actually ended up holding a fully-extracted, ready-to-convert per-country directory
tree (confirmed by browsing the Dataset's Data Explorer: data/zips/RDD2022_released_through_CRDDC2022/
RDD2022/<Country>/<Country>/{train,test}/... for all 7 countries, ~85.8k files, 13.83GB). That's
actually a BETTER cache than a zip would have been - it skips the extraction step too, not just the
download - so find_kaggle_cached_extracted_countries() below is checked FIRST and, when it covers
every requested country, this script skips straight to "done" (no download, no MD5 verify, nothing -
extract_convert_per_country.py finds and uses the same cache independently). find_kaggle_cached_zip()
is kept as a fallback for a cache Dataset built some other way (e.g. zip auto-extraction turned off),
and the plain FigShare download remains the final fallback - so this script still works unchanged
off-Kaggle, or on a fresh Kaggle account with no cache Dataset set up yet.

Usage:
    python scripts/download_rdd2022.py                        # download + extract + link all found countries
    python scripts/download_rdd2022.py --countries Japan Czech # only link these two after extraction
    python scripts/download_rdd2022.py --skip-extract          # download (+ verify) only
    python scripts/download_rdd2022.py --connections 32        # more aria2c connections (default 16)
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"
FIGSHARE_ZIP_NAME = "RDD2022_released_through_CRDDC2022.zip"
FIGSHARE_MD5 = "b62bd51d2ffcfaa76c60f234f0cc2bb3"

# Logical country name -> name(s) we'll look for (case-insensitively) among directories inside the
# extracted zip, since the exact internal layout wasn't independently verified (see module docstring).
COUNTRY_ALIASES = {
    "Japan": ["Japan"],
    "India": ["India"],
    "Czech": ["Czech"],
    "Norway": ["Norway"],
    "United_States": ["United_States", "United States", "US", "USA"],
    "China_MotorBike": ["China_MotorBike", "China-MotorBike", "China_Motorbike"],
    "China_Drone": ["China_Drone", "China-Drone"],
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_with_aria2(url, dest_path, connections=16, retries=3):
    """Segmented, multi-connection download via the aria2c CLI - typically several times faster than
    a single urllib stream on hosts like FigShare's S3-backed CDN, where per-connection throughput
    is often capped well below the link's actual bandwidth. aria2c handles its own retries/resume
    (via its .aria2 control file next to the output), so this just shells out and lets it manage
    that; --continue=true means re-running after an interrupted aria2c download resumes rather than
    restarting from zero.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "-x", str(connections),          # max connections per server
        "-s", str(connections),          # split file into this many pieces
        "-k", "1M",                      # min split size
        "--continue=true",
        "--max-tries", str(retries),
        "--retry-wait=5",
        "--file-allocation=none",
        "--summary-interval=5",
        # FigShare's CDN (via ndownloader.figshare.com -> a presigned S3 URL) returns 403 to aria2c's
        # default "aria2/x.y.z" User-Agent - matching the header the plain-urllib path already used
        # successfully fixes it. --auto-file-renaming=false avoids aria2 silently writing to a
        # "-1" suffixed file if dest_path.name already exists from an earlier failed attempt.
        "--user-agent=Mozilla/5.0",
        "--auto-file-renaming=false",
        "-d", str(dest_path.parent),
        "-o", dest_path.name,
        url,
    ]
    print(f"  using aria2c with {connections} parallel connections (much faster than a single stream)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"aria2c exited with code {result.returncode} - see its output above for details")


def download_one(url, dest_path, retries=3, connections=16):
    """Download url to dest_path. Skips entirely if dest_path already exists and is non-empty - this
    does NOT check the existing file's checksum, so a corrupted prior download won't be caught here;
    verify_md5() below is what actually guarantees integrity, run separately after this.

    Uses aria2c (multi-connection, much faster) when it's installed on PATH; otherwise falls back to
    a plain single-connection urllib download and prints a one-time hint about installing aria2c for
    a large file like this one. `brew install aria2` on macOS. If aria2c is present but fails (a
    misbehaving CDN, a network that blocks it, etc.), falls back to the urllib path automatically
    rather than giving up outright - not verified against every possible aria2c/network combination,
    so this fallback matters.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  already have {dest_path.name} ({_format_bytes(dest_path.stat().st_size)}), skipping download")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("aria2c"):
        try:
            download_with_aria2(url, dest_path, connections=connections, retries=retries)
            return
        except RuntimeError as exc:
            print(f"  aria2c failed ({exc}); falling back to a plain single-connection download instead.")
            # clean up whatever partial/zero-byte file aria2c may have left behind before falling back
            if dest_path.exists() and dest_path.stat().st_size == 0:
                dest_path.unlink()
    else:
        print("  NOTE: aria2c not found on PATH - using a single-connection download, which will be "
              "noticeably slower for a file this size. `brew install aria2` (macOS) and re-run for a "
              "multi-connection download instead.")

    _download_urllib(url, dest_path, retries=retries)


def find_kaggle_cached_zip():
    """Look for a pre-cached copy of the combined zip under /kaggle/input/ - a private Kaggle
    Dataset added as this notebook's input, built once by a "cache builder" kernel that just runs
    `download_rdd2022.py --skip-extract` and turns its output into a Dataset (see the module
    docstring's 2026-09-17 note). Kaggle mounts every added dataset read-only at
    /kaggle/input/<dataset-slug>/, so this searches by filename across ALL mounted datasets rather
    than assuming a specific slug - the cache dataset can be renamed, or other unrelated datasets
    can be mounted alongside it, without breaking this. Returns None (not an error) when
    /kaggle/input doesn't exist at all (i.e. not running on Kaggle) or no dataset has the file.
    """
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None
    matches = list(kaggle_input.rglob(FIGSHARE_ZIP_NAME))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  NOTE: found {len(matches)} copies of {FIGSHARE_ZIP_NAME} under /kaggle/input/, "
              f"using the first: {matches[0]}")
    return matches[0]


def find_kaggle_cached_extracted_countries(root=None):
    """Look for a pre-EXTRACTED per-country RDD2022 tree under /kaggle/input/ - what the cache
    Dataset actually contains (see the module docstring's 2026-09-17 UPDATE). Kaggle's "New Dataset
    from notebook output" recursively auto-unzips every .zip it finds in the output, including zips
    nested inside other zips - so the cache-builder notebook's output (which only ever contained the
    still-zipped combined archive on disk) turned into a fully-extracted directory tree by the time
    it became a Dataset: both the outer combined zip AND all 7 nested per-country zips got expanded
    in place, not just the outer one. That's a BETTER cache than a re-downloadable zip (skips
    extraction too, not just download), so this is checked before find_kaggle_cached_zip() above.

    Returns {country: path} for every COUNTRY_ALIASES key found as a non-empty directory somewhere
    under root (default /kaggle/input). Does ONE pass over the whole mounted-input tree rather than
    one rglob per country - /kaggle/input can hold 80k+ files once a dataset like this is extracted,
    so scanning it 7x over would be wasteful. Returns {} (not an error) when root doesn't exist (e.g.
    off-Kaggle) or nothing matches. `root` is overridable for self-testing without touching the real
    /kaggle/input.
    """
    kaggle_input = Path(root) if root is not None else Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return {}
    found = {}
    for path in kaggle_input.rglob("*"):
        if len(found) == len(COUNTRY_ALIASES):
            break
        if path.name not in COUNTRY_ALIASES or path.name in found:
            continue
        if not path.is_dir():
            continue
        try:
            if any(path.iterdir()):
                found[path.name] = path
        except OSError:
            continue
    return found


def _download_urllib(url, dest_path, retries=3):
    """Plain single-connection streaming download - the fallback used when aria2c isn't available or
    didn't work."""
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url} -> {dest_path} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                last_print = time.time()
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_print > 2:
                        pct = f"{downloaded / total:.0%}" if total else "?"
                        print(f"    {_format_bytes(downloaded)}" + (f" / {_format_bytes(total)} ({pct})" if total else ""),
                              end="\r", file=sys.stderr)
                        last_print = time.time()
            tmp_path.rename(dest_path)
            print(f"\n  done: {dest_path.name} ({_format_bytes(dest_path.stat().st_size)})")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as exc:
            print(f"\n  attempt {attempt} failed: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def verify_md5(path, expected_md5):
    """Stream-hash path and compare against expected_md5. Caches a good result next to the file
    (a .md5ok marker) so re-running this script doesn't re-hash a ~12GB file every time - if you
    ever suspect the downloaded file got corrupted after the fact, delete the .md5ok marker (or the
    zip itself) to force a real re-check.
    """
    marker = path.with_suffix(path.suffix + ".md5ok")
    if marker.exists():
        print(f"  {path.name}: MD5 previously verified (delete {marker.name} to re-check)")
        return True

    print(f"  verifying MD5 of {path.name} ({_format_bytes(path.stat().st_size)}, this can take a minute)...")
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_md5:
        print(f"  MD5 MISMATCH: expected {expected_md5}, got {actual}")
        print(f"  {path} is corrupt or incomplete - delete it and re-run this script to re-download.")
        return False
    marker.write_text("ok\n")
    print(f"  MD5 verified: {actual}")
    return True


def extract_one(zip_path, extract_root):
    """Extract zip_path into extract_root, skipping if already extracted (marker-based, not by
    checking for a specific internal folder name - see reorganize_countries() for why)."""
    marker = extract_root / ".extracted"
    if marker.exists():
        print(f"  already extracted into {extract_root}, skipping")
        return
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} -> {extract_root} (large archive, this can take several minutes)")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"{zip_path} is corrupt (bad member: {bad}) - delete it and re-run to re-download")
        zf.extractall(extract_root)
    marker.write_text("ok\n")


def reorganize_countries(extract_root, raw_dir, wanted_countries):
    """Find each wanted country's directory somewhere inside the extracted tree and link it to
    raw_dir/<country>, so convert_voc_to_yolo.py (which expects data/raw/<country>/ folders) keeps
    working unchanged regardless of whatever top-level wrapper folder(s) FigShare's zip actually
    uses internally.

    Uses a symlink rather than copying, since the extracted tree is already ~12GB+ and duplicating
    it serves no purpose. For each country, searches for directories whose name matches one of its
    known aliases (case-insensitive) and picks the SHALLOWEST match; if more than one directory at
    that same shallowest depth matches (e.g. the zip ships both a train/ and test/ split each with
    their own per-country subfolder), this prints every candidate found and picks the first
    alphabetically - flagged clearly so you can sanity-check it, rather than silently guessing.
    This is exactly the "raw data doesn't match documentation" messiness this project is meant to
    surface, so warn rather than hide it.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    found_any = False

    for country in wanted_countries:
        link_path = raw_dir / country
        if link_path.exists() or link_path.is_symlink():
            print(f"  {country}: {link_path} already exists, leaving as-is")
            found_any = True
            continue

        aliases_lower = {a.lower() for a in COUNTRY_ALIASES[country]}
        candidates = [
            p for p in extract_root.rglob("*")
            if p.is_dir() and p.name.lower() in aliases_lower
        ]
        if not candidates:
            print(f"  WARNING: no directory matching {COUNTRY_ALIASES[country]} found under {extract_root} "
                  f"- {country} will be missing from data/raw/. Check the extracted tree by hand "
                  f"(e.g. `find {extract_root} -iname '*{country.split('_')[0]}*' -type d`) and symlink "
                  f"it manually if this script guessed wrong.")
            continue

        min_depth = min(len(p.relative_to(extract_root).parts) for p in candidates)
        shallowest = sorted(p for p in candidates if len(p.relative_to(extract_root).parts) == min_depth)
        if len(shallowest) > 1:
            print(f"  NOTE: multiple equally-shallow matches for {country}, picking the first:")
            for p in shallowest:
                print(f"    - {p.relative_to(extract_root)}")
        chosen = shallowest[0]
        link_path.symlink_to(chosen, target_is_directory=True)
        print(f"  {country}: linked data/raw/{country} -> {chosen.relative_to(extract_root)}")
        found_any = True

    if not found_any:
        print("  WARNING: none of the requested countries were found - inspect the extracted tree "
              f"under {extract_root} directly; the assumed alias names in COUNTRY_ALIASES may not "
              "match this zip's actual layout.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                         help="root data directory (default: %(default)s)")
    parser.add_argument("--countries", nargs="+", choices=list(COUNTRY_ALIASES) + ["China"], default=None,
                         help="subset of countries to link into data/raw/ after extraction (default: all). "
                              "Does NOT reduce download size - FigShare ships one combined zip for all "
                              "six countries. Pass 'China' for both China_MotorBike and China_Drone.")
    parser.add_argument("--skip-extract", action="store_true", help="download (+ verify) only, don't unzip")
    parser.add_argument("--connections", type=int, default=16,
                         help="parallel connections for aria2c downloads (default: %(default)s, ignored "
                              "if aria2c isn't installed)")
    args = parser.parse_args(argv)

    countries = args.countries
    if countries is None:
        countries = list(COUNTRY_ALIASES)
    elif "China" in countries:
        countries = [c for c in countries if c != "China"] + ["China_MotorBike", "China_Drone"]

    cached_extracted = find_kaggle_cached_extracted_countries()
    missing_from_extracted_cache = [c for c in countries if c not in cached_extracted]
    if cached_extracted and not missing_from_extracted_cache:
        print(f"Found all {len(countries)} requested countries already pre-extracted under "
              f"/kaggle/input/ (see the module docstring's 2026-09-17 UPDATE) - skipping the download "
              f"entirely, no zip needed at all:")
        for country in countries:
            print(f"  {country}: {cached_extracted[country]}")
        print("\nDone (nothing downloaded/extracted here). Next: python scripts/extract_convert_per_country.py "
              "will independently find and use this same cache.")
        return 0
    elif cached_extracted:
        print(f"NOTE: found a partial pre-extracted cache under /kaggle/input/ ({sorted(cached_extracted)}) "
              f"but it's missing {missing_from_extracted_cache} - falling back to the normal "
              f"download/zip-cache path below for all requested countries (not mixing sources).")

    data_dir = Path(args.data_dir)
    zips_dir = data_dir / "zips"
    extract_root = data_dir / "raw" / "_extracted_all"
    raw_dir = data_dir / "raw"

    zip_path = zips_dir / FIGSHARE_ZIP_NAME
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Already have {zip_path} ({_format_bytes(zip_path.stat().st_size)}), skipping download/cache-copy")
    else:
        cached = find_kaggle_cached_zip()
        if cached:
            print(f"Found a pre-cached copy at {cached} (Kaggle input dataset) - copying into "
                  f"{zip_path} instead of downloading ~12.35GB from FigShare again")
            zips_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, zip_path)
            print(f"  copied {_format_bytes(zip_path.stat().st_size)}")
        else:
            print(f"Downloading combined RDD2022 archive (~12.35GB) from FigShare into {zip_path}")
            download_one(FIGSHARE_URL, zip_path, connections=args.connections)

    if not verify_md5(zip_path, FIGSHARE_MD5):
        print("\nAborting - downloaded file failed MD5 verification. Delete the zip and re-run.")
        return 1

    if args.skip_extract:
        print("\n--skip-extract set, stopping after download+verify.")
        return 0

    print(f"\nExtracting into {extract_root}")
    extract_one(zip_path, extract_root)

    print(f"\nLinking {len(countries)} countries into {raw_dir}")
    reorganize_countries(extract_root, raw_dir, countries)

    print("\nDone. Raw data is under:", raw_dir)
    print("Next: python scripts/convert_voc_to_yolo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
