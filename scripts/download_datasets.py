"""Download the catalogued Kaggle datasets into dataset/machine_learning and dataset/deep_learning.

Requires the Kaggle API:
    pip install kaggle
    kaggle auth login                  (OAuth in the browser, recommended)

Alternatives to OAuth, for Kaggle CLI 2.2 and newer:
    set KAGGLE_API_TOKEN=<token from https://www.kaggle.com/settings/api>
    or save that token to ~/.kaggle/access_token
Older clients also accept ~/.kaggle/kaggle.json.

Examples:
    python download_datasets.py --list
    python download_datasets.py --category ml
    python download_datasets.py --category dl --max-size 500
    python download_datasets.py --only creditcardfraud --only fashionmnist
    python download_datasets.py --all --dry-run
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CATALOG = os.path.join(ROOT, "catalog", "datasets.json")


def load_catalog():
    with open(CATALOG, encoding="utf-8") as fh:
        return json.load(fh)


def human(n):
    if not n:
        return "unknown"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024


def check_credentials():
    """True if the Kaggle client has some usable credential.

    Covers OAuth (kaggle auth login), the 2.2+ access token, and legacy kaggle.json.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".kaggle")
    for name in ("access_token", "kaggle.json", "oauth_token.json", "credentials.json"):
        if os.path.isfile(os.path.join(config_dir, name)):
            return True
    return False


CRED_HELP = """Kaggle credentials not found. Authenticate once with either:

  kaggle auth login        opens a browser, nothing to store yourself

or generate a token at https://www.kaggle.com/settings/api and then:

  set KAGGLE_API_TOKEN=<token>        (this shell only)
  or save the token to %s"""


def kaggle_base():
    """Prefer the kaggle CLI on PATH, fall back to the module in this interpreter."""
    from shutil import which
    if which("kaggle"):
        return ["kaggle"]
    return [sys.executable, "-m", "kaggle"]


def kaggle_cmd(entry, target):
    base = kaggle_base()
    if entry["kind"] == "competition":
        return base + ["competitions", "download", "-c", entry["slug"], "-p", target]
    return base + ["datasets", "download", "-d", entry["slug"], "-p", target]


def unzip_all(target):
    for name in sorted(os.listdir(target)):
        if not name.endswith(".zip"):
            continue
        path = os.path.join(target, name)
        try:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
            os.remove(path)
        except zipfile.BadZipFile:
            print("    warning: %s is not a valid zip, left in place" % name)


def oversized_files(target, ceiling):
    """Extracted files at or over the ceiling. A small zip can hide a huge CSV."""
    found = []
    for dirpath, _, names in os.walk(target):
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size >= ceiling:
                found.append((size, os.path.relpath(path, target)))
    return sorted(found, reverse=True)


def already_downloaded(target):
    if not os.path.isdir(target):
        return False
    return any(n != ".gitkeep" and not n.endswith(".zip") for n in os.listdir(target))


def main():
    p = argparse.ArgumentParser(description="Download catalogued Kaggle datasets.")
    p.add_argument("--category", choices=["ml", "dl", "all"], default="all")
    p.add_argument("--only", action="append", default=[],
                   help="download only slugs containing this text (repeatable)")
    p.add_argument("--max-size", type=float, default=None,
                   help="skip datasets larger than this many MB")
    p.add_argument("--all", action="store_true", help="same as --category all")
    p.add_argument("--list", action="store_true", help="print the catalogue and exit")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    p.add_argument("--keep-zip", action="store_true", help="do not unzip after download")
    p.add_argument("--cap", type=float, default=100.0,
                   help="hard ceiling in MB, strictly enforced on the download size AND on every "
                        "extracted file (default 100, 0 disables)")
    p.add_argument("--keep-oversize", action="store_true",
                   help="report files at or over the cap instead of deleting the dataset")
    args = p.parse_args()

    entries = load_catalog()
    if args.category != "all" and not args.all:
        entries = [e for e in entries if e["category"] == args.category]
    if args.only:
        entries = [e for e in entries
                   if any(t.lower() in e["slug"].lower() for t in args.only)]
    if args.cap:
        # the catalogue is curated to stay under this ceiling; this enforces it at download time.
        # strictly less than: a dataset exactly at the cap is rejected too
        ceiling = int(args.cap * 1_000_000)
        oversized = [e for e in entries if (e["size_bytes"] or 0) >= ceiling]
        for e in oversized:
            print("skipping %s, %s is at or over the %.0f MB cap" % (e["slug"], human(e["size_bytes"]), args.cap))
        entries = [e for e in entries if e not in oversized]

    if args.max_size is not None:
        # competition entries have no published size, so they are excluded here
        limit = args.max_size * 1024 * 1024
        entries = [e for e in entries if e["size_bytes"] and e["size_bytes"] <= limit]

    if args.list:
        for e in entries:
            print("%-3s %-62s %10s  %s" % (e["category"], e["slug"], human(e["size_bytes"]), e["title"]))
        print("\n%d entries, about %s total" % (len(entries), human(sum(e["size_bytes"] or 0 for e in entries))))
        return 0

    if not entries:
        print("No entries matched the filters.")
        return 1

    total = sum(e["size_bytes"] or 0 for e in entries)
    print("Selected %d datasets, about %s to download.\n" % (len(entries), human(total)))

    if args.dry_run:
        for e in entries:
            print("would download %s -> %s" % (e["slug"], e["folder"]))
        return 0

    if not check_credentials():
        token_path = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
        print(CRED_HELP % token_path)
        return 1

    ok, failed, skipped, violations = [], [], [], []
    for i, e in enumerate(entries, 1):
        target = os.path.join(ROOT, *e["folder"].split("/"))
        print("[%d/%d] %s (%s)" % (i, len(entries), e["slug"], human(e["size_bytes"])))
        if already_downloaded(target) and not args.force:
            print("    already present, skipping")
            skipped.append(e["slug"])
            continue
        os.makedirs(target, exist_ok=True)
        try:
            result = subprocess.run(kaggle_cmd(e, target), capture_output=True, text=True)
        except FileNotFoundError:
            print("    kaggle CLI not found. Run: pip install kaggle")
            return 1
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip().splitlines()
            print("    failed: %s" % (msg[-1] if msg else "unknown error"))
            if e["kind"] == "competition":
                print("    competitions require accepting the rules once at %s" % e["url"])
            failed.append(e["slug"])
            continue
        if not args.keep_zip:
            unzip_all(target)
        if args.cap:
            big = oversized_files(target, int(args.cap * 1_000_000))
            if big:
                print("    CAP VIOLATION: %s contains %s (%s)"
                      % (e["slug"], big[0][1], human(big[0][0])))
                if args.keep_oversize:
                    violations.append((e["slug"], big[0][1], big[0][0]))
                else:
                    shutil.rmtree(target, ignore_errors=True)
                    print("    removed, no file under this tree may reach %.0f MB" % args.cap)
                    violations.append((e["slug"], big[0][1], big[0][0]))
                    continue
        print("    done -> %s" % e["folder"])
        ok.append(e["slug"])

    print("\nDownloaded %d, skipped %d, failed %d" % (len(ok), len(skipped), len(failed)))
    for slug in failed:
        print("  failed: %s" % slug)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
