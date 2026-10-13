"""
download_data.py  —  Acquire the CSE-CIC-IDS2018 dataset.

Designed to run in Google Colab or Kaggle. Three routes are supported;
pick whichever matches where you are running.

------------------------------------------------------------------
ROUTE A — Kaggle notebook (easiest)
------------------------------------------------------------------
If you run this inside a Kaggle notebook, add the dataset to the
notebook ("+ Add Data" -> search "ids-intrusion-csv") and it will be
mounted read-only at:
    /kaggle/input/ids-intrusion-csv/
No download needed. Set RAW_DIR below to that path (or symlink it).

------------------------------------------------------------------
ROUTE B — Google Colab via the Kaggle API
------------------------------------------------------------------
1. On kaggle.com -> Account -> "Create New API Token" -> downloads kaggle.json
2. In Colab, upload kaggle.json when prompted (code below handles it),
   then this script pulls the dataset into data/raw/.

------------------------------------------------------------------
ROUTE C — Official source (Canadian Institute for Cybersecurity)
------------------------------------------------------------------
The original is distributed via AWS S3 (see the UNB CIC IDS-2018 page).
It is large (~220 GB of raw PCAPs; the processed CSVs are a few GB).
For this project the processed CSVs (Route A/B) are what you want.
------------------------------------------------------------------
"""

import os
import sys
import glob
import argparse
from pathlib import Path


def find_project_root() -> Path:
    """Walk up until we find config.yaml, so paths work from anywhere."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "config.yaml").exists():
            return p
    return Path.cwd()


ROOT = find_project_root()
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_DATASET = "solarmainframe/ids-intrusion-csv"
KAGGLE_INPUT = Path("/kaggle/input/ids-intrusion-csv")


def route_kaggle_mounted() -> bool:
    """If running on Kaggle with the dataset added, just point at it."""
    if KAGGLE_INPUT.exists():
        csvs = list(KAGGLE_INPUT.glob("*.csv"))
        print(f"[Kaggle] Dataset already mounted at {KAGGLE_INPUT}")
        print(f"[Kaggle] Found {len(csvs)} CSV file(s).")
        print("[Kaggle] No download needed. Point config.yaml raw_dir at that path,")
        print("         or run:  ln -s /kaggle/input/ids-intrusion-csv data/raw")
        return True
    return False


def route_kaggle_api():
    """Download via the Kaggle API (works in Colab once kaggle.json is set up)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure the kaggle package is available.
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[setup] Installing the kaggle package...")
        os.system(f"{sys.executable} -m pip install -q kaggle")

    # Check credentials.
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists() and not os.environ.get("KAGGLE_USERNAME"):
        print("\n[!] Kaggle credentials not found.")
        print("    In Colab, run this first to upload kaggle.json:")
        print("        from google.colab import files; files.upload()")
        print("        !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
        print("    Then re-run this script.")
        return

    print(f"[download] Pulling {KAGGLE_DATASET} into {RAW_DIR} ...")
    # --unzip extracts the CSVs directly.
    code = os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p "{RAW_DIR}" --unzip'
    )
    if code != 0:
        print("[!] Download failed. Check your Kaggle token and internet access.")
        return

    csvs = glob.glob(str(RAW_DIR / "*.csv"))
    print(f"[done] {len(csvs)} CSV file(s) now in {RAW_DIR}")
    for c in sorted(csvs):
        size_mb = os.path.getsize(c) / 1e6
        print(f"    {os.path.basename(c):50s} {size_mb:8.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Acquire CSE-CIC-IDS2018.")
    ap.add_argument(
        "--route",
        choices=["auto", "kaggle-api"],
        default="auto",
        help="auto = use mounted Kaggle data if present, else fall back to the API.",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("CSE-CIC-IDS2018 data acquisition")
    print("=" * 60)

    if args.route in ("auto",) and route_kaggle_mounted():
        return
    route_kaggle_api()


if __name__ == "__main__":
    main()
