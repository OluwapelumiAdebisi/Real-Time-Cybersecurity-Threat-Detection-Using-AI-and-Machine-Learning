"""
inspect_data.py  —  First-look inspection and sanity-check of CSE-CIC-IDS2018.

Run this straight after downloading the data. It does NOT modify anything;
it reports what the data actually contains so preprocessing decisions are
grounded in fact rather than assumption. It writes a short report to
results/data_inspection_report.txt and prints the same to screen.

Checks performed:
  1. Which CSV files exist and how big they are.
  2. Columns actually present (and which configured leakage columns are found).
  3. Row counts (total across files).
  4. Class distribution  ->  benign vs each attack family, with imbalance ratio.
  5. Data-quality flags: NaN counts, infinity values, duplicate rows,
     constant (zero-variance) columns, non-numeric feature columns.

Usage:
    python inspect_data.py                 # uses raw_dir from config.yaml
    python inspect_data.py --raw-dir PATH  # override (e.g. Kaggle mount)
    python inspect_data.py --nrows 100000  # sample rows per file for a fast look
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "config.yaml").exists():
            return p
    return Path.cwd()


ROOT = find_project_root()


def load_config():
    cfg_path = ROOT / "config.yaml"
    if yaml and cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    # Fallback defaults if PyYAML missing.
    return {
        "paths": {"raw_dir": "data/raw", "results_dir": "results"},
        "dataset": {
            "label_column": "Label",
            "benign_label": "Benign",
            "leakage_columns": [
                "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
                "Src Port", "Source Port", "Dst Port", "Destination Port", "Timestamp",
            ],
        },
    }


class Tee:
    """Write report lines to both stdout and a file."""
    def __init__(self, path):
        self.file = open(path, "w")
    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.file.write(line + "\n")
    def close(self):
        self.file.close()


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--nrows", type=int, default=None,
                    help="Rows to read per file (for a fast preview). Omit to read all.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else ROOT / cfg["paths"]["raw_dir"]
    results_dir = ROOT / cfg["paths"].get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    label_col = cfg["dataset"]["label_column"]
    benign = cfg["dataset"]["benign_label"]
    leakage_cfg = set(cfg["dataset"].get("leakage_columns", []))

    report_path = results_dir / "data_inspection_report.txt"
    log = Tee(report_path)

    log("=" * 70)
    log("CSE-CIC-IDS2018  —  DATA INSPECTION REPORT")
    log("=" * 70)
    log(f"Raw data directory: {raw_dir}")

    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        log(f"\n[!] No CSV files found in {raw_dir}.")
        log("    Run download_data.py first, or pass --raw-dir with the correct path.")
        log.close()
        sys.exit(1)

    # ---- 1. Files -------------------------------------------------------
    log(f"\n[1] FILES  ({len(csv_files)} found)")
    log("-" * 70)
    total_bytes = 0
    for f in csv_files:
        sz = os.path.getsize(f)
        total_bytes += sz
        log(f"    {os.path.basename(f):55s} {sz/1e6:9.1f} MB")
    log(f"    {'TOTAL':55s} {total_bytes/1e6:9.1f} MB")

    # ---- 2+3. Read, columns, rows --------------------------------------
    log(f"\n[2] READING DATA" + (f"  (first {args.nrows:,} rows per file)" if args.nrows else "  (all rows)"))
    log("-" * 70)

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=args.nrows, low_memory=False, encoding="latin-1")
        except Exception as e:
            log(f"    [!] Could not read {os.path.basename(f)}: {e}")
            continue
        # Normalise column whitespace (these CSVs often have leading spaces).
        df.columns = [c.strip() for c in df.columns]
        log(f"    {os.path.basename(f):55s} {len(df):>10,} rows  {df.shape[1]} cols")
        frames.append(df)

    if not frames:
        log("\n[!] No readable data. Stopping.")
        log.close()
        sys.exit(1)

    # Align on common columns before concat (guards against schema drift between files).
    common = set(frames[0].columns)
    for fr in frames[1:]:
        common &= set(fr.columns)
    common = [c for c in frames[0].columns if c in common]
    data = pd.concat([fr[common] for fr in frames], ignore_index=True)
    log(f"\n    Combined shape: {data.shape[0]:,} rows x {data.shape[1]} columns")

    # ---- Columns --------------------------------------------------------
    log(f"\n[3] COLUMNS  ({data.shape[1]} total)")
    log("-" * 70)
    for i, c in enumerate(data.columns):
        log(f"    {i:3d}  {c}")

    if label_col not in data.columns:
        log(f"\n[!] Expected label column '{label_col}' not found.")
        log(f"    Columns containing 'label': {[c for c in data.columns if 'label' in c.lower()]}")
        log.close()
        sys.exit(1)

    # ---- Leakage columns present ---------------------------------------
    present_leak = [c for c in data.columns if c in leakage_cfg]
    log(f"\n[4] LEAKAGE-PRONE COLUMNS PRESENT  (to drop in preprocessing)")
    log("-" * 70)
    if present_leak:
        for c in present_leak:
            log(f"    {c}")
    else:
        log("    (none of the configured leakage columns are present under those names)")

    # ---- 5. Class distribution -----------------------------------------
    log(f"\n[5] CLASS DISTRIBUTION  (column: '{label_col}')")
    log("-" * 70)
    labels = data[label_col].astype(str).str.strip()
    counts = labels.value_counts()
    total = len(labels)
    for lab, n in counts.items():
        tag = "  <- benign" if lab == benign else ""
        log(f"    {lab:45s} {n:>10,}  ({100*n/total:5.2f}%){tag}")

    n_benign = int(counts.get(benign, 0))
    n_attack = total - n_benign
    log(f"\n    Benign : {n_benign:>10,}  ({100*n_benign/total:.2f}%)")
    log(f"    Attack : {n_attack:>10,}  ({100*n_attack/total:.2f}%)")
    if n_attack > 0:
        log(f"    Imbalance ratio (benign : attack) = {n_benign/max(n_attack,1):.1f} : 1")
    log(f"    Distinct attack families: {labels.nunique() - (1 if n_benign else 0)}")

    # ---- Data quality ---------------------------------------------------
    log(f"\n[6] DATA-QUALITY CHECKS")
    log("-" * 70)

    feature_cols = [c for c in data.columns if c != label_col]
    coerced = data[feature_cols].apply(pd.to_numeric, errors="coerce")

    # A column is "non-numeric" if coercion introduced NaNs that weren't already there.
    non_numeric = [c for c in feature_cols
                   if coerced[c].isna().sum() > data[c].isna().sum()]
    log(f"    Non-numeric / mixed feature columns : {len(non_numeric)}")
    for c in non_numeric[:15]:
        log(f"        {c}")
    if len(non_numeric) > 15:
        log(f"        ... and {len(non_numeric)-15} more")

    # Quality checks below count only genuinely numeric feature columns,
    # so text columns (dropped later anyway) don't inflate the NaN figure.
    numeric_cols = [c for c in feature_cols if c not in non_numeric]
    numeric = coerced[numeric_cols]

    nan_total = int(numeric.isna().sum().sum())
    inf_total = int(np.isinf(numeric.to_numpy(dtype='float64', na_value=np.nan)).sum())
    log(f"    NaN cells in numeric features              : {nan_total:,}")
    log(f"    Cells that are +/-Infinity                 : {inf_total:,}")
    cols_with_inf = [c for c in numeric.columns
                     if np.isinf(numeric[c].to_numpy(dtype='float64', na_value=np.nan)).any()]
    log(f"    Columns containing infinity (common in Flow Bytes/s, Flow Packets/s):")
    for c in cols_with_inf[:10]:
        log(f"        {c}")
    if len(cols_with_inf) > 10:
        log(f"        ... and {len(cols_with_inf)-10} more")

    dup = int(data.duplicated().sum())
    log(f"    Fully duplicated rows                      : {dup:,}  ({100*dup/total:.2f}%)")

    zero_var = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    log(f"    Constant (zero-variance) columns           : {len(zero_var)}")
    for c in zero_var[:15]:
        log(f"        {c}")
    if len(zero_var) > 15:
        log(f"        ... and {len(zero_var)-15} more")

    # ---- Summary / recommendations -------------------------------------
    log(f"\n[7] SUMMARY & NEXT-STEP FLAGS")
    log("-" * 70)
    log(f"    * Total rows ............... {total:,}")
    log(f"    * Feature columns .......... {len(feature_cols)}")
    log(f"    * Attack families .......... {labels.nunique() - (1 if n_benign else 0)}")
    log(f"    * Class imbalance .......... {n_benign/max(n_attack,1):.1f} : 1  (handle with SMOTE / class weights)")
    log(f"    * Rows to clean ............ {dup:,} duplicates, {nan_total:,} NaNs, {inf_total:,} infinities")
    log(f"    * Columns to drop .......... {len(present_leak)} leakage + {len(zero_var)} constant")
    log(f"\n    Report written to: {report_path}")
    log("=" * 70)
    log.close()


if __name__ == "__main__":
    main()
