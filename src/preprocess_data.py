"""
preprocess_data.py  —  Clean, split and scale CSE-CIC-IDS2018 for modelling.

MEMORY-SAFE streaming version: the full 16M-row dataset is NEVER held in memory
at once. Each CSV is read in chunks; from each chunk we keep (a) all zero-day
hold-out rows (they're rare) and (b) a random fraction of the rest. Only these
small kept slices accumulate, so peak memory stays low enough for Kaggle.

Pipeline (in order):
  1. Pass 1 (cheap): count rows per file to size the sampling fraction.
  2. Pass 2 (chunked): per chunk -> strip headers, drop leakage cols, coerce
     numeric, inf->NaN; keep all zero-day rows + a sampled fraction of the rest.
  3. Concatenate the (small) kept slices.
  4. Drop exact duplicates; median-impute residual NaNs.
  5. Labels: family label + binary label (Benign=0, Attack=1).
  6. Separate zero-day hold-out (web attacks) from main data.
  7. Train/test split (stratified on binary label).
  8. Scale: fit StandardScaler on TRAIN ONLY, apply to test + zero-day.
  9. Save processed CSVs + scaler + feature list + report.

Usage:
  python preprocess_data.py --raw-dir /kaggle/input/datasets/solarmainframe/ids-intrusion-csv
  python preprocess_data.py --raw-dir <path> --target-rows 200000   # sample size
  python preprocess_data.py --raw-dir <path> --chunksize 250000      # rows per chunk
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


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
    raise SystemExit("config.yaml not found or PyYAML missing.")


class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.file.write(line + "\n")
    def close(self):
        self.file.close()


def count_rows(path):
    """Fast row count without loading the file into a DataFrame."""
    with open(path, "r", encoding="latin-1", errors="ignore") as fh:
        return sum(1 for _ in fh) - 1  # minus header


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--full", action="store_true",
                    help="Keep all rows (still chunked, but no down-sampling).")
    ap.add_argument("--target-rows", type=int, default=None,
                    help="Approx. total rows to keep from the non-zero-day data.")
    ap.add_argument("--chunksize", type=int, default=250000,
                    help="Rows per read chunk (lower this if memory is tight).")
    args = ap.parse_args()

    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    raw_dir = Path(args.raw_dir) if args.raw_dir else ROOT / cfg["paths"]["raw_dir"]
    proc_dir = ROOT / cfg["paths"]["processed_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    label_col = cfg["dataset"]["label_column"]
    benign = cfg["dataset"]["benign_label"]
    leakage_cols = set(cfg["dataset"].get("leakage_columns", []))
    drop_label_values = set(cfg.get("cleaning", {}).get("drop_label_values", []))
    do_dedup = cfg.get("cleaning", {}).get("drop_duplicates", True)
    zero_day = set(cfg["evaluation"].get("zero_day_holdout", []))
    test_size = cfg["evaluation"].get("test_size", 0.2)
    target_rows = args.target_rows or cfg["sampling"].get("subset_rows", 200000)

    log = Tee(results_dir / "preprocessing_report.txt")
    log("=" * 70)
    log("CSE-CIC-IDS2018  —  PREPROCESSING REPORT  (memory-safe streaming)")
    log("=" * 70)
    log(f"Raw dir: {raw_dir}")
    log(f"Seed: {seed} | chunksize: {args.chunksize:,} | "
        f"{'FULL (no downsample)' if args.full else f'target ~{target_rows:,} rows'}")
    log(f"Zero-day hold-out families: {sorted(zero_day)}")

    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        log(f"[!] No CSVs in {raw_dir}")
        log.close(); sys.exit(1)

    # ---- Pass 1: total row count -> global sampling fraction -----------
    log(f"\n[1] COUNTING ROWS ({len(csv_files)} files)")
    log("-" * 70)
    total = 0
    for f in csv_files:
        n = count_rows(f)
        total += n
        log(f"    {os.path.basename(f):20s} {n:>12,} rows")
    log(f"    {'TOTAL':20s} {total:>12,} rows")

    if args.full:
        keep_frac = 1.0
    else:
        # Keep a bit extra; zero-day rows are kept in full regardless.
        keep_frac = min(1.0, target_rows / max(total, 1))
    log(f"    Sampling fraction for non-zero-day rows: {keep_frac:.4f}")

    # ---- Pass 2: chunked read + sample ---------------------------------
    log(f"\n[2] STREAMING + CLEANING (chunksize {args.chunksize:,})")
    log("-" * 70)
    kept = []           # list of small DataFrames
    seen_leak = set()
    for f in csv_files:
        file_kept = 0
        for chunk in pd.read_csv(f, chunksize=args.chunksize,
                                 low_memory=False, encoding="latin-1"):
            chunk.columns = [c.strip() for c in chunk.columns]
            # drop stray header rows
            if label_col in chunk.columns:
                chunk = chunk[~chunk[label_col].astype(str).str.strip()
                              .isin(drop_label_values)]
            if chunk.empty:
                continue
            # drop leakage columns
            present_leak = [c for c in chunk.columns if c in leakage_cols]
            seen_leak.update(present_leak)
            chunk = chunk.drop(columns=present_leak)
            # split label / numeric features
            lab = chunk[label_col].astype(str).str.strip()
            feats = (chunk.drop(columns=[label_col])
                     .apply(pd.to_numeric, errors="coerce")
                     .replace([np.inf, -np.inf], np.nan))
            feats[label_col] = lab.values
            # keep ALL zero-day rows; sample the rest
            is_zero = feats[label_col].isin(zero_day)
            zero_part = feats[is_zero]
            rest = feats[~is_zero]
            if keep_frac < 1.0 and len(rest):
                mask = rng.random(len(rest)) < keep_frac
                rest = rest[mask]
            piece = pd.concat([zero_part, rest], ignore_index=True)
            if len(piece):
                kept.append(piece)
                file_kept += len(piece)
        log(f"    {os.path.basename(f):20s} kept {file_kept:>9,} rows")

    data = pd.concat(kept, ignore_index=True)
    del kept
    log(f"\n    Collected sample: {data.shape[0]:,} rows x {data.shape[1]-1} features")
    if seen_leak:
        log(f"    Leakage columns dropped: {sorted(seen_leak)}")

    # ---- 4. Dedup + impute ---------------------------------------------
    if do_dedup:
        before = len(data)
        data = data.drop_duplicates().reset_index(drop=True)
        log(f"[3] Dropped {before-len(data):,} duplicates -> {len(data):,} rows")

    feat_cols = [c for c in data.columns if c != label_col]
    nan_rows = int(data[feat_cols].isna().any(axis=1).sum())
    if nan_rows:
        data[feat_cols] = data[feat_cols].fillna(data[feat_cols].median(numeric_only=True))
        log(f"[4] Imputed NaNs in {nan_rows:,} rows with column medians")

    # ---- 5. Labels ------------------------------------------------------
    data[label_col] = data[label_col].astype(str).str.strip()
    data["binary_label"] = (data[label_col] != benign).astype(int)

    # ---- 6. Zero-day split ---------------------------------------------
    is_zero = data[label_col].isin(zero_day)
    zeroday = data[is_zero].copy()
    main_data = data[~is_zero].copy()
    log(f"\n[5] Zero-day hold-out: {len(zeroday):,} rows set aside")
    log(f"    Remaining for train/test: {len(main_data):,} rows")

    # ---- 7. Train/test split -------------------------------------------
    feat_cols = [c for c in main_data.columns if c not in (label_col, "binary_label")]
    X = main_data[feat_cols]
    strat = main_data["binary_label"] if main_data["binary_label"].nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, main_data[[label_col, "binary_label"]],
        test_size=test_size, random_state=seed, stratify=strat)
    log(f"[6] Split: train {len(X_tr):,} | test {len(X_te):,}")

    # ---- 8. Scale (fit on train only) ----------------------------------
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_cols)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=feat_cols)
    Xz_s = pd.DataFrame(scaler.transform(zeroday[feat_cols]), columns=feat_cols)
    log(f"[7] Scaled (StandardScaler fit on train only — no leakage)")

    # ---- 9. Save --------------------------------------------------------
    X_tr_s.to_csv(proc_dir / "X_train.csv", index=False)
    X_te_s.to_csv(proc_dir / "X_test.csv", index=False)
    Xz_s.to_csv(proc_dir / "X_zeroday.csv", index=False)
    y_tr.reset_index(drop=True).to_csv(proc_dir / "y_train.csv", index=False)
    y_te.reset_index(drop=True).to_csv(proc_dir / "y_test.csv", index=False)
    zeroday[[label_col, "binary_label"]].reset_index(drop=True).to_csv(
        proc_dir / "y_zeroday.csv", index=False)
    joblib.dump(scaler, proc_dir / "scaler.joblib")
    with open(proc_dir / "feature_columns.json", "w") as fh:
        json.dump(feat_cols, fh, indent=2)

    # ---- Summary --------------------------------------------------------
    log(f"\n[8] SAVED TO {proc_dir}")
    log("-" * 70)
    def dist(yf, name):
        b = int((yf['binary_label'] == 0).sum()); a = int((yf['binary_label'] == 1).sum())
        log(f"    {name:10s} {len(yf):>9,} rows  |  benign {b:>8,}  attack {a:>8,}")
    dist(y_tr, "train"); dist(y_te, "test")
    dist(zeroday[[label_col, 'binary_label']], "zero-day")
    log(f"    features: {len(feat_cols)}")
    log(f"\n    Autoencoders: train on benign rows only (y_train.binary_label == 0).")
    log(f"    Report: {results_dir/'preprocessing_report.txt'}")
    log("=" * 70)
    log.close()


if __name__ == "__main__":
    main()
