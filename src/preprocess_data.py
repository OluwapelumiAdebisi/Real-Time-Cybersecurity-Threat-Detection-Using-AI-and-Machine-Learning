"""
preprocess_data.py  —  Clean, split and scale CSE-CIC-IDS2018 for modelling.

Reads the raw CSVs file-by-file (memory-safe), applies the cleaning decisions
recorded in config.yaml, carves out the zero-day hold-out, builds a stratified
working subset, scales features without leakage, and saves everything to
data/processed/.

Pipeline (in order):
  1. Read each raw CSV, strip column whitespace.
  2. Drop stray header rows (Label == "Label").
  3. Drop leakage-prone columns (Dst Port, Timestamp, ...).
  4. Coerce features to numeric; clean +/-inf -> NaN -> drop/fill.
  5. Concatenate, then drop exact-duplicate rows.
  6. Build labels: family label (original) + binary label (Benign=0, Attack=1).
  7. Separate the zero-day hold-out families (never seen in training).
  8. Stratified subsample of the remaining data (for manageable experiments).
  9. Train/test split (stratified on binary label).
 10. Scale features: fit scaler on TRAIN ONLY, apply to test + zero-day.
 11. Save processed arrays/frames + a scaler + a preprocessing report.

Outputs (in data/processed/):
  X_train.csv, X_test.csv, X_zeroday.csv        (scaled features)
  y_train.csv, y_test.csv, y_zeroday.csv        (binary + family labels)
  scaler.joblib                                 (fitted StandardScaler)
  feature_columns.json                          (final feature order)
Also writes results/preprocessing_report.txt.

Usage:
    python preprocess_data.py --raw-dir /kaggle/input/datasets/solarmainframe/ids-intrusion-csv
    python preprocess_data.py --raw-dir <path> --full      # use ALL rows, no subsample
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


def clean_one_file(path, label_col, drop_label_values, leakage_cols, log):
    """Read one CSV and apply per-file cleaning. Returns a cleaned DataFrame."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)

    # Drop stray header rows.
    if label_col in df.columns:
        df = df[~df[label_col].astype(str).str.strip().isin(drop_label_values)]
    n_hdr = n0 - len(df)

    # Drop leakage columns that are present.
    present_leak = [c for c in df.columns if c in leakage_cols]
    df = df.drop(columns=present_leak)

    # Separate label, coerce the rest to numeric.
    labels = df[label_col].astype(str).str.strip()
    feats = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    # Replace +/- infinity with NaN.
    feats = feats.replace([np.inf, -np.inf], np.nan)

    out = feats.copy()
    out[label_col] = labels.values
    log(f"    {os.path.basename(path):20s} read {n0:>9,}  dropped {n_hdr} header  "
        f"dropped {len(present_leak)} leak cols  -> {out.shape[1]-1} feats")
    return out, present_leak


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--full", action="store_true",
                    help="Use all rows (no stratified subsample).")
    args = ap.parse_args()

    seed = cfg.get("seed", 42)
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
    use_subset = cfg["sampling"].get("use_subset", True) and not args.full
    subset_rows = cfg["sampling"].get("subset_rows", 200000)

    log = Tee(results_dir / "preprocessing_report.txt")
    log("=" * 70)
    log("CSE-CIC-IDS2018  —  PREPROCESSING REPORT")
    log("=" * 70)
    log(f"Raw dir: {raw_dir}")
    log(f"Seed: {seed} | subset: {use_subset} ({subset_rows:,} rows) | test_size: {test_size}")
    log(f"Zero-day hold-out families: {sorted(zero_day)}")

    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not csv_files:
        log(f"[!] No CSVs in {raw_dir}")
        log.close(); sys.exit(1)

    # ---- 1-4. Clean each file ------------------------------------------
    log(f"\n[1] CLEANING {len(csv_files)} FILES")
    log("-" * 70)
    frames = []
    for f in csv_files:
        df, _ = clean_one_file(f, label_col, drop_label_values, leakage_cols, log)
        frames.append(df)

    # Align on common columns, then concat.
    common = set(frames[0].columns)
    for fr in frames[1:]:
        common &= set(fr.columns)
    common = [c for c in frames[0].columns if c in common]
    data = pd.concat([fr[common] for fr in frames], ignore_index=True)
    log(f"\n    Combined: {data.shape[0]:,} rows x {data.shape[1]-1} features")

    # ---- 5. Drop duplicates --------------------------------------------
    if do_dedup:
        before = len(data)
        data = data.drop_duplicates().reset_index(drop=True)
        log(f"[2] Dropped {before-len(data):,} duplicate rows -> {len(data):,} remain")

    # ---- Handle residual NaNs (from inf or parse) ----------------------
    feat_cols = [c for c in data.columns if c != label_col]
    nan_rows = data[feat_cols].isna().any(axis=1).sum()
    if nan_rows:
        # Median-impute (robust); alternative is to drop. Median keeps rows.
        medians = data[feat_cols].median(numeric_only=True)
        data[feat_cols] = data[feat_cols].fillna(medians)
        log(f"[3] Imputed NaNs in {nan_rows:,} rows with column medians")

    # ---- 6. Labels: family + binary ------------------------------------
    data[label_col] = data[label_col].astype(str).str.strip()
    data["binary_label"] = (data[label_col] != benign).astype(int)  # 1 = attack

    # ---- 7. Zero-day hold-out ------------------------------------------
    is_zero = data[label_col].isin(zero_day)
    zeroday = data[is_zero].copy()
    main_data = data[~is_zero].copy()
    log(f"\n[4] Zero-day hold-out: {len(zeroday):,} rows "
        f"({', '.join(sorted(zero_day))}) set aside")
    log(f"    Remaining for train/test: {len(main_data):,} rows")

    # ---- 8. Stratified subsample ---------------------------------------
    if use_subset and len(main_data) > subset_rows:
        # stratify on family label so rare classes survive
        frac = subset_rows / len(main_data)
        main_data = (main_data
                     .groupby(label_col, group_keys=False)
                     .apply(lambda g: g.sample(frac=frac, random_state=seed)))
        main_data = main_data.reset_index(drop=True)
        log(f"[5] Stratified subsample -> {len(main_data):,} rows")

    # ---- 9. Train/test split (stratified on binary) --------------------
    feat_cols = [c for c in main_data.columns
                 if c not in (label_col, "binary_label")]
    X = main_data[feat_cols]
    y = main_data[[label_col, "binary_label"]]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=main_data["binary_label"])
    log(f"[6] Split: train {len(X_tr):,} | test {len(X_te):,}")

    # ---- 10. Scale (fit on TRAIN only) ---------------------------------
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_cols, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=feat_cols, index=X_te.index)
    Xz = zeroday[feat_cols]
    Xz_s = pd.DataFrame(scaler.transform(Xz), columns=feat_cols, index=zeroday.index)
    yz = zeroday[[label_col, "binary_label"]]
    log(f"[7] Scaled features (StandardScaler fit on train only — no leakage)")

    # ---- 11. Save -------------------------------------------------------
    X_tr_s.to_csv(proc_dir / "X_train.csv", index=False)
    X_te_s.to_csv(proc_dir / "X_test.csv", index=False)
    Xz_s.to_csv(proc_dir / "X_zeroday.csv", index=False)
    y_tr.to_csv(proc_dir / "y_train.csv", index=False)
    y_te.to_csv(proc_dir / "y_test.csv", index=False)
    yz.to_csv(proc_dir / "y_zeroday.csv", index=False)
    joblib.dump(scaler, proc_dir / "scaler.joblib")
    with open(proc_dir / "feature_columns.json", "w") as fh:
        json.dump(feat_cols, fh, indent=2)

    # ---- Summary --------------------------------------------------------
    log(f"\n[8] SAVED TO {proc_dir}")
    log("-" * 70)
    def dist(yframe, name):
        b = int((yframe['binary_label'] == 0).sum())
        a = int((yframe['binary_label'] == 1).sum())
        log(f"    {name:10s} {len(yframe):>9,} rows  |  benign {b:>8,}  attack {a:>8,}")
    dist(y_tr, "train"); dist(y_te, "test"); dist(yz, "zero-day")
    log(f"    features: {len(feat_cols)}")
    log(f"\n    For autoencoders: train on benign rows of X_train only")
    log(f"      (y_train.binary_label == 0).")
    log(f"    Report: {results_dir/'preprocessing_report.txt'}")
    log("=" * 70)
    log.close()


if __name__ == "__main__":
    main()
