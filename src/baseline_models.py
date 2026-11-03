"""
baseline_models.py  —  Baseline anomaly detectors for zero-day intrusion detection.

Two unsupervised baselines, evaluated identically:
  1. Isolation Forest  — classical anomaly detector (sklearn).
  2. Dense Autoencoder — neural reconstruction model, trained on BENIGN ONLY;
     flows with high reconstruction error are flagged as anomalous.

Both are scored on:
  * the held-out TEST set  (seen attack families + benign), and
  * the ZERO-DAY set       (unseen web attacks) — the key research question.

Reported per model & per set: precision, recall, F1, false-positive rate (FPR),
ROC-AUC, confusion matrix, plus fit/inference timing. Results saved to
results/baseline_results.json and results/baseline_report.txt; figures to results/.

Reads the processed data produced by preprocess_data.py.

Usage:
  python baseline_models.py                       # uses data/processed from config
  python baseline_models.py --epochs 30           # autoencoder training epochs
  python baseline_models.py --contamination 0.1   # Isolation Forest contamination
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    return {"seed": 42, "paths": {"processed_dir": "data/processed",
            "results_dir": "results", "models_dir": "models"}}


class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line); self.file.write(line + "\n")
    def close(self):
        self.file.close()


def metrics_block(y_true, y_pred, scores=None):
    """Compute the standard IDS metrics. y=1 is attack (the positive class)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr":       float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["roc_auc"] = None
    return out


def zero_day_recall(y_pred_zero):
    """On the zero-day set every row is an attack, so recall = detection rate."""
    return float(np.mean(y_pred_zero == 1))


def run_isolation_forest(X_tr, X_te, y_te, X_zero, contamination, seed, log):
    log("\n" + "=" * 60)
    log("MODEL 1 — ISOLATION FOREST")
    log("=" * 60)
    t0 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    iso.fit(X_tr)
    fit_t = time.time() - t0

    # sklearn: predict -> 1 (inlier) / -1 (outlier). Map outlier -> attack (1).
    t0 = time.time()
    te_pred = (iso.predict(X_te) == -1).astype(int)
    infer_t = time.time() - t0
    per_sample_ms = 1000 * infer_t / len(X_te)

    # score_samples: lower = more anomalous -> negate so higher = more attack-like
    te_scores = -iso.score_samples(X_te)
    zero_pred = (iso.predict(X_zero) == -1).astype(int)

    m = metrics_block(y_te, te_pred, te_scores)
    zdr = zero_day_recall(zero_pred)
    log(f"  fit: {fit_t:.1f}s | inference: {per_sample_ms:.4f} ms/sample")
    log(f"  TEST  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
        f"F1 {m['f1']:.3f}  FPR {m['fpr']:.3f}  AUC {m.get('roc_auc')}")
    log(f"  ZERO-DAY detection rate (recall on unseen web attacks): {zdr:.3f}")
    return {"test": m, "zero_day_recall": zdr,
            "fit_time_s": fit_t, "infer_ms_per_sample": per_sample_ms}


def run_autoencoder(X_tr, y_tr, X_te, y_te, X_zero, epochs, seed, log, results_dir):
    log("\n" + "=" * 60)
    log("MODEL 2 — DENSE AUTOENCODER (trained on BENIGN only)")
    log("=" * 60)
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Model

    # Train on benign rows only — the model learns 'normal'.
    benign_mask = (y_tr["binary_label"].values == 0)
    X_tr_benign = X_tr[benign_mask]
    log(f"  training on {len(X_tr_benign):,} benign flows (of {len(X_tr):,})")

    n_features = X_tr.shape[1]
    inp = layers.Input(shape=(n_features,))
    e = layers.Dense(64, activation="relu")(inp)
    e = layers.Dense(32, activation="relu")(e)
    z = layers.Dense(16, activation="relu")(e)         # bottleneck
    d = layers.Dense(32, activation="relu")(z)
    d = layers.Dense(64, activation="relu")(d)
    out = layers.Dense(n_features, activation="linear")(d)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    t0 = time.time()
    ae.fit(X_tr_benign, X_tr_benign, epochs=epochs, batch_size=256,
           validation_split=0.1, verbose=0)
    fit_t = time.time() - t0

    def recon_error(X):
        pred = ae.predict(X, verbose=0)
        return np.mean(np.square(X - pred), axis=1)

    # Threshold from benign TRAIN errors (95th percentile) — no test leakage.
    tr_err = recon_error(X_tr_benign)
    threshold = float(np.percentile(tr_err, 95))
    log(f"  fit: {fit_t:.1f}s | anomaly threshold (95th pct benign): {threshold:.5f}")

    t0 = time.time()
    te_err = recon_error(X_te)
    infer_t = time.time() - t0
    per_sample_ms = 1000 * infer_t / len(X_te)

    te_pred = (te_err > threshold).astype(int)
    zero_err = recon_error(X_zero)
    zero_pred = (zero_err > threshold).astype(int)

    m = metrics_block(y_te["binary_label"].values, te_pred, te_err)
    zdr = zero_day_recall(zero_pred)
    log(f"  inference: {per_sample_ms:.4f} ms/sample")
    log(f"  TEST  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
        f"F1 {m['f1']:.3f}  FPR {m['fpr']:.3f}  AUC {m.get('roc_auc')}")
    log(f"  ZERO-DAY detection rate (recall on unseen web attacks): {zdr:.3f}")

    # Reconstruction-error distribution figure (benign vs attack on test).
    try:
        plt.figure(figsize=(7, 4))
        tb = te_err[y_te["binary_label"].values == 0]
        ta = te_err[y_te["binary_label"].values == 1]
        plt.hist(tb, bins=60, alpha=0.6, label="Benign", density=True)
        plt.hist(ta, bins=60, alpha=0.6, label="Attack", density=True)
        plt.axvline(threshold, color="red", ls="--", label="Threshold")
        plt.xlabel("Reconstruction error"); plt.ylabel("Density")
        plt.title("Autoencoder reconstruction error — test set")
        plt.legend(); plt.tight_layout()
        plt.savefig(results_dir / "autoencoder_recon_error.png", dpi=140)
        plt.close()
        log(f"  figure saved: results/autoencoder_recon_error.png")
    except Exception as e:
        log(f"  (figure skipped: {e})")

    return {"test": m, "zero_day_recall": zdr, "threshold": threshold,
            "fit_time_s": fit_t, "infer_ms_per_sample": per_sample_ms}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--contamination", type=float, default=None,
                    help="Isolation Forest contamination; default = train attack rate.")
    args = ap.parse_args()

    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    proc = ROOT / cfg["paths"]["processed_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    log = Tee(results_dir / "baseline_report.txt")
    log("=" * 60)
    log("BASELINE MODELS — Isolation Forest & Dense Autoencoder")
    log("=" * 60)

    # Load processed data.
    X_tr = pd.read_csv(proc / "X_train.csv").values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    X_zero = pd.read_csv(proc / "X_zeroday.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")
    y_te = pd.read_csv(proc / "y_test.csv")
    log(f"Loaded: train {X_tr.shape}, test {X_te.shape}, zero-day {X_zero.shape}")

    attack_rate = float(y_tr["binary_label"].mean())
    contamination = args.contamination if args.contamination is not None else max(0.01, min(0.5, attack_rate))
    log(f"Train attack rate: {attack_rate:.3f} | IF contamination: {contamination:.3f}")

    results = {}
    results["isolation_forest"] = run_isolation_forest(
        X_tr, X_te, y_te["binary_label"].values, X_zero, contamination, seed, log)
    results["autoencoder"] = run_autoencoder(
        X_tr, y_tr, X_te, y_te, X_zero, args.epochs, seed, log, results_dir)

    # Comparison summary.
    log("\n" + "=" * 60)
    log("SUMMARY  (higher is better except FPR)")
    log("=" * 60)
    log(f"  {'model':20s} {'F1':>6s} {'recall':>7s} {'FPR':>6s} {'zero-day':>9s} {'ms/sample':>10s}")
    for name, r in results.items():
        log(f"  {name:20s} {r['test']['f1']:>6.3f} {r['test']['recall']:>7.3f} "
            f"{r['test']['fpr']:>6.3f} {r['zero_day_recall']:>9.3f} "
            f"{r['infer_ms_per_sample']:>10.4f}")

    with open(results_dir / "baseline_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nResults: results/baseline_results.json")
    log(f"Report:  results/baseline_report.txt")
    log("=" * 60)
    log.close()


if __name__ == "__main__":
    main()
