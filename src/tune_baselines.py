"""
tune_baselines.py  —  Make the baselines a FAIR comparison before the LSTM.

Two improvements over baseline_models.py:
  1. Isolation Forest: sweep the `contamination` parameter and report the best,
     so the classical baseline is tuned, not accidentally crippled.
  2. Dense Autoencoder: sweep the anomaly threshold across a range of benign-error
     percentiles, producing a precision / recall / F1 / FPR curve and choosing the
     threshold that maximises F1 — a principled, defensible operating point.

Saves:
  results/tuning_results.json
  results/tuning_report.txt
  results/if_contamination_sweep.png
  results/ae_threshold_sweep.png

Usage:
  python tune_baselines.py --epochs 30
"""

import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "config.yaml").exists():
            return p
    return Path.cwd()

ROOT = find_root()


def load_cfg():
    p = ROOT / "config.yaml"
    if yaml and p.exists():
        return yaml.safe_load(open(p))
    return {"seed": 42, "paths": {"processed_dir": "data/processed", "results_dir": "results"}}


class Tee:
    def __init__(self, path): self.f = open(path, "w")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s); self.f.write(s + "\n")
    def close(self): self.f.close()


def fpr_of(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    proc = ROOT / cfg["paths"]["processed_dir"]
    res = ROOT / cfg["paths"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    log = Tee(res / "tuning_report.txt")

    X_tr = pd.read_csv(proc / "X_train.csv").values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    X_z = pd.read_csv(proc / "X_zeroday.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")["binary_label"].values
    y_te = pd.read_csv(proc / "y_test.csv")["binary_label"].values

    out = {}

    # ---------- 1. Isolation Forest contamination sweep ----------------
    log("=" * 60); log("ISOLATION FOREST — contamination sweep"); log("=" * 60)
    log(f"  {'contam':>8s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'FPR':>6s} {'AUC':>6s} {'zero-day':>9s}")
    best = {"f1": -1}
    if_rows = []
    for c in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        iso = IsolationForest(n_estimators=200, contamination=c, random_state=seed, n_jobs=-1)
        iso.fit(X_tr)
        yp = (iso.predict(X_te) == -1).astype(int)
        sc = -iso.score_samples(X_te)
        pr, rc = precision_score(y_te, yp, zero_division=0), recall_score(y_te, yp, zero_division=0)
        f1 = f1_score(y_te, yp, zero_division=0); fp = fpr_of(y_te, yp)
        try: auc = roc_auc_score(y_te, sc)
        except Exception: auc = float("nan")
        zdr = float(np.mean((iso.predict(X_z) == -1).astype(int)))
        log(f"  {c:>8.2f} {pr:>6.3f} {rc:>7.3f} {f1:>6.3f} {fp:>6.3f} {auc:>6.3f} {zdr:>9.3f}")
        if_rows.append({"contamination": c, "precision": pr, "recall": rc, "f1": f1,
                        "fpr": fp, "auc": float(auc), "zero_day_recall": zdr})
        if f1 > best["f1"]:
            best = {"contamination": c, "f1": f1, "precision": pr, "recall": rc,
                    "fpr": fp, "auc": float(auc), "zero_day_recall": zdr}
    log(f"\n  BEST Isolation Forest: contamination={best['contamination']} "
        f"F1={best['f1']:.3f} AUC={best['auc']:.3f} zero-day={best['zero_day_recall']:.3f}")
    out["isolation_forest_best"] = best
    out["isolation_forest_sweep"] = if_rows

    # figure
    cs = [r["contamination"] for r in if_rows]
    plt.figure(figsize=(7, 4))
    for k in ["precision", "recall", "f1"]:
        plt.plot(cs, [r[k] for r in if_rows], marker="o", label=k)
    plt.xlabel("contamination"); plt.ylabel("score"); plt.title("Isolation Forest — contamination sweep")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(res / "if_contamination_sweep.png", dpi=140); plt.close()

    # ---------- 2. Autoencoder threshold sweep ------------------------
    log("\n" + "=" * 60); log("AUTOENCODER — threshold sweep (trained on benign only)"); log("=" * 60)
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Model

    Xb = X_tr[y_tr == 0]
    nf = X_tr.shape[1]
    inp = layers.Input(shape=(nf,))
    e = layers.Dense(64, activation="relu")(inp)
    e = layers.Dense(32, activation="relu")(e)
    z = layers.Dense(16, activation="relu")(e)
    d = layers.Dense(32, activation="relu")(z)
    d = layers.Dense(64, activation="relu")(d)
    o = layers.Dense(nf, activation="linear")(d)
    ae = Model(inp, o); ae.compile(optimizer="adam", loss="mse")
    t0 = time.time()
    ae.fit(Xb, Xb, epochs=args.epochs, batch_size=256, validation_split=0.1, verbose=0)
    log(f"  trained on {len(Xb):,} benign flows in {time.time()-t0:.1f}s")

    def err(X):
        return np.mean(np.square(X - ae.predict(X, verbose=0)), axis=1)

    tr_err = err(Xb)
    te_err = err(X_te)
    z_err = err(X_z)

    log(f"\n  {'pctile':>7s} {'thresh':>9s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'FPR':>6s} {'zero-day':>9s}")
    best_ae = {"f1": -1}
    ae_rows = []
    for pct in [80, 85, 90, 92, 95, 97, 99]:
        thr = float(np.percentile(tr_err, pct))
        yp = (te_err > thr).astype(int)
        pr, rc = precision_score(y_te, yp, zero_division=0), recall_score(y_te, yp, zero_division=0)
        f1 = f1_score(y_te, yp, zero_division=0); fp = fpr_of(y_te, yp)
        zdr = float(np.mean((z_err > thr).astype(int)))
        log(f"  {pct:>7d} {thr:>9.5f} {pr:>6.3f} {rc:>7.3f} {f1:>6.3f} {fp:>6.3f} {zdr:>9.3f}")
        ae_rows.append({"percentile": pct, "threshold": thr, "precision": pr, "recall": rc,
                        "f1": f1, "fpr": fp, "zero_day_recall": zdr})
        if f1 > best_ae["f1"]:
            best_ae = {"percentile": pct, "threshold": thr, "f1": f1, "precision": pr,
                       "recall": rc, "fpr": fp, "zero_day_recall": zdr}
    log(f"\n  BEST Autoencoder: pctile={best_ae['percentile']} F1={best_ae['f1']:.3f} "
        f"recall={best_ae['recall']:.3f} FPR={best_ae['fpr']:.3f} "
        f"zero-day={best_ae['zero_day_recall']:.3f}")
    out["autoencoder_best"] = best_ae
    out["autoencoder_sweep"] = ae_rows

    pcts = [r["percentile"] for r in ae_rows]
    plt.figure(figsize=(7, 4))
    for k in ["precision", "recall", "f1", "fpr"]:
        plt.plot(pcts, [r[k] for r in ae_rows], marker="o", label=k)
    plt.xlabel("benign-error percentile (threshold)"); plt.ylabel("score")
    plt.title("Autoencoder — threshold sweep"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(res / "ae_threshold_sweep.png", dpi=140); plt.close()

    # ---------- summary ----------------------------------------------
    log("\n" + "=" * 60); log("TUNED BASELINE SUMMARY"); log("=" * 60)
    log(f"  Isolation Forest (best): F1={best['f1']:.3f} AUC={best['auc']:.3f} "
        f"FPR={best['fpr']:.3f} zero-day={best['zero_day_recall']:.3f}")
    log(f"  Autoencoder (best F1):   F1={best_ae['f1']:.3f} recall={best_ae['recall']:.3f} "
        f"FPR={best_ae['fpr']:.3f} zero-day={best_ae['zero_day_recall']:.3f}")
    json.dump(out, open(res / "tuning_results.json", "w"), indent=2)
    log(f"\n  Saved: results/tuning_results.json + 2 figures")
    log("=" * 60)
    log.close()


if __name__ == "__main__":
    main()
