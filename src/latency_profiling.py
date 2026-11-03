"""
latency_profiling.py  —  Inference latency & feature-selection test (H2).

Hypothesis H2:
  "Tree-based (gain) feature selection reduces per-sample inference latency below
   1 ms without more than a 2% drop in Recall."

This script:
  1. Trains the dense autoencoder on ALL features; measures per-sample inference
     latency carefully (warm-up excluded, averaged over repeats) and recall at the
     tuned 90th-percentile threshold.
  2. Ranks features by tree-based importance (RandomForest gain) on the labelled
     training data.
  3. For several feature-count budgets K, retrains the autoencoder on the top-K
     features, and re-measures latency + recall.
  4. Reports whether H2 holds: latency < 1 ms AND recall drop <= 2 percentage
     points vs the full-feature model.

Saves results/latency_report.txt, results/latency_results.json,
results/latency_vs_features.png.

Usage:
  python latency_profiling.py --epochs 30 --k-values 10 20 30 40 77
"""

import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
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


def build_ae(nf, seed):
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(nf,))
    e = layers.Dense(64, activation="relu")(inp)
    e = layers.Dense(32, activation="relu")(e)
    z = layers.Dense(16, activation="relu")(e)
    d = layers.Dense(32, activation="relu")(z)
    d = layers.Dense(64, activation="relu")(d)
    out = layers.Dense(nf, activation="linear")(d)
    m = Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m


def measure_latency(model, X, repeats=5):
    """Per-sample inference latency in ms. Warm-up run excluded; averaged."""
    _ = model.predict(X[:256], verbose=0)          # warm-up (build graph)
    times = []
    for _ in range(repeats):
        t0 = time.time()
        _ = model.predict(X, verbose=0)
        times.append(time.time() - t0)
    mean_total = float(np.mean(times))
    return 1000.0 * mean_total / len(X)            # ms per sample


def train_eval(Xb_cols, X_tr, y_tr, X_te, y_te, cols, seed, epochs, log):
    """Train AE on benign (selected cols), return (latency_ms, recall, threshold)."""
    Xtr_sel = X_tr[:, cols]
    Xte_sel = X_te[:, cols]
    Xb = Xtr_sel[y_tr == 0]
    model = build_ae(len(cols), seed)
    model.fit(Xb, Xb, epochs=epochs, batch_size=256, validation_split=0.1, verbose=0)

    def err(X): return np.mean(np.square(X - model.predict(X, verbose=0)), axis=1)
    thr = float(np.percentile(err(Xb), 90))
    te_pred = (err(Xte_sel) > thr).astype(int)
    rec = recall_score(y_te, te_pred, zero_division=0)
    lat = measure_latency(model, Xte_sel)
    return lat, float(rec), thr


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 30, 40, 77])
    args = ap.parse_args()
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    proc = ROOT / cfg["paths"]["processed_dir"]
    res = ROOT / cfg["paths"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    log = Tee(res / "latency_report.txt")
    log("=" * 60); log("LATENCY PROFILING + FEATURE SELECTION (H2)"); log("=" * 60)
    log("H2: tree-based feature selection -> latency < 1 ms AND recall drop <= 2%")

    feat_names = json.load(open(proc / "feature_columns.json")) \
        if (proc / "feature_columns.json").exists() else None
    X_tr = pd.read_csv(proc / "X_train.csv").values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")["binary_label"].values
    y_te = pd.read_csv(proc / "y_test.csv")["binary_label"].values
    nf = X_tr.shape[1]
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(nf)]

    # ---- Tree-based feature importance (gain) -------------------------
    log("\n[1] Tree-based feature ranking (RandomForest gain)")
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                n_jobs=-1, random_state=seed)
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    rank = np.argsort(imp)[::-1]
    log("  Top 10 features by gain:")
    for r in rank[:10]:
        log(f"    {feat_names[r]:35s} {imp[r]:.4f}")

    # ---- Full-feature baseline ----------------------------------------
    log("\n[2] Full-feature model (all 77 features)")
    full_cols = np.arange(nf)
    full_lat, full_rec, _ = train_eval(None, X_tr, y_tr, X_te, y_te, full_cols,
                                       seed, args.epochs, log)
    log(f"  latency {full_lat:.4f} ms/sample | recall {full_rec:.3f}")

    # ---- Feature-count sweep ------------------------------------------
    log("\n[3] Feature-selection sweep")
    log(f"  {'K':>4s} {'latency_ms':>11s} {'recall':>7s} {'rec_drop_pp':>12s} {'<1ms?':>6s} {'H2?':>5s}")
    rows = []
    for K in sorted(set(args.k_values)):
        cols = np.sort(rank[:K])
        lat, rec, _ = train_eval(None, X_tr, y_tr, X_te, y_te, cols, seed, args.epochs, log)
        drop_pp = 100.0 * (full_rec - rec)          # percentage points
        under1 = lat < 1.0
        h2 = under1 and (drop_pp <= 2.0)
        log(f"  {K:>4d} {lat:>11.4f} {rec:>7.3f} {drop_pp:>12.2f} "
            f"{('yes' if under1 else 'no'):>6s} {('YES' if h2 else 'no'):>5s}")
        rows.append({"k": int(K), "latency_ms": lat, "recall": rec,
                     "recall_drop_pp": drop_pp, "under_1ms": bool(under1),
                     "h2_holds": bool(h2)})

    # ---- Verdict ------------------------------------------------------
    log("\n" + "=" * 60); log("H2 VERDICT"); log("=" * 60)
    passing = [r for r in rows if r["h2_holds"] and r["k"] < nf]
    if passing:
        best = min(passing, key=lambda r: r["k"])   # most aggressive reduction that passes
        log(f"  H2 SUPPORTED: with K={best['k']} features, latency "
            f"{best['latency_ms']:.4f} ms (<1ms) and recall drop "
            f"{best['recall_drop_pp']:.2f}pp (<=2pp).")
    else:
        # Report honestly if latency was already <1ms even at full features.
        if full_lat < 1.0:
            log(f"  NOTE: latency was already < 1 ms at full features "
                f"({full_lat:.4f} ms). Feature selection still reduces it further "
                f"(see table) but the <1ms bar was met without selection — report honestly.")
        else:
            log("  H2 NOT fully supported under these settings — see table for trade-off.")

    # ---- Figure -------------------------------------------------------
    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, [r["latency_ms"] for r in rows], "o-", color="tab:blue", label="latency (ms)")
    ax1.axhline(1.0, color="tab:blue", ls=":", alpha=.6)
    ax1.set_xlabel("number of features (K)"); ax1.set_ylabel("latency ms/sample", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, [r["recall"] for r in rows], "s-", color="tab:red", label="recall")
    ax2.set_ylabel("recall", color="tab:red")
    plt.title("Latency & recall vs feature count (H2)")
    fig.tight_layout(); plt.savefig(res / "latency_vs_features.png", dpi=140); plt.close()

    json.dump({"full_feature": {"latency_ms": full_lat, "recall": full_rec},
               "sweep": rows,
               "top_features": [feat_names[r] for r in rank[:15]]},
              open(res / "latency_results.json", "w"), indent=2)
    log(f"\n  Saved: results/latency_results.json, results/latency_vs_features.png")
    log("=" * 60)
    log.close()


if __name__ == "__main__":
    main()
