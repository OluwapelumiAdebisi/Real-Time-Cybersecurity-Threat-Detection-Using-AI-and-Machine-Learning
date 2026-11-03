"""
explainability.py  —  Explain WHY a flow was flagged (threat attribution).

Two complementary techniques (both named in the supervisor's requirements):

  1. Reconstruction-error feature ranking (autoencoder-native, exact & fast):
     for a flagged flow, rank features by how badly the autoencoder reconstructed
     them. The worst-reconstructed features are the ones driving the anomaly —
     e.g. an abnormal packet length or an unusual port-related feature.

  2. SHAP (KernelExplainer) on the anomaly score: attributes the per-flow anomaly
     score across input features, giving a principled, literature-standard
     explanation comparable across models.

Produces, for a sample of flagged attack flows:
  * results/explain_feature_ranking.csv   (top features per flagged flow)
  * results/explain_global_importance.png (which features most drive anomalies)
  * results/shap_summary.png              (SHAP summary, if shap installed)
  * results/explainability_report.txt

Uses the DENSE autoencoder (per-flow, so each explanation maps to one flow —
cleaner for analyst attribution than a windowed LSTM explanation).

Usage:
  python explainability.py --epochs 30 --n-explain 200
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

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


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-explain", type=int, default=200,
                    help="How many flagged flows to explain in detail.")
    args = ap.parse_args()
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    proc = ROOT / cfg["paths"]["processed_dir"]
    res = ROOT / cfg["paths"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    log = Tee(res / "explainability_report.txt")
    log("=" * 60); log("EXPLAINABILITY — threat attribution for flagged flows"); log("=" * 60)

    # feature names
    fc_path = proc / "feature_columns.json"
    feat_names = json.load(open(fc_path)) if fc_path.exists() else None

    X_tr = pd.read_csv(proc / "X_train.csv")
    if feat_names is None:
        feat_names = list(X_tr.columns)
    X_tr = X_tr.values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")["binary_label"].values
    y_te = pd.read_csv(proc / "y_test.csv")["binary_label"].values

    # ---- Train the per-flow dense autoencoder on benign ---------------
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
    out = layers.Dense(nf, activation="linear")(d)
    ae = Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xb, Xb, epochs=args.epochs, batch_size=256, validation_split=0.1, verbose=0)
    log(f"Autoencoder trained on {len(Xb):,} benign flows | {nf} features")

    def recon_err_vec(X):
        """Per-feature squared error (n, nf)."""
        return np.square(X - ae.predict(X, verbose=0))

    thr = float(np.percentile(recon_err_vec(Xb).mean(axis=1), 90))

    # Flagged ATTACK flows in the test set (true positives we want to explain).
    te_err = recon_err_vec(X_te)
    te_score = te_err.mean(axis=1)
    flagged = (te_score > thr) & (y_te == 1)
    idx = np.where(flagged)[0]
    log(f"Flagged attack flows available to explain: {len(idx):,}")
    if len(idx) == 0:
        log("No flagged attacks to explain (threshold too high)."); log.close(); return
    sample_idx = idx[:args.n_explain]

    # ---- 1. Reconstruction-error feature ranking ----------------------
    log("\n" + "-" * 60)
    log("[1] RECONSTRUCTION-ERROR FEATURE RANKING")
    log("-" * 60)
    per_feat = te_err[sample_idx]                 # (n, nf)
    # global: mean per-feature error over flagged flows
    global_imp = per_feat.mean(axis=0)
    order = np.argsort(global_imp)[::-1]
    log("Top 10 features driving anomalies (across flagged flows):")
    for r in order[:10]:
        log(f"    {feat_names[r]:35s} mean recon-error {global_imp[r]:.4f}")

    # per-flow top-3 features -> CSV for analyst use
    rows = []
    for i in sample_idx:
        o = np.argsort(te_err[i])[::-1][:3]
        rows.append({
            "flow_index": int(i),
            "anomaly_score": float(te_score[i]),
            "top_feature_1": feat_names[o[0]],
            "top_feature_2": feat_names[o[1]],
            "top_feature_3": feat_names[o[2]],
        })
    pd.DataFrame(rows).to_csv(res / "explain_feature_ranking.csv", index=False)
    log(f"Per-flow top-3 attribution saved: results/explain_feature_ranking.csv")

    # global importance figure
    plt.figure(figsize=(7, 5))
    top = order[:15][::-1]
    plt.barh([feat_names[r] for r in top], [global_imp[r] for r in top])
    plt.xlabel("Mean reconstruction error (flagged flows)")
    plt.title("Which features drive anomaly detection")
    plt.tight_layout(); plt.savefig(res / "explain_global_importance.png", dpi=140); plt.close()
    log("Global importance figure: results/explain_global_importance.png")

    # ---- 2. SHAP (optional) -------------------------------------------
    log("\n" + "-" * 60)
    log("[2] SHAP ATTRIBUTION")
    log("-" * 60)
    try:
        import shap
        # anomaly score function for SHAP: mean reconstruction error per row
        def score_fn(X):
            X = np.asarray(X, dtype="float32")
            return np.mean(np.square(X - ae.predict(X, verbose=0)), axis=1)
        bg = Xb[np.random.choice(len(Xb), 100, replace=False)]   # background
        explainer = shap.KernelExplainer(score_fn, bg)
        shap_sample = X_te[sample_idx[:50]]                       # keep it tractable
        sv = explainer.shap_values(shap_sample, nsamples=100)
        shap.summary_plot(sv, shap_sample, feature_names=feat_names, show=False)
        plt.tight_layout(); plt.savefig(res / "shap_summary.png", dpi=140, bbox_inches="tight")
        plt.close()
        log("SHAP summary figure: results/shap_summary.png")
        mean_abs = np.abs(sv).mean(axis=0)
        sorder = np.argsort(mean_abs)[::-1][:10]
        log("Top 10 features by mean |SHAP value|:")
        for r in sorder:
            log(f"    {feat_names[r]:35s} {mean_abs[r]:.4f}")
    except ImportError:
        log("shap not installed — run `pip install shap`. Feature ranking (above) still provided.")
    except Exception as ex:
        log(f"SHAP step skipped ({ex}). Feature ranking (above) still provided.")

    log("\n" + "=" * 60)
    log("Explainability complete. An analyst can read, per flagged flow, the")
    log("features most responsible — turning an opaque score into an attack")
    log("attribution (e.g. abnormal packet length / port behaviour).")
    log("=" * 60)
    log.close()


if __name__ == "__main__":
    main()
