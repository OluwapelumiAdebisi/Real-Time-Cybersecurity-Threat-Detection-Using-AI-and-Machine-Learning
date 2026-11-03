"""
lstm_autoencoder.py  —  LSTM-Autoencoder for zero-day intrusion detection.

The project's core model. Like the dense autoencoder it is trained on BENIGN
traffic only and flags high-reconstruction-error inputs as anomalies — but it
uses LSTM layers over SEQUENCES of consecutive flows, so it can capture temporal
structure that a per-flow model cannot.

Pipeline:
  1. Load processed data (same inputs as the baselines).
  2. Build fixed-length sequences (windows) of consecutive flows.
     - Train sequences: benign-only windows (learn 'normal' behaviour).
     - Test / zero-day: windows labelled attack if ANY flow in the window is an
       attack (standard windowed-IDS labelling).
  3. Train an LSTM-Autoencoder to reconstruct benign sequences.
  4. Anomaly score = per-sequence mean reconstruction error.
  5. Threshold from the 90th percentile of benign-train errors (matches the
     baseline's tuned operating point for a fair comparison).
  6. Evaluate on test + zero-day with the same metrics as the baselines.

IMPORTANT LIMITATION (documented for the dissertation):
  Preprocessing shuffled the sampled rows, so these windows group flows that are
  adjacent in the *sample*, not strictly in capture-time order. This still gives
  the LSTM short local context and a valid first result, but a stronger version
  would sort by timestamp before windowing. Flagged here honestly.

Usage:
  python lstm_autoencoder.py --epochs 30 --seq-len 10
"""

import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

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


def make_sequences(X, y, seq_len):
    """Non-overlapping windows. Returns (seqs, seq_labels).
    seqs: (n_windows, seq_len, n_features); label = 1 if ANY flow in window is attack."""
    n = (len(X) // seq_len) * seq_len
    X = X[:n]; y = y[:n]
    seqs = X.reshape(-1, seq_len, X.shape[1])
    labs = y.reshape(-1, seq_len).max(axis=1)   # attack if any flow is attack
    return seqs.astype("float32"), labs.astype(int)


def fpr_of(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="Benign-error percentile for the anomaly threshold.")
    args = ap.parse_args()

    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    proc = ROOT / cfg["paths"]["processed_dir"]
    res = ROOT / cfg["paths"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    log = Tee(res / "lstm_report.txt")

    log("=" * 60); log("LSTM-AUTOENCODER — zero-day intrusion detection"); log("=" * 60)

    X_tr = pd.read_csv(proc / "X_train.csv").values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    X_z = pd.read_csv(proc / "X_zeroday.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")["binary_label"].values
    y_te = pd.read_csv(proc / "y_test.csv")["binary_label"].values
    y_z = pd.read_csv(proc / "y_zeroday.csv")["binary_label"].values
    nf = X_tr.shape[1]
    L = args.seq_len
    log(f"Loaded train {X_tr.shape}, test {X_te.shape}, zero-day {X_z.shape} | seq_len={L}")

    # ---- Sequences -----------------------------------------------------
    # Train on benign-only windows: filter to benign rows first, then window.
    Xtr_benign = X_tr[y_tr == 0]
    tr_seq, _ = make_sequences(Xtr_benign, np.zeros(len(Xtr_benign)), L)
    te_seq, te_lab = make_sequences(X_te, y_te, L)
    z_seq, z_lab = make_sequences(X_z, y_z, L)
    log(f"Sequences: train(benign) {tr_seq.shape}, test {te_seq.shape}, zero-day {z_seq.shape}")
    log(f"Test window attack rate: {te_lab.mean():.3f} | zero-day windows all-attack: {bool((z_lab==1).all())}")

    # ---- Model ---------------------------------------------------------
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(L, nf))
    # Encoder
    e = layers.LSTM(64, activation="tanh", return_sequences=True)(inp)
    e = layers.LSTM(32, activation="tanh", return_sequences=False)(e)   # bottleneck state
    # Bridge: repeat the encoded vector across the sequence length
    b = layers.RepeatVector(L)(e)
    # Decoder
    d = layers.LSTM(32, activation="tanh", return_sequences=True)(b)
    d = layers.LSTM(64, activation="tanh", return_sequences=True)(d)
    out = layers.TimeDistributed(layers.Dense(nf))(d)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    log(f"Model params: {ae.count_params():,}")

    t0 = time.time()
    ae.fit(tr_seq, tr_seq, epochs=args.epochs, batch_size=128,
           validation_split=0.1, verbose=0)
    fit_t = time.time() - t0
    log(f"Trained in {fit_t:.1f}s on {len(tr_seq):,} benign sequences")

    def seq_error(S):
        pred = ae.predict(S, verbose=0)
        # mean squared error per sequence (over time and features)
        return np.mean(np.square(S - pred), axis=(1, 2))

    tr_err = seq_error(tr_seq)
    threshold = float(np.percentile(tr_err, args.percentile))
    log(f"Anomaly threshold ({args.percentile:.0f}th pct benign): {threshold:.6f}")

    # ---- Evaluate ------------------------------------------------------
    t0 = time.time()
    te_err = seq_error(te_seq)
    infer_t = time.time() - t0
    ms_per_seq = 1000 * infer_t / len(te_seq)
    ms_per_flow = ms_per_seq / L

    te_pred = (te_err > threshold).astype(int)
    z_err = seq_error(z_seq)
    z_pred = (z_err > threshold).astype(int)

    pr = precision_score(te_lab, te_pred, zero_division=0)
    rc = recall_score(te_lab, te_pred, zero_division=0)
    f1 = f1_score(te_lab, te_pred, zero_division=0)
    fp = fpr_of(te_lab, te_pred)
    try: auc = float(roc_auc_score(te_lab, te_err))
    except Exception: auc = None
    zdr = float(np.mean(z_pred == 1))

    log("\n" + "-" * 60)
    log(f"  TEST   precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}  FPR {fp:.3f}  AUC {auc}")
    log(f"  ZERO-DAY detection rate (unseen web attacks): {zdr:.3f}")
    log(f"  inference: {ms_per_seq:.4f} ms/sequence  (~{ms_per_flow:.4f} ms/flow)")

    # ---- Figure --------------------------------------------------------
    try:
        plt.figure(figsize=(7, 4))
        plt.hist(te_err[te_lab == 0], bins=60, alpha=.6, density=True, label="Benign window")
        plt.hist(te_err[te_lab == 1], bins=60, alpha=.6, density=True, label="Attack window")
        plt.axvline(threshold, color="red", ls="--", label="Threshold")
        plt.xlabel("Reconstruction error"); plt.ylabel("Density")
        plt.title("LSTM-AE reconstruction error — test windows")
        plt.legend(); plt.tight_layout()
        plt.savefig(res / "lstm_recon_error.png", dpi=140); plt.close()
        log("  figure: results/lstm_recon_error.png")
    except Exception as e:
        log(f"  (figure skipped: {e})")

    # ---- Save + compare to baseline -----------------------------------
    out_json = {
        "seq_len": L, "epochs": args.epochs, "threshold": threshold,
        "test": {"precision": pr, "recall": rc, "f1": f1, "fpr": fp, "roc_auc": auc},
        "zero_day_recall": zdr, "fit_time_s": fit_t,
        "ms_per_sequence": ms_per_seq, "ms_per_flow": ms_per_flow,
    }
    json.dump(out_json, open(res / "lstm_results.json", "w"), indent=2)

    log("\n" + "=" * 60); log("COMPARISON TO TUNED BASELINE (H1)"); log("=" * 60)
    log(f"  {'model':28s} {'F1':>6s} {'zero-day':>9s} {'FPR':>6s}")
    log(f"  {'Dense AE (baseline best)':28s} {0.568:>6.3f} {0.480:>9.3f} {0.100:>6.3f}")
    log(f"  {'LSTM-AE (this run)':28s} {f1:>6.3f} {zdr:>9.3f} {fp:>6.3f}")
    beat_f1 = "YES" if f1 > 0.568 else "no"
    beat_zd = "YES" if zdr > 0.480 else "no"
    log(f"  Beats baseline F1? {beat_f1}   Beats baseline zero-day? {beat_zd}")
    log(f"\n  Saved: results/lstm_results.json, results/lstm_report.txt")
    log("=" * 60)
    log.close()


if __name__ == "__main__":
    main()
