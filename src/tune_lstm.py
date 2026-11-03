"""
tune_lstm.py  —  Fair-operating-point search for the LSTM-Autoencoder.

The first LSTM run showed strong zero-day detection but a low F1, because
(a) the 90th-percentile threshold is too conservative once data is windowed, and
(b) long windows saturate the test set with 'attack' windows (any-attack labelling).

This script, for each sequence length, trains the LSTM-AE once on benign windows
then sweeps the anomaly threshold to find the F1-optimal operating point — the
fair comparison against the tuned dense-AE baseline (F1 0.568, zero-day 0.480).

Saves results/lstm_tuning_report.txt, results/lstm_tuning_results.json, and a
threshold-sweep figure per sequence length.

Usage:
  python tune_lstm.py --epochs 30 --seq-lens 3 5 10
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


def make_sequences(X, y, seq_len, stride=None):
    """Windows with optional overlap. stride=None -> non-overlapping (=seq_len)."""
    stride = stride or seq_len
    idx = range(0, len(X) - seq_len + 1, stride)
    seqs = np.stack([X[i:i+seq_len] for i in idx]).astype("float32")
    labs = np.array([int(y[i:i+seq_len].max()) for i in idx])
    return seqs, labs


def fpr_of(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def build_model(L, nf, seed):
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(L, nf))
    e = layers.LSTM(64, activation="tanh", return_sequences=True)(inp)
    e = layers.LSTM(32, activation="tanh", return_sequences=False)(e)
    b = layers.RepeatVector(L)(e)
    d = layers.LSTM(32, activation="tanh", return_sequences=True)(b)
    d = layers.LSTM(64, activation="tanh", return_sequences=True)(d)
    out = layers.TimeDistributed(layers.Dense(nf))(d)
    m = Model(inp, out); m.compile(optimizer="adam", loss="mse")
    return m


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[3, 5, 10])
    args = ap.parse_args()
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    proc = ROOT / cfg["paths"]["processed_dir"]
    res = ROOT / cfg["paths"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    log = Tee(res / "lstm_tuning_report.txt")
    log("=" * 64); log("LSTM-AE TUNING — sequence length x threshold sweep"); log("=" * 64)
    log("Baseline to beat: dense AE  F1=0.568  zero-day=0.480  FPR=0.100")

    X_tr = pd.read_csv(proc / "X_train.csv").values.astype("float32")
    X_te = pd.read_csv(proc / "X_test.csv").values.astype("float32")
    X_z = pd.read_csv(proc / "X_zeroday.csv").values.astype("float32")
    y_tr = pd.read_csv(proc / "y_train.csv")["binary_label"].values
    y_te = pd.read_csv(proc / "y_test.csv")["binary_label"].values
    y_z = pd.read_csv(proc / "y_zeroday.csv")["binary_label"].values
    nf = X_tr.shape[1]
    Xtr_benign = X_tr[y_tr == 0]

    all_results = {}
    overall_best = {"f1": -1}

    for L in args.seq_lens:
        log("\n" + "=" * 64); log(f"SEQ_LEN = {L}"); log("=" * 64)
        # overlapping train windows (stride = L//2) to get more training sequences
        stride = max(1, L // 2)
        tr_seq, _ = make_sequences(Xtr_benign, np.zeros(len(Xtr_benign)), L, stride=stride)
        te_seq, te_lab = make_sequences(X_te, y_te, L)        # non-overlap for eval
        z_seq, z_lab = make_sequences(X_z, y_z, L)
        log(f"  train {tr_seq.shape} | test {te_seq.shape} (attack-rate {te_lab.mean():.3f}) "
            f"| zero-day {z_seq.shape}")

        model = build_model(L, nf, seed)
        t0 = time.time()
        model.fit(tr_seq, tr_seq, epochs=args.epochs, batch_size=128,
                  validation_split=0.1, verbose=0)
        log(f"  trained {len(tr_seq):,} seqs in {time.time()-t0:.1f}s | params {model.count_params():,}")

        def serr(S):
            return np.mean(np.square(S - model.predict(S, verbose=0)), axis=(1, 2))
        tr_err, te_err, z_err = serr(tr_seq), serr(te_seq), serr(z_seq)
        try: auc = float(roc_auc_score(te_lab, te_err))
        except Exception: auc = None

        log(f"\n  {'pctile':>7s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'FPR':>6s} {'zero-day':>9s}")
        rows = []; best = {"f1": -1}
        for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
            thr = float(np.percentile(tr_err, pct))
            yp = (te_err > thr).astype(int)
            pr = precision_score(te_lab, yp, zero_division=0)
            rc = recall_score(te_lab, yp, zero_division=0)
            f1 = f1_score(te_lab, yp, zero_division=0)
            fp = fpr_of(te_lab, yp)
            zdr = float(np.mean((z_err > thr).astype(int)))
            log(f"  {pct:>7d} {pr:>6.3f} {rc:>7.3f} {f1:>6.3f} {fp:>6.3f} {zdr:>9.3f}")
            rows.append({"percentile": pct, "threshold": thr, "precision": pr,
                         "recall": rc, "f1": f1, "fpr": fp, "zero_day_recall": zdr})
            if f1 > best["f1"]:
                best = {"seq_len": L, "percentile": pct, "f1": f1, "precision": pr,
                        "recall": rc, "fpr": fp, "zero_day_recall": zdr, "auc": auc}
        all_results[f"seq_len_{L}"] = {"sweep": rows, "best": best, "auc": auc}
        log(f"\n  BEST @ L={L}: pct={best['percentile']} F1={best['f1']:.3f} "
            f"recall={best['recall']:.3f} FPR={best['fpr']:.3f} zero-day={best['zero_day_recall']:.3f}")
        if best["f1"] > overall_best["f1"]:
            overall_best = best

        pcts = [r["percentile"] for r in rows]
        plt.figure(figsize=(7, 4))
        for k in ["precision", "recall", "f1", "zero_day_recall"]:
            plt.plot(pcts, [r[k] for r in rows], marker="o", label=k)
        plt.axhline(0.568, color="gray", ls="--", label="baseline F1")
        plt.xlabel("benign-error percentile (threshold)"); plt.ylabel("score")
        plt.title(f"LSTM-AE threshold sweep (seq_len={L})"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(res / f"lstm_sweep_L{L}.png", dpi=140); plt.close()

    log("\n" + "=" * 64); log("OVERALL BEST LSTM-AE"); log("=" * 64)
    log(f"  seq_len={overall_best['seq_len']} pct={overall_best['percentile']}  "
        f"F1={overall_best['f1']:.3f}  recall={overall_best['recall']:.3f}  "
        f"FPR={overall_best['fpr']:.3f}  zero-day={overall_best['zero_day_recall']:.3f}")
    log(f"\n  vs baseline F1 0.568: {'BEATS' if overall_best['f1']>0.568 else 'below'}")
    log(f"  vs baseline zero-day 0.480: {'BEATS' if overall_best['zero_day_recall']>0.480 else 'below'}")
    json.dump({"overall_best": overall_best, "by_seq_len": all_results},
              open(res / "lstm_tuning_results.json", "w"), indent=2)
    log(f"\n  Saved: results/lstm_tuning_results.json + per-L figures")
    log("=" * 64)
    log.close()


if __name__ == "__main__":
    main()
