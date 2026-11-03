# Results Log

Model results for the dissertation, newest findings appended. Keep this separate
from `DATA_FINDINGS.md` (which covers the data). Figures referenced here live in
`results/`.

Evaluation sets (from preprocessing):
- **Test set:** 34,742 flows (seen attack families + benign), ~14.5% attack.
- **Zero-day set:** 867 flows — unseen **web attacks** (SQL Injection, XSS,
  Web brute force) held out entirely from training. Every row is an attack, so
  "zero-day recall" = the fraction of unseen attacks the model detects.

Positive class = attack. FPR = false-positive rate (lower is better); all other
metrics higher is better.

---

## 1. Baseline models — INITIAL (untuned) — 3 Nov 2026

First run of the two baselines (`src/baseline_models.py`), default settings.
**Preliminary — superseded by the tuned results below once run.**

| Model | F1 | Recall | FPR | ROC-AUC | Zero-day recall | ms/sample |
|-------|-----|--------|-----|---------|-----------------|-----------|
| Isolation Forest | 0.050 | 0.050 | 0.160 | 0.394 | 0.483 | 0.009 |
| Dense Autoencoder (benign-only) | 0.342 | 0.267 | 0.050 | 0.826 | 0.469 | 0.074 |

**Interpretation (initial):**
- The **autoencoder clearly outperforms Isolation Forest** (AUC 0.826 vs 0.394).
  An AUC below 0.5 means the untuned Isolation Forest is not separating attacks
  from benign at all on this scaled, high-dimensional data — a known weakness of
  the method here, and a tuning target (contamination) before any fair comparison.
- The autoencoder already achieves a **low false-positive rate (0.050)**, the
  metric singled out in the proposal for analyst alert-fatigue.
- Both detect ~47–48% of **unseen zero-day** web attacks — the benchmark the
  LSTM-Autoencoder must beat for hypothesis **H1**.
- Inference is fast for both (<0.1 ms/sample), relevant to hypothesis **H2**.

**Caveat:** these are untuned. The Isolation Forest figure in particular is not a
fair baseline until its contamination parameter is tuned (see §2). Do not quote
the untuned Isolation Forest numbers as final.

---

## 2. Baseline models — TUNED — (to be filled after running tune_baselines.py)

`src/tune_baselines.py` sweeps Isolation Forest `contamination` and the
autoencoder anomaly threshold (benign-error percentile), choosing the best-F1
operating point for each. Produces `if_contamination_sweep.png` and
`ae_threshold_sweep.png`.

*(Paste the tuned summary table here after the run.)*

---

*Baselines established the reference point for H1 (zero-day F1) and H2
(latency). Next model: LSTM-Autoencoder.*
