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

**Tuned results (3 Nov 2026):**

*Isolation Forest — contamination sweep (AUC is flat at 0.394 because it is
threshold-independent):*

| contamination | precision | recall | F1 | FPR | zero-day |
|---|---|---|---|---|---|
| 0.02 | 0.053 | 0.007 | 0.013 | 0.022 | 0.426 |
| 0.05 | 0.028 | 0.010 | 0.014 | 0.058 | 0.433 |
| 0.10 | 0.022 | 0.015 | 0.018 | 0.113 | 0.461 |
| 0.15 | 0.064 | 0.065 | 0.065 | 0.162 | 0.484 |
| 0.20 | 0.066 | 0.090 | 0.076 | 0.216 | 0.499 |
| 0.30 | 0.086 | 0.178 | **0.116** | 0.321 | 0.672 |

*Autoencoder — threshold sweep (benign-error percentile):*

| percentile | threshold | precision | recall | F1 | FPR | zero-day |
|---|---|---|---|---|---|---|
| 80 | 0.00688 | 0.378 | 0.712 | 0.494 | 0.199 | 0.649 |
| 85 | 0.01068 | 0.419 | 0.640 | 0.506 | 0.151 | 0.645 |
| **90** | 0.01736 | 0.517 | 0.631 | **0.568** | 0.100 | 0.480 |
| 92 | 0.02190 | 0.475 | 0.424 | 0.448 | 0.080 | 0.476 |
| 95 | 0.03623 | 0.492 | 0.279 | 0.356 | 0.049 | 0.471 |
| 97 | 0.05821 | 0.545 | 0.212 | 0.305 | 0.030 | 0.468 |
| 99 | 0.12596 | 0.592 | 0.080 | 0.141 | 0.009 | 0.294 |

**Final tuned baseline benchmark (the line the LSTM must beat):**

| Model | F1 | Recall | FPR | AUC | Zero-day recall |
|-------|-----|--------|-----|-----|-----------------|
| Isolation Forest (best, c=0.30) | 0.116 | 0.178 | 0.321 | 0.394 | 0.672 |
| **Dense Autoencoder (best F1, 90th pct)** | **0.568** | 0.631 | 0.100 | 0.826* | 0.480 |

\*AE AUC from the initial run (threshold-independent).

**Key findings for the write-up:**
1. **Deep reconstruction >> classical anomaly detection here.** Even after a fair
   6-point contamination sweep, Isolation Forest tops out at F1 0.116 / AUC 0.394.
   The autoencoder reaches F1 0.568. This supports the project's premise and is a
   citable result: classical methods struggle on high-dimensional scaled flow data.
2. **The FPR/recall trade-off is explicit.** The AE threshold sweep shows the
   alert-fatigue tension directly — 71% recall at 20% FPR (80th pct) vs 8% recall
   at ~1% FPR (99th pct). The 90th percentile is the F1-optimal compromise.
3. **Seen-attack vs zero-day tension.** The F1-optimal threshold (90th pct, zero-day
   recall 0.480) is NOT the zero-day-optimal one (80th pct, zero-day recall 0.649).
   Detecting unseen attacks favours a lower threshold than detecting seen ones — a
   discussion point, and a gap the temporal LSTM-Autoencoder aims to close.

**LSTM must beat (H1):** test F1 **0.568** and/or zero-day recall **0.480** at a
comparable or better FPR.

---

*Baselines established the reference point for H1 (zero-day F1) and H2
(latency). Next model: LSTM-Autoencoder.*
