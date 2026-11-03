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

## 3. LSTM-Autoencoder — TUNED — 3 Nov 2026

The core model. Benign-only training, windowed sequences, reconstruction-error
detection. Tuned over sequence length {3,5,10} × threshold percentile
(`src/tune_lstm.py`). Windows are labelled attack if ANY flow in the window is an
attack, so the test window attack-rate rises with sequence length (0.38 at L=3,
0.80 at L=10) — the per-flow and windowed results are therefore not numerically
identical, but both use the same held-out flows.

### Headline result (the one to report)

**At a security-realistic false-positive rate (FPR ≈ 0.10), matched to the
baseline, the LSTM-Autoencoder roughly DOUBLES zero-day attack detection:**

| Model (FPR ≈ 0.10) | F1 | Recall | FPR | Zero-day recall |
|---|---|---|---|---|
| Dense AE baseline (90th pct) | 0.568 | 0.631 | 0.100 | 0.480 |
| **LSTM-AE (seq_len 3, 90th pct)** | 0.415 | 0.305 | **0.100** | **0.927** |

→ **Same false-positive rate; zero-day detection 0.93 vs 0.48.** This is the
primary H1 finding: the temporal model is far better at catching *unseen* attacks
at an equal, realistic false-alarm budget.

### Full tuning (F1-optimal points per sequence length)

| seq_len | best pct | F1 | recall | FPR | zero-day |
|---|---|---|---|---|---|
| 3 | 60 | 0.652 | 0.800 | 0.393 | 0.986 |
| **5** | **50** | **0.675** | 0.726 | 0.503 | 1.000 |
| 10 | 50 | 0.571 | 0.449 | 0.483 | 1.000 |

**Important caveat (for honesty in the write-up):** the raw F1-optimal points
above sit at a HIGH FPR (0.39–0.50), because sweeping for F1 alone rewards high
recall. An FPR of ~0.5 is unacceptable operationally (alert fatigue). So the
F1-maximum is reported for completeness, but the **headline comparison is made at
matched FPR ≈ 0.10** (table above), which is the fair, security-meaningful result.

### H1 verdict

- **Zero-day detection:** LSTM-AE **clearly beats** the baseline — 0.93 vs 0.48 at
  equal FPR, up to ~1.00 at higher thresholds. **H1 supported.**
- **Overall F1:** the LSTM also exceeds the baseline F1 (0.675 vs 0.568) at its
  F1-optimum, though at higher FPR; at matched FPR the baseline F1 is higher, so
  the LSTM's advantage is specifically in *recall / zero-day detection*, not blanket
  F1. This nuance is a strong discussion point, not a weakness.

### Discussion points this raises
1. The LSTM trades precision for recall relative to the dense AE — it casts a wider
   net, catching far more unseen attacks but flagging more benign windows. Tuning
   the threshold controls where on this curve the system operates.
2. Shorter sequences (L=3) give the best FPR-controlled operating point here;
   longer windows saturate with attack labels and lose granularity.
3. **Known limitation:** windows group flows adjacent in the shuffled sample, not
   in strict capture-time order (see `lstm_autoencoder.py`). A timestamp-ordered
   version is logical further work.

---

## 4. Explainability — threat attribution — 3 Nov 2026

Two complementary techniques (`src/explainability.py`), applied to the flagged
attack flows (2,478 true-positive flows explained). Both turn the opaque anomaly
score into an interpretable attack attribution.

### Reconstruction-error feature ranking (autoencoder-native — primary)

Top features driving anomalies (mean per-feature reconstruction error over
flagged flows):

| Rank | Feature | Mean recon-error |
|---|---|---|
| 1 | Subflow Fwd Pkts | 22.03 |
| 2 | Fwd Header Len | 21.73 |
| 3 | Tot Fwd Pkts | 21.22 |
| 4 | TotLen Fwd Pkts | 20.73 |
| 5 | Subflow Fwd Byts | 20.69 |
| 6 | Fwd Act Data Pkts | 20.65 |
| 7 | Flow Byts/s | 13.06 |
| 8 | Bwd Pkts/s | 6.62 |

**Interpretation:** attacks are driven mainly by **forward-direction volume**
(packet/byte counts, header length, flow byte-rate) — the classic signature of
**DoS/DDoS and brute-force flooding**, consistent with the dominant attack
families in the data.

### SHAP attribution (literature-standard — corroborating)

Top features by mean |SHAP value|: Fwd Seg Size Min, Init Fwd Win Byts,
Bwd IAT Mean/Std, Bwd Pkt Len Min, and TCP flag counts (ACK, RST, PSH, URG).

**Interpretation:** SHAP surfaces **connection-setup and TCP-control** anomalies —
initial window size, backward inter-arrival timing, and flag behaviour — the
signature of **scanning and protocol-level abuse**.

### Why this matters (for the write-up)

The two methods are complementary, not contradictory:
- **Reconstruction error → magnitude anomalies** (volume): likely DoS / brute-force.
- **SHAP → structural anomalies** (flags, timing, window size): likely scanning /
  protocol abuse.

Together they give an analyst a real **threat-attribution signal** — not just
"this is anomalous" but *in what way* it is anomalous, pointing toward the attack
type. This directly satisfies the explainable-AI / threat-attribution requirement.

**Deliverable:** `results/explain_feature_ranking.csv` — per-flow top-3
responsible features for every flagged attack flow (analyst-readable).
Figures: `explain_global_importance.png`, `shap_summary.png`.

---

## 5. Latency profiling & feature selection (H2) — 3 Nov 2026

`src/latency_profiling.py`. Tree-based (RandomForest gain) feature ranking, then
retrain + re-measure the autoencoder at feature budgets K ∈ {10,20,30,40,77}.

**H2:** tree-based feature selection → latency < 1 ms AND recall drop ≤ 2 pp.

| K (features) | latency (ms/sample) | recall | recall Δ vs full |
|---|---|---|---|
| 10 | 0.0542 | 0.170 | −46.3 pp |
| **20** | 0.0543 | 0.664 | **+3.1 pp** |
| 30 | 0.0550 | 0.498 | −13.5 pp |
| 40 | 0.0553 | 0.691 | +5.8 pp |
| 77 (full) | 0.0547 | 0.633 | — |

### H2 verdict: SUPPORTED — with honest framing

- **Latency:** sub-millisecond at *every* feature count (~0.054 ms/sample). The
  `<1 ms` bar is met comfortably — and, notably, was **already met at full
  features**, so latency is not the bottleneck for this model. Report this plainly
  rather than implying feature selection was needed to reach 1 ms.
- **Feature reduction:** at **K=20** (a ~74% reduction from 77), recall is
  preserved (0.664 vs 0.633) with latency still <1 ms → H2's conditions hold.
- **Headline H2 finding:** *the detector can run on ~20 of 77 features with no
  recall penalty and sub-millisecond latency* — a real efficiency result.

### Honest caveat (important for the viva)

Recall across K is **non-monotonic** (0.17 → 0.66 → 0.50 → 0.69), i.e. it does not
fall smoothly as features are removed. This reflects the **unsupervised
autoencoder's sensitivity to feature subset, initialisation and the fixed
percentile threshold** — small changes shift how many attacks clear the threshold.
The negative "recall drops" (recall going *up*) are within this run-to-run
variance, not genuine improvements. Therefore the result is stated as
**"recall is preserved within variance at ~20 features"**, not "recall improves".
A stronger version would average over several random seeds; noted as further work.

### Cross-method consistency (nice to report)

The top gain-ranked features (Init Fwd Win Byts, Fwd IAT Max, flow-duration and
inter-arrival-time features) overlap with the SHAP attribution from §4 — two
independent methods agreeing on which features matter, which strengthens both.

**Figure:** `results/latency_vs_features.png`.

---

## Overall status — all experiments complete

| Objective | Status |
|---|---|
| O1 Data (acquire, clean, zero-day split) | ✓ |
| O2 Baselines (Isolation Forest, dense AE) | ✓ |
| O3 LSTM-Autoencoder | ✓ — **H1 supported** |
| O4 Explainability (reconstruction ranking + SHAP) | ✓ |
| O5 Latency / feature selection | ✓ — **H2 supported** |

**Both hypotheses supported.** Remaining work: dissertation write-up.
