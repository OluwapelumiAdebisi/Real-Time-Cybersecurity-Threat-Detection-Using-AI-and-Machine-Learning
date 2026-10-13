# Real-Time Cybersecurity Threat Detection Using AI and Machine Learning

**Explainable, Unsupervised Deep Learning for Zero-Day Intrusion Detection**

MSc Computer Science Dissertation — Ulster University (COM748)

---

## Project aim

Design, implement and evaluate an explainable, unsupervised deep learning system
that detects zero-day cybersecurity threats in network traffic in near real time,
balancing detection accuracy, false-positive suppression and inference latency.

## Approach (three models, one primary dataset)

1. **Isolation Forest** — classical anomaly-detection baseline
2. **Dense Autoencoder** — reconstruction-error baseline
3. **LSTM-Autoencoder** — main contribution; models temporal structure in flow sequences

Anomalies are detected via reconstruction-error / anomaly-score thresholds, so the
models flag *unseen* (zero-day) attacks rather than memorising known signatures.
Explainability (SHAP + reconstruction-error feature ranking) tells an analyst *why*
a flow was flagged.

## Research hypotheses

- **H1:** An unsupervised LSTM-Autoencoder achieves a statistically significantly higher
  F1-score on zero-day attack traffic (CSE-CIC-IDS2018) than an Isolation Forest baseline.
- **H2:** Tree-based (gain) feature selection reduces per-sample inference latency below
  1 ms without more than a 2% drop in recall.

## Primary dataset

**CSE-CIC-IDS2018** (Canadian Institute for Cybersecurity). UNSW-NB15 is held as an
optional stretch source for cross-dataset validation.

## Repo structure

```
ids-project/
├── data/
│   ├── raw/          # original dataset CSVs (gitignored — never committed)
│   └── processed/    # cleaned / sampled data produced by the pipeline
├── notebooks/        # Jupyter/Colab notebooks (exploration, experiments)
├── src/              # reusable Python modules (pipeline code)
├── models/           # saved trained models (gitignored)
├── results/          # figures, metrics, tables for the dissertation
├── config.yaml       # central settings (paths, dataset choices, seeds)
├── requirements.txt  # reproducible environment
└── README.md
```

## Setup

See `docs/SETUP.md` for step-by-step environment and data instructions
(Google Colab / Kaggle).

## Reproducibility

All randomness is seeded (`config.yaml` → `seed: 42`). Install the pinned
environment with `pip install -r requirements.txt`.
