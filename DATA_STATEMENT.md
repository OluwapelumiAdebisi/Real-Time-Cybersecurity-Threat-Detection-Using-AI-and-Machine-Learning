# Data Statement

This document records the dataset used in this project, its provenance, its
known limitations, and how this project's pipeline processes it. It exists so
the work is fully reproducible and the data's origin is traceable, without
redistributing the raw dataset itself.

---

## Dataset

**Name:** CSE-CIC-IDS2018 (Intrusion Detection Evaluation Dataset)

**Authors / owner:** Canadian Institute for Cybersecurity (CIC), University of
New Brunswick, in collaboration with the Communications Security Establishment
(CSE), Canada.

**Official source (dataset of record):**
https://www.unb.ca/cic/datasets/ids-2018.html

**Accessed in this project via:** the Kaggle mirror
`solarmainframe/ids-intrusion-csv` ("IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)"),
used for convenience and because it provides a free GPU runtime alongside the
data. The official CIC release is cited as the authoritative source.

**Year generated:** 2018, on a large-scale AWS testbed using real attack tooling.

**Content:** Labelled network-flow records (~80 flow features per record)
captured across several days, each day featuring different attack scenarios:
Brute Force, DoS, DDoS, Web attacks, Infiltration, and Botnet traffic, alongside
benign background traffic.

**Label column:** `Label` — `Benign` denotes normal traffic; attack rows are
labelled by family.

---

## Why this dataset is appropriate

- Produced by a recognised national cybersecurity research body (CIC / CSE),
  not an informal source.
- One of the standard, widely cited benchmarks in modern intrusion-detection
  research.
- Named explicitly in this project's research hypothesis H1, i.e. it is the
  dataset specified for the study.
- Modern and realistic attack families, making it suitable for evaluating
  detection of unseen (zero-day) attacks — unlike older datasets such as
  KDD'99 / NSL-KDD.

---

## Known limitations (addressed deliberately in preprocessing)

The literature documents imperfections in CSE-CIC-IDS2018. These are handled
as explicit preprocessing steps rather than ignored:

- **Duplicate flows** — removed during cleaning.
- **Leakage-prone identifiers** (source/destination IP, port, timestamp, flow
  ID) that can artificially inflate accuracy — dropped, so models learn traffic
  *behaviour* rather than dataset artefacts.
- **Infinity / NaN values** in rate features (e.g. Flow Bytes/s on divide-by-zero)
  — cleaned.
- **Severe class imbalance** (benign traffic vastly outnumbers attacks) —
  handled with class-weighted loss and, where appropriate, SMOTE.

An independently cleaned variant (the Distrinet / KU Leuven "Improved
CSE-CIC-IDS2018") exists and documents these issues; this project performs its
own documented cleaning for transparency.

---

## Licensing and redistribution

CSE-CIC-IDS2018 is made publicly available by CIC for research use, subject to
citation of the dataset. The raw dataset is **not redistributed** in this
repository. Users should obtain it from the official CIC source (or the cited
mirror). Only this project's own processed/sampled outputs, documentation and
code are included here.

**Required citation** (for the dissertation):

> I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New
> Intrusion Detection Dataset and Intrusion Traffic Characterization," in Proc.
> 4th Int. Conf. on Information Systems Security and Privacy (ICISSP), 2018,
> pp. 108–116.

---

## How this project processes the data

1. **Acquisition** (`src/download_data.py`) — accesses the dataset on Kaggle
   (mounted) or via the Kaggle API on Colab. The raw CSVs are never committed
   to version control (see `.gitignore`).
2. **Inspection** (`src/inspect_data.py`) — reports files, columns, class
   distribution, imbalance ratio and data-quality flags; output saved to
   `results/data_inspection_report.txt`. No data is modified at this stage.
3. **Preprocessing** (next stage) — removes duplicates and leakage columns,
   cleans NaN/infinity values, scales features, handles class imbalance, and
   writes a cleaned, sampled dataset to `data/processed/`. This processed
   output is the data artifact submitted as a deliverable; the raw dataset is
   referenced, not submitted.

---

## What is submitted as a deliverable

- This data statement.
- The project **code** (acquisition, inspection, preprocessing, modelling).
- The **processed / sampled dataset** produced by the pipeline (small enough to
  submit; regenerable from the code).
- The **inspection report** and resulting figures.

The raw CSE-CIC-IDS2018 dataset is **referenced via its official source**, not
uploaded, in line with reproducible-research practice.
