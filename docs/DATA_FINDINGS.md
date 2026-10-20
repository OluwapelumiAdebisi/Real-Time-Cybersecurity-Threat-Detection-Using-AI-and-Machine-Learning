# Data Findings — CSE-CIC-IDS2018 Inspection

Notes captured from the data acquisition and inspection stage (Gantt task 2,
milestone O1). These are the verified facts and decisions to write up in the
dissertation's **Data** chapter. Figures below are from the full dataset unless
stated otherwise.

---

## 1. Dataset at a glance (for the Data chapter)

- **Source:** CSE-CIC-IDS2018, Canadian Institute for Cybersecurity (CIC) / CSE.
  Accessed via the Kaggle mirror `solarmainframe/ids-intrusion-csv`; official
  source cited as dataset of record.
- **Scale:** **16,232,943** labelled network flows across **10 capture days**
  (14 Feb – 02 Mar 2018).
- **Features:** **79 flow features** + 1 `Label` column (80 columns total).
- **Attack coverage:** **13 attack families** spanning DDoS, DoS, botnet,
  brute-force and web attacks.

## 2. Class distribution (TRUE figures, full dataset)

Computed across all 10 files (not sampled):

| Class  | Count | Share |
|--------|-------|-------|
| Benign | 13,484,708 | **83.07%** |
| Attack | 2,748,235  | **16.93%** |

- **Imbalance ratio = 4.91 : 1 (benign : attack).**
- This matches the ~83% benign figure reported in the literature, confirming the
  data is intact and the mirror is faithful.
- **This is the figure that justifies the class-imbalance handling** (class-weighted
  loss / SMOTE for the Isolation Forest baseline; benign-only training for the
  autoencoders).

> Methodological note for the write-up: an initial inspection that read the
> *first* 100,000 rows per file reported a misleading ~47% / 53% split. This was
> a **sampling artifact** — attacks are clustered within each day's file, so
> reading the head of each file oversamples them. The true distribution above was
> obtained by counting labels across the complete data. Worth mentioning as an
> example of methodological care (head-sampling ≠ random sampling).

## 3. Attack families present (sampled counts, 1M-row preview)

Full-dataset per-family counts to be recomputed during preprocessing; the
families themselves are confirmed present:

DDoS attacks-LOIC-HTTP, DDOS attack-HOIC, DDOS attack-LOIC-UDP,
FTP-BruteForce, Bot, DoS attacks-SlowHTTPTest, DoS attacks-GoldenEye,
DoS attacks-Slowloris, DoS attacks-Hulk, Brute Force -Web, Brute Force -XSS,
SQL Injection.

## 4. Data-quality issues found (and how each is handled)

These are the documented CSE-CIC-IDS2018 quality problems, confirmed in this
data and handled deliberately in preprocessing — a point of critical engagement
for the dissertation:

| Issue | Evidence | Handling |
|-------|----------|----------|
| **Repeated header rows** | 59 rows across the dataset have `Label == "Label"` | Dropped before any processing |
| **Duplicate flows** | ~15.5% fully duplicated rows (in the 1M preview) | Removed (`drop_duplicates`) |
| **Leakage-prone identifiers** | `Dst Port`, `Timestamp` present | Dropped, so models learn behaviour, not artefacts |
| **NaN / infinity in rate features** | 0 in preview, but known to occur in full data (Flow Byts/s etc. on divide-by-zero) | Cleaned if present |

> Note: the preview flagged "79 non-numeric columns", which was an artefact of the
> 59 header rows contaminating type inference — not a genuine data problem. Once
> the header rows are dropped the feature columns are properly numeric.

## 5. Zero-day hold-out decision

To simulate **zero-day** (unseen) attacks, the following **web-attack family** is
held out entirely from training and used only at test time:

- `SQL Injection`
- `Brute Force -XSS`
- `Brute Force -Web`

**Justification (for the viva / methodology):**
1. A **coherent attack type**, not an arbitrary single class — the research
   question becomes "can the model detect web-layer attacks it never trained on?"
2. **Genuinely rare** (~928 flows in the preview), so an unsupervised model has
   good reason not to have characterised them — a realistic zero-day scenario.
3. **Subtle** (low volume, small payloads), so detecting them unseen via
   reconstruction error is a **strong, honest result** — not a soft target like a
   high-volume DDoS flood.

## 6. Modelling note — training regime for autoencoders

For the unsupervised autoencoders (dense AE, LSTM-AE), the strongest and most
standard setup is to **train on benign traffic only**: the model learns "normal",
then flags poorly-reconstructed flows as anomalous. This is cleaner than SMOTE
for autoencoders. **SMOTE / class-weighting is applied to the Isolation Forest
baseline comparison.** Both regimes will be reported, which strengthens the
evaluation.

---

*Captured: 20 Oct 2026. Decisions recorded in `config.yaml`
(`evaluation.zero_day_holdout`, `cleaning.*`).*
