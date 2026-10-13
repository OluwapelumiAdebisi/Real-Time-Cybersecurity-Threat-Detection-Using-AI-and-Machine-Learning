# Setup Guide — Environment & Data (Google Colab / Kaggle)

This guide gets you from nothing to inspected, sanity-checked data.
Follow it once at the start of the project (Gantt task 2: Data acquisition
& environment setup).

---

## Option 1 — Kaggle Notebook (recommended, simplest)

Kaggle hosts the dataset *and* gives you a free GPU, so there is no download.

1. Go to kaggle.com → **Create → New Notebook**.
2. In the right-hand panel: **+ Add Input** → search **`ids-intrusion-csv`**
   (title: "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)") → **Add**.
   It mounts read-only at `/kaggle/input/ids-intrusion-csv/`.
3. Turn on the GPU: **Settings → Accelerator → GPU** (for the LSTM stage later).
4. Upload the project code (the `src/` folder and `config.yaml`), or clone your
   GitHub repo in the first cell:
   ```python
   !git clone https://github.com/<your-username>/ids-project.git
   %cd ids-project
   ```
5. Inspect the data straight from the mount:
   ```python
   !python src/inspect_data.py --raw-dir /kaggle/input/ids-intrusion-csv
   ```

---

## Option 2 — Google Colab (via the Kaggle API)

Use this if you prefer Colab. You pull the dataset in with your Kaggle token.

1. New notebook at colab.research.google.com. **Runtime → Change runtime type → GPU**.
2. Get your Kaggle token: kaggle.com → **Account → Create New API Token**
   (downloads `kaggle.json`).
3. In the first cell, upload the token and set it up:
   ```python
   from google.colab import files
   files.upload()                       # choose kaggle.json
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
4. Get the project code (clone your repo, or upload `src/` + `config.yaml`):
   ```python
   !git clone https://github.com/<your-username>/ids-project.git
   %cd ids-project
   !pip install -q -r requirements.txt
   ```
5. Download the data, then inspect it:
   ```python
   !python src/download_data.py          # pulls into data/raw/
   !python src/inspect_data.py           # reads data/raw/ from config.yaml
   ```

> **Colab note:** downloaded files vanish when the runtime resets. To avoid
> re-downloading, mount Google Drive and point `raw_dir` at a Drive folder:
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> ```

---

## What each script does

| Script | Purpose |
|--------|---------|
| `src/download_data.py` | Acquires CSE-CIC-IDS2018 (Kaggle mount or API). |
| `src/inspect_data.py`  | Reports files, columns, class balance, and data-quality flags. Writes `results/data_inspection_report.txt`. No data is modified. |

## Fast preview vs full read

The dataset is several GB. For a quick first look, sample rows per file:
```python
!python src/inspect_data.py --nrows 100000
```
Drop `--nrows` for the full, accurate class distribution once you're ready.

## After inspection — what to record

Open `results/data_inspection_report.txt` and note, for your dissertation's
data chapter and for the next step (preprocessing):

- The exact **attack family names** (you'll choose which to hold out as "zero-day"
  in `config.yaml → evaluation.zero_day_holdout`).
- The **imbalance ratio** (justifies SMOTE / class-weighting).
- Which **leakage and constant columns** are present (justifies dropping them).
- Counts of **NaNs, infinities, duplicates** (justifies the cleaning steps).

That report is evidence: screenshots and figures from it go straight into the
"Data" section of the thesis.
