# Tabular Foundation Models for Outlier Detection

Evaluation of **Tabular Foundation Models (TFMs)** for outlier detection. The TFMs are benchmarked against baselines, and additional strategies for exploiting **free-text features** are investigated.

**Author:** Florian Veitz — TH Köln, in cooperation with inovex GmbH.

---

## Setup

### Environment
- **Package manager:** [`uv`](https://docs.astral.sh/uv/) — must be installed. `uv sync` sets up **Python 3.14** (pinned in `.python-version`) and all dependencies from `uv.lock` automatically.
- **Hardware:** a CUDA-capable GPU for the TFMs (TabPFN, ConTextTab, AnoLLM, FoMo-0D); this work used an **NVIDIA A40-12C** vGPU. The PyOD baselines and the evaluation (`experiment_summary.ipynb`) also run without a GPU.
- **Internet:** required once to download the model weights from HuggingFace (see [Model weights](#model-weights)).

### Quick start (view results)
1. **Clone the repo:** `git clone https://github.com/florivz/tfm-anomaly-detection.git`
2. **`uv sync`** — install dependencies.
3. **Set up `mlruns`** — create the folder in the project root and add the data from [Sciebo](https://th-koeln.sciebo.de/s/DyxqLqP3yAS2Z2J).
4. **Run `experiment_summary.ipynb`** — aggregates the existing runs from `./mlruns`.

### Reproduce experiments
1. **Clone the repo** and run **`uv sync`**.
2. **Raw data:** download from [Sciebo](https://th-koeln.sciebo.de/s/FJzqqF7RpSr8Hdj/download) into `data/raw/`.
3. **Preprocessed data:** download from [Sciebo](https://th-koeln.sciebo.de/s/wyKrtxdTy4ACPr8/download) into `data/preprocessed/` (or run the notebooks under `airbnb_notebooks` / `fake_job_notebooks`).
4. **Foundation models & weights:** AnoLLM (`anollm_src/`) and FoMo-0D (`FoMo-0D/`) are vendored in the repo (see [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md)); TabPFN and ConTextTab come via `uv sync`. FoMo-0D additionally needs its checkpoint — see [Model weights](#model-weights).
5. **MLflow base:** if not done yet, provide `mlruns` from [Sciebo](https://th-koeln.sciebo.de/s/DyxqLqP3yAS2Z2J/download) in the root.
6. **Run the notebooks:** `<dataset>_notebooks/<exp1–exp4>/` (logs to `./mlruns`).
7. **Evaluate:** run `experiment_summary.ipynb`.

### Model weights
- **Automatic from HuggingFace** (on first run, internet required, **no token / no `.env`**): TabPFN, ConTextTab (`rpt-1-oss`), and the AnoLLM base model `HuggingFaceTB/SmolLM-135M`.
- **Manual — FoMo-0D only:** the checkpoint is **not** fetched automatically. Download `ckpt.zip` (~32 MB) from HuggingFace [`YuchenShen/FoMo-0D`](https://huggingface.co/YuchenShen/FoMo-0D) into `FoMo-0D/ckpt.zip` (the notebook unpacks it to `FoMo-0D/ckpt/`).
- **PyOD baselines** (iForest, LODA, ECOD, Auto-Encoder) have no pretrained weights — they are trained per run.

---

## Repository structure

* **`data/`** — `raw/` (unprocessed) and `preprocessed/` (cleaned/transformed, 9 variants)
* **`<dataset>_notebooks/`** — `preprocessing/` and `exp/` (Experiments 1–4)
* **`descriptions/`** — technical appendix: [datasets & preprocessing variants](descriptions/data_description.md), [TFM input requirements](descriptions/tfm_model_requirements.md)
* **`diagrams/`** — generated result plots (bar charts, SHAP plots)
* **`result_tables/`** — exported result tables (CSV/LaTeX)
* **`mlruns/`** — local MLflow directory (run logs)
* **`experiment_summary.ipynb`** — central evaluation notebook

---

## Data

| Dataset | Label | Logic |
|---|---|---|
| **Fake Job Postings** | natural | `fraudulent = 1` → outlier |
| **Airbnb Paris** | In-Class Classification (ICC) | `review_score_rating == 5` → inlier, `<= 3` → outlier |

The label logic is identical across all preprocessing variants. See [detailed data description](descriptions/data_description.md).

---

## Train/test setup

- **Split:** stratified 70/30 train/test split that preserves the outlier rate in both parts, keyed on the stable `row_id` (seed 42) → an **identical test set** for every comparison.
- **Test set:** always keeps the original in-/outlier distribution and is **never** modified (Exp 1/2), so AP, AUPRC, and AUC-ROC stay interpretable. **Exp 3:** balanced 1:4 subset (train 120/480, test 30/120). **Exp 4:** two variants per model — original distribution (train 4000 / test 3000) and balanced 1:4.
- **Train set:** filled differently depending on the model's training paradigm:

| Group | Models | Train set |
|---|---|---|
| Unsupervised | iForest, LODA, ECOD, Auto-Encoder, TabPFN (unsup.), FoMo-0D, AnoLLM | original distribution, **without** labels |
| Supervised / Enhanced | TabPFN-Enhanced-Embeddings, ConTextTab, TabPFN-classification | original distribution, **with** labels |

No model is trained on outliers only.

---

## Model requirements & preprocessing

- **TFM input requirements:** see [model requirements](descriptions/tfm_model_requirements.md).
- **Preprocessing pipelines:** both datasets are prepared in **9 variants** (`cleaned`, `cleaned_text`, `semantic`, `semantic_pca100/30`, `fast_text_pca100/30`, `enhanced`, `enhanced_pca30`) under `data/preprocessed/`, with notebooks in `<dataset>_notebooks/preprocessing/`. See [preprocessing variants](descriptions/data_description.md#3-preprocessing-variants).

---

## Experiments

Layout: `<dataset>_notebooks/<exp>/notebook.ipynb`. **MLflow** logs all results as `<dataset> → experiment_x` into `mlruns` (no database). Logged metrics: **AUPRC**, **AP** (Average Precision), **AUC-ROC**, and each model's **runtime**.

**Numbering in the repo vs. in the thesis:** the repository keeps its original numbering (notebook folders, MLflow experiment names, `result_tables/exp*`), because renaming would mean re-running everything. In the thesis, the SHAP study is only a **preliminary experiment**, which shifts the numbering of the following experiment:

| Repo (notebooks, MLflow, `result_tables/`) | Thesis | Topic |
|---|---|---|
| Experiment 1 | Experiment 1 | Unsupervised outlier detection |
| Experiment 2 | Experiment 2 | Enhanced baseline models |
| **Experiment 3** | **Preliminary experiment (Vorexperiment)** | Semantic relevance (SHAP) |
| **Experiment 4** | **Experiment 3** | TFMs for binary classification |

### Experiment 1 — Unsupervised outlier detection
*Thesis: Experiment 1*

How do TFMs perform in an unsupervised setting?

- **TFMs:** TabPFN (unsupervised), AnoLLM, FoMo-0D
- **Baselines (PyOD):** iForest, Auto-Encoder, LODA, ECOD — hyperparameter-tuned via grid search
- **Data:** `cleaned` for all models. Exception: **AnoLLM** receives **readable raw values** from `raw` (categories/location/date as strings, rates as numbers, free texts; leakage/ID columns removed), since it serializes each row as text.
- **Score direction:** each method is evaluated with its **native** score orientation (higher = more anomalous); scores are **not** flipped based on the test labels.

### Experiment 2 — Enhanced baseline models
*Thesis: Experiment 2*

Baselines trained and compared on **8 representations**: `cleaned` (numeric only), `semantic_pca100`, `semantic_pca30`, `fast_text_pca100`, `fast_text_pca30`, `enhanced`, `enhanced_pca30`, and `enhanced_semantic_pca30` (join of `enhanced_pca30` + `semantic_pca30`, built in the notebook itself). The question is which free-text representation helps the classical detectors most — see [preprocessing variants](descriptions/data_description.md#3-preprocessing-variants) for what each one contains.

### Experiment 3 — Semantic relevance
*Thesis: preliminary experiment (Vorexperiment), not a numbered experiment*

SHAP analysis with **SAP ConTextTab** as classifier: it processes numeric features and free texts natively (one column = one feature), so the relevance of each free-text column is directly measurable.

- Data: `cleaned_text`; balanced 1:4 subset (120/480) as in-context train, a sample of rows is explained.
- **KernelExplainer** (model-agnostic) yields a mean |SHAP| value per feature.
- Output: per-feature SHAP table + aggregated sum **numeric vs. free text**.

### Experiment 4 — TFMs for binary classification
*Thesis: Experiment 3*

Which model is better for binary classification as outlier detection: **ConTextTab** vs. **TabPFN classification**?

- **Two distribution variants per model** (logged as MLflow param `distribution`):
  - `original`: original distribution (~4–5 % outliers), train context 4000 (TabPFN limit), test 3000
  - `balanced_1to4`: artificially balanced (train 120/480, test 30/120)
  - shared rows for both models (seed 42) → fair comparison
- TabPFN: `cleaned` (numeric scaled, categories frequency-encoded, free texts removed)
- ConTextTab: `cleaned_text` (same numeric features + original free texts, processed natively)

### Results (`experiment_summary.ipynb`)
- **Per experiment & dataset:** table with **AUPRC**, **AUROC**, and `n_runs` (Exp 1/2/4); bar charts for Exp 1/2/4.
- **Experiment 3:** SHAP relevance table (per feature and numeric vs. free text) plus bar charts (top-20 features and text vs. numeric).
- Source: aggregated MLflow runs from `./mlruns`; missing experiments are skipped.
- All names and file prefixes (`exp1`–`exp4`) follow the **repo numbering** — see the mapping table above for the thesis numbering.

---

## Baseline hyperparameters (Experiment 1)

Grid search; best model chosen on the validation split.

| Model | Search space | Fake Jobs | Airbnb Paris |
|---|---|---|---|
| iForest | `n_estimators ∈ {100, 200}`, `max_features ∈ {0.5, 1.0}` | `n_estimators=100`, `max_features=1.0` | `n_estimators=100`, `max_features=1.0` |
| LODA | `n_bins ∈ {10, 20}`, `n_random_cuts ∈ {100, 200}` | `n_bins=20`, `n_random_cuts=200` | `n_bins=10`, `n_random_cuts=100` |
| ECOD | parameter-free | — | — |
| AutoEncoder | `hidden_neuron_list ∈ {[64,32], [32,16]}`, `epoch_num ∈ {20, 50}` | `[64,32]`, `epoch_num=20` | `[64,32]`, `epoch_num=50` |

---

## Limitations

### Methodological
- **Data leakage in the Enhanced variants (Exp 2):** the TabPFN embeddings are trained on the label, so the features carry label information. This systematically overestimates Enhanced performance — the variants are deliberately classified as **semi-supervised** and are not directly comparable to the unsupervised pipelines.
- **Artificial label for Airbnb Paris (ICC):** there is no natural anomaly label; the proxy is defined via `review_score_rating`. Ratings between 3 and 5 are excluded, removing the transition region and making the problem artificially easier. Since ratings are subjective (≠ true anomaly), transferability to real scenarios is limited.

### Technical (A40-12C vGPU limits)
- **AnoLLM:** batch size = 2 (otherwise CUDA OOM).
- **TabPFN:** max. 4000 context rows (attention scales quadratically in the number of samples).
- **Semantic embeddings:** PCA is mandatory — without dimensionality reduction the free-text embeddings produce thousands of features and exceed GPU memory.

---

## License

The code written for this thesis (notebooks, evaluation, documentation) is released under the **MIT License** — see [`LICENSE`](LICENSE).

This does **not** cover:
- **Vendored third-party code** in `anollm_src/` and `FoMo-0D/` — each ships its own license; see [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md) and the `LICENSE`/`NOTICE` files inside those folders.
- **Model weights** downloaded at runtime (TabPFN, ConTextTab, the AnoLLM base model, the FoMo-0D checkpoint) — subject to the terms of their respective providers.
- **The datasets**, which are not redistributed in this repository. Airbnb Paris comes from [Inside Airbnb](https://insideairbnb.com/) and Fake Job Postings from Kaggle; both remain under their original terms of use.
