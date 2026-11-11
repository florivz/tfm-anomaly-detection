# Tabular Foundation Models for Outlier Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24141080.svg)](https://doi.org/10.5281/zenodo.24141080)

Evaluation of **Tabular Foundation Models (TFMs)** for outlier detection. The TFMs are benchmarked against baselines, and additional strategies for exploiting **free-text features** are investigated.

This repository accompanies the paper *Evaluating Foundation Model Embeddings for Anomaly Detection in Multivariate Tabular Data*. The paper is based on the first author's master's thesis at TH Köln, written in cooperation with inovex GmbH.

**Authors:** Florian Veitz¹, Gernot Heisenberg¹, Mehmet Hakan Akdag¹, Darjan Salaj²
¹ Institute of Information Science, TH Köln – University of Applied Sciences · ² inovex GmbH

**Contact:** contact@florian-veitz.de

---

## Setup

### Environment

- **Package manager:** [`uv`](https://docs.astral.sh/uv/) — must be installed. `uv sync` sets up Python 3.12 (pinned in `.python-version`) and installs the exact package versions recorded in `uv.lock`.
- **Build toolchain:** `fasttext-wheel 0.9.2` ships prebuilt Python 3.12 wheels for Linux (x86_64), Windows and Intel macOS, but none for Apple Silicon (macOS arm64). There, `uv sync` compiles it from source and needs a C/C++ compiler: Xcode Command Line Tools (`xcode-select --install`).
- **Hardware:** a CUDA-capable GPU for the TFMs (TabPFN, ConTextTab, AnoLLM, FoMo-0D); this work used an **NVIDIA A40-12C** vGPU. The PyOD baselines and the evaluation (`experiment_summary.ipynb`) also run without a GPU.
- **Internet:** required once to download the model weights from Hugging Face (see [Model weights](#model-weights--requirements)).
- **Notebook outputs:** kept out of git via `nbstripout`. After `uv sync`, run **`uv run nbstripout --install`** once per clone (the filter itself isn't versionable, only its `.gitattributes` mapping is).

### Reproduce

1. **Clone the repo** and run **`uv sync`**.
2. **Raw data:** download both datasets from their original sources into `data/raw/` (see [Raw data](#raw-data)).
3. **Foundation models & weights:** [AnoLLM](https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection) (`anollm_src/`) and [FoMo-0D](https://github.com/A-Chicharito-S/FoMo-0D) (`FoMo-0D/`) are vendored in the repo (see [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md)); [TabPFN](https://github.com/PriorLabs/TabPFN) and [ConTextTab](https://github.com/SAP-samples/sap-rpt-1-oss) come via `uv sync` — see their repos for further installation details. FoMo-0D additionally needs its checkpoint — see [Model weights](#model-weights--requirements).
4. **Run preprocessing, then the experiment notebooks** — see [Preprocessing order](#preprocessing-order) for the exact sequence.
5. **Evaluate:** run `experiment_summary.ipynb` to produce `result_tables/` and `diagrams/`.

Steps 3 and 4 can be skipped if you only want to inspect or re-aggregate the reported results: download the archived MLflow logs (see [Experiment logs](#experiment-logs)) and run step 5.

### Raw data

The datasets are not redistributed in this repository. Download them from the original sources and save them under the file names below.

| File | Source | Notes |
| --- | --- | --- |
| `data/raw/fake_job_postings.csv` | [Kaggle: Real / Fake Job Postings Prediction](https://www.kaggle.com/datasets/hassanlegends/fake-job-postings), a re-upload of EMSCAD ([Vidros et al., 2017](https://doi.org/10.3390/fi9010006)) | 17,880 rows, 18 columns |
| `data/raw/airbnb_paris.csv` | [Inside Airbnb](https://insideairbnb.com/), Paris, detailed `listings.csv` | scrape of **12 September 2025** (`scrape_id` 20250912034503), 81,853 rows, 79 columns |

**Airbnb Paris — data version matters.** Inside Airbnb offers only the scrapes of the last twelve months for direct download. The September 2025 scrape used here has to be obtained via an [archived data request](https://insideairbnb.com/data-requests/). A newer scrape has a different set of listings and a partly different schema, so the results will not match the reported ones. Inside Airbnb data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); credit Inside Airbnb as the source when you use it.

### Experiment logs

The MLflow logs of all reported runs are archived on Zenodo: **[10.5281/zenodo.24141080](https://doi.org/10.5281/zenodo.24141080)**. The archive covers the 8 MLflow experiments (preliminary experiment and Experiments 1–3, each on both datasets) with 240 completed runs over the seeds 1, 2 and 3. Each run records its parameters, including the hyperparameter configuration selected by the grid search, AUPRC, AUC-ROC, runtime and seed. It contains no data and no model artifacts.

Download `mlruns_zenodo.zip` and extract it in the project root; this creates `mlruns/`, which `experiment_summary.ipynb` reads. The stored `artifact_location` entries are absolute paths of the original compute environment (`/home/debian/...`). Since the runs hold no artifacts, this only matters for tools that resolve artifact paths.

### Additional notes

- **Result tables and diagrams** are already included in the repository. To regenerate them, run `experiment_summary.ipynb`.
- **Preprocessed data** is **not** distributed: it is seed-dependent by construction and is regenerated by step 4.

### Model weights & requirements

- **Pinned versions:** `uv.lock` is committed and pins every package; `uv sync` installs exactly these versions, among them `tabpfn 8.0.3`, `tabpfn-extensions 0.4.1`, `sap-rpt-oss 1.0.1` and `mlflow 3.12.0`. `pyproject.toml` itself sets no upper bounds, so `uv lock --upgrade` or adding packages can move to newer releases that were not validated for this work.
- **TabPFN model version:** `tabpfn 8.0.3` loads the **TabPFN-3** checkpoints by default, and the notebooks use this default. A newer `tabpfn` release may load a different default model and change the results.
- **TabPFN license gate:** `tabpfn 8.0.3` gates **local** inference behind a free Prior Labs account: on first use it opens a browser login (or reads `TABPFN_TOKEN` in headless environments) against https://ux.priorlabs.ai, requires accepting the license under the account's "Licenses" tab, then the token from the account page. No payment and no cloud billing — inference still runs on your own hardware; this is a license-acceptance gate, not a switch to a hosted API.
- **ConTextTab (`SAP/sap-rpt-1-oss`) is a gated Hugging Face repo** (`"gated": "auto"`) — it needs a free Hugging Face account, accepting the license on the model page, and an `HF_TOKEN` (`huggingface-cli login` or env var) before the download succeeds.
- **Automatic from Hugging Face, no account needed:** the Sentence-Transformer `all-mpnet-base-v2` and the AnoLLM base model (`Qwen/Qwen2.5-0.5B`) are public and ungated. The generic "unauthenticated requests" warning that `huggingface_hub` prints for those is just a rate-limit hint.
- **Manual — FoMo-0D only:** the checkpoint is **not** fetched automatically. Download `ckpt.zip` (~32 MB) from Hugging Face [`YuchenShen/FoMo-0D`](https://huggingface.co/YuchenShen/FoMo-0D) into `FoMo-0D/ckpt.zip` (the notebook unpacks it to `FoMo-0D/ckpt/`).
- **PyOD baselines** (iForest, LODA, ECOD, Auto-Encoder) have no pretrained weights — they are trained per run.

---

## Repository structure

- **`data/`** — `raw/` (original datasets, not in git), `preprocessed/` (generated, per seed), `splits/` (row universe + train/val/test assignment per seed)
- **`<dataset>_notebooks/`** — `preprocessing/`, `exp1/`, `exp2/`, `preliminary_experiment/`, `exp3/`
- **`descriptions/`** — technical appendix: [datasets & preprocessing variants](descriptions/data_description.md), [TFM input requirements](descriptions/tfm_model_requirements.md)
- **`diagrams/`**, **`result_tables/`** — generated plots and exported tables
- **`mlruns/`** — local MLflow directory (run logs; not in git, archived on [Zenodo](https://doi.org/10.5281/zenodo.24141080))
- **`experiment_summary.ipynb`** — central evaluation notebook

---

## Data

| Dataset               | Label             | Logic                                                                                  |
| --------------------- | ----------------- | -------------------------------------------------------------------------------------- |
| **Fake Job Postings** | native            | `fraudulent = 1` → outlier                                                             |
| **Airbnb Paris**      | constructed label | `review_scores_rating == 5.0` → inlier, `<= 3.0` → outlier; listings in between or without rating are discarded |

The label logic is identical across all preprocessing variants. See the [detailed data description](descriptions/data_description.md).

---

## Model requirements & preprocessing

- **TFM input requirements:** see [model requirements](descriptions/tfm_model_requirements.md).
- **Preprocessing pipelines:** notebooks under `<dataset>_notebooks/preprocessing/`, artefacts under `data/preprocessed/`. See [preprocessing variants](descriptions/data_description.md#3-preprocessing-variants).

### Preprocessing order

Per dataset, the notebooks under `<dataset>_notebooks/preprocessing/` each read files written by an earlier one — run them in this order:

1. **`cleaned.ipynb`** — reads `data/raw/`; writes the split (`data/splits/split_<dataset>_seed<k>.csv`) and the numeric table `cleaned_<dataset>_seed<k>.csv`. Everything below depends on it. Run **per seed** (1, 2, 3).
2. **`cleaned_text.ipynb`** — reads the row universe written by step 1; writes `cleaned_text_<dataset>.csv` (free-text columns only). Seed-independent — run **once**.
3. **`semantic.ipynb`** — reads step 2's output; writes `semantic_emb_<dataset>.parquet` (raw Sentence-Transformer embeddings). The embedding model is frozen, nothing is fitted, so this is also seed-independent — run **once**.
4. Three independent branches from here, each run **per seed**:
   - **`fast_text.ipynb`** (needs steps 1 + 2) — fits a fastText model on train, writes `fast_text_pca{30,100}_<dataset>_seed<k>.csv`.
   - **`semantic_pca.ipynb`** (needs steps 1 + 3) — PCA fitted on train over the semantic embeddings, writes `semantic_pca{30,100}_<dataset>_seed<k>.csv`.
   - **`enhanced.ipynb`** (needs step 1; extracts **TabPFN embeddings**, GPU) → **`enhanced_pca30.ipynb`** (needs `enhanced.ipynb`'s output) — writes `enhanced_<dataset>_seed<k>.csv` and `enhanced_pca30_<dataset>_seed<k>.csv`.

`cleaned` alone is sufficient for Experiment 1. Experiment 2 (`exp2.ipynb`) reads all seven representations at once (`fast_text_pca{30,100}`, `semantic_pca{30,100}`, `enhanced`, `enhanced_pca30`, plus `enhanced_semantic_pca30` built inline from the other two) — without a GPU, `enhanced`/`enhanced_pca30` can't be generated, so `exp2.ipynb` needs its representation list trimmed accordingly before it will run.

---

## Experiments

Layout: `<dataset>_notebooks/<experiment>/notebook.ipynb`. **MLflow** logs **AUPRC**, **AUC-ROC** and runtime per run into `mlruns`; a classification report is printed per model, using a top-`k` threshold (`k = test size × outlier rate in train`) for score-only models (AnoLLM, TabPFN-unsupervised, FoMo-0D).

### Experiment 1 — Unsupervised outlier detection

- **TFMs:** TabPFN (unsupervised), AnoLLM, FoMo-0D
- **Baselines (PyOD):** iForest, Auto-Encoder, LODA, ECOD — grid-searched per seed on validation (AUPRC), refitted on train
- **Data:** `cleaned` for all models, except **AnoLLM**, which gets readable raw values since it serialises each row as text
- Each method uses its **native** score orientation, unflipped

### Experiment 2 — Enhanced baseline models

Which free-text representation helps the classical detectors most? Baselines are trained on **7 representations**: `semantic_pca{30,100}`, `fast_text_pca{30,100}`, `enhanced`, `enhanced_pca30`, and `enhanced_semantic_pca30` (join of the latter two). `cleaned` is excluded — it's covered by Experiment 1. Each detector/representation pair gets its own grid search on validation.

### Hyperparameter search space (Experiments 1 and 2)

Both experiments share one exhaustive grid (`sklearn.model_selection.ParameterGrid`, 13 candidates); the winner is refitted on train and evaluated once on test.

| Model        | Search space                                                       | Candidates |
| ------------ | ------------------------------------------------------------------ | ---------- |
| iForest      | `n_estimators ∈ {100, 200}` × `max_features ∈ {0.5, 1.0}`          | 4          |
| LODA         | `n_bins ∈ {10, 20}` × `n_random_cuts ∈ {100, 200}`                 | 4          |
| Auto-Encoder | `hidden_neuron_list ∈ {[64,32], [32,16]}` × `epoch_num ∈ {20, 50}` | 4          |
| ECOD         | parameter-free                                                     | 1          |

The optimum is flat — winners are not stable across seeds (only 7 of 48 combinations pick the same configuration all three times) — so the selected configurations are logged as MLflow parameters (see [Experiment logs](#experiment-logs)) rather than reported per seed.

### Preliminary experiment — Semantic relevance (SHAP)

SHAP analysis with **SAP ConTextTab** as classifier, since it processes numeric features and free texts natively. Context and SHAP background come from train (balanced 1:4); the performance gate runs on the full, natural-distribution test split. **KernelExplainer** yields mean |SHAP| per feature; output is a per-feature table plus a numeric-vs-free-text comparison (sum and mean, since the groups differ in size).

### Experiment 3 — TFMs for binary classification

**ConTextTab** vs. **TabPFN classification**, each with two context variants (`natural`, `balanced_1to4`), both fixed at **1500 in-context rows** so the contrast isolates composition rather than size. Both are evaluated on the same untouched, natural-distribution test split. TabPFN uses numeric features only; ConTextTab adds the original free texts natively.

### Results (`experiment_summary.ipynb`)

Aggregates MLflow runs from `./mlruns` into per-experiment/dataset tables (AUPRC, AUC-ROC, runtime, `n_runs`, mean ± standard deviation across seeds) and bar charts; the preliminary experiment additionally gets a SHAP relevance table.

---

## Limitations

### Methodological

- **Label-guided hyperparameter selection:** the baselines are algorithmically unsupervised — no `fit` ever sees a label — but the hyperparameters are chosen by AUPRC on a labelled validation split. In a strictly unsupervised setting that information would not exist. This favours the baselines over the untuned TFMs.
- **Label leakage in the Enhanced variants (Exp 2):** the TabPFN embeddings are fitted on the label by construction. Fitting on train only removes the test leakage, but the variants remain **semi-supervised** and are not directly comparable to the unsupervised pipelines.
- **Constructed label for Airbnb Paris:** there is no native anomaly label; the proxy is defined via `review_scores_rating`. Ratings between 3 and 5 are excluded, removing the transition region and making the problem artificially easier. Since ratings are subjective, transferability to real scenarios is limited.
- **The in-context training size in Experiment 3 is capped by the number of outliers, not by the model.** A 1:4 balanced context of _n_ rows needs _n_/5 outliers, and the Airbnb training split contains only 360. The largest size feasible on both datasets with some headroom is therefore 1500 rows — well below TabPFN's 4000-row context limit. The experiment consequently reports how the two models behave at 1500 in-context rows, not at their maximum capacity.
- **SHAP ranking uncertainty is not quantified:** the three seeds yield three rankings with a mean and standard deviation, but no bootstrap confidence intervals and no rank-stability measure.
- **One-hot level set (Airbnb)** is determined over all rows. The rarest level has 36 rows, so the probability that it falls entirely outside a 50 % stratified training sample is negligible.
- **fastText residual nondeterminism:** `thread=1` removes the Hogwild nondeterminism, but the internal RNG is fixed at 0 and not exposed by the Python binding. Variation across seeds therefore comes from the differing training corpus.

### Technical (A40-12C vGPU limits)

- **AnoLLM:** batch size = 2 and `max_steps = 1000` (otherwise CUDA OOM / excessive runtime); Airbnb additionally needs gradient checkpointing.
- **TabPFN:** max. 4000 context rows for classification, 9000 for the embedding extraction (attention scales quadratically in the number of samples).
- **Semantic embeddings:** PCA is mandatory — without dimensionality reduction the free-text embeddings produce thousands of features and exceed GPU memory.

---

## How to cite

The paper is forthcoming; its citation will be added here once it is published.

The experiment logs can be cited as:

> Veitz, F. (2026). *MLflow experiment logs for "Evaluating Foundation Model Embeddings for Anomaly Detection in Multivariate Tabular Data"* (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24141080

---

## License

The code written for this work (notebooks, evaluation, documentation) is released under the **MIT License** — see [`LICENSE`](LICENSE). The MLflow logs on Zenodo are released under **CC BY 4.0**.

This does **not** cover:

- **Vendored third-party code** in `anollm_src/` and `FoMo-0D/` — each ships its own license; see [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md) and the `LICENSE`/`NOTICE` files inside those folders.
- **Model weights** downloaded at runtime (TabPFN, ConTextTab, the AnoLLM base model, the FoMo-0D checkpoint) — subject to the terms of their respective providers.
- **The datasets**, which are not redistributed in this repository. Airbnb Paris comes from [Inside Airbnb](https://insideairbnb.com/) and Fake Job Postings from Kaggle; both remain under their original terms of use.
