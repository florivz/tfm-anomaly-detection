# TFM model requirements

Input requirements of the Tabular Foundation Models (TFMs) used in this project — what each model structurally expects as input, independent of its concrete use in the notebooks.

Models covered: TabPFN 2.5, TabPFN-Unsupervised, ConTextTab, AnoLLM, FoMo-0D. A compact overview is at the end.

---

## 1. TabPFN 2.5 (classifier → embedding extractor)

Used not as an OD model but as an **embedding-refinement step** (`get_embeddings`) — the pretrained model is fit on inlier context and returns a representation vector per row, passed downstream to classical detectors (kNN, AE, iForest, …).

| Requirement | Detail |
|---|---|
| **Task** | classification / regression (in-context); embedding via `get_embeddings()` |
| **Data type** | numeric; categorical features as integer codes |
| **Text** | not supported — free texts must be embedded beforehand (e.g. Sentence-Transformer + PCA) |
| **Missing values** | partially tolerated, but not robust |
| **# Features** | **practically ≤ 500** — architectural soft limit; higher dimensions cause performance drops and memory issues |
| **# Samples** | **practically ≤ 10,000** — attention scales quadratically in N |
| **Label** | **required** — in-context learning is supervised (dummy labels suffice for pure embedding extraction, typically `y=0` for all inliers) |
| **Format** | NumPy array or PyTorch tensor (`X_train`, `y_train`, `X_test`) |

**Core issue:** TabPFN is a classifier, not an OD model. For outlier detection only the embedding path is used, and the embeddings are useful only with enough inlier context (≥ a few hundred samples). High-dimensional inputs (e.g. ST embeddings 1,500+ d) must be PCA-reduced first.

---

## 2. TabPFN-Unsupervised (native OD)

Used as a **direct OD detector** — distilled TabPFN variant via `tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel` + `OutlierDetectionUnsupervisedExperiment`. Returns outlier scores directly, no labels.

| Requirement | Detail |
|---|---|
| **Task** | outlier detection (unsupervised) |
| **Data type** | numeric; categorical features as integer codes |
| **Text** | not supported — same restriction as TabPFN |
| **Missing values** | partially tolerated, but not robust |
| **# Features** | **practically ≤ 50** — no hard limit, but strong degradation above (attention over the feature dimension); rarely useful > 100 |
| **# Samples** | **practically ≤ 3,000** — inference quadratic in N |
| **Label** | **not required** (fully unsupervised) |
| **Format** | NumPy array / PyTorch tensor; internally combines classifier + regressor into a density model |

**Core issue:** the double limit on features (~50) and samples (~3,000) makes the model usable on large, high-dimensional datasets only with substantial preprocessing (PCA, subsampling) — and results then hold only for the subsample.

---

## 3. ConTextTab (SAP-rpt-1-oss)

Used as a **native classifier over tabular data incl. free text** — applied in Experiment 3 for SHAP feature-importance analysis.

| Requirement | Detail |
|---|---|
| **Task** | classification / regression (in-context, transductive) |
| **Data type** | **all types native** — numbers, text, categories, dates |
| **Text** | native — embedded internally via Sentence-Transformer (`all-MiniLM-L6-v2`) |
| **Categorical** | native — embedded as text cell value |
| **Numeric** | native — quantile-based embedding (64 quantile levels) |
| **Date** | native — year, month, day, weekday separately |
| **Missing values** | handled internally as their own state — no imputation needed |
| **# Features** | **max. 500 columns (hard)** — random subsampling above |
| **# Samples (context)** | **max. 8,192 context samples per inference call** — subsampling above |
| **Label** | **required** (a dummy label is needed even for OD — e.g. all inliers `y=0`) |
| **Context** | purely **transductive** — each query row is embedded relative to a given inlier context; no standalone embedding |
| **Format** | pandas DataFrame with original column types (str, float, int, datetime) |

**Core issue:** the model is inherently **context-dependent** — a row's embedding is not absolute but always relative to the given context. Without a reference context (inlier set) no meaningful representations exist. The architecture also nominally requires a target label, even if it is only a dummy in the OD setting.

---

## 4. AnoLLM (Qwen2.5-0.5B + LoRA)

Used for **semi-supervised OD** — LoRA fine-tuning of a small language model on inlier rows; outlier score = negative log-likelihood (NLL) of the serialized row.

| Requirement | Detail |
|---|---|
| **Task** | language-model density estimation → OD score via NLL |
| **Data type** | all types — **serialized to text** (`"col is val, col is val, …"`) |
| **Text** | native — taken directly into the serialization |
| **Categorical** | native — as string value |
| **Numeric** | serialized as string (`"salary is 50000"`) — no mathematical processing, numbers are tokenized |
| **Missing values** | serialized as empty string / `"unknown"` — the model learns the pattern |
| **# Features** | no structural limit, but each column consumes token budget |
| **# Samples (inference)** | unlimited; training only on inlier rows |
| **Token limit** | **hard: base model context window** (Qwen2.5-0.5B → 32k nominal, typically capped at a few hundred tokens/row) — long text columns must be truncated |
| **Label** | training on inliers only (`y=0`); labels are not passed as input, only used to filter the train set |
| **Format** | each row as one free-text string (serialization), tokenized in batches |

**Core issue:** numeric relations (e.g. `salary > 100,000`) are not understood as numbers but as token sequences. Datasets with many long text columns (e.g. Fake Job Postings) require aggressive truncation and hence information loss. The per-row NLL is also not directly comparable across datasets — it is calibrated relative to the inlier score.

---

## 5. FoMo-0D

Used for **zero-shot OD** — a pretrained PFN model applied to new datasets without further training.

| Requirement | Detail |
|---|---|
| **Task** | zero-shot outlier detection |
| **Data type** | numeric (float) only |
| **Text** | not supported |
| **Categorical** | not supported — must be encoded beforehand (OHE / frequency encoding) |
| **Missing values** | not allowed — must be imputed |
| **# Features** | **exactly 100 (hard)** — architectural constraint; fewer features → zero-padding, more features → subsampling/reduction to 100 |
| **# Samples (context)** | no hard limit; inference context typically ≤ 5,000 samples |
| **Label** | not required; context should consist of inlier samples (semi-supervised assumption) |
| **Format** | numeric matrix (N × 100), `float32` |

**Core issue:** the fixed input of **exactly 100 features** is the strongest constraint in the project. Datasets must always be brought to this number — via padding (few features, e.g. Fake Jobs with 13) or reduction (many features, e.g. ST-embedding pipelines with 1,920 d). The choice of reduction method (PCA vs. feature selection vs. subsampling) strongly affects performance.

---

## Constraints at a glance

| Model | Numeric | Text | Categorical | NaN | Max features | Max samples | Label |
|---|---|---|---|---|---|---|---|
| **TabPFN 2.5 (embed)** | ✅ native | ❌ | ⚠️ as int | ⚠️ | ~500 (practical) | ~10,000 | ✅ (dummy ok) |
| **TabPFN-Unsupervised** | ✅ native | ❌ | ⚠️ as int | ⚠️ | **~50 (practical)** | **~3,000 (practical)** | ❌ |
| **ConTextTab** | ✅ native | ✅ native | ✅ native | ✅ native | **≤ 500 (hard)** | ≤ 8,192/call | ⚠️ dummy |
| **AnoLLM** | ⚠️ as token | ✅ native | ✅ as token | ✅ as token | ⚠️ token budget | unlimited | ❌ (inlier-only) |
| **FoMo-0D** | ✅ native | ❌ | ❌ | ❌ | **= 100 (hard)** | ~5,000 | ❌ |

**Legend:** ✅ native · ⚠️ with limitations · ❌ not supported

---

## Implications for pipeline choice

| Data situation | Suitable TFMs |
|---|---|
| Few features (≤ 50), no text | TabPFN-Unsupervised, FoMo-0D (padded to 100) |
| Many features (>> 100), no text | TabPFN 2.5 (embed path after PCA), FoMo-0D after PCA(100) |
| Text is central | AnoLLM (semi-supervised), ConTextTab (for SHAP / classification) |
| Heterogeneous mix (num + text + cat + NaN) | ConTextTab (native), AnoLLM (via serialization) |
| Pure zero-shot, no training | FoMo-0D, TabPFN-Unsupervised |
