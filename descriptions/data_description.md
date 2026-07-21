# Datasets & preprocessing

Two datasets for the outlier-detection experiments — both a mix of structured and free-text columns. Sections 1–2 describe the raw data and the dataset-specific cleaning; [section 3](#3-preprocessing-variants) describes the preprocessing variants built on top of it.

| Dataset | Task | Outlier definition | Outlier rate | Free-text cols |
|---|---|---|---|---|
| **Fake Job Postings** | fraud detection on job ads | natural (`fraudulent = 1`) | 4.84 % | 5 |
| **Airbnb Listings (Paris)** | detecting poorly rated listings | constructed (`is_top_rating = 0`) | 3.93 % | 4 |

---

## 1. Fake Job Postings

Source: Kaggle — *Real or Fake – Fake Job Posting Prediction*. Each row is a job ad; the label `fraudulent` marks fake jobs.

### Key figures

| Metric | Value |
|---|---|
| Total entries | 17,880 |
| Raw columns | 18 (16 features + `job_id` + `fraudulent`) |
| Outliers (`fraudulent = 1`) | 866 |
| Outlier rate | **4.84 %** |
| Raw file | `data/raw/fake_job_postings.csv` |

### Columns

| Type | Columns |
|---|---|
| ID | `job_id` |
| Binary | `telecommuting`, `has_company_logo`, `has_questions` |
| Categorical | `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `department` |
| Structured (substring) | `location` (`"Country, State, City"`), `salary_range` (`"min-max"`) |
| **Free text** | `title`, `company_profile`, `description`, `requirements`, `benefits` |
| Label | `fraudulent` (0 = real, 1 = fake) |

Several columns have a high share of missing values (up to 84 % for `salary_range`). Handling by type during preprocessing: categorical → `"missing"`, text → `""`, numeric → median.

### Feature engineering

The cleaned pipeline (`fake_job_notebooks/preprocessing/cleaned.ipynb`):

1. **Split `location`** → `country`, `state`, `city`.
2. **Parse `salary_range`** → `salary_avg` (range mean, median-imputed).
3. Keep **`job_id`** as `row_id` (join key); extract the label `fraudulent`.
4. Frequency-encode all categories; StandardScaler on numeric columns.

`cleaned` yields **13 numeric features** + `row_id` + label. The variants built on top of it are described in [section 3](#3-preprocessing-variants).

---

## 2. Airbnb Listings (Paris)

Source: **Inside Airbnb** (`listings.csv`, scrape **September 2025**). Property, host, and review metadata for short-term rentals in **Paris**. Inside Airbnb provides no outlier label — outlier status is constructed from the rating.

> **Data version:** this is a recent scrape (Sep 2025), not the older v4.3 (Aug 2022). The schema contains newer columns (`source`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `availability_eoy`, `number_of_reviews_ly`), and several columns are **100 % empty** in this version (see below).

### Key figures

| Metric | Value |
|---|---|
| Raw entries | 81,853 |
| After cleaning | 18,350 |
| Raw columns | 79 |
| Outliers (`is_top_rating = 0`) | 721 |
| Outlier rate | **3.93 %** |
| Raw file | `data/raw/airbnb_paris.csv` |
| Cleaned file | `data/preprocessed/cleaned_airbnb_paris.csv` |

### Fully empty columns (Sep-2025 scrape)

These columns are 100 % empty and drive part of the feature engineering:

| Column | Consequence |
|---|---|
| `price` | no price feature possible (the "parse price" step is dropped) |
| `beds` | drop; optionally replace with `accommodates` |
| `bathrooms` | parse from `bathrooms_text` (`"1 bath"`, `"1.5 baths"`, `"shared bath"`) |
| `neighbourhood_group_cleansed`, `calendar_updated`, `estimated_revenue_l365d` | drop |

### Columns (selection)

| Type | Columns |
|---|---|
| ID / metadata | `id`, `host_id`, `listing_url`, `scrape_id`, `last_scraped` *(removed in cleaning)* |
| Property | `property_type` (61 values, freq-enc), `room_type` (4 values, OHE), `accommodates`, `bedrooms`; `bathrooms` from `bathrooms_text` |
| Host | `host_since`, `host_response_rate`, `host_acceptance_rate`, `host_is_superhost`, `host_identity_verified`, `host_verifications`, `host_location` |
| Geo | `latitude`, `longitude`, `neighbourhood_cleansed` (arrondissements, freq-encoded) |
| Booking | `minimum_nights`, `maximum_nights`, `instant_bookable`, `availability_30/60/90/365` |
| Reviews | `number_of_reviews`, `review_scores_rating` *(label source, then dropped)* |
| **Free text** | `name`, `description`, `neighborhood_overview`, `host_about` |
| Label (constructed) | `is_top_rating` (0 = outlier, 1 = inlier) |

Missing-value handling in cleaning: numeric → median, categorical → `"unknown"`, boolean → mode, free texts → kept as NaN / empty string and filled only at embedding time. Free-text fill rates: `name` (100 %), `description` (96.7 %), `neighborhood_overview` (48.4 %), `host_about` (44.6 %).

### Label construction from the rating

In `airbnb_notebooks/preprocessing/cleaned.ipynb`:

1. **Inlier (`is_top_rating = 1`):** `review_scores_rating == 5.0` (top rating).
2. **Outlier (`is_top_rating = 0`):** `review_scores_rating <= 3.0` (clearly weak rating).
3. The **mid-range** (`3 < rating < 5`) is discarded so the classes are cleanly separable.
4. Listings without a rating are removed.
5. All granular `review_scores_*` columns are dropped to avoid label leakage.

### Feature engineering

`airbnb_notebooks/preprocessing/cleaned.ipynb`:

1. **`host_since`** → `host_tenure_days` (days since registration).
2. **`host_response_rate`, `host_acceptance_rate`** → strip `%`, cast to float.
3. **`bathrooms`** → parse from `bathrooms_text` (`"shared/half bath"` → 0.5).
4. **`amenities`, `host_verifications`** → list lengths as counts.
5. **`host_location`** → flags `host_in_paris`, `host_in_france`, `host_location_missing`; raw column dropped.
6. **Boolean (`t`/`f`)** → 0/1.
7. **OHE** for low cardinality: `host_response_time` (5 columns incl. `unknown`), `room_type` (4 columns).
8. **Frequency encoding** for high cardinality: `neighbourhood_cleansed` (20 arrondissements), `property_type` (61 values).
9. **StandardScaler** on all numeric / frequency-encoded columns (boolean, OHE, label, free texts excluded).
10. Column names normalized to `snake_case` (ASCII, lowercase).

The variants built on top of `cleaned` are described in the next section.

---

## 3. Preprocessing variants

Nine variants are stored as CSV under `data/preprocessed/<variant>_<dataset>.csv`; the notebooks live in `<dataset>_notebooks/preprocessing/`. The tenth (`enhanced_semantic_pca30`) is assembled on the fly in the `exp2` notebook. The label logic is identical across all variants, and every file keeps `row_id` as the join key.

| Variant | Notebook | Content |
|---|---|---|
| `cleaned` | `cleaned.ipynb` | numeric/encoded features only (free text removed) |
| `cleaned_text` | `cleaned_text.ipynb` | `cleaned` + the original free-text columns |
| `semantic` | `semantic.ipynb` | `cleaned` + Sentence-Transformer text embeddings (high-dimensional) |
| `semantic_pca100` / `semantic_pca30` | `semantic_pca*.ipynb` | `semantic` with the text block reduced to 100 / 30 PCA components |
| `fast_text_pca100` / `fast_text_pca30` | `fast_text.ipynb` | `cleaned` + fastText text embeddings, PCA 100 / 30 |
| `enhanced` | `enhanced.ipynb` | TabPFN embeddings only (~192-dim), no raw features |
| `enhanced_pca30` | `enhanced_pca30.ipynb` | `enhanced` reduced to 30 PCA components |
| `enhanced_semantic_pca30` | — (built in `exp2`) | join of `enhanced_pca30` + `semantic_pca30` |

### cleaned
Purely numeric table: free-text and leakage columns removed, categoricals frequency-/one-hot-encoded, numeric median-imputed and **StandardScaler**-normalized. Since the two datasets differ in content, `cleaned` is implemented **per dataset** — see the feature-engineering sections above ([Fake Jobs](#feature-engineering), [Airbnb Paris](#feature-engineering-1)).

### cleaned_text
`cleaned` **+** the original free-text columns (joined via `row_id`) — 5 texts for Fake Jobs, 4 for Airbnb. Input format for the models that process text natively (AnoLLM, ConTextTab).

### semantic / semantic_pca100 / semantic_pca30
Each **free-text cell** is embedded with a Sentence-Transformer (`all-mpnet-base-v2`), with the column-name embedding added on top; the free-text columns are replaced by their embedding vectors. `semantic` keeps the full block (1,500+ dimensions), the `_pca*` variants reduce it to **100** or **30** components — PCA 30 still retains > 80 % explained variance.

### fast_text_pca100 / fast_text_pca30
Same pipeline as `semantic`, but the cells are embedded with unsupervised **fastText** (skipgram, 100d per column, column-name embedding added, per-vector LayerNorm) and then PCA-reduced to **30** or **100** components. Serves as the non-Transformer alternative for the free-text representation.

### enhanced / enhanced_pca30
`cleaned` (text removed) → **TabPFN** is fit on the label, and `get_embeddings` yields a ~192-dim representation per row, which *replaces* the raw features. `enhanced_pca30` reduces this to 30 components (> 80 % explained variance).

> **Leakage:** the TabPFN embeddings are trained on the label, so these two variants carry label information and count as **semi-supervised** — they are not directly comparable to the unsupervised pipelines.

### enhanced_semantic_pca30
Inner join (on `row_id`) of **`enhanced_pca30`** + **`semantic_pca30`** → numeric/encoded features + 30 text-PCA columns (`pca_*_sem`) + 30 TabPFN-embedding-PCA columns (`pca_*_enh`). Built directly in the `exp2` notebook, no separate preprocessing notebook.
