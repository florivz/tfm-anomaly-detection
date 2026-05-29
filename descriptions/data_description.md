# Datensätze

Zwei Datensätze für die Outlier-Detection-Experimente — beide mit Mischung aus strukturierten und Freitext-Spalten.

| Datensatz | Aufgabe | Outlier-Definition | Outlier-Rate | Freitextspalten |
|---|---|---|---|---|
| **Fake Job Postings** | Fraud Detection auf Stellenanzeigen | natürlich (`fraudulent = 1`) | 4,84 % | 5 |
| **Airbnb Listings (Paris)** | Erkennung schlecht bewerteter Listings | konstruiert (`is_top_rating = 0`) | 3,93 % | 4 |

---

## 1. Fake Job Postings

Quelle: Kaggle — *Real or Fake – Fake Job Posting Prediction*. Jede Zeile ist eine Stellenanzeige; das Label `fraudulent` markiert Fake-Jobs.

### Kenngrößen

| Kenngröße | Wert |
|---|---|
| Gesamteinträge | 17.880 |
| Spalten roh | 18 (16 Features + `job_id` + `fraudulent`) |
| Outlier (`fraudulent = 1`) | 866 |
| Outlier-Rate | **4,84 %** |
| Datei (roh) | `data/raw/fake_job_postings.csv` |

### Spaltenübersicht

| Typ | Spalten |
|---|---|
| ID | `job_id` |
| Binär | `telecommuting`, `has_company_logo`, `has_questions` |
| Kategorisch | `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `department` |
| Strukturiert (Substring) | `location` (`"Country, State, City"`), `salary_range` (`"min-max"`) |
| **Freitext** | `title`, `company_profile`, `description`, `requirements`, `benefits` |
| Label | `fraudulent` (0 = echt, 1 = Fake) |

### Fehlende Werte (roh)

Hoher Missing-Anteil in mehreren Spalten — im Preprocessing je nach Typ behandelt (kategorisch → `"missing"`, Text → `""`, numerisch → Median).

| Spalte | Missing | Anteil |
|---|---|---|
| `salary_range` | 15.012 | 84,0 % |
| `department` | 11.547 | 64,6 % |
| `required_education` | 8.105 | 45,3 % |
| `benefits` | 7.212 | 40,3 % |
| `required_experience` | 7.050 | 39,4 % |
| `function` | 6.455 | 36,1 % |
| `industry` | 4.903 | 27,4 % |
| `employment_type` | 3.471 | 19,4 % |
| `company_profile` | 3.308 | 18,5 % |
| `requirements` | 2.696 | 15,1 % |
| `location` | 346 | 1,9 % |
| `description` | 1 | < 0,1 % |

### OD-Label

Natürlich vorhanden: `fraudulent ∈ {0, 1}`. Keine Konstruktion nötig.

### Feature Engineering

Beide Pipelines (`fake_job_notebooks/preprocessing/prep_fake_jobs.ipynb`, `prep_embed_fake_jobs.ipynb`) wenden dieselben Ableitungen an:

1. **`location` aufsplitten** → `country`, `state`, `city`.
2. **`salary_range` parsen** → `salary_avg` (Mittel der Range) + `salary_missing` (Binär-Indikator).
3. **`job_id` droppen** (reine ID).
4. **`fraudulent`** wird als Label extrahiert.

### Pipelines

| Pipeline | Text-Behandlung | Ergebnis | Datei |
|---|---|---|---|
| **Manuelle Baseline** | Freitexte droppen | 13 Features | `data/preprocessed/baseline_fake_jobs.csv` |
| **Embedding-Pipeline** | Sentence-Transformer (`all-MiniLM-L6-v2`, 384 Dim/Spalte) | 1.959 Features (5 num + 6 freq + 28 OHE + 1.920 emb) | `data/preprocessed/embed_fake_jobs.csv` |

---

## 2. Airbnb Listings (Paris)

Quelle: **Inside Airbnb** (`listings.csv`, Version 4.3, August 2022). Property-, Host- und Review-Metadaten von Kurzzeit-Vermietungen für **Paris**. Inside-Airbnb liefert kein OD-Label — der Outlier-Status wird über die Bewertung konstruiert.

### Kenngrößen

| Kenngröße | Wert |
|---|---|
| Roh-Einträge | 81.853 |
| Nach Cleaning | 18.350 |
| Spalten roh | 79 |
| Spalten nach Cleaning | **45** (40 Features + 4 Freitext + 1 Label) |
| Features (40) | 27 numerisch/binär · 2 freq-encoded · 9 OHE (5 `host_response_time` + 4 `room_type`) |
| Outlier (`is_top_rating = 0`) | 721 |
| Outlier-Rate | **3,93 %** |
| Datei (roh) | `data/raw/airbnb_paris.csv` |
| Datei (cleaned) | `data/preprocessed/cleaned_airbnb_paris.csv` |

### Spaltenübersicht

| Typ | Spalten (Auswahl) |
|---|---|
| ID / Metadaten | `id`, `host_id`, `listing_url`, `scrape_id`, `last_scraped` *(im Cleaning entfernt)* |
| Property | `property_type`, `room_type`, `accommodates`, `bedrooms`, `beds`, `bathrooms` |
| Host | `host_since`, `host_response_rate`, `host_acceptance_rate`, `host_is_superhost`, `host_identity_verified`, `host_verifications`, `host_location` |
| Geo | `latitude`, `longitude`, `neighbourhood_cleansed` (Arrondissements, freq-encoded) |
| Booking / Pricing | `price`, `minimum_nights`, `maximum_nights`, `instant_bookable` |
| Verfügbarkeit | `availability_30/60/90/365` |
| Reviews | `number_of_reviews`, `review_scores_rating` *(Quelle für Label, danach gedroppt)* |
| **Freitext** | `name`, `description`, `neighborhood_overview`, `host_about` |
| Label (konstruiert) | `is_top_rating` (0 = Outlier, 1 = Inlier) |

### Fehlende Werte (roh)

Typische Missing-Spalten vor dem Cleaning: `host_about`, `neighborhood_overview`, `host_location`, `host_response_rate`, `host_acceptance_rate`, `host_response_time`. Behandlung im Cleaning: numerisch → Median, kategorisch → `"unknown"`, Boolean → Mode, Freitexte → bleiben als NaN bzw. leerer String und werden erst beim Embedding gefüllt.

### OD-Label — Konstruktion via Bewertung

Konstruktion in `airbnb_notebooks/preprocessing/data_cleaning.ipynb`:

1. **Inlier (`is_top_rating = 1`):** `review_scores_rating == 5.0` (Top-Bewertung).
2. **Outlier (`is_top_rating = 0`):** `review_scores_rating <= 3.0` (klar schwache Bewertung).
3. **Mittelfeld** (`3 < rating < 5`) wird verworfen, damit die Klassen sauber trennbar sind.
4. Listings ohne Rating werden ebenfalls entfernt.
5. Alle granularen `review_scores_*`-Spalten werden gedroppt, um Label-Leak auszuschließen.

### Feature Engineering

`airbnb_notebooks/preprocessing/data_cleaning.ipynb`:

1. **`host_since`** → `host_tenure_days` (Tage seit Registrierung).
2. **`host_response_rate`, `host_acceptance_rate`, `price`** → Sonderzeichen entfernen, in Float.
3. **`amenities`, `host_verifications`** → Listenlängen als Counts.
4. **`host_location`** → Flags `host_in_paris`, `host_in_france`, `host_location_missing`; Rohspalte gedroppt.
5. **Boolean (`t`/`f`)** → 0/1.
6. **OHE** für niedrige Kardinalität: `host_response_time` (5 Spalten), `room_type` (4 Spalten).
7. **Frequency-Encoding** für hohe Kardinalität: `neighbourhood_cleansed`, `property_type`.
8. **`StandardScaler`** auf alle numerischen / frequency-encoded Spalten (Boolean, OHE, Label, Freitexte ausgenommen).
9. Spaltennamen in `snake_case` normalisiert (Umlaute transliteriert).

### Pipelines

| Pipeline | Text-Behandlung | Ergebnis | Datei |
|---|---|---|---|
| **Cleaning / Baseline** | Freitexte bleiben als Rohspalten erhalten | 40 Features + 4 Text + Label = **45 Spalten** | `data/preprocessed/cleaned_airbnb_paris.csv` |
| **Embedding-Pipeline** | Sentence-Transformer (`all-mpnet-base-v2`, 384 Dim/Spalte) | 40 Features + 4×384 emb + Label = **1.577 Spalten** | `data/preprocessed/embed_airbnb_paris.csv` |
| **Embedding-Cache (mpnet)** | `all-mpnet-base-v2` (768 Dim/Spalte) als `.npz`-Cache | 4×768 = 3.072 Embedding-Dim | `data/preprocessed/embed_airbnb_paris_mpnet.npz` |
