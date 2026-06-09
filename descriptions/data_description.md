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

Die Cleaned-Pipeline (`fake_job_notebooks/preprocessing/cleaned.ipynb`) leitet ab:

1. **`location` aufsplitten** → `country`, `state`, `city`.
2. **`salary_range` parsen** → `salary_avg` (Mittel der Range, Median-imputiert).
3. **`job_id`** als `row_id` (Join-Key) behalten; Label `fraudulent` extrahiert.
4. Alle Kategorien **frequency-encoded**, numerische Spalten StandardScaler-skaliert.

### Pipelines

| Pipeline | Notebook | Ergebnis | Datei |
|---|---|---|---|
| **Cleaned** | `preprocessing/cleaned.ipynb` | 13 numerische Features (Text entfernt) + `row_id` | `cleaned_fake_jobs.csv` |
| **Cleaned + Freitexte** | `preprocessing/cleaned_text.ipynb` | Cleaned + 5 Rohtexte | `cleaned_text_fake_jobs.csv` |
| **Semantisch / Enhanced** | `preprocessing/{semantic,enhanced}*.ipynb` | Sentence-Transformer- bzw. TabPFN-Embeddings | `{semantic,enhanced}*_fake_jobs.csv` |

---

## 2. Airbnb Listings (Paris)

Quelle: **Inside Airbnb** (`listings.csv`, Scrape **September 2025**, `last_scraped` 2025-09-12 bis 2025-09-15). Property-, Host- und Review-Metadaten von Kurzzeit-Vermietungen für **Paris**. Inside-Airbnb liefert kein OD-Label — der Outlier-Status wird über die Bewertung konstruiert.

> ⚠️ **Datenstand:** Es handelt sich um einen aktuellen Scrape (Sep. 2025), nicht um die ältere v4.3 (Aug. 2022). Das Schema enthält neuere Spalten (`source`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `availability_eoy`, `number_of_reviews_ly`).

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

> Hinweis: `Features (40)` / `Spalten nach Cleaning 45` stammen aus der älteren Datenversion und müssen für den Sep-2025-Scrape neu bestimmt werden — siehe **Vollständig leere Spalten** unten.

### Vollständig leere Spalten (Sep-2025-Scrape)

In dieser Datenversion sind mehrere Spalten zu **100 % leer** und damit unbrauchbar — das betrifft direkt das Feature Engineering:

| Spalte | Status | Konsequenz |
|---|---|---|
| `price` | komplett leer | **Kein** Preis-Feature möglich (Feature-Engineering-Schritt „price parsen" entfällt) |
| `beds` | komplett leer | droppen; ggf. durch `accommodates` ersetzen |
| `bathrooms` | komplett leer | aus `bathrooms_text` (`"1 bath"`, `"1.5 baths"`, `"shared bath"`) parsen |
| `neighbourhood_group_cleansed` | komplett leer | droppen |
| `calendar_updated` | komplett leer | droppen |
| `estimated_revenue_l365d` | komplett leer | droppen |

### Spaltenübersicht

| Typ | Spalten (Auswahl) |
|---|---|
| ID / Metadaten | `id`, `host_id`, `listing_url`, `scrape_id`, `last_scraped` *(im Cleaning entfernt)* |
| Property | `property_type` (61 Werte, freq-enc), `room_type` (4 Werte, OHE), `accommodates`, `bedrooms` (81 % gefüllt); `beds`/`bathrooms` leer → `bathrooms` aus `bathrooms_text` |
| Host | `host_since`, `host_response_rate`, `host_acceptance_rate`, `host_is_superhost`, `host_identity_verified`, `host_verifications`, `host_location` |
| Geo | `latitude`, `longitude`, `neighbourhood_cleansed` (Arrondissements, freq-encoded) |
| Booking / Pricing | `price`, `minimum_nights`, `maximum_nights`, `instant_bookable` |
| Verfügbarkeit | `availability_30/60/90/365` |
| Reviews | `number_of_reviews`, `review_scores_rating` *(Quelle für Label, danach gedroppt)* |
| **Freitext** | `name`, `description`, `neighborhood_overview`, `host_about` |
| Label (konstruiert) | `is_top_rating` (0 = Outlier, 1 = Inlier) |

### Fehlende Werte (roh)

Behandlung im Cleaning: numerisch → Median, kategorisch → `"unknown"`, Boolean → Mode, Freitexte → bleiben als NaN bzw. leerer String und werden erst beim Embedding gefüllt.

| Spalte | Missing | Anteil |
|---|---|---|
| `beds`, `bathrooms`, `price`, `calendar_updated`, `neighbourhood_group_cleansed`, `estimated_revenue_l365d` | 81.853 | 100,0 % |
| `host_neighbourhood` | 53.832 | 65,8 % |
| `host_about` | 45.350 | 55,4 % |
| `neighbourhood`, `neighborhood_overview` | 42.253 | 51,6 % |
| `host_response_time`, `host_response_rate` | ~32.200 | 39,3 % |
| `host_acceptance_rate` | 26.107 | 31,9 % |
| `review_scores_*` | ~18.000 | 22,0 % |
| `review_scores_rating` (Label-Quelle) | 17.960 | 21,9 % |
| `reviews_per_month` | 17.960 | 22,0 % |
| `bedrooms` | 15.427 | 18,9 % |
| `description` | 2.713 | 3,3 % |
| `bathrooms_text` | 67 | < 0,1 % |
| `name` | 0 | 0,0 % |

**Freitextspalten:** `name` (0 %), `description` (3,3 %), `neighborhood_overview` (51,6 %), `host_about` (55,4 %).

### OD-Label — Konstruktion via Bewertung

Konstruktion in `airbnb_notebooks/preprocessing/cleaned.ipynb`:

1. **Inlier (`is_top_rating = 1`):** `review_scores_rating == 5.0` (Top-Bewertung).
2. **Outlier (`is_top_rating = 0`):** `review_scores_rating <= 3.0` (klar schwache Bewertung).
3. **Mittelfeld** (`3 < rating < 5`) wird verworfen, damit die Klassen sauber trennbar sind.
4. Listings ohne Rating werden ebenfalls entfernt.
5. Alle granularen `review_scores_*`-Spalten werden gedroppt, um Label-Leak auszuschließen.

### Feature Engineering

`airbnb_notebooks/preprocessing/cleaned.ipynb`:

1. **`host_since`** → `host_tenure_days` (Tage seit Registrierung).
2. **`host_response_rate`, `host_acceptance_rate`** → `%` entfernen, in Float (`price` entfällt, da leer).
3. **`bathrooms`** → aus `bathrooms_text` parsen (Zahl extrahieren; `"shared/half bath"` → 0,5).
5. **`amenities`, `host_verifications`** → Listenlängen als Counts.
6. **`host_location`** → Flags `host_in_paris`, `host_in_france`, `host_location_missing`; Rohspalte gedroppt.
7. **Boolean (`t`/`f`)** → 0/1.
8. **OHE** für niedrige Kardinalität: `host_response_time` (5 Spalten inkl. `unknown`), `room_type` (4 Spalten).
9. **Frequency-Encoding** für hohe Kardinalität: `neighbourhood_cleansed` (20 Arrondissements), `property_type` (61 Werte).
10. **`StandardScaler`** auf alle numerischen / frequency-encoded Spalten (Boolean, OHE, Label, Freitexte ausgenommen).
11. Spaltennamen in `snake_case` normalisiert (Umlaute transliteriert).

### Pipelines

| Pipeline | Notebook | Ergebnis | Datei |
|---|---|---|---|
| **Cleaned** | `preprocessing/cleaned.ipynb` | numerische Features (Text entfernt) + `row_id` | `cleaned_airbnb_paris.csv` |
| **Cleaned + Freitexte** | `preprocessing/cleaned_text.ipynb` | Cleaned + 4 Rohtexte | `cleaned_text_airbnb_paris.csv` |
| **Semantisch / Enhanced** | `preprocessing/{semantic,enhanced}*.ipynb` | Sentence-Transformer- bzw. TabPFN-Embeddings | `{semantic,enhanced}*_airbnb_paris.csv` |
