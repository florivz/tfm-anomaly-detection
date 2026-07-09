# Pipeline-Beschreibungen

Kurzüberblick, was jede Preprocessing-Variante grob Schritt für Schritt macht.
Quelle sind die Rohdaten unter `data/raw/` (`airbnb_paris.csv`, `fake_job_postings.csv`),
Output je Variante eine CSV unter `data/preprocessed/`.

## cleaned
Rein numerische Tabelle (Freitext-/Leakage-Spalten entfernt, kategorisch frequency-/one-hot-encoded,
numerisch median-imputiert + **StandardScaler**). Da beide Datensätze fachlich verschieden sind,
wurde das `cleaned`-Preprocessing **pro Datensatz individuell** umgesetzt.

### cleaned – Airbnb Paris
`data/raw/airbnb_paris.csv` → `data/preprocessed/cleaned_airbnb_paris.csv`
- **Spalten-Drop:** IDs/URLs/Scrape-Metadaten, redundante Review-Subscores und Review-Counts/-Daten,
  `price`/`estimated_revenue_l365d` u. a. (Leakage bzw. Rauschen).
- **Target (künstliches Proxy-Label):** Zeilen ohne `review_scores_rating` verworfen, dann nur
  Bewertungen `== 5.0` oder `<= 3.0` behalten (Übergangsbereich 3–5 entfällt); `is_top_rating = (rating == 5)`,
  Anomalie = nicht-Top-Rating.
- **Feature Engineering:** `host_tenure_days` aus `host_since`; `host_response_rate`/`host_acceptance_rate`
  von "%"-Strings zu Zahlen; `amenities_count`/`host_verifications_count` aus Listen-Strings;
  `host_in_paris`/`host_in_france`/`host_location_missing` aus `host_location`; t/f-Spalten → 1/0.
- **Imputation:** numerisch Median, Boolean Modus, kategorisch `"unknown"`; konstante Spalten gedroppt.
- **Encoding:** `room_type`, `host_response_time` one-hot; `neighbourhood_cleansed`, `property_type`
  **frequency-encoded**.
- **Skalierung & Cleanup:** numerische + frequency-Spalten via **StandardScaler**; Spaltennamen
  normalisiert (ASCII/lowercase); `review_scores_rating` und Freitextspalten entfernt.

### cleaned – Fake Job Postings
`data/raw/fake_job_postings.csv` → `data/preprocessed/cleaned_fake_jobs.csv`
- **Target (natürliches Label):** `fraudulent` (1 = Anomalie), als Spalte am Ende beibehalten.
- **Feature Engineering:** `location` in `country`/`state`/`city` gesplittet; `salary_range` →
  `salary_avg` (Mittel aus Unter-/Obergrenze).
- **Spalten-Drop:** IDs sowie alle Freitextspalten (`title`, `company_profile`, `description`,
  `requirements`, `benefits`) und die schon zerlegten Roh-Spalten.
- **Imputation:** kategorisch `"missing"`, `salary_avg` per Median.
- **Encoding:** alle kategorischen Spalten (Industry, Function, Department, Land/State/City,
  Employment-/Experience-/Education-Felder) **frequency-encoded**.
- **Skalierung:** `salary_avg`, Binär-Flags (`telecommuting`, `has_company_logo`, `has_questions`)
  und frequency-Spalten via **StandardScaler** (13 Features).

## cleaned_text
`cleaned` **+** die originalen Freitextspalten (per `row_id` angehängt).
→ numerisch/encoded + Roh-Freitexte (für AnoLLM / ConTextTab).

## semantic
`cleaned_text` → jede **Freitextzelle** per Sentence-Transformer (`all-mpnet-base-v2`) einbetten (+ Spaltennamen-Embedding addiert); Freitextspalten ersetzt durch ihre Embedding-Vektoren.
→ numerisch/encoded + Text-Embeddings (hochdimensional).

## semantic_pca100
`semantic` → **PCA 100** über die Text-Embeddings.
→ numerisch/encoded + 100 Text-PCA-Komponenten.

## semantic_pca30
`semantic` → **PCA 30** über die Text-Embeddings (erklärte Varianz > 80 %).
→ numerisch/encoded + 30 Text-PCA-Komponenten.

## fast_text_pca30 / fast_text_pca100
`cleaned_text` → jede **Freitextzelle** per unsupervised **fastText** (skipgram, 100d) einbetten (+ Spaltennamen-Embedding addiert, per-Vektor-LayerNorm) → **PCA 30** bzw. **PCA 100** über den Embedding-Block; Freitextspalten entfernt. Gleiche Pipeline wie `semantic`, nur mit fastText statt Sentence-Transformer.
→ numerisch/encoded + 30 bzw. 100 Text-PCA-Komponenten.

## enhanced
`cleaned` (Text entfernt) → **TabPFN** auf das Label fitten → `get_embeddings` (~192-dim).
→ nur TabPFN-Embeddings (keine Roh-Features).

## enhanced_pca30
`enhanced` → **PCA 30** über die TabPFN-Embeddings (erklärte Varianz > 80 %).
→ nur 30 TabPFN-Embedding-PCA-Komponenten.

## enhanced_semantic_pca30
Inner-Join (über `row_id`) von **`enhanced_pca30`** + **`semantic_pca30`** (nur in `exp2`, kein eigenes Notebook).
→ numerisch/encoded + 30 Text-PCA (`pca_*_sem`) + 30 TabPFN-Emb-PCA (`pca_*_enh`).
