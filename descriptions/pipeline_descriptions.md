# Pipeline-Beschreibungen

Kurzüberblick, was jede Preprocessing-Variante grob Schritt für Schritt macht.
Quelle sind die Rohdaten (`data/raw`), Output je Variante eine CSV unter `data/preprocessed/`.

## cleaned
Roh → Freitext-/Leakage-Spalten entfernen → kategorisch **frequency-encoden**, numerisch median-imputieren + **StandardScaler** → Feature Engineering (z. B. `salary_avg`, Airbnb-ICC).
→ rein numerische Tabelle.

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

## enhanced
`cleaned` (Text entfernt) → **TabPFN** auf das Label fitten → `get_embeddings` (~192-dim).
→ nur TabPFN-Embeddings (keine Roh-Features).

## enhanced_pca30
`enhanced` → **PCA 30** über die TabPFN-Embeddings (erklärte Varianz > 80 %).
→ nur 30 TabPFN-Embedding-PCA-Komponenten.

## enhanced_semantic_pca30
Inner-Join (über `row_id`) von **`enhanced_pca30`** + **`semantic_pca30`** (nur in `exp2`, kein eigenes Notebook).
→ numerisch/encoded + 30 Text-PCA (`pca_*_sem`) + 30 TabPFN-Emb-PCA (`pca_*_enh`).
