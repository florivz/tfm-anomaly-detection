# TFM Master Thesis

Evaluierung von **Tabular Foundation Models (TFMs)** für die Outlier Detection. Die TFMs werden mit Baseline-Modellen verglichen, zusätzlich werden Strategien zur besseren Nutzung von **Freitext-Features** untersucht.

---

## Setup

### Umgebung
- **Paketmanager:** [`uv`](https://docs.astral.sh/uv/) — muss installiert sein.
- **Ausführung:** lokal in PyCharm, per SSH an eine VM gekoppelt (`ssh debian@185.113.124.164`); lokale Änderungen werden auf die VM hochgeladen.
- **GPU:** vGPU **NVIDIA A40-12C** auf der VM (für TabPFN, ConTextTab, AnoLLM nötig).

### Schnellstart (vorhandene Ergebnisse ansehen)
1. `uv sync` — Abhängigkeiten installieren.
2. `experiment_summary.ipynb` ausführen — aggregiert die bestehenden MLflow-Runs aus `./mlruns` zu Tabellen/Charts.

### Eigene Experimente reproduzieren
1. **Roh-Daten** laden: <https://th-koeln.sciebo.de/s/QDa6WLKEPFMrM5g/download> → nach `data/raw/`.
2. **Preprocessing-Pipelines** ausführen (`<datensatz>_notebooks/preprocessing/`) → erzeugt die CSVs in `data/preprocessed/`.
3. **Foundation-Modelle** je nach Experiment als Submodul/Clone hinzufügen (entsprechend dem jeweiligen Paper; für Exp 5 siehe dort die separate `venv_explainerpfn`).
4. **Experiment-Notebooks** ausführen (`<datensatz>_notebooks/<exp>/`) → loggt nach `./mlruns`.
5. `mlruns` ggf. mit der VM synchronisieren.
6. `experiment_summary.ipynb` für die Gesamtauswertung ausführen.

---

## Daten

| Datensatz | Label | Logik |
|---|---|---|
| **Fake Job Postings** (`data/raw`) | natürlich | `Fraudulent` = Outlier, sonst Inlier |
| **Airbnb Paris** | per ICC (In-Class Classification) | `review_score_rating == 5` = Inlier, `<= 3` = Outlier |

Die Label-Logik ist über alle Preprocessing-Varianten hinweg identisch.

---

## Train/Test-Aufbau

- **Split:** stratifizierter Train/Test-Split (70/30), der die Outlier-Rate in beiden Teilen erhält.
- **Test-Set:** behält **immer** die Originalverteilung an In- und Outliern und wird **nie** verändert (Exp 1/2). Nur so bleiben AP und AUC-ROC interpretierbar. **Ausnahme Exp 3/4:** balanciertes 1:4-Subset (Train 120/480, Test 30/120), da reiner TFM-Klassifikationsvergleich.
- **Train-Set:** je nach Trainingsparadigma des Modells unterschiedlich gefüllt:

| Modellgruppe | Modelle | Train-Set |
|---|---|---|
| Unsupervised | iForest, LODA, ECOD, Auto-Encoder, TabPFN (unsup.), FoMo-OD, AnoLLM | Originalverteilung, **ohne** Labels |
| Supervised / Enhanced | TabPFN-Enhanced-Embeddings, ContextTab, TabPFN-classification | Originalverteilung, **mit** Labels |

Kein Modell wird auf reinen Outliern trainiert.

Weitere Informationen siehe: **descriptions &rArr; data_descriptions.md**

---

## TFM Modellanforderungen

siehe: **descriptions &rArr; tfm_model_requirements.md**

## Preprocessing-Pipelines

Beide Datensätze werden in **8 Varianten** aufbereitet und unter
`data/preprocessed/<sinnvoller_name_englisch>.csv` gespeichert.
Die zugehörigen Notebooks liegen unter
`<datensatz>_notebooks/preprocessing/<sinnvoller_name_englisch>.ipynb`.

- **Kein Train/Test-Split im Preprocessing:** Jede Variante wird als **eine einzige CSV** gespeichert (nicht in `_train`/`_test` unterteilt). Der stratifizierte 70/30-Split erfolgt erst in den Experiment-Notebooks.

### 1. cleaned
Roh → Freitext-/Leakage-Spalten entfernen → kategorisch **frequency-encoden**, numerisch median-imputieren + **StandardScaler** → Feature Engineering (z. B. `salary_avg`; Airbnb: ICC, `host_tenure_days`, Listen-Counts, Host-Flags).
→ rein numerische Tabelle.

### 2. cleaned_text
`cleaned` **+** die originalen Freitextspalten (per `row_id` angehängt).
→ numerisch/encoded + Roh-Freitexte (für **AnoLLM** / **ConTextTab**).

### 3. semantic
`cleaned_text` → jede **Freitextzelle** per Sentence-Transformer (`all-mpnet-base-v2`) einbetten (+ Spaltennamen-Embedding addiert); Freitextspalten ersetzt durch ihre Embedding-Vektoren.
→ numerisch/encoded + Text-Embeddings (hochdimensional).

### 4. semantic_pca100
`semantic` → **PCA 100** über die Text-Embeddings.
→ numerisch/encoded + 100 Text-PCA-Komponenten.

### 5. semantic_pca30
`semantic` → **PCA 30** über die Text-Embeddings (erklärte Varianz > 80 %).
→ numerisch/encoded + 30 Text-PCA-Komponenten.

### 6. enhanced
`cleaned` (Text entfernt) → **TabPFN** auf das Label fitten → `get_embeddings` (~192-dim).
→ nur TabPFN-Embeddings (keine Roh-Features).

### 7. enhanced_pca30
`enhanced` → **PCA 30** über die TabPFN-Embeddings (erklärte Varianz > 80 %).
→ nur 30 TabPFN-Embedding-PCA-Komponenten.

### 8. enhanced_semantic_pca30
Join aus `enhanced_pca30` + `semantic_pca30`. Hat kein eigenes Notebook und wird erst in **Experiment 2** gebaut. Details zu allen Pipelines: `descriptions/pipeline_descriptions.md`.

---

## Experiment-Struktur

- Aufbau im Repo: `<datensatz>_notebooks/<exp>/notebook.ipynb`
- **MLflow** loggt alle Ergebnisse nach dem Schema `<datensatz> → experiment_x`.
- Speicherung im Ordner `mlruns` (keine Datenbank).
- Geloggte Metriken: **AP** (Average Precision), **AUCPR** (Fläche unter der Precision-Recall-Kurve), **AUC-ROC** sowie die **Laufzeit** jedes Modells.

---

## Experimente

### Experiment 1 — Unsupervised Outlier Detection
Wie schlagen sich TFMs im unsupervised Setting?

- **TFMs:** TabPFN (unsupervised), AnoLLM, FoMo-OD
- **Baselines (PyOD):** iForest, Auto-Encoder, LODA, ECOD — per GridSearch hyperparameter-optimiert
- **Daten:** `cleaned` für alle Modelle.
  Ausnahme: **AnoLLM** bekommt **lesbare Rohwerte** direkt aus `raw` (Kategorien/Ort/Datum als Strings, %-Raten als Zahl, 5/4 Freitexte; Leakage-/ID-Spalten entfernt) — es serialisiert jede Zeile als Text, daher sind encodierte/skalierte Werte ungeeignet.

### Experiment 2 — Enhanced Baseline-Modelle
Die Baseline-Modelle werden auf verschiedenen Repräsentationen trainiert und verglichen:
`cleaned` (rein numerisch), `semantic_pca100`, `semantic_pca30`, `enhanced`, `enhanced_pca30` sowie `enhanced_semantic_pca30` (Join aus enhanced_pca30 + semantic_pca30).

### Experiment 3 — Semantische Relevanz
SHAP-Analyse mit **SAP ConTextTab** als Klassifikator: es verarbeitet numerische Features und Freitexte nativ (eine Spalte = ein Feature), daher ist die Relevanz jeder Freitextspalte direkt messbar.

- Daten: `cleaned_text`; balanciertes 1:4-Subset (120/480) als In-Context-Train, eine Stichprobe Zeilen wird erklärt.
- **KernelExplainer** (modell-agnostisch) liefert pro Feature einen mittleren |SHAP|-Wert.
- Ausgabe: Tabelle der SHAP-Werte je Feature + aggregierte Summe **numerisch vs. Freitext**.

### Experiment 4 — TFMs zur binären Klassifikation
Welches Modell ist besser für die binäre Klassifikation als Outlier Detection: **ContextTab** vs. **TabPFN classification**?

- gemeinsames balanciertes 1:4-Subset (Train 120/480, Test 30/120, seed 42)
- binäre Klassifikation des Labels (Airbnb per ICC, Fake Jobs natürliches `fraudulent`)
- TabPFN: `cleaned` (numerisch skaliert, Kategorien frequency-encoded, Freitexte entfernt)
- ConTextTab: `cleaned_text` (dieselben numerischen Features + originale Freitexte, nativ verarbeitet)

### Experiment 5 – TFM Erklärbarkeit

Vergleich zweier moderner TFM-Explainer gegen KernelSHAP als Baseline auf einem TabPFN-Klassifikator
(`cleaned`-Features, numerisch). Setup analog Exp. 3: seed 42, 1:4 Outlier/Inlier-Ratio,
30 Erklär-Zeilen. Verglichen werden `mean|attr|` je Feature, Fidelity vs. KernelSHAP
(Spearman + Cosine) und Laufzeit.

- **KernelSHAP** – modell-agnostische Baseline (Gold-Standard)
- **ShapPFN** – SHAP-Werte als Nebenprodukt des Forward Pass (`kunumi/ShapPFN`)
- **ExplainerPFN** – Zero-Shot Explainer; benötigt TabPFN 2.1.2 → separate `venv_explainerpfn`
  (Haupt-Env nutzt TabPFN 2.5)

**Limitationen:** Alle Methoden werden einheitlich auf m ≤ 8 Features und n ≤ 200 Zeilen
evaluiert (gemeinsame Obergrenze beider PFN-Explainer). Stratifiziertes 1:4-Sampling sichert
stabile Attributionen trotz kleiner Stichprobe. ExplainerPFN erfordert zusätzlich einen
Standard Scaler auf den `cleaned`-Features. Experiment 5 beschränkt sich auf die `cleaned`-Pipeline (numerische Features).
Semantische und Freitext-Pipelines werden nicht evaluiert

**Notebooks:** `explainers.ipynb` (KernelSHAP + ShapPFN, Haupt-`.venv`, erzeugt
`ref_kernelshap.csv`) → danach `explainerpfn.ipynb` (`venv_explainerpfn`). Beide Repos
als Clone im Projektverzeichnis (`ShapPFN/`, `ExplainerPFN/`, gitignored).

### 📊 Resultate (`experiment_summary.ipynb`)

* **Pro Experiment & Datensatz:** Tabelle mit **AP**, **AUCPR**, **AUROC** und `n_runs` (Exp 1/2/4); Bar-Charts (je Metrik) für Exp 1/2/4.
* **Experiment 3:** SHAP-Relevanz-Tabelle (Feature bzw. numerisch vs. Freitext) **plus** Bar-Charts (Top-Features und Text vs. numerisch).
* **Aggregation:** Durchschnitt über die **neuesten 3 Runs** je Modell (bzw. Detektor × Repräsentation).
* **Export:** alle Tabellen werden zusätzlich als CSV unter `result_tables/expX_<datensatz>.csv` gespeichert.
* Quelle: aggregierte MLflow-Runs aus `./mlruns`; fehlende Experimente werden übersprungen.

---

## Baseline-Hyperparameter (Experiment 1)

GridSearch, bestes Modell per AP auf dem Val-Split gewählt (beste Werte nach dem Lauf aus MLflow eintragen).

### Fake Jobs

| Modell | Suchraum | Beste Hyperparameter |
|---|---|---|
| iForest | `n_estimators ∈ {100, 200}`, `max_features ∈ {0.5, 1.0}` | `n_estimators=200`, `max_features=1.0` |
| LODA | `n_bins ∈ {10, 20}`, `n_random_cuts ∈ {100, 200}` | `n_bins=20`, `n_random_cuts=100` |
| ECOD | parameterfrei | — |
| AutoEncoder | `hidden_neuron_list ∈ {[64,32], [32,16]}`, `epoch_num ∈ {20, 50}` | `hidden_neuron_list=[64,32]`, `epoch_num=20` |

### Airbnb Paris

| Modell | Suchraum | Beste Hyperparameter |
|---|---|---|
| iForest | `n_estimators ∈ {100, 200}`, `max_features ∈ {0.5, 1.0}` | `n_estimators=100`, `max_features=0.5` |
| LODA | `n_bins ∈ {10, 20}`, `n_random_cuts ∈ {100, 200}` | `n_bins=10`, `n_random_cuts=200` |
| ECOD | parameterfrei | — |
| AutoEncoder | `hidden_neuron_list ∈ {[64,32], [32,16]}`, `epoch_num ∈ {20, 50}` | `hidden_neuron_list=[64,32]`, `epoch_num=20` |
---

## Limitationen

### Methodisch
- **Data Leakage in den Enhanced-Varianten (Exp 2):** Die TabPFN-Embeddings werden auf dem Label trainiert, die Features tragen also Label-Information. Das überschätzt die Enhanced-Performance systematisch — die Varianten sind bewusst als **semi-supervised** eingeordnet und nicht direkt mit den unsupervised Pipelines vergleichbar.
- **Künstliches Label bei Airbnb Paris (ICC):** Es gibt kein natürliches Anomalie-Label; der Proxy wird über `review_score_rating` definiert. Bewertungen 3–5 werden ausgeschlossen, wodurch der Übergangsbereich fehlt und das Problem künstlich leichter wird. Da Bewertungen subjektiv sind (≠ echte Anomalie), ist die Übertragbarkeit auf reale Szenarien eingeschränkt.
- **Eingeschränkte Vergleichbarkeit von Exp 3/4:** Beide nutzen ein balanciertes 1:4-Subset statt der Originalverteilung. AP/AUROC sind dadurch nicht mit der natürlichen Outlier-Rate (~4–5 %) vergleichbar und beruhen auf kleinen Stichproben (150 Testzeilen).

### Technisch (Hardware-Grenzen der A40-12C vGPU)
- **AnoLLM:** Batchsize = 2 (sonst CUDA OOM).
- **TabPFN:** max. 4000 Zeilen als Kontext (Attention skaliert quadratisch in der Sample-Anzahl).
- **Semantische Embeddings:** PCA zwingend nötig — ohne Dimensionsreduktion erzeugen die Freitext-Embeddings tausende Features und sprengen den GPU-Speicher.
