# TFM Master Thesis

Evaluierung von **Tabular Foundation Models (TFMs)** für die Outlier Detection. Die TFMs werden mit Baseline-Modellen verglichen, zusätzlich werden Strategien zur besseren Nutzung von **Freitext-Features** untersucht.

---

## Setup

- **uv** ist das Standard-Tool zum Installieren von Abhängigkeiten.
- Das Projekt liegt lokal in PyCharm und ist per SSH mit einer VM verbunden:
  `ssh debian@185.113.124.164` und die Daten von der lokalen Umgebung werden automatisch hochgeladen. 
- Dort läuft eine vGPU **NVIDIA A40-12C**.
- **Paketmanager:** `uv` muss installiert sein.
- **API-Key:** `TABPFN_API_KEY=dein_key` in der `.env` eintragen.
- **Aktivierung:** `uv sync` im Terminal ausführen.

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
| Unsupervised | iForest, LODA, ECOD, TabPFN (unsup.), FoMo-OD, AnoLLM | Originalverteilung, **ohne** Labels |
| Semi-supervised (One-Class) | Auto-Encoder | **nur Inlier** |
| Supervised / Enhanced | TabPFN-Enhanced-Embeddings, ContextTab, TabPFN-classification | Originalverteilung, **mit** Labels |

Kein Modell wird auf reinen Outliern trainiert.

---

## Preprocessing-Pipelines

Beide Datensätze werden in **7 Varianten** aufbereitet und unter
`data/preprocessed/<sinnvoller_name_englisch>.csv` gespeichert.
Die zugehörigen Notebooks liegen unter
`<datensatz>_notebooks/preprocessing/<sinnvoller_name_englisch>.ipynb`.

- **Kein Train/Test-Split im Preprocessing:** Jede Variante wird als **eine einzige CSV** gespeichert (nicht in `_train`/`_test` unterteilt). Der stratifizierte 70/30-Split erfolgt erst in den Experiment-Notebooks.

### 1. Cleaned
- Kategorische Daten: **Frequency-Encoding** (Häufigkeit je Kategorie).
- Numerische Werte: Median-imputiert und StandardScaler-skaliert.
- Freitext- sowie Leakage-/Müll-Spalten entfernt; `row_id` als Join-Key behalten.
- Datensatz-spezifisches Feature Engineering (z. B. `salary_avg`; Airbnb: ICC, `host_tenure_days`, Listen-Counts, Host-Flags).

### 2. Cleaned + Freitexte
Cleaned-Dataset plus die originalen Freitextspalten (per `row_id` angehängt) — für **AnoLLM** und **ConTextTab**.

### 3. Semantisch (no PCA)
Standard-Text-Embedding: `all-mpnet-base-v2`.
Nutzt das *Cleaned + Freitexte*-Dataset, wobei die Freitexte hier durch das Embedding-Modell eingebettet werden:

- **Freitext-Features:** Jede Textzelle wird per Sentence-Transformer in einen semantischen Vektor eingebettet.
- **Spaltennamen:** Mit demselben Modell eingebettet (einmal pro Spalte gecacht, da konstant über alle Zeilen) und auf den Zellvektor **addiert** (eine Art Positionskodierung).
- **Reihenfolge:** Zelle und Spaltenname werden getrennt embedded und erst danach elementweise addiert: `v_zelle + v_spalte`.
- **Abschluss:** LayerNorm bzw. Standardisierung über die resultierenden Vektoren. Die Dimensionsanpassung erfolgt in den PCA-Stufen (Varianten 4/5), daher keine separaten lernbaren Projektionsschichten.

### 4. Semantisch (PCA 100)
Wie Variante 3, anschließend PCA auf 100 Komponenten aus den Sentence-Transformer-Embeddings.

### 5. Semantisch (PCA 30)
Wie Variante 3, anschließend PCA auf 30 Komponenten. Die erklärte Varianz muss größer als 80 % sein.

### 6. Enhanced (no PCA)
TabPFN wird auf den **cleaned**-Daten für eine binäre Klassifikation des Labels trainiert (Texte vorher entfernt, da TabPFN nativ keinen Text verarbeitet). Die extrahierten Embeddings werden den Baseline-Modellen übergeben.

### 7. Enhanced (PCA 30)
Wie Variante 6, anschließend PCA auf die TabPFN-Embeddings. Die erklärte Varianz muss größer als 80 % sein.

---

## Experiment-Struktur

- Aufbau im Repo: `<datensatz>_notebooks/<exp>/notebook.ipynb`
- **MLflow** loggt alle Ergebnisse nach dem Schema `<datensatz> → experiment_x`.
- Speicherung im Ordner `mlruns` (keine Datenbank).
- Geloggte Metriken: **AP** (Average Precision), **AUC-ROC** sowie die **Laufzeit** jedes Modells.

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

### 📊 Resultate (`experiment_summary.ipynb`)

* **Pro Experiment & Datensatz:** Tabelle mit **AP**, **AUROC** und `n_runs` (Exp 1/2/4); Bar-Charts für Exp 1/2/4.
* **Experiment 3:** nur die SHAP-Relevanz-Tabelle (Feature bzw. numerisch vs. Freitext), kein Chart.
* Quelle: aggregierte MLflow-Runs aus `./mlruns`; fehlende Experimente werden übersprungen.

---

## Baseline-Hyperparameter (Experiment 1)

GridSearch, bestes Modell per AP auf dem Val-Split gewählt (beste Werte nach dem Lauf aus MLflow eintragen).

### Fake Jobs

| Modell | Suchraum | Beste Hyperparameter |
|---|---|---|
| iForest | `n_estimators ∈ {100, 200}`, `max_features ∈ {0.5, 1.0}` | `n_estimators=100`, `max_features=1.0` |
| LODA | `n_bins ∈ {10, 20}`, `n_random_cuts ∈ {100, 200}` | `n_bins=20`, `n_random_cuts=200` |
| ECOD | parameterfrei | — |
| AutoEncoder | `hidden_neuron_list ∈ {[64,32], [32,16]}`, `epoch_num ∈ {20, 50}` | `hidden_neuron_list=[64,32]`, `epoch_num=20` |

### Airbnb Paris

| Modell | Suchraum | Beste Hyperparameter |
|---|---|---|
| iForest | `n_estimators ∈ {100, 200}`, `max_features ∈ {0.5, 1.0}` | `n_estimators=100`, `max_features=1.0` |
| LODA | `n_bins ∈ {10, 20}`, `n_random_cuts ∈ {100, 200}` | `n_bins=10`, `n_random_cuts=100` |
| ECOD | parameterfrei | — |
| AutoEncoder | `hidden_neuron_list ∈ {[64,32], [32,16]}`, `epoch_num ∈ {20, 50}` | `hidden_neuron_list=[64,32]`, `epoch_num=50` |
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
