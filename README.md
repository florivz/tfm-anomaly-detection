# TFM Master Thesis

Evaluierung von **Tabular Foundation Models (TFMs)** für die Outlier Detection. Die TFMs werden mit Baseline-Modellen verglichen, zusätzlich werden Strategien zur besseren Nutzung von **Freitext-Features** untersucht.

---

## Setup

- **uv** ist das Standard-Tool zum Installieren von Abhängigkeiten.
- Das Projekt liegt lokal in PyCharm und ist per SSH mit einer VM verbunden:
  `ssh debian@185.113.124.164` und die Daten von der lokalen Umgebung werden automatisch hochgeladen. 
- Dort läuft eine vGPU **NVIDIA A40-12C**.

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
- **Test-Set:** behält **immer** die Originalverteilung an In- und Outliern und wird **nie** verändert (gilt für alle Modelle). Nur so bleiben AP und AUC-ROC interpretierbar.
- **Train-Set:** je nach Trainingsparadigma des Modells unterschiedlich gefüllt:

| Modellgruppe | Modelle | Train-Set |
|---|---|---|
| Unsupervised | iForest, LODA, ECOD, TabPFN (unsup.), FoMo-OD, AnoLLM | Originalverteilung, **ohne** Labels |
| Semi-supervised (One-Class) | Auto-Encoder | **nur Inlier** |
| Supervised / Enhanced | TabPFN-Enhanced-Embeddings, ContextTab, TabPFN-classification | Originalverteilung, **mit** Labels |

Kein Modell wird auf reinen Outliern trainiert.

---

## Preprocessing-Pipelines

Beide Datensätze werden in **6 Varianten** aufbereitet und unter
`data/preprocessed/<sinnvoller_name_englisch>.csv` gespeichert.
Die zugehörigen Notebooks liegen unter
`<datensatz>/preprocessing/<sinnvoller_name_englisch>.ipynb`.

### 1. Cleaned
- Kategorische Daten: One-Hot- oder Frequency-Encoding (je nach Bedarf).
- Numerische Werte: skaliert.
- Freitextspalten: entfernt.
- Unnötige Zeilen entfernt.
- Airbnb Paris: ICC angewandt.

### 2. Cleaned + Freitexte
Erzeugt das Cleaned-Dataset zuzüglich der originalen Freitexte für das **AnoLLM**-Modell.

### 3. Semantisch (no PCA)
Orientiert sich am **ContextTab-Paper** (SAP-Modell). Standard-Text-Embedding: `all-mpnet-base-v2`.
Nutzt das *Cleaned + Freitexte*-Dataset, wobei die Freitexte hier durch das Embedding-Modell entsprechend der folgenden Beschreibung eingebettet werden.

- **Text & kategoriale Merkmale:** Jede Zelle bzw. jedes Label wird in einen semantischen Vektor eingebettet und über eine lernbare lineare Schicht auf die Zieldimension projiziert.
- **Spaltennamen:** Mit demselben Modell eingebettet, über eine eigene lineare Schicht projiziert und auf den Zellvektor addiert (eine Art Positionskodierung).

### 4. Semantisch (PCA 30)
Wie Variante 3, anschließend PCA auf 30 Komponenten aus den Sentence-Transformer-Embeddings. Die erklärte Varianz muss größer als 80 % sein.

### 5. Enhanced (no PCA)
TabPFN wird auf den **cleaned**-Daten für eine binäre Klassifikation des Labels trainiert (Texte vorher entfernt, da TabPFN nativ keinen Text verarbeitet). Die extrahierten Embeddings werden den Baseline-Modellen übergeben.

### 6. Enhanced (PCA 30)
Wie Variante 5, anschließend PCA auf die TabPFN-Embeddings. Die erklärte Varianz muss größer als 80 % sein.

---

## Experiment-Struktur

- Aufbau im Repo: `root/<datensatz>/<experiment>/notebook.ipynb`
- **MLflow** loggt alle Ergebnisse nach dem Schema `<datensatz> → Experiment_x`.
- Speicherung im Ordner `mlruns` (keine Datenbank).
- Geloggte Metriken: **AP** (Average Precision), **AUC-ROC** sowie die **Laufzeit** jedes Modells.

---

## Experimente

### Experiment 1 — Unsupervised Outlier Detection
Wie schlagen sich TFMs im unsupervised Setting?

- **TFMs:** TabPFN (unsupervised), AnoLLM, FoMo-OD
- **Baselines (PyOD):** iForest, Auto-Encoder, LODA, ECOD — per GridSearch hyperparameter-optimiert
- **Daten:** `cleaned` für alle Modelle.
  Ausnahme: **AnoLLM** erhält eine eigene Pipeline (`cleaned` + die Freitextspalten aus dem `raw`-Datensatz), da es als einziges Modell nativ mit Zahlen umgehen kann.

### Experiment 2 — Enhanced Baseline-Modelle
Die Baseline-Modelle werden auf den **Semantisch**- (PCA / no PCA) gegen die **Enhanced**-Daten (PCA / no PCA) trainiert und verglichen.
Das heißt, die Baseline-Modelle bekommen folgende Daten:
Semantisch vs. rein numerisch (cleaned) vs. enhanced vs. enhanced + semantisch.

### Experiment 3 — Semantische Relevanz
SHAP-Analyse auf dem besten Baseline-Modell aus Experiment 2:

- Wie wichtig ist jede einzelne Freitextspalte (semantisch vs. TabPFN-Embeddings)?
- Aggregierter Vergleich: Relevanz aller numerischen Spalten vs. der semantischen Variante. Bei PCA bedeutet es die Relevanz aller PCA Spalten zu summieren,
ohne PCA sollen alle Embedding Spalten aufsummiert werden.

### Experiment 4 — TFMs zur binären Klassifikation
Welches Modell ist besser für die binäre Klassifikation als Outlier Detection: **ContextTab** vs. **TabPFN classification**?
