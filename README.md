# TFM Master Thesis

Evaluierung von **Tabular Foundation Models (TFMs)** für die Outlier Detection. Die TFMs werden mit Baseline-Modellen verglichen, zusätzlich werden Strategien zur besseren Nutzung von **Freitext-Features** untersucht.

---

## Projektdaten

* **Institution:** TH Köln
* **Kooperationspartner:** inovex GmbH

### Beteiligte Personen
* **Autor:** Florian Veitz ([fveitz@smail.th-koeln.de](mailto:fveitz@smail.th-koeln.de))
* **Prüfer:** Prof. Dr. Gernot Heisenberg ([gernot.heisenberg@th-koeln.de](mailto:gernot.heisenberg@th-koeln.de))
* **Zweitprüfer:** Dr. Mehmet Hakan Akdag ([hakan.akdag@th-koeln.de](mailto:hakan.akdag@th-koeln.de))
* **Ansprechpartner inovex:** Dr. Darjan Salaj ([darjan.salaj@inovex.de](mailto:darjan.salaj@inovex.de))
---

## Setup

### Umgebung
- **Paketmanager:** [`uv`](https://docs.astral.sh/uv/) — muss installiert sein. `uv sync` richtet Python 3.13 und alle Abhängigkeiten automatisch ein.
- **Hardware:** CUDA-fähige GPU für die TFMs (TabPFN, ConTextTab, AnoLLM, FoMo-0D); in dieser Arbeit vGPU **NVIDIA A40-12C**. PyOD-Baselines und die Auswertung (`experiment_summary.ipynb`) laufen auch ohne GPU.
- **Internet:** für den einmaligen Download der Modellgewichte von HuggingFace (siehe [Modellgewichte](#modellgewichte)).

### Schnellstart (Ergebnisse ansehen)
1. **Github Repo clonen:** git clone https://github.com/florivz/TFM_master_thesis.git
2. **`uv sync`** — Abhängigkeiten installieren.
3. **`mlruns` einrichten** — Ordner im Projekt-Root erstellen und Daten aus [Sciebo](https://th-koeln.sciebo.de/s/DyxqLqP3yAS2Z2J) einfügen.
4. **`experiment_summary.ipynb` ausführen** — Aggregiert bestehende Runs aus `./mlruns`.

### Eigene Experimente reproduzieren
1. **Github Repo clonen:** git clone https://github.com/florivz/TFM_master_thesis.git
2. **`uv sync`** — Python-Umgebung + Abhängigkeiten installieren.
3. **Rohdaten:** Von [Sciebo](https://th-koeln.sciebo.de/s/FJzqqF7RpSr8Hdj/download) nach `data/raw/` laden.
4. **Preprocessing:** Vorverarbeitete Daten von [Sciebo](https://th-koeln.sciebo.de/s/wyKrtxdTy4ACPr8/download) nach `data/preprocessed/` laden (optional Notebooks unter `airbnb_notebooks` / `fakejobs_notebooks` ausführen).
5. **Foundation-Modelle & Gewichte:** AnoLLM (`anollm_src/`) und FoMo-0D (`FoMo-0D/`) liegen bereits im Repo (vendored, siehe [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md)); TabPFN und ConTextTab kommen über `uv sync`. Für FoMo-0D zusätzlich den Checkpoint besorgen — Details unter [Modellgewichte](#modellgewichte).
6. **MLflow-Basis:** Falls noch nicht geschehen, `mlruns` via [Sciebo](https://th-koeln.sciebo.de/s/DyxqLqP3yAS2Z2J/download) im Root bereitstellen.
7. **Notebooks ausführen:** `<datensatz>_notebooks/<exp1–exp4>/` starten (loggt nach `./mlruns`). *Exp 5 (ShapPFN/ExplainerPFN) ist nicht im Repo enthalten und hier nicht abgedeckt.*
8. **Auswertung:** `mlruns` ggf. mit VM synchronisieren und `experiment_summary.ipynb` ausführen.

### Modellgewichte
- **Automatisch von HuggingFace** (beim ersten Lauf, Internet nötig, **kein Token / kein `.env`**): TabPFN, ConTextTab (`rpt-1-oss`) und das AnoLLM-Basismodell `HuggingFaceTB/SmolLM-135M`.
- **Manuell — nur FoMo-0D:** Der Checkpoint wird **nicht** automatisch geladen. `ckpt.zip` (~32 MB) von HuggingFace [`YuchenShen/FoMo-0D`](https://huggingface.co/YuchenShen/FoMo-0D) herunterladen und nach `FoMo-0D/ckpt.zip` legen (das Notebook entpackt es selbst nach `FoMo-0D/ckpt/`).
- **PyOD-Baselines** (iForest, LODA, ECOD, Auto-Encoder) haben keine vortrainierten Gewichte — sie werden pro Lauf trainiert.

---

## Repository-Struktur

* **`data/`**
  * `raw/` — Unverarbeitete Rohdaten
  * `preprocessed/` — Bereinigte und transformierte Daten (8 Varianten)
* **`<dataset>_notebooks/`**
  * `preprocessing/` — Notebooks zur Datenvorverarbeitung
  * `exp/` — Notebooks zur Durchführung der Experimente 1–5
* **`descriptions/`** — Ausführliche Markdown-Dokumentationen (`.md`)
* **`diagrams/`** — Generierte Ergebnisdiagramme (Bar-Charts, SHAP-Plots)
* **`result_tables/`** — Exportierte Ergebnistabellen (CSV/LaTeX)
* **`mlruns/`** — Lokales MLflow-Verzeichnis (Logdateien der Modell-Runs)
* **`experiment_summary.ipynb`** — Zentrales Notebook zur Gesamtauswertung

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
- **Test-Set:** behält **immer** die Originalverteilung an In- und Outliern und wird **nie** verändert (Exp 1/2). Nur so bleiben AP, AUPRC und AUC-ROC interpretierbar. **Ausnahme Exp 3:** balanciertes 1:4-Subset (Train 120/480, Test 30/120). **Exp 4:** zwei Varianten je Modell — Originalverteilung (Train 4000 / Test 3000) und balanciert 1:4.
- **Train-Set:** je nach Trainingsparadigma des Modells unterschiedlich gefüllt:

| Modellgruppe | Modelle | Train-Set |
|---|---|---|
| Unsupervised | iForest, LODA, ECOD, Auto-Encoder, TabPFN (unsup.), FoMo-OD, AnoLLM | Originalverteilung, **ohne** Labels |
| Supervised / Enhanced | TabPFN-Enhanced-Embeddings, ContextTab, TabPFN-classification | Originalverteilung, **mit** Labels |

Kein Modell wird auf reinen Outliern trainiert.

Weitere Informationen siehe: [Detaillierte Beschreibung der Daten](descriptions/data_description.md)

---

## TFM Modellanforderungen

siehe [Modellanforderungen](descriptions/tfm_model_requirements.md)

## Preprocessing-Pipelines

Beide Datensätze werden in **8 Varianten** aufbereitet und unter
`data/preprocessed/<datensatz>.csv` gespeichert.
Die zugehörigen Notebooks liegen unter
`<datensatz>_notebooks/preprocessing/<notebook>.ipynb`.

- für weitere Infos siehe [Detaillierte Pipeline Beschreibung](descriptions/pipeline_description.md)

## Experiment-Struktur

- Aufbau im Repo: `<datensatz>_notebooks/<exp>/notebook.ipynb`
- **MLflow** loggt alle Ergebnisse nach dem Schema `<datensatz> → experiment_x`.
- Speicherung im Ordner `mlruns` (keine Datenbank).
- Geloggte Metriken:  **AUPRC** (Area under the Precision-Recall Curve), **AP** (Average Precision), **AUC-ROC** sowie die **Laufzeit** jedes Modells.

---

## Experimente

### Experiment 1 — Unsupervised Outlier Detection
Wie schlagen sich TFMs im unsupervised Setting?

- **TFMs:** TabPFN (unsupervised), AnoLLM, FoMo-OD
- **Baselines (PyOD):** iForest, Auto-Encoder, LODA, ECOD — per GridSearch hyperparameter-optimiert
- **Daten:** `cleaned` für alle Modelle.
  Ausnahme: **AnoLLM** bekommt **lesbare Rohwerte** direkt aus `raw` (Kategorien/Ort/Datum als Strings, %-Raten als Zahl, 5/4 Freitexte; Leakage-/ID-Spalten entfernt) — es serialisiert jede Zeile als Text, daher sind encodierte/skalierte Werte ungeeignet.
- **Gemeinsamer Split:** alle Modelle nutzen denselben stratifizierten 70/30-Split, definiert über den stabilen `row_id`-Schlüssel (seed 42) → **identisches Test-Set** für jeden Vergleich; nur die *Repräsentation* je Zeile unterscheidet sich (AnoLLM Text, Rest numerisch).
- **Score-Richtung:** jede Methode wird mit ihrer **nativen** Score-Orientierung ausgewertet (höher = anomaler); es wird **nicht** anhand der Test-Labels umgedreht — gilt einheitlich für TFMs und Baselines.

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

- **zwei Verteilungs-Varianten je Modell** (als MLflow-Param `distribution` geloggt):
  - `original`: Originalverteilung (~4–5 % Outlier), Train-Kontext 4000 (TabPFN-Limit), Test 3000
  - `balanced_1to4`: künstlich balanciert (Train 120/480, Test 30/120)
  - gemeinsame Zeilen für beide Modelle (seed 42) → fairer Vergleich
- binäre Klassifikation des Labels (Airbnb per ICC, Fake Jobs natürliches `fraudulent`)
- TabPFN: `cleaned` (numerisch skaliert, Kategorien frequency-encoded, Freitexte entfernt)
- ConTextTab: `cleaned_text` (dieselben numerischen Features + originale Freitexte, nativ verarbeitet)

### 📊 Resultate (`experiment_summary.ipynb`)

* **Pro Experiment & Datensatz:** Tabelle mit **AUPRC**, **AUROC** und `n_runs` (Exp 1/2/4); Bar-Charts für Exp 1/2/4/5.
* **Experiment 3:** SHAP-Relevanz-Tabelle (Feature bzw. numerisch vs. Freitext) plus Bar-Charts (Top-20-Features und Text vs. numerisch).
* Quelle: aggregierte MLflow-Runs aus `./mlruns`; fehlende Experimente werden übersprungen.

---

## Baseline-Hyperparameter (Experiment 1)

GridSearch, bestes Modell auf dem Val-Split gewählt (beste Werte nach dem Lauf aus MLflow).

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
- **Eingeschränkte Vergleichbarkeit von Exp 3:** nutzt ein balanciertes 1:4-Subset statt der Originalverteilung; AP/AUROC sind dadurch nicht mit der natürlichen Outlier-Rate (~4–5 %) vergleichbar und beruhen auf einer kleinen Stichprobe (150 Testzeilen). **Exp 4** liefert zusätzlich die `original`-Variante (Originalverteilung, Test 3000), die diese Einschränkung adressiert; die `balanced_1to4`-Variante bleibt für den direkten Vergleich erhalten.

### Technisch (Hardware-Grenzen der A40-12C vGPU)
- **AnoLLM:** Batchsize = 2 (sonst CUDA OOM).
- **TabPFN:** max. 4000 Zeilen als Kontext (Attention skaliert quadratisch in der Sample-Anzahl).
- **Semantische Embeddings:** PCA zwingend nötig — ohne Dimensionsreduktion erzeugen die Freitext-Embeddings tausende Features und sprengen den GPU-Speicher.
