# TFM-Modell-Anforderungen

Dieses Dokument beschreibt die **datentechnischen Eingabe-Anforderungen** der im Projekt verwendeten Tabular Foundation Models (TFMs). Es geht um das, was jedes Modell strukturell als Input erwartet — unabhängig vom konkreten Einsatz im Notebook.

Behandelte Modelle:

1. [TabPFN 2.5 (Klassifikator → Embedding-Extraktor)](#1-tabpfn-25-klassifikator--embedding-extraktor)
2. [TabPFN-Unsupervised (Native OD)](#2-tabpfn-unsupervised-native-od)
3. [ConTextTab (SAP-rpt-1-oss)](#3-contexttab-sap-rpt-1-oss)
4. [AnoLLM (Qwen2.5-0.5B + LoRA)](#4-anollm-qwen25-05b--lora)
5. [FoMo-OD](#5-fomo-od)

Eine kompakte Übersicht steht am Ende: [Constraints auf einen Blick](#-constraints-auf-einen-blick).

---

## 1. TabPFN 2.5 (Klassifikator → Embedding-Extraktor)

Verwendung im Projekt: nicht als OD-Modell, sondern als **Embedding-Refinement-Schritt** (`get_embeddings`) — das vortrainierte Modell wird auf Inlier-Kontext gefittet und liefert für jede Zeile einen Repräsentations-Vektor, der nachgelagert an klassische Detektoren (kNN, AE, iForest, …) übergeben wird.

| Anforderung | Detail |
|---|---|
| **Aufgabe** | Klassifikation / Regression (in-context); Embedding über `get_embeddings()` |
| **Datentyp** | Numerisch; kategorische Features als Integer-Codes |
| **Text** | Nicht unterstützt — Freitexte müssen vorab eingebettet werden (z. B. Sentence-Transformer + PCA) |
| **Kategorisch** | Über Integer-Codes; intern als kategorial deklarierbar |
| **Fehlende Werte** | Werden teilweise toleriert, jedoch nicht robust |
| **Feature-Anzahl** | **Praktisch ≤ 500** — Architektur-Soft-Limit; bei höheren Dimensionen Performance-Einbruch und Speicherprobleme |
| **Sample-Anzahl** | **Praktisch ≤ 10.000** Trainings-/Kontext-Samples — Attention skaliert quadratisch in N |
| **Label** | **Benötigt** — In-Context-Lernen ist supervised (für reine Embedding-Extraktion reichen Dummy-Labels, üblich `y=0` für alle Inlier) |
| **Format** | NumPy-Array oder PyTorch-Tensor (`X_train`, `y_train`, `X_test`) |

**Kernproblem:** TabPFN ist kein OD-Modell, sondern ein Klassifikator. Für Outlier Detection wird ausschließlich der Embedding-Pfad genutzt — die Embeddings selbst sind erst dann brauchbar, wenn ausreichend Inlier-Kontext (≥ einige Hundert Samples) übergeben wird. Hochdimensionale Eingaben (z. B. ST-Embeddings 1.500 + d) müssen vorher per PCA reduziert werden.

---

## 2. TabPFN-Unsupervised (Native OD)

Verwendung im Projekt: **direkter OD-Detektor** — destillierte TabPFN-Variante über `tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel` + `OutlierDetectionUnsupervisedExperiment`. Liefert direkt Outlier-Scores ohne Labels.

| Anforderung | Detail |
|---|---|
| **Aufgabe** | Outlier Detection (unüberwacht) |
| **Datentyp** | Numerisch; kategorische Features als Integer-Codes |
| **Text** | Nicht unterstützt — gleiche Restriktion wie TabPFN |
| **Kategorisch** | Über Integer-Codes; explizite Kennzeichnung möglich |
| **Fehlende Werte** | Werden teilweise toleriert, jedoch nicht robust |
| **Feature-Anzahl** | **Praktisch ≤ 50** — kein hartes Limit, aber starke Degradation oberhalb (Attention über Feature-Dimension); bei > 100 selten sinnvoll |
| **Sample-Anzahl** | **Praktisch ≤ 3.000** — Inferenz quadratisch in N, oberhalb stark eingeschränkte Praktikabilität |
| **Label** | **Nicht benötigt** (vollständig unüberwacht) |
| **Format** | NumPy-Array / PyTorch-Tensor; intern werden Klassifikator + Regressor zu einem Density-Modell kombiniert |

**Kernproblem:** Die doppelte Einschränkung auf Features (~50) und Samples (~3.000) macht das Modell für große, hochdimensionale Datensätze nur mit erheblicher Vorverarbeitung (PCA, Subsampling) einsetzbar — und Ergebnisse gelten dann nur für das Subsample.

---

## 3. ConTextTab (SAP-rpt-1-oss)

Verwendung im Projekt: **nativer Klassifikator über tabellarische Daten inkl. Freitext** — eingesetzt in Experiment 2 für SHAP-Analyse der Feature-Importance.

| Anforderung | Detail |
|---|---|
| **Aufgabe** | Klassifikation / Regression (in-context, transduktiv) |
| **Datentyp** | **Alle Typen nativ** — Zahlen, Text, Kategorien, Datum |
| **Text** | Nativ unterstützt — wird intern via Sentence-Transformer (`all-MiniLM-L6-v2`) eingebettet |
| **Kategorisch** | Nativ unterstützt — wird als Text-Zellwert eingebettet |
| **Numerisch** | Nativ unterstützt — quantil-basierte Einbettung (64 Quantil-Levels) |
| **Datum** | Nativ unterstützt — Jahr, Monat, Tag, Wochentag separat |
| **Fehlende Werte** | Werden intern als eigener Zustand behandelt — kein Imputieren nötig |
| **Feature-Anzahl** | **Max. 500 Spalten (hart)** — bei mehr wird zufällig subsampelt |
| **Sample-Anzahl (Kontext)** | **Max. 8.192 Kontext-Samples pro Inferenz-Aufruf** — darüber Subsampling |
| **Label** | **Benötigt** (auch für OD ein Dummy-Label nötig — z. B. alle Inlier mit `y=0`) |
| **Kontext-Anforderung** | Rein **transduktiv** — jede Query-Zeile wird relativ zu einem übergebenen Inlier-Kontext eingebettet, kein eigenständiges Embedding möglich |
| **Format** | Pandas-DataFrame mit originalen Spaltentypen (str, float, int, datetime) |

**Kernproblem:** Das Modell ist inhärent **kontextabhängig** — ein Embedding einer Zeile ist nicht absolut, sondern stets relativ zum übergebenen Kontext. Ohne Referenz-Kontext (Inlier-Set) sind keine sinnvollen Repräsentationen berechenbar. Außerdem erfordert die Architektur nominell ein Ziel-Label, auch wenn es im OD-Setting nur ein Dummy ist.

---

## 4. AnoLLM (Qwen2.5-0.5B + LoRA)

Verwendung im Projekt: **semi-supervised OD** — LoRA-Fine-Tuning eines kleinen Sprachmodells auf Inlier-Zeilen; Outlier-Score = negative Log-Likelihood (NLL) der serialisierten Zeile.

| Anforderung | Detail |
|---|---|
| **Aufgabe** | Sprachmodell-basierte Dichte-Schätzung → OD-Score über NLL |
| **Datentyp** | Alle Typen — wird in **Text serialisiert** (`"col is val, col is val, …"`) |
| **Text** | Nativ unterstützt — direkt in die Serialisierung übernommen |
| **Kategorisch** | Nativ unterstützt — als String-Wert |
| **Numerisch** | Als String serialisiert (`"salary is 50000"`) — keine mathematische Verarbeitung, Zahlen werden tokenisiert |
| **Fehlende Werte** | Werden als leerer String / `"unknown"` serialisiert — Modell lernt das Muster |
| **Feature-Anzahl** | Kein strukturelles Limit; jede Spalte belegt aber Token-Budget |
| **Sample-Anzahl (Inferenz)** | Unbegrenzt; Training ausschließlich auf Inlier-Zeilen |
| **Token-Limit** | **Hart: Kontextfenster des Basismodells** (Qwen2.5-0.5B → 32k nominal, im Setup typischerweise auf wenige Hundert Tokens/Zeile begrenzt) — lange Textspalten müssen gekürzt werden |
| **Label** | Training auf Inlier-only (`y=0`); Labels werden nicht als Input übergeben, nur zur Filterung des Train-Sets |
| **Format** | Jede Zeile als ein Freitext-String (Serialisierung), batchweise tokenisiert |

**Kernproblem:** Numerische Relationen (z. B. `salary > 100.000`) werden nicht als Zahlen verstanden, sondern als Token-Sequenzen. Bei Datensätzen mit vielen langen Textspalten (z. B. Fake Job Postings) muss aggressiv gekürzt werden, was Informationsverlust bedeuten kann. Außerdem ist die NLL pro Zeile nicht direkt vergleichbar zwischen Datensätzen — kalibriert wird relativ zum Inlier-Score.

---

## 5. FoMo-OD

Verwendung im Projekt: **Zero-Shot OD** — vortrainiertes PFN-Modell wird ohne weiteres Training auf neue Datensätze angewendet.

| Anforderung | Detail |
|---|---|
| **Aufgabe** | Zero-Shot Outlier Detection |
| **Datentyp** | Ausschließlich numerisch (float) |
| **Text** | Nicht unterstützt |
| **Kategorisch** | Nicht unterstützt — muss vorab kodiert werden (OHE / Frequency-Encoding) |
| **Fehlende Werte** | Nicht erlaubt — müssen imputiert werden |
| **Feature-Anzahl** | **Exakt 100 (hart)** — Architektur-Constraint; weniger Features → Padding mit Nullen, mehr Features → Subsampling / Reduktion auf 100 |
| **Sample-Anzahl (Kontext)** | Kein hartes Limit; Inferenz-Kontext typischerweise ≤ 5.000 Samples |
| **Label** | Nicht benötigt; Kontext sollte aus Inlier-Samples bestehen (semi-supervised Annahme) |
| **Format** | Numerische Matrix (N × 100), `float32` |

**Kernproblem:** Die feste Eingabedimension von **genau 100 Features** ist das stärkste Constraint im Projekt. Datensätze müssen immer auf diese Zahl gebracht werden — entweder per Padding (bei wenigen Features wie Fake Jobs mit 13 Features) oder per Reduktion (bei vielen Features wie ST-Embedding-Pipelines mit 1.920 d). Die Wahl der Reduktionsmethode (PCA vs. Feature-Selection vs. Subsampling) hat starken Einfluss auf die Performance.

---

## 📊 Constraints auf einen Blick

| Modell | Numerisch | Text | Kategorisch | NaN | Max. Features | Max. Samples | Label nötig |
|---|---|---|---|---|---|---|---|
| **TabPFN 2.5 (Embed)** | ✅ nativ | ❌ | ⚠️ als Int | ⚠️ | ~500 (praktisch) | ~10.000 | ✅ (Dummy-Label reicht) |
| **TabPFN-Unsupervised** | ✅ nativ | ❌ | ⚠️ als Int | ⚠️ | **~50 (praktisch)** | **~3.000 (praktisch)** | ❌ |
| **ConTextTab** | ✅ nativ | ✅ nativ | ✅ nativ | ✅ nativ | **≤ 500 (hart)** | ≤ 8.192/Aufruf | ⚠️ Dummy |
| **AnoLLM** | ⚠️ als Token | ✅ nativ | ✅ als Token | ✅ als Token | ⚠️ Token-Budget | unbegrenzt | ❌ (Inlier-only Training) |
| **FoMo-OD** | ✅ nativ | ❌ | ❌ | ❌ | **= 100 (hart)** | ~5.000 | ❌ |

**Legende:** ✅ = nativ unterstützt | ⚠️ = mit Einschränkung | ❌ = nicht unterstützt

---

## Implikationen für die Pipeline-Wahl

| Datenlage | Geeignete TFMs |
|---|---|
| Wenige Features (≤ 50), keine Texte | TabPFN-Unsupervised, FoMo-OD (mit Padding auf 100) |
| Viele Features (>> 100), keine Texte | TabPFN 2.5 (Embed-Pfad nach PCA), FoMo-OD nach PCA(100) |
| Texte zentral wichtig | AnoLLM (semi-supervised), ConTextTab (für SHAP / Klassifikation) |
| Heterogene Mischung (num + text + cat + NaN) | ConTextTab (nativ), AnoLLM (über Serialisierung) |
| Pure Zero-Shot ohne Training | FoMo-OD, TabPFN-Unsupervised |