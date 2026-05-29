# CLAUDE.md – Coding Guidelines

## Kontext
Lies vor dem Start folgende Dateien:
- `readme.md`
- `descriptions/data_description`
- `descriptions/tfm_modell_requirements`

---

## Notebook-Stil

- Jeder Abschnitt beginnt mit einer **Markdown-Zelle**: Titel + 1–2 Stichpunkte auf Deutsch.
- Alle Imports kommen **an den Anfang** des Notebooks.
- Kein Helper-Funktionen – Code direkt und linear wie typisch für Notebooks.

## Code

- So **kurz und einfach** wie möglich.
- Nur **wenige Kommentare**, auf Englisch.
- Kein validierender, absichernder oder unnötiger Overhead-Code.

## Ausgabe

Pro Modell werden **nur** diese Metriken ausgegeben:
- Average Precision (AP)
- AUC-ROC
- Classification Report

## Limitationen
1. Data Leakage in Experiment 2 (Enhanced)

TabPFN-Embeddings werden auf dem Label trainiert → Features tragen Label-Information.
Überschätzt die Enhanced-Performance systematisch.
Bewusst als semi-supervised eingeordnet, daher nicht direkt mit den unsupervised Varianten vergleichbar.

2. In-Class Classification bei Airbnb Paris

Kein natürliches Label → künstliche Proxy-Definition über review_score_rating.
Bewertungen 3–5 ausgeschlossen → Übergangsbereich fehlt, Problem künstlich leichter.
Bewertungen subjektiv, nicht zwingend = echte Anomalie.
Nur eingeschränkt auf reale Szenarien übertragbar.
