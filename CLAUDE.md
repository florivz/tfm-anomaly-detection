# CLAUDE.md – Coding Guidelines

## Kontext
Lies vor dem Start folgende Dateien:
- `readme.md`
- `descriptions/data_description`
- `tfm_modell_requirements`

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
