#!/usr/bin/env python3
"""
run_experiments.py — orchestriert Exp 1-4 (beide Datensaetze) auf der VM-vGPU.

Laeuft LOKAL auf dem Mac und ruft jedes Notebook per SSH mit `uv run` auf der VM auf
(wie deine Notebooks: lokal gestartet, auf der VM gerechnet). Nach jedem Notebook werden
die vGPU-Prozesse gekillt, damit der Speicher frei wird und der naechste Lauf nicht OOM
geht. Jedes Notebook wird ROUNDS-mal ausgefuehrt (Default 3) -> n_runs = 3 pro Modell.
Vor dem Lauf werden die lokalen Notebooks auf die VM gepusht (kein veralteter Code);
am Ende wird mlruns nach lokal geholt.

Nutzung:
  python3 run_experiments.py --dry-run    # nur anzeigen: wie viele Runs pro Modell (liest VM-mlruns)
  python3 run_experiments.py              # Notebooks pushen + 3x laufen lassen + GPU-Kill + mlruns-Pull
  python3 run_experiments.py --rounds 1   # nur 1 Runde
  python3 run_experiments.py --sync       # nur lokale Notebooks auf die VM pushen
  python3 run_experiments.py --pull       # nur mlruns von der VM nach lokal holen

Tipp: lange Laufzeit -> in einem lokalen tmux starten, damit ein SSH-Abbruch nichts kostet.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ----------------------------- Konfiguration ------------------------------
VM = "debian@185.113.124.164"
REPO = "/home/debian/TFM_master_thesis"          # Repo-Pfad auf der VM
UV = "/home/debian/.local/bin/uv"                # uv-Binary auf der VM
ROUNDS_DEFAULT = 3                               # n_runs pro Modell
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10", "-o", "ConnectTimeout=15"]
LOCAL_REPO = Path(__file__).resolve().parent     # lokales Repo (= dieser Ordner)

# Notebooks (Exp 1-4, beide Datensaetze) als (verzeichnis, datei) in Laufreihenfolge.
NOTEBOOKS = [
    ("fake_job_notebooks/exp1", "baselines.ipynb"),
    ("fake_job_notebooks/exp1", "anollm.ipynb"),
    ("fake_job_notebooks/exp1", "tabpfn_fomo.ipynb"),
    ("fake_job_notebooks/exp2", "exp2.ipynb"),
    ("fake_job_notebooks/exp3", "exp3.ipynb"),
    ("fake_job_notebooks/exp4", "exp4.ipynb"),
    ("airbnb_notebooks/exp1", "baselines.ipynb"),
    ("airbnb_notebooks/exp1", "anollm.ipynb"),
    ("airbnb_notebooks/exp1", "tabpfn_fomo.ipynb"),
    ("airbnb_notebooks/exp2", "exp2.ipynb"),
    ("airbnb_notebooks/exp3", "exp3.ipynb"),
    ("airbnb_notebooks/exp4", "exp4.ipynb"),
]

# Zaehlt MLflow-Runs pro Experiment/Modell direkt aus dem mlruns-Verzeichnis (laeuft auf der VM).
DRY_SNIPPET = r"""
import os, sys
from collections import defaultdict
root, target = sys.argv[1], int(sys.argv[2])
if not os.path.isdir(root):
    print("kein mlruns-Verzeichnis auf der VM -> noch keine Runs."); sys.exit(0)
counts = defaultdict(int)
for exp_id in os.listdir(root):
    exp_dir = os.path.join(root, exp_id)
    meta = os.path.join(exp_dir, "meta.yaml")
    if not os.path.isfile(meta):
        continue
    name = exp_id
    for line in open(meta):
        s = line.strip()
        if s.startswith("name:"):
            name = s.split("name:", 1)[1].strip().strip("'\""); break
    if name == "Default":
        continue
    for run_id in os.listdir(exp_dir):
        run_dir = os.path.join(exp_dir, run_id)
        if not os.path.isdir(run_dir) or not os.path.isfile(os.path.join(run_dir, "meta.yaml")):
            continue
        rn = os.path.join(run_dir, "tags", "mlflow.runName")
        model = open(rn).read().strip() if os.path.isfile(rn) else run_id
        counts[(name, model)] += 1
if not counts:
    print("mlruns vorhanden, aber noch keine Runs geloggt."); sys.exit(0)
print("%-30s %-26s %4s" % ("experiment", "model", "runs"))
print("-" * 64)
total = missing = 0
for key in sorted(counts):
    name, model = key; c = counts[key]
    flag = "" if c >= target else "   <- %d fehlen" % (target - c)
    print("%-30s %-26s %4d%s" % (name, model, c, flag))
    total += c; missing += c < target
print("-" * 64)
print("Summe Runs: %d | Modelle unter Ziel: %d" % (total, missing))
"""

# ------------------------------- Helfer -----------------------------------

def ssh(remote_cmd, **kw):
    """Fuehrt einen Shell-Befehl auf der VM aus (stdout/stderr durchgereicht)."""
    return subprocess.run(["ssh", *SSH_OPTS, VM, remote_cmd], **kw)


def dry_run(rounds):
    print(f"=== Runs pro Modell (VM: {REPO}/mlruns, Ziel n_runs={rounds}) ===")
    ssh(f"python3 - '{REPO}/mlruns' '{rounds}'", input=DRY_SNIPPET, text=True)


def sync_notebooks():
    """Lokale Exp-1-4-Notebooks per tar-over-ssh auf die VM pushen (VM hat kein rsync)."""
    print("--- lokale Notebooks (Exp 1-4) auf die VM pushen ---")
    files = sorted(
        str(p.relative_to(LOCAL_REPO))
        for d in ("fake_job_notebooks", "airbnb_notebooks")
        for p in (LOCAL_REPO / d).rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )
    tar = subprocess.Popen(["tar", "czf", "-", "--no-xattrs", *files],
                           cwd=LOCAL_REPO, stdout=subprocess.PIPE)
    rx = subprocess.Popen(["ssh", *SSH_OPTS, VM, f"tar xzf - -C '{REPO}'"], stdin=tar.stdout)
    tar.stdout.close()
    rx.communicate()
    if tar.wait() == 0 and rx.returncode == 0:
        print(f"Notebooks synchronisiert ({len(files)} Dateien).")
    else:
        print("Notebook-Sync FEHLGESCHLAGEN -> Abbruch.")
        sys.exit(1)


def pull_mlruns():
    print("\n--- mlruns von der VM nach lokal holen ---")
    if ssh(f"test -d '{REPO}/mlruns'").returncode != 0:
        print("kein mlruns auf der VM vorhanden.")
        return
    tx = subprocess.Popen(["ssh", *SSH_OPTS, VM, f"tar czf - -C '{REPO}' mlruns"],
                          stdout=subprocess.PIPE)
    rx = subprocess.Popen(["tar", "xzf", "-", "-C", str(LOCAL_REPO)], stdin=tx.stdout)
    tx.stdout.close()
    rx.communicate()
    if tx.wait() == 0 and rx.returncode == 0:
        print(f"mlruns lokal aktualisiert: {LOCAL_REPO}/mlruns")
    else:
        print("mlruns-Pull fehlgeschlagen (Daten bleiben auf der VM).")


def free_gpu():
    """vGPU freigeben: Compute-Prozesse + uebrig gebliebene Jupyter-Kernel killen."""
    print("--- vGPU freigeben ---")
    ssh(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null "
        "| grep -Eo '^[0-9]+' | xargs -r -n1 kill -9 2>/dev/null; "
        "pkill -9 -f '[i]pykernel_launcher' 2>/dev/null; sleep 3; "
        "printf 'VRAM belegt: '; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def run_one(directory, file, rnd, rounds, failures):
    """Ein Notebook ausfuehren (executed copy -> run_logs/, Quelle bleibt unveraendert)."""
    base = file[:-len(".ipynb")]
    tag = f"{directory.replace('/', '_')}_{base}_r{rnd}"
    from datetime import datetime
    print("\n" + "=" * 66)
    print(f">>> [{datetime.now():%H:%M:%S}] Runde {rnd}/{rounds} | {directory}/{file}")
    print("=" * 66)
    remote = (
        f"mkdir -p '{REPO}/run_logs'; cd '{REPO}/{directory}' && '{UV}' run --no-sync "
        f"jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 "
        f"--ExecutePreprocessor.kernel_name=python3 --output-dir '{REPO}/run_logs' "
        f"--output '{tag}.ipynb' '{file}'"
    )
    rc = ssh(remote).returncode
    if rc != 0:
        print(f"!!! FEHLGESCHLAGEN (rc={rc}) -> Log: run_logs/{tag}.ipynb")
        failures.append(f"{directory}/{file} (Runde {rnd}, rc={rc})")
    else:
        print(f"<<< OK: {directory}/{file} (Runde {rnd})")
    free_gpu()


# --------------------------------- Main -----------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)  # lokale prints + SSH-Ausgabe in richtiger Reihenfolge
    ap = argparse.ArgumentParser(description="Exp 1-4 auf der VM-vGPU orchestrieren.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", "-n", action="store_true", help="nur Runs pro Modell anzeigen")
    g.add_argument("--sync", action="store_true", help="nur Notebooks auf die VM pushen")
    g.add_argument("--pull", action="store_true", help="nur mlruns von der VM holen")
    ap.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT, help="Anzahl Runden (n_runs, Default 3)")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.rounds)
        return
    if args.sync:
        sync_notebooks()
        return
    if args.pull:
        pull_mlruns()
        return

    # Voller Lauf
    print(f"Starte {args.rounds} Runden ueber {len(NOTEBOOKS)} Notebooks (Exp 1-4, beide Datensaetze).")
    print(f"VM: {VM} | Repo: {REPO}")
    print("Hinweis: keine PyCharm-Remote-Kernel parallel offen halten (werden mitgekillt).")
    sync_notebooks()
    failures = []
    for rnd in range(1, args.rounds + 1):
        for directory, file in NOTEBOOKS:
            run_one(directory, file, rnd, args.rounds, failures)

    print("\n" + "=" * 32 + " FERTIG " + "=" * 26)
    if failures:
        print(f"Fehlgeschlagen ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("Alle Notebooks ohne Fehler durchgelaufen.")
    print()
    dry_run(args.rounds)
    pull_mlruns()


if __name__ == "__main__":
    main()
