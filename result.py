#!/usr/bin/env python
"""
result.py — project entry point for summarizing and plotting the latency benchmark.

Wraps latency_bench/summarize_end_removal.py (which produces a markdown/CSV table
per run) and additionally renders a bar chart of end-to-end latency by k/mode.
All output (the raw per-k JSON, the summary table, and the chart) is copied into
data/<run-name>/ so result artifacts live at the project root, per the course
template's "data directory" requirement.

Usage:
    python result.py [--results_dir latency_bench/results/<run>]

With no --results_dir, the most recently modified run under latency_bench/results/
is used.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LATENCY_BENCH = os.path.join(ROOT, "latency_bench")
DATA_DIR = os.path.join(ROOT, "data")


def find_latest_results_dir():
    candidates = [
        d for d in glob.glob(os.path.join(LATENCY_BENCH, "results", "*")) if os.path.isdir(d)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def plot_latency(csv_path, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib to enable it).", file=sys.stderr)
        return False

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows found in {csv_path}; skipping plot.", file=sys.stderr)
        return False

    labels = [f"k={r['k']}\n{r['mode']}" for r in rows]
    medians = [float(r["e2e_median_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, medians, color="#4C72B0")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("End-to-end median latency (ms)")
    ax.set_title("VLA-Adapter inference latency vs. layers dropped (k)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=None,
                   help="latency_bench/results/<run> folder to summarize. "
                        "Defaults to the most recently modified one.")
    args = p.parse_args()

    results_dir = args.results_dir or find_latest_results_dir()
    if results_dir is None:
        sys.exit(
            "No results found under latency_bench/results/. Run run.sh (or "
            "latency_bench/bench_end_removal.py directly) first."
        )
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        sys.exit(f"--results_dir does not exist: {results_dir}")

    # Produce latency_bench's own markdown/CSV table inside results_dir.
    subprocess.run(
        [sys.executable, os.path.join(LATENCY_BENCH, "summarize_end_removal.py"),
         "--results_dir", results_dir],
        check=True,
    )

    # Copy everything (raw per-k JSON + the summary table) into data/<run-name>/.
    run_name = os.path.basename(results_dir)
    out_dir = os.path.join(DATA_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    for src in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    csv_path = os.path.join(out_dir, "end_removal_summary.csv")
    jpg_path = os.path.join(out_dir, "end_removal_latency.jpg")
    if os.path.exists(csv_path) and plot_latency(csv_path, jpg_path):
        print(f"Wrote {jpg_path}")

    print(f"Result artifacts in {out_dir}")


if __name__ == "__main__":
    main()
