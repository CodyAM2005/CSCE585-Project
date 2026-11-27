#!/usr/bin/env bash
# Project entry point: runs the end-of-backbone layer-removal latency benchmark
# (compares k=0,1,3 — baseline vs. dropping the final 1/3 decoder layers of the
# VLA-Adapter LLM backbone). See latency_bench/bench_end_removal.py for details,
# and README.md's "How to Run" section for full setup instructions.
#
# On a Slurm cluster (what this project runs on today), this submits the batch
# job in latency_bench/run_end_removal_latency.sh, which benchmarks all three
# k values back-to-back on one GPU/node. Edit that script's ROOT and CKPT_K*
# variables first to point at your own checkpoint paths.
#
# After the job finishes, run `python result.py` to summarize and plot the results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v sbatch >/dev/null 2>&1; then
  sbatch latency_bench/run_end_removal_latency.sh
else
  echo "No Slurm (sbatch) found on this machine." >&2
  echo "Run the benchmark directly instead, once per k value, e.g.:" >&2
  echo "  python latency_bench/bench_end_removal.py \\" >&2
  echo "    --repo_dir EfficientVLA --pretrained_checkpoint <ckpt_dir> \\" >&2
  echo "    --num_end_layers_drop 0 --output_json latency_bench/results/k0.json" >&2
  exit 1
fi
