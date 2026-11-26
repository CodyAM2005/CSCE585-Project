#!/usr/bin/env python
"""Combine bench_end_removal.py JSONs into one table (markdown + CSV). Speedups are vs k=0 `full`."""

import argparse
import csv
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.results_dir, "end_removal_k*.json")))]
    runs.sort(key=lambda r: r["k"])
    base = next((r["results"]["full"] for r in runs if r["k"] == 0), None)

    rows = []
    for r in runs:
        eq = r.get("equivalence_full_ignore_vs_truncated")
        for mode, res in r["results"].items():
            e2e, pa, llm = res["e2e"], res["predict_action"], res["llm_forward"]
            row = {
                "k": r["k"],
                "mode": mode,
                "layers_run": res["layers_executed"],
                "e2e_mean_ms": e2e["mean_ms"],
                "e2e_median_ms": e2e["median_ms"],
                "e2e_p90_ms": e2e["p90_ms"],
                "e2e_std_ms": e2e["std_ms"],
                "predict_action_median_ms": pa.get("median_ms"),
                "llm_median_ms": llm.get("median_ms"),
                "e2e_speedup_vs_k0": (base["e2e"]["median_ms"] / e2e["median_ms"]) if base else None,
                "llm_speedup_vs_k0": (base["llm_forward"]["median_ms"] / llm["median_ms"])
                if base and llm.get("n") and base["llm_forward"].get("n") else None,
                "equivalence": ("PASS" if eq["pass"] else "FAIL") if (eq and mode == "truncated") else "",
                "gpu": r["gpu"],
            }
            rows.append(row)

    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    cols = ["k", "mode", "layers_run", "e2e_mean_ms", "e2e_median_ms", "e2e_p90_ms", "predict_action_median_ms",
            "llm_median_ms", "e2e_speedup_vs_k0", "llm_speedup_vs_k0", "equivalence"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in rows]
    table = "\n".join(lines)
    print(f"GPU: {rows[0]['gpu'] if rows else '?'}\n")
    print(table)

    with open(os.path.join(args.results_dir, "end_removal_summary.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.results_dir, "end_removal_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
