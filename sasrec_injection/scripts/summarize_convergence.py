"""Convergence speed summary across all trained methods.

Reads train_summary.json from every seed directory and emits a
convergence table: method × dataset → mean best epoch, mean wall
time (seconds), total epochs run. Useful for the paper's efficiency
comparison (e.g. A1 converges in 16 epochs vs SASRec's 54).

Usage:
    uv run python sasrec_injection/scripts/summarize_convergence.py
"""

from __future__ import annotations
import json
from pathlib import Path

OUTPUTS = Path("sasrec_injection/outputs")

RUNS = {
    "SASRec (VG)":        OUTPUTS / "p1_video_games",
    "SAILS (VG)":         OUTPUTS / "sasrec_injection_video_games",
    "LLM-Init (VG)":      OUTPUTS / "ablations/A1_llm_init",
    "SAILS+LLM-Init (VG)":OUTPUTS / "ablations/A6_sails_llm_init",
    "Freq-Weighted (VG)": OUTPUTS / "ablations/A7_freq_weighted",
    "Init+FW (VG)":       OUTPUTS / "ablations/A8_init_freq_weighted",
    "Input-Fusion (VG)":  OUTPUTS / "ablations/A2_input_fusion",
    "Hidden-Distill (VG)":OUTPUTS / "ablations/A3_hidden_distill",
    "Seq-InfoNCE (VG)":   OUTPUTS / "ablations/A4_seq_infonce",
    "SASRec (Yelp)":      OUTPUTS / "p1_yelp",
    "SAILS (Yelp)":       OUTPUTS / "sasrec_injection_yelp",
    "Freq-Weighted (Yelp)":OUTPUTS / "ablations/A7_yelp",
    "Init+FW (Yelp)":     OUTPUTS / "ablations/A8_yelp",
    "SASRec (Beauty18)":  OUTPUTS / "p1_beauty2018",
    "SAILS (Beauty18)":   OUTPUTS / "sasrec_injection_beauty2018",
    "LLM-Init (Beauty18)":OUTPUTS / "ablations/A1_beauty2018",
    "Freq-Weighted (B18)":OUTPUTS / "ablations/A7_beauty2018",
    "Init+FW (Beauty18)": OUTPUTS / "ablations/A8_beauty2018",
    "SASRec (Sports18)":  OUTPUTS / "p1_sports2018",
    "SAILS (Sports18)":   OUTPUTS / "sasrec_injection_sports2018",
    "Init+FW (Sports18)": OUTPUTS / "ablations/A8_sports2018",
}


def load_summaries(run_dir: Path) -> list[dict]:
    summaries = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        f = seed_dir / "train_summary.json"
        if f.exists():
            summaries.append(json.load(open(f)))
    # Also check lambda subdirs (SAILS λ-sweep layout)
    for lam_dir in sorted(run_dir.glob("lambda_*")):
        for seed_dir in sorted(lam_dir.glob("seed_*")):
            f = seed_dir / "train_summary.json"
            if f.exists():
                summaries.append(json.load(open(f)))
    return summaries


print(f"\n{'Method':<26} {'Seeds':>5} {'Best epoch':>10} {'Total epochs':>12} "
      f"{'Wall time (min)':>16}")
print("-" * 76)

for label, run_dir in RUNS.items():
    if not run_dir.exists():
        continue
    summaries = load_summaries(run_dir)
    if not summaries:
        continue
    best_epochs = [s["best_epoch"] for s in summaries]
    total_epochs = [s["total_epochs"] for s in summaries]
    times_s = [s.get("total_time_s", 0) for s in summaries]
    n = len(summaries)
    mean_best = sum(best_epochs) / n
    mean_total = sum(total_epochs) / n
    mean_min = sum(times_s) / n / 60
    print(f"{label:<26} {n:>5} {mean_best:>10.1f} {mean_total:>12.1f} "
          f"{mean_min:>16.1f}")
