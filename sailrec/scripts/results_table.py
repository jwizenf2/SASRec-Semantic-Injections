"""Build presentation-ready results tables, split by eval protocol.

Reads ``aggregate_fullrank_results.json`` from each ``<method>_<dataset>``
output directory and emits one CSV per protocol:

* ``sailrec/outputs/full_rank_results.csv`` — full-catalog ranking.
* ``sailrec/outputs/sampled_results.csv``   — sampled@100 (LLM-ESR
  reproduction protocol).

Each CSV has rows ``(dataset, method)``; columns are the four metric
values first (``HR@10, HR@20, NDCG@10, NDCG@20``), then two gain
columns at the end (``HR@10 gain %, NDCG@10 gain %``). The K=20 gains
are intentionally omitted — HR@10 / NDCG@10 are the headline metrics
in the LLM-ESR / LLMEmb literature and adding K=20 gains just clutters
the table.

Cells are ``mean ± std`` (always with ±, even ``± 0.0000`` for
single-seed rows so adding more seeds later doesn't shift the layout).
Empty gain cells (the SASRec baseline row) render as ``-``.

Usage:
    uv run python sailrec/scripts/results_table.py
    uv run python sailrec/scripts/results_table.py --datasets video_games
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "sailrec" / "outputs"

# Metric value columns (in display order). Anything outside this list is
# dropped from the table (e.g. HR/NDCG@5 — too low-K to be informative —
# and recall@k, which duplicates hr@k under leave-one-out).
METRIC_COLUMNS = [
    ("HR", 10), ("HR", 20),
    ("NDCG", 10), ("NDCG", 20),
]

# Subset of METRIC_COLUMNS that gets a ``gain %`` column. K=10 only —
# K=20 gains are correlated and clutter the table.
GAIN_COLUMNS = [("HR", 10), ("NDCG", 10)]

# Placeholder for empty cells (baseline rows in the gain columns).
# Used in both the CSV and the displayed table; pandas reads it back as
# the string "-", not NaN, so notebook display stays clean.
EMPTY_CELL = "-"

# One CSV per protocol, in display order.
PROTOCOLS = [
    ("full_rank", "full_rank_results.csv"),
    ("sampled@100", "sampled_results.csv"),
]

# Method directory prefix → display label. Used for top-level
# ``sailrec/outputs/<prefix><dataset>/`` runs.
METHOD_PREFIXES = {
    "p1_": "SASRec",
    "sailrec_": "SAILRec",
}
BASELINE_LABEL = "SASRec"

# Ablation runs live under ``sailrec/outputs/ablations/<key>/`` and
# always run on Video_Games (per the locked ablation plan). Each entry
# maps the directory key to a presentation method label and the dataset
# slug used in DATASET_DISPLAY_NAMES.
ABLATION_METHODS: dict[str, tuple[str, str]] = {
    "A1_llm_init":            ("LLM-Init",         "video_games"),
    "A2_input_fusion":        ("Input-Fusion",     "video_games"),
    "A3_hidden_distill":      ("Hidden-Distill",   "video_games"),
    "A4_seq_infonce":         ("Seq-InfoNCE",      "video_games"),
    "A6_sails_llm_init":      ("SAILS+LLM-Init",   "video_games"),
    "A7_freq_weighted":       ("Freq-Weighted",    "video_games"),
    "A8_init_freq_weighted":  ("Init+Freq-Weighted", "video_games"),
    "A8_beauty2018":          ("Init+Freq-Weighted", "beauty2018"),
    "A8_sports2018":          ("Init+Freq-Weighted", "sports2018"),
    "A1_beauty2018":          ("LLM-Init",          "beauty2018"),
    "A7_beauty2018":          ("Freq-Weighted",     "beauty2018"),
    "A1_sports2018":          ("LLM-Init",          "sports2018"),
    "A7_sports2018":          ("Freq-Weighted",     "sports2018"),
    "A7_yelp":                ("Freq-Weighted",     "yelp"),
    "A8_yelp":                ("Init+Freq-Weighted", "yelp"),
}

# Output-directory dataset slug → human-readable label for the table.
# Keep these explicit (one entry per dataset we run on) rather than
# auto-generating, because the displayed name needs to disambiguate
# dataset *vintage* — e.g. Amazon Reviews has had multiple releases and
# our Video_Games numbers are not bit-comparable to the original SASRec
# paper's "Games" numbers, which used the 2014/2018 release. Calling out
# the release year up-front avoids that confusion.
DATASET_DISPLAY_NAMES = {
    "video_games":  "Amazon Reviews 2023 — Video_Games (5-core)",
    "yelp":         "Yelp Open Dataset (5-core)",
    "beauty2018":   "Amazon Reviews 2018 — Beauty (3-core, LLMEmb preprocessing)",
    "sports2018":   "Amazon Reviews 2018 — Sports (3-core, LLMEmb preprocessing)",
}


@dataclass
class MethodSpec:
    dataset: str
    method: str               # display label
    aggregate_path: Path


def discover_methods(datasets: list[str] | None = None) -> list[MethodSpec]:
    """Find all aggregate-results JSONs across baseline + ablation runs."""
    specs: list[MethodSpec] = []

    # 1. Top-level baseline / SAILRec runs: <prefix><dataset>/.
    for child in sorted(OUTPUTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        agg = child / "aggregate_fullrank_results.json"
        if not agg.exists():
            continue
        for prefix, label in METHOD_PREFIXES.items():
            if child.name.startswith(prefix):
                ds = child.name[len(prefix):]
                if datasets is not None and ds not in datasets:
                    continue
                specs.append(MethodSpec(ds, label, agg))
                break

    # 2. Ablation runs: ablations/<key>/. Each ABLATION_METHODS entry
    # maps the directory key to a method label + dataset slug.
    abl_root = OUTPUTS_DIR / "ablations"
    if abl_root.is_dir():
        for key, (label, ds) in ABLATION_METHODS.items():
            agg = abl_root / key / "aggregate_fullrank_results.json"
            if not agg.exists():
                continue
            if datasets is not None and ds not in datasets:
                continue
            specs.append(MethodSpec(ds, label, agg))

    return specs


def parse_metric_key(key: str) -> tuple[str, str, int] | None:
    """``hr@10`` → ``("sampled@100", "HR", 10)``; ``full_ndcg@20`` → full_rank."""
    if "@" not in key:
        return None
    name, k_str = key.split("@", 1)
    try:
        k = int(k_str)
    except ValueError:
        return None
    if name.startswith("full_"):
        protocol, metric_raw = "full_rank", name[len("full_"):]
    else:
        protocol, metric_raw = "sampled@100", name
    if metric_raw not in {"hr", "ndcg"}:
        return None
    return protocol, metric_raw.upper(), k


def format_cell(mean: float, std: float) -> str:
    """Always ``mean ± std`` for consistent formatting; ``± 0.0000`` flags
    a single-seed run, kept explicit so adding more seeds later doesn't
    visually shift the column layout."""
    return f"{mean:.4f} ± {std:.4f}"


def load_long(specs: list[MethodSpec]) -> pd.DataFrame:
    """Long-form intermediate: (dataset, method, protocol, metric, k) → mean/std."""
    rows: list[dict] = []
    for spec in specs:
        with spec.aggregate_path.open() as f:
            agg = json.load(f)
        for raw_key, payload in agg.items():
            parsed = parse_metric_key(raw_key)
            if parsed is None:
                continue
            protocol, metric, k = parsed
            rows.append({
                "dataset": spec.dataset,
                "method": spec.method,
                "protocol": protocol,
                "metric": metric,
                "k": k,
                "mean": float(payload["mean"]),
                "std": float(payload["std"]),
            })
    return pd.DataFrame(rows)


def build_protocol_table(long_df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """One protocol slice: rows = (dataset, method); columns = metric + gain."""
    sub = long_df[long_df["protocol"] == protocol]
    baseline_mean = (
        sub[sub["method"] == BASELINE_LABEL]
        .set_index(["dataset", "metric", "k"])["mean"]
    )

    keys = (
        sub[["dataset", "method"]]
        .drop_duplicates()
        .sort_values(["dataset", "method"])
    )
    out_rows: list[dict] = []
    for _, key in keys.iterrows():
        cell_src = sub[
            (sub["dataset"] == key["dataset"]) & (sub["method"] == key["method"])
        ].set_index(["metric", "k"])
        row: dict[str, object] = {
            "dataset": DATASET_DISPLAY_NAMES.get(key["dataset"], key["dataset"]),
            "method": key["method"],
        }
        # Metric value columns first.
        for metric, k in METRIC_COLUMNS:
            col = f"{metric}@{k}"
            if (metric, k) in cell_src.index:
                r = cell_src.loc[(metric, k)]
                row[col] = format_cell(r["mean"], r["std"])
            else:
                row[col] = EMPTY_CELL
        # Gain columns at the end (K=10 only).
        for metric, k in GAIN_COLUMNS:
            gain_col = f"{metric}@{k} gain %"
            if key["method"] == BASELINE_LABEL:
                row[gain_col] = EMPTY_CELL
                continue
            bkey = (key["dataset"], metric, k)
            if (metric, k) not in cell_src.index or bkey not in baseline_mean.index:
                row[gain_col] = EMPTY_CELL
                continue
            m = float(cell_src.loc[(metric, k), "mean"])
            b = float(baseline_mean.loc[bkey])
            row[gain_col] = EMPTY_CELL if b == 0 else f"{(m - b) / b * 100:+.1f}%"
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    # Build ordered label list, deduplicating while preserving first
    # occurrence order. Multiple datasets share the same display label
    # (e.g. "LLM-Init" appears for vg, beauty2018, sports2018) — pandas
    # CategoricalDtype requires unique categories, so we deduplicate.
    ablation_labels = [label for label, _ in ABLATION_METHODS.values()]
    seen: set[str] = set()
    ordered_labels: list[str] = []
    for lbl in [BASELINE_LABEL, *ablation_labels, "SAILRec"]:
        if lbl not in seen:
            ordered_labels.append(lbl)
            seen.add(lbl)
    method_order = pd.CategoricalDtype(ordered_labels, ordered=True)
    df["method"] = df["method"].astype(method_order)
    df = df.sort_values(["dataset", "method"]).reset_index(drop=True)
    df["method"] = df["method"].astype(str)

    ordered_cols = (
        ["dataset", "method"]
        + [f"{m}@{k}" for m, k in METRIC_COLUMNS]
        + [f"{m}@{k} gain %" for m, k in GAIN_COLUMNS]
    )
    return df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUTS_DIR,
        help="Directory to write the per-protocol CSVs into.",
    )
    args = parser.parse_args()

    specs = discover_methods(args.datasets)
    if not specs:
        raise SystemExit(
            f"No aggregate_fullrank_results.json found under {OUTPUTS_DIR}."
        )

    long_df = load_long(specs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for protocol, filename in PROTOCOLS:
        if not (long_df["protocol"] == protocol).any():
            continue
        df = build_protocol_table(long_df, protocol)
        out_path = args.out_dir / filename
        df.to_csv(out_path, index=False)
        print(f"\n=== {protocol} → {out_path.name} ===")
        print(df.to_string(index=False))
    print(
        "\nNote: '± 0.0000' means the row was run on a single seed. The "
        "± is always shown so the column layout is stable as more seeds "
        "are added."
    )


if __name__ == "__main__":
    main()
