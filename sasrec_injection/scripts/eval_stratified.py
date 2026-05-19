"""Stratified full-rank evaluation: per-bucket HR@10 / NDCG@10.

Why this exists
---------------

The aggregate full-rank numbers (e.g. A1 HR@10 = 0.0776 vs A0 0.0574)
tell us *that* LLM-init helps, not *where*. Three stories are
empirically distinguishable:

* If the gain concentrates in items at the **right edge of 5-core**
  (5-10 interactions), the bandit-prior framing is right: items with
  the least training signal benefit most from informative init.
* If it concentrates in **head items** (frequent training items),
  LLM init is providing better discriminative geometry on items the
  model could already rank well.
* If it's **uniform across buckets**, the convergence-speed framing
  is the right one: LLM init is a general optimization win, not a
  tail-specific one.

We bucket on two axes:

* **Item frequency** (by training-set interaction count of the test
  target item):

    * Head: top 20% of items by frequency.
    * Torso: middle 60%.
    * Tail: bottom 20%. **Comparable to LLM-ESR's "tail" bucket.**
    * Edge-tail (overlay): items with ≤ 10 training interactions
      (the right edge of the 5-core threshold). Disjoint from the
      head; significantly overlaps the tail.

* **User history length** (training-set sequence length, padding
  excluded):

    * Short: 5–9 interactions.
    * Medium: 10–29.
    * Long: ≥ 30.

For each (item-bucket, user-bucket) cell we report HR@10 / NDCG@10
across the test set restricted to that cell. We also report the
marginals over each axis separately.

Output
------

Per checkpoint::

    sasrec_injection/outputs/<run_dir>/seed_<S>/stratified_fullrank.json

Aggregate (mean ± std across seeds, written next to the existing
``aggregate_fullrank_results.json``)::

    sasrec_injection/outputs/<run_dir>/aggregate_stratified_fullrank.json

Run on existing checkpoints — no retraining. ~3 min/seed on M2 Max.

Usage
-----

::

    uv run python sasrec_injection/scripts/eval_stratified.py \\
        --config sasrec_injection/configs/p1_video_games.yaml --seeds 42 7 18

    uv run python sasrec_injection/scripts/eval_stratified.py \\
        --config sasrec_injection/configs/ablations/A1_llm_init.yaml --seeds 42 7 18

    uv run python sasrec_injection/scripts/eval_stratified.py \\
        --config sasrec_injection/configs/sasrec_injection_video_games.yaml --seeds 42 7 18
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasrec_injection.config import SASRecConfig
from sasrec_injection.data.dataset import FullRankEvalDataset
from sasrec_injection.data.loaders import build_user_sequences, load_ratings, preprocess
from sasrec_injection.data.splitting import leave_one_out_split
from sasrec_injection.models.sasrec import SASRec


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def item_frequency_buckets(
    train_seqs: dict[int, list[int]], num_items: int
) -> tuple[dict[int, str], dict[int, int]]:
    """Bucket items by training-set interaction count.

    Buckets:
        * "head"  — top 20% by frequency.
        * "torso" — middle 60%.
        * "tail"  — bottom 20%.

    The percentile cuts are computed over the *non-zero-frequency*
    items so a 5-core dataset's items (which always have ≥ 5
    interactions) are partitioned cleanly. Items that are never seen
    at training time (rare; only happens with held-out splits) get
    bucketed as "tail" by convention.

    Returns:
        ``(item_to_bucket, item_to_count)`` — both keyed by the
        remapped int item id (1..num_items).
    """
    counts = Counter()
    for seq in train_seqs.values():
        counts.update(seq)
    item_to_count = {iid: counts.get(iid, 0) for iid in range(1, num_items + 1)}

    # Sort items by frequency. Within a frequency tier we sort by id
    # for determinism — the percentile cuts are robust to within-tier
    # ordering anyway.
    sorted_items = sorted(
        range(1, num_items + 1),
        key=lambda i: (item_to_count[i], i),
        reverse=False,  # ascending by count
    )
    n = len(sorted_items)
    tail_cut = int(n * 0.20)
    head_cut = int(n * 0.80)
    item_to_bucket: dict[int, str] = {}
    for rank, iid in enumerate(sorted_items):
        if rank < tail_cut:
            item_to_bucket[iid] = "tail"
        elif rank < head_cut:
            item_to_bucket[iid] = "torso"
        else:
            item_to_bucket[iid] = "head"
    return item_to_bucket, item_to_count


def edge_tail_set(item_to_count: dict[int, int], threshold: int = 10) -> set[int]:
    """Items with ≤ ``threshold`` training interactions.

    On 5-core data this is "items at the right edge of the 5-core
    filter," typically 25–40% of the catalog on Amazon Video_Games.
    Overlaps the percentile-tail bucket but is defined by an absolute
    cutoff rather than a relative one — both are useful framings.
    """
    return {iid for iid, c in item_to_count.items() if c <= threshold}


def user_history_buckets(
    train_seqs: dict[int, list[int]],
) -> dict[int, str]:
    """Bucket users by training-history length.

    Buckets:
        * "short" — 5–9 interactions (right at the 5-core lower edge).
        * "medium" — 10–29.
        * "long" — ≥ 30.
    """
    out: dict[int, str] = {}
    for uid, seq in train_seqs.items():
        n = len(seq)
        if n < 10:
            out[uid] = "short"
        elif n < 30:
            out[uid] = "medium"
        else:
            out[uid] = "long"
    return out


# ---------------------------------------------------------------------------
# Stratified metric computation
# ---------------------------------------------------------------------------


def hr_at_k(rankings: torch.Tensor, k: int) -> float:
    return (rankings < k).float().mean().item() if len(rankings) else 0.0


def ndcg_at_k(rankings: torch.Tensor, k: int) -> float:
    if not len(rankings):
        return 0.0
    hits = (rankings < k)
    # rankings here are 0-indexed positions of the GT.
    log2_ranks = torch.log2(rankings.float() + 2.0)
    return (hits.float() / log2_ranks).mean().item()


def stratified_metrics(
    rankings: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    item_buckets: dict[int, str],
    edge_tail: set[int],
    user_buckets: dict[int, str],
) -> dict[str, float]:
    """Compute HR@10 / NDCG@10 stratified by item-bucket × user-bucket.

    Args:
        rankings: ``(N,)`` 0-indexed rank of the GT against the full
            (post-mask) catalog.
        user_ids: ``(N,)`` test-user ids matching ``rankings``.
        item_ids: ``(N,)`` test target item ids matching ``rankings``.
        item_buckets: ``iid -> 'head'|'torso'|'tail'``.
        edge_tail: set of edge-tail item ids (≤ 10 interactions).
        user_buckets: ``uid -> 'short'|'medium'|'long'``.

    Returns:
        Flat dict like
        ``{"item_tail_hr@10": ..., "user_short_ndcg@10": ...,
           "item_tail_user_short_hr@10": ..., "edge_tail_hr@10": ...}``.
    """
    out: dict[str, float] = {}
    item_b = [item_buckets[int(i)] for i in item_ids.tolist()]
    user_b = [user_buckets[int(u)] for u in user_ids.tolist()]
    edge_mask = torch.tensor(
        [int(i) in edge_tail for i in item_ids.tolist()], dtype=torch.bool
    )

    item_t = torch.tensor([{"head": 0, "torso": 1, "tail": 2}[b] for b in item_b])
    user_t = torch.tensor(
        [{"short": 0, "medium": 1, "long": 2}[b] for b in user_b]
    )

    # Marginal: item-axis only.
    for label, code in [("head", 0), ("torso", 1), ("tail", 2)]:
        sel = rankings[item_t == code]
        n = len(sel)
        out[f"item_{label}_n"] = float(n)
        out[f"item_{label}_hr@10"] = hr_at_k(sel, 10)
        out[f"item_{label}_ndcg@10"] = ndcg_at_k(sel, 10)

    # Edge-tail overlay (≤10 interactions).
    sel = rankings[edge_mask]
    out["edge_tail_n"] = float(len(sel))
    out["edge_tail_hr@10"] = hr_at_k(sel, 10)
    out["edge_tail_ndcg@10"] = ndcg_at_k(sel, 10)

    # Marginal: user-axis only.
    for label, code in [("short", 0), ("medium", 1), ("long", 2)]:
        sel = rankings[user_t == code]
        n = len(sel)
        out[f"user_{label}_n"] = float(n)
        out[f"user_{label}_hr@10"] = hr_at_k(sel, 10)
        out[f"user_{label}_ndcg@10"] = ndcg_at_k(sel, 10)

    # Joint cells (3x3 = 9 combinations).
    for ilab, ic in [("head", 0), ("torso", 1), ("tail", 2)]:
        for ulab, uc in [("short", 0), ("medium", 1), ("long", 2)]:
            mask = (item_t == ic) & (user_t == uc)
            sel = rankings[mask]
            out[f"item_{ilab}_user_{ulab}_n"] = float(len(sel))
            out[f"item_{ilab}_user_{ulab}_hr@10"] = hr_at_k(sel, 10)
            out[f"item_{ilab}_user_{ulab}_ndcg@10"] = ndcg_at_k(sel, 10)

    # Overall (sanity check — should match aggregate_fullrank's HR@10).
    out["overall_n"] = float(len(rankings))
    out["overall_hr@10"] = hr_at_k(rankings, 10)
    out["overall_ndcg@10"] = ndcg_at_k(rankings, 10)
    return out


# ---------------------------------------------------------------------------
# Full-rank ranking computation, returned per user (vs the aggregate
# in evaluation/metrics.py which loses user/item ids).
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_rankings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    test_user_ids: list[int],
    test_target_items: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the GT rank for every test user, preserving (user, item) ids."""
    model.eval()
    all_rankings: list[torch.Tensor] = []
    for seq, gt_items, exclude_masks in loader:
        seq = seq.to(device)
        exclude_masks = exclude_masks.to(device)
        scores = model.score_all_items(seq)
        scores[exclude_masks] = float("-inf")
        gt_indices = (gt_items - 1).to(device)
        gt_scores = scores[torch.arange(len(gt_indices), device=device), gt_indices]
        rankings = (scores > gt_scores.unsqueeze(1)).sum(dim=1)
        all_rankings.append(rankings.cpu())
    rankings_cat = torch.cat(all_rankings, dim=0)
    user_ids_cat = torch.tensor(test_user_ids, dtype=torch.long)
    item_ids_cat = torch.tensor(test_target_items, dtype=torch.long)
    return rankings_cat, user_ids_cat, item_ids_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="sasrec_injection/configs/p1_video_games.yaml",
        help="P1-style config whose output_dir contains seed_<S>/best_model.pt.",
    )
    parser.add_argument("--base-config", default="sasrec_injection/configs/base.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = SASRecConfig.from_yaml(args.config, args.base_config)
    seeds = args.seeds or cfg.seeds
    device = torch.device(cfg.device)

    log(f"Loading {cfg.dataset_name}...")
    ratings = load_ratings("data", dataset=cfg.dataset_name)
    df, _, item_map = preprocess(ratings, min_interactions=cfg.min_interactions)
    num_items = len(item_map)
    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"Users: {split.num_users}, Items: {split.num_items}")

    # Compute the bucketing once (deterministic given the data).
    item_buckets, item_counts = item_frequency_buckets(
        split.train_seqs, split.num_items
    )
    edge_tail = edge_tail_set(item_counts, threshold=10)
    user_buckets = user_history_buckets(split.train_seqs)
    n_per_item = Counter(item_buckets.values())
    n_per_user = Counter(user_buckets.values())
    n_edge = len(edge_tail)
    log(
        f"Items by bucket — head: {n_per_item['head']}, "
        f"torso: {n_per_item['torso']}, tail: {n_per_item['tail']}; "
        f"edge_tail (≤10): {n_edge}"
    )
    log(
        f"Users by bucket — short: {n_per_user['short']}, "
        f"medium: {n_per_user['medium']}, long: {n_per_user['long']}"
    )

    # Build the same full-rank test loader as eval_fullrank.py.
    test_seqs = {
        uid: split.train_seqs[uid] + [split.val_targets[uid]]
        for uid in split.test_targets
    }
    exclude_items = {
        uid: set(split.train_seqs[uid]) | {split.val_targets[uid]}
        for uid in split.test_targets
    }
    test_dataset = FullRankEvalDataset(
        user_seqs=test_seqs,
        targets=split.test_targets,
        num_items=num_items,
        exclude_items=exclude_items,
        max_seq_len=cfg.model.max_seq_len,
    )
    # FullRankEvalDataset.__init__ sorts targets.keys() internally;
    # use its canonical ordering instead of relying on dict insertion.
    ordered_uids = test_dataset.user_ids
    ordered_targets = [split.test_targets[uid] for uid in ordered_uids]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    output_dir = Path(cfg.output_dir)
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        ckpt_path = output_dir / f"seed_{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            log(f"[seed {seed}] no checkpoint at {ckpt_path}, skipping")
            continue

        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        state_dict = (
            ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        )
        has_fusion = any(k.startswith("llm_emb") for k in state_dict)
        if has_fusion:
            import yaml as _yaml
            raw_cfg = _yaml.safe_load(open(args.config))
            llm_path = raw_cfg.get("llm_embeddings_path",
                                   raw_cfg.get("align", {}).get("embeddings_path", ""))
            llm_dim = raw_cfg.get("align", {}).get("llm_dim", 1024)
            llm_emb = torch.load(llm_path, map_location="cpu", weights_only=True)
            model = SASRec(
                num_items=num_items, embed_dim=cfg.model.embed_dim,
                num_blocks=cfg.model.num_blocks, num_heads=cfg.model.num_heads,
                max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
                llm_item_emb=llm_emb, llm_dim=llm_dim, fusion_mode="add",
            ).to(device)
        else:
            model = SASRec(
                num_items=num_items, embed_dim=cfg.model.embed_dim,
                num_blocks=cfg.model.num_blocks, num_heads=cfg.model.num_heads,
                max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout,
            ).to(device)
        model.load_state_dict(state_dict)

        rankings, uids, iids = collect_rankings(
            model, test_loader, device, ordered_uids, ordered_targets
        )
        metrics = stratified_metrics(
            rankings, uids, iids, item_buckets, edge_tail, user_buckets
        )

        seed_out = output_dir / f"seed_{seed}" / "stratified_fullrank.json"
        with open(seed_out, "w") as f:
            json.dump(metrics, f, indent=2)
        per_seed.append(metrics)

        log(f"\n[seed {seed}] item-axis HR@10:")
        for lab in ["head", "torso", "tail"]:
            log(f"  item_{lab}: hr@10={metrics[f'item_{lab}_hr@10']:.4f}  "
                f"ndcg@10={metrics[f'item_{lab}_ndcg@10']:.4f}  "
                f"n={int(metrics[f'item_{lab}_n'])}")
        log(f"  edge_tail (≤10 interactions): "
            f"hr@10={metrics['edge_tail_hr@10']:.4f}  "
            f"ndcg@10={metrics['edge_tail_ndcg@10']:.4f}  "
            f"n={int(metrics['edge_tail_n'])}")

    if not per_seed:
        log("no checkpoints found.")
        return

    # Aggregate across seeds: mean ± std on every numeric metric except
    # the ``_n`` counts (which are deterministic given the data).
    bar = "=" * 60
    log(f"\n{bar}\nAGGREGATE STRATIFIED RESULTS\n{bar}")
    keys = sorted(k for k in per_seed[0] if not k.endswith("_n"))
    summary: dict[str, dict] = {}
    for key in keys:
        values = [m[key] for m in per_seed]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
    # Carry through the deterministic counts as-is.
    for key in [k for k in per_seed[0] if k.endswith("_n")]:
        summary[key] = {"value": per_seed[0][key]}

    for lab in ["head", "torso", "tail"]:
        m = summary[f"item_{lab}_hr@10"]
        log(f"item_{lab}: hr@10 = {m['mean']:.4f} ± {m['std']:.4f}")
    e = summary["edge_tail_hr@10"]
    log(f"edge_tail: hr@10 = {e['mean']:.4f} ± {e['std']:.4f}")
    for lab in ["short", "medium", "long"]:
        m = summary[f"user_{lab}_hr@10"]
        log(f"user_{lab}: hr@10 = {m['mean']:.4f} ± {m['std']:.4f}")
    o = summary["overall_hr@10"]
    log(f"overall: hr@10 = {o['mean']:.4f} ± {o['std']:.4f}  (sanity check)")

    out_path = output_dir / "aggregate_stratified_fullrank.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
