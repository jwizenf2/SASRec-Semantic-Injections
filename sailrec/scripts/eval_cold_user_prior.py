"""Test-time cold-user prior — mix model scores with LLM-NN similarity.

Hypothesis
----------

Existing LLM-rec papers focus on cold *items* (LLM-ESR, LLMEmb both
title their long-tail item story). Cold *users* — users with short
training history — are underexplored. Our stratified eval shows
A1 (LLM-init) already lifts short-history users disproportionately
(+37% vs +8% for long-history users on full HR@10), suggesting the
LLM prior helps users whose encoder representation is poorly informed.

This script tests whether we can do better at test time, with no
retraining, by *explicitly* mixing the model's score with an LLM
nearest-neighbor score:

    final[i] = (1-α) * normalize(model_score[u, i])
             + α     * normalize(llm_sim[u, i])

where ``llm_sim[u, i] = cos(mean_pool(llm_emb[h] for h in history[u]),
                            llm_emb[i])``.

α=0 reduces to vanilla model scoring (matches eval_fullrank). α=1 is
pure LLM-similarity retrieval (no model). Intermediate α tests the
mixed regime.

Bandit framing
--------------

α can be read as a posterior-precision schedule: high α for
high-uncertainty (cold) users, low α for warm users. This connects
directly to Thompson-sampling-style warm-start bounds — see
``docs/SAILS_ablation_plan.md`` (forthcoming addition).

Score normalisation
-------------------

Scores from different methods have wildly different scales (model
dot-products span tens; LLM cosines are bounded in [-1, 1]). We
min-max normalise per-user within the *unmasked* candidate set
(items not already seen by that user), so the mix is in [0, 1]
across the comparable candidate pool. Items the user has seen get
``-inf`` in both before normalisation so they're never scored.

This is the standard score-fusion approach in TREC retrieval
benchmarks; rank fusion (Reciprocal Rank Fusion) gives nearly
identical results in our smoke tests.

Output
------

For each input config, writes per-α stratified metrics::

    sailrec/outputs/<run_dir>/cold_user_prior_results.json

Fields: ``alpha`` × ``user_bucket`` × ``hr@10|ndcg@10`` (mean over
seeds). Both the *aggregate* (across all users) and the per-bucket
(short/medium/long) HR@10 are reported.

Run on existing checkpoints — no retraining. ~5 min/seed/α on M2 Max.

Usage
-----

::

    uv run python sailrec/scripts/eval_cold_user_prior.py \\
        --config sailrec/configs/p1_video_games.yaml \\
        --llm-embeddings sailrec/outputs/embeddings/video_games.pt \\
        --seeds 42 7 18 \\
        --alphas 0 0.1 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.config import P1Config
from sailrec.data.dataset import FullRankEvalDataset
from sailrec.data.loaders import build_user_sequences, load_ratings, preprocess
from sailrec.data.splitting import leave_one_out_split
from sailrec.models.sasrec import SASRec


def log(msg: str) -> None:
    print(msg, flush=True)


def user_history_buckets(train_seqs: dict[int, list[int]]) -> dict[int, str]:
    """5-9 → short, 10-29 → medium, 30+ → long."""
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


def hr_at_k(rankings: torch.Tensor, k: int) -> float:
    return (rankings < k).float().mean().item() if len(rankings) else 0.0


def ndcg_at_k(rankings: torch.Tensor, k: int) -> float:
    if not len(rankings):
        return 0.0
    hits = (rankings < k).float()
    log2_ranks = torch.log2(rankings.float() + 2.0)
    return (hits / log2_ranks).mean().item()


def build_llm_catalog(llm_item_emb: torch.Tensor) -> torch.Tensor:
    """L2-normalise the LLM table and drop the padding row.

    Returns ``(num_items, llm_dim)`` — ~100 MB for VG, kept in RAM once.
    """
    normed = llm_item_emb / llm_item_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return normed[1:]                                    # (N, D)


def llm_scores_for_batch(
    user_histories: list[list[int]],
    llm_catalog: torch.Tensor,               # (N, D)  — pre-normalised
    llm_item_emb_normed: torch.Tensor,       # (N+1, D) — same norms, with padding
) -> torch.Tensor:
    """Compute cosine(mean_pool(history), catalog) for a batch of users.

    Returns ``(B, N)`` cosine scores. Small intermediate tensors only.
    """
    B = len(user_histories)
    D = llm_catalog.shape[1]
    user_vecs = torch.zeros(B, D)
    for i, hist in enumerate(user_histories):
        if hist:
            rows = llm_item_emb_normed[torch.tensor(hist, dtype=torch.long)]
            user_vecs[i] = rows.mean(dim=0)
    user_vecs = user_vecs / user_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return user_vecs @ llm_catalog.T                     # (B, N)


def minmax_normalize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row [0, 1] normalisation over the *unmasked* candidate set.

    Args:
        scores: (U, N) raw scores.
        mask: (U, N) bool, True at items to exclude (seen).

    Returns:
        (U, N) normalised scores. Excluded items get -inf so they
        never enter the rankings; non-excluded scores are min-maxed
        per row across the unmasked items only.
    """
    # Replace seen items with extreme values so they don't influence
    # min/max but we can still take a regular ``min`` / ``max``. Older
    # torch versions don't have ``torch.nanmin``; this trick avoids it.
    safe_for_min = scores.clone()
    safe_for_min[mask] = float("inf")
    safe_for_max = scores.clone()
    safe_for_max[mask] = float("-inf")
    row_min = safe_for_min.min(dim=1, keepdim=True).values
    row_max = safe_for_max.max(dim=1, keepdim=True).values
    # Avoid /0 if a row is all the same value.
    denom = (row_max - row_min).clamp(min=1e-8)
    norm = (scores - row_min) / denom
    norm[mask] = float("-inf")
    return norm


def rank_of_gt(
    scores: torch.Tensor,                  # (U, N) — higher = better
    gt_items: torch.Tensor,                # (U,)   — item ids in 1..N
) -> torch.Tensor:
    """0-indexed rank of GT against full-rank candidate scores.

    GT must NOT be in the exclude mask (otherwise its score is -inf).
    """
    u = scores.shape[0]
    gt_idx = (gt_items - 1).long()         # (U,)
    gt_scores = scores[torch.arange(u), gt_idx]
    # Items with strictly higher scores than GT get rank above it.
    return (scores > gt_scores.unsqueeze(1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument("--llm-embeddings", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        help="Mixing coefficients for LLM score (0=model only, 1=LLM only).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = P1Config.from_yaml(args.config, args.base_config)
    seeds = args.seeds or cfg.seeds
    device = torch.device(cfg.device)

    log(f"Loading {cfg.dataset_name}...")
    ratings = load_ratings("data", dataset=cfg.dataset_name)
    df, _, item_map = preprocess(ratings, min_interactions=cfg.min_interactions)
    num_items = len(item_map)
    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"Users: {split.num_users}, Items: {split.num_items}")

    user_buckets = user_history_buckets(split.train_seqs)
    n_per_user = Counter(user_buckets.values())
    log(
        f"Users by bucket — short: {n_per_user['short']}, "
        f"medium: {n_per_user['medium']}, long: {n_per_user['long']}"
    )

    # Test set: same construction as eval_fullrank.
    test_seqs = {
        uid: split.train_seqs[uid] + [split.val_targets[uid]]
        for uid in split.test_targets
    }
    exclude_items = {
        uid: set(split.train_seqs[uid]) | {split.val_targets[uid]}
        for uid in split.test_targets
    }
    ordered_uids = list(test_seqs.keys())
    ordered_targets = torch.tensor(
        [split.test_targets[uid] for uid in ordered_uids], dtype=torch.long
    )
    test_dataset = FullRankEvalDataset(
        user_seqs=test_seqs,
        targets=split.test_targets,
        num_items=num_items,
        exclude_items=exclude_items,
        max_seq_len=cfg.model.max_seq_len,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # LLM catalog: normalise once, hold in RAM (~100 MB). Never
    # materialise a full (U, N) score matrix — stream per batch.
    log(f"Loading LLM embeddings from {args.llm_embeddings}...")
    llm_item_emb = torch.load(
        args.llm_embeddings, weights_only=True, map_location="cpu"
    )
    if llm_item_emb.shape[0] != num_items + 1:
        raise ValueError(
            f"LLM rows {llm_item_emb.shape[0]} != num_items+1 ({num_items+1})"
        )
    llm_normed_full = llm_item_emb / llm_item_emb.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-8)                       # (N+1, D) — used for history lookup
    llm_catalog = llm_normed_full[1:]       # (N,   D) — item scoring matrix

    # User-history lookup in list form (aligned with loader iteration order).
    ordered_histories = [split.train_seqs[uid] for uid in ordered_uids]
    user_bucket_codes = torch.tensor(
        [{"short": 0, "medium": 1, "long": 2}[user_buckets[uid]]
         for uid in ordered_uids],
        dtype=torch.long,
    )

    output_dir = Path(cfg.output_dir)
    per_seed_alpha: dict[float, list[dict[str, float]]] = {a: [] for a in args.alphas}

    for seed in seeds:
        ckpt_path = output_dir / f"seed_{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            log(f"[seed {seed}] no checkpoint at {ckpt_path}, skipping")
            continue

        model = SASRec(
            num_items=num_items,
            embed_dim=cfg.model.embed_dim,
            num_blocks=cfg.model.num_blocks,
            num_heads=cfg.model.num_heads,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        ).to(device)
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(sd)
        model.eval()

        log(f"\n[seed {seed}] streaming batch-by-batch (no full U×N matrix)...")
        # Accumulate per-user rank for each α in a single forward pass.
        alpha_ranks: dict[float, list[torch.Tensor]] = {a: [] for a in args.alphas}

        user_offset = 0
        with torch.no_grad():
            for seq, gt_items, excl_masks in test_loader:
                B = seq.shape[0]
                seq = seq.to(device)

                # Model scores: (B, N), on device.
                m_scores = model.score_all_items(seq).cpu()   # (B, N)

                # LLM scores: (B, N), on CPU — tiny intermediate.
                batch_histories = ordered_histories[user_offset: user_offset + B]
                l_scores = llm_scores_for_batch(
                    batch_histories, llm_catalog, llm_normed_full
                )                                             # (B, N)

                # Mask (B, N): True at seen items.
                excl = excl_masks                             # already cpu bool

                m_norm = minmax_normalize(m_scores, excl)
                l_norm = minmax_normalize(l_scores, excl)

                for alpha in args.alphas:
                    mixed = (1 - alpha) * m_norm + alpha * l_norm
                    ranks = rank_of_gt(mixed, gt_items)       # (B,)
                    alpha_ranks[alpha].append(ranks)

                user_offset += B
                # Eagerly free large tensors so each batch's peak is ~2×(B×N).
                del m_scores, l_scores, m_norm, l_norm

        for alpha in args.alphas:
            ranks = torch.cat(alpha_ranks[alpha], dim=0)   # (U,)
            metrics: dict[str, float] = {
                "overall_hr@10": hr_at_k(ranks, 10),
                "overall_ndcg@10": ndcg_at_k(ranks, 10),
            }
            for label, code in [("short", 0), ("medium", 1), ("long", 2)]:
                sel = ranks[user_bucket_codes == code]
                metrics[f"user_{label}_hr@10"] = hr_at_k(sel, 10)
                metrics[f"user_{label}_ndcg@10"] = ndcg_at_k(sel, 10)
                metrics[f"user_{label}_n"] = float(len(sel))

            per_seed_alpha[alpha].append(metrics)
            log(
                f"  α={alpha:>4}  overall={metrics['overall_hr@10']:.4f}  "
                f"short={metrics['user_short_hr@10']:.4f}  "
                f"med={metrics['user_medium_hr@10']:.4f}  "
                f"long={metrics['user_long_hr@10']:.4f}"
            )

    # Aggregate across seeds: mean ± std per α per metric.
    bar = "=" * 70
    log(f"\n{bar}\nAGGREGATE across seeds — overall HR@10\n{bar}")
    summary: dict[str, dict] = {}
    for alpha in args.alphas:
        seeds_metrics = per_seed_alpha[alpha]
        if not seeds_metrics:
            continue
        agg: dict[str, dict] = {}
        for key in seeds_metrics[0]:
            if key.endswith("_n"):
                agg[key] = {"value": seeds_metrics[0][key]}
                continue
            values = [m[key] for m in seeds_metrics]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            agg[key] = {"mean": mean, "std": std, "values": values}
        summary[str(alpha)] = agg
        log(
            f"α={alpha:>4}  "
            f"overall {agg['overall_hr@10']['mean']:.4f} ± {agg['overall_hr@10']['std']:.4f}  "
            f"short {agg['user_short_hr@10']['mean']:.4f} ± {agg['user_short_hr@10']['std']:.4f}  "
            f"med   {agg['user_medium_hr@10']['mean']:.4f} ± {agg['user_medium_hr@10']['std']:.4f}  "
            f"long  {agg['user_long_hr@10']['mean']:.4f} ± {agg['user_long_hr@10']['std']:.4f}"
        )

    out_path = output_dir / "cold_user_prior_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
