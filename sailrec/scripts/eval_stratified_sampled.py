"""Stratified sampled@100 evaluation — head/torso/tail × short/medium/long.

Mirrors eval_stratified.py but uses the sampled@100 protocol
(rank the positive against 100 fixed negatives) rather than full-rank.
This is the protocol reported by LLM-ESR, LLMEmb, and most published
LLM-rec papers, so stratified sampled@100 numbers are needed for
direct table comparisons.

Why sampled@100 differs from full-rank for tail items
------------------------------------------------------

Negatives are sampled uniformly from the catalog. On a Zipf-distributed
catalog (like Video_Games) a uniform draw produces mostly head items.
So a tail item's positive competes against ~100 head items — much
easier than full-rank where it competes against all 25K items.
Sampled@100 therefore OVER-ESTIMATES tail performance compared to
full-rank. Both numbers should be reported; sampled for comparisons
with published work, full-rank for honest assessment.

Usage
-----

::

    uv run python sailrec/scripts/eval_stratified_sampled.py \\
        --config sailrec/configs/p1_video_games.yaml --seeds 42 7 18

    uv run python sailrec/scripts/eval_stratified_sampled.py \\
        --config sailrec/configs/ablations/A7_freq_weighted.yaml --seeds 42
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
from sailrec.data.dataset import SASRecEvalDataset, create_eval_loader
from sailrec.data.loaders import build_user_sequences, load_ratings, preprocess
from sailrec.data.splitting import (
    generate_negative_samples,
    leave_one_out_split,
    load_negative_samples,
    save_negative_samples,
)
from sailrec.models.sasrec import SASRec


def log(msg: str) -> None:
    print(msg, flush=True)


def item_frequency_buckets(
    train_seqs: dict[int, list[int]], num_items: int
) -> tuple[dict[int, str], dict[int, int]]:
    counts = Counter(iid for seq in train_seqs.values() for iid in seq)
    item_to_count = {iid: counts.get(iid, 0) for iid in range(1, num_items + 1)}
    sorted_items = sorted(range(1, num_items + 1), key=lambda i: (item_to_count[i], i))
    n = len(sorted_items)
    tail_cut, head_cut = int(n * 0.20), int(n * 0.80)
    item_to_bucket: dict[int, str] = {}
    for rank, iid in enumerate(sorted_items):
        item_to_bucket[iid] = "tail" if rank < tail_cut else ("torso" if rank < head_cut else "head")
    return item_to_bucket, item_to_count


def edge_tail_set(item_to_count: dict[int, int], threshold: int = 10) -> set[int]:
    return {iid for iid, c in item_to_count.items() if c <= threshold}


def user_history_buckets(train_seqs: dict[int, list[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for uid, seq in train_seqs.items():
        n = len(seq)
        out[uid] = "short" if n < 10 else ("medium" if n < 30 else "long")
    return out


def hr_at_k(rankings: torch.Tensor, k: int) -> float:
    return (rankings < k).float().mean().item() if len(rankings) else 0.0


def ndcg_at_k(rankings: torch.Tensor, k: int) -> float:
    if not len(rankings):
        return 0.0
    hits = (rankings < k).float()
    return (hits / torch.log2(rankings.float() + 2.0)).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
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

    item_buckets, item_counts = item_frequency_buckets(split.train_seqs, split.num_items)
    edge_tail = edge_tail_set(item_counts, threshold=10)
    user_buckets = user_history_buckets(split.train_seqs)

    # Load / generate shared negatives.
    neg_path = Path(cfg.output_dir) / "neg_samples.npz"
    if neg_path.exists():
        neg_samples = load_negative_samples(str(neg_path))
    else:
        neg_samples = generate_negative_samples(split, num_neg=cfg.evaluation.num_neg_samples)
        save_negative_samples(neg_samples, str(neg_path))

    # Build test sequences (train + val as context, test target as GT).
    test_seqs = {
        uid: split.train_seqs[uid] + [split.val_targets[uid]]
        for uid in split.test_targets
    }
    test_dataset = SASRecEvalDataset(
        user_seqs=test_seqs,
        targets=split.test_targets,
        neg_samples=neg_samples,
        max_seq_len=cfg.model.max_seq_len,
    )
    # Preserve uid order to align with ranks.
    ordered_uids = list(test_seqs.keys())
    ordered_targets = [split.test_targets[uid] for uid in ordered_uids]

    output_dir = Path(cfg.output_dir)
    per_seed: list[dict[str, float]] = []

    for seed in seeds:
        ckpt_path = output_dir / f"seed_{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            log(f"[seed {seed}] no checkpoint at {ckpt_path}, skipping")
            continue

        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        has_fusion = any(k.startswith("llm_emb") for k in sd)
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
        model.load_state_dict(sd)
        model.eval()

        # Score 101 candidates (1 pos + 100 neg) per user. The positive
        # is always index 0 in the candidates tensor by dataset convention.
        loader = create_eval_loader(test_dataset)
        all_ranks: list[torch.Tensor] = []
        with torch.no_grad():
            for seq, candidates in loader:
                seq = seq.to(device)
                candidates = candidates.to(device)
                scores = model.predict(seq, candidates)   # (B, 101)
                # Rank of positive (index 0) among all 101 candidates.
                pos_score = scores[:, 0:1]                # (B, 1)
                ranks = (scores > pos_score).sum(dim=1)   # (B,)
                all_ranks.append(ranks.cpu())

        ranks = torch.cat(all_ranks, dim=0)               # (U,)

        # Bucket by item and user axes.
        item_b = torch.tensor(
            [{"head":0,"torso":1,"tail":2}[item_buckets[t]] for t in ordered_targets]
        )
        user_b = torch.tensor(
            [{"short":0,"medium":1,"long":2}[user_buckets[uid]] for uid in ordered_uids]
        )
        edge_mask = torch.tensor([t in edge_tail for t in ordered_targets])

        metrics: dict[str, float] = {}
        for lab, code in [("head",0),("torso",1),("tail",2)]:
            sel = ranks[item_b == code]
            metrics[f"item_{lab}_n"] = float(len(sel))
            metrics[f"item_{lab}_hr@10"] = hr_at_k(sel, 10)
            metrics[f"item_{lab}_ndcg@10"] = ndcg_at_k(sel, 10)
        sel = ranks[edge_mask]
        metrics["edge_tail_n"] = float(len(sel))
        metrics["edge_tail_hr@10"] = hr_at_k(sel, 10)
        metrics["edge_tail_ndcg@10"] = ndcg_at_k(sel, 10)
        for lab, code in [("short",0),("medium",1),("long",2)]:
            sel = ranks[user_b == code]
            metrics[f"user_{lab}_n"] = float(len(sel))
            metrics[f"user_{lab}_hr@10"] = hr_at_k(sel, 10)
            metrics[f"user_{lab}_ndcg@10"] = ndcg_at_k(sel, 10)
        metrics["overall_hr@10"] = hr_at_k(ranks, 10)
        metrics["overall_ndcg@10"] = ndcg_at_k(ranks, 10)

        per_seed.append(metrics)
        log(f"\n[seed {seed}] sampled@100 item-axis HR@10:")
        for lab in ["head","torso","tail"]:
            log(f"  item_{lab}: hr@10={metrics[f'item_{lab}_hr@10']:.4f}  "
                f"ndcg@10={metrics[f'item_{lab}_ndcg@10']:.4f}  "
                f"n={int(metrics[f'item_{lab}_n'])}")
        log(f"  edge_tail: hr@10={metrics['edge_tail_hr@10']:.4f}  "
            f"n={int(metrics['edge_tail_n'])}")

    if not per_seed:
        log("No checkpoints found.")
        return

    bar = "=" * 60
    log(f"\n{bar}\nAGGREGATE sampled@100 STRATIFIED\n{bar}")
    keys = sorted(k for k in per_seed[0] if not k.endswith("_n"))
    summary: dict = {}
    for key in keys:
        values = [m[key] for m in per_seed]
        mean = sum(values) / len(values)
        std = (sum((v-mean)**2 for v in values)/len(values))**0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
    for key in [k for k in per_seed[0] if k.endswith("_n")]:
        summary[key] = {"value": per_seed[0][key]}

    for lab in ["head","torso","tail"]:
        m = summary[f"item_{lab}_hr@10"]
        log(f"item_{lab}: hr@10 = {m['mean']:.4f} ± {m['std']:.4f}")
    e = summary["edge_tail_hr@10"]
    log(f"edge_tail: hr@10 = {e['mean']:.4f} ± {e['std']:.4f}")
    o = summary["overall_hr@10"]
    log(f"overall:   hr@10 = {o['mean']:.4f} ± {o['std']:.4f}")

    out_path = output_dir / "aggregate_stratified_sampled.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
