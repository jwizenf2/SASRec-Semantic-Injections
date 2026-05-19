"""End-of-training evaluation of saved SASRec checkpoints (P1 or SAILRec).

For each ``seed_<S>/best_model.pt`` under ``cfg.output_dir`` we report
**two** evaluation protocols, side by side:

* **Full-rank** — rank the test ground truth against every item in the
  catalog, with previously-seen items masked. The honest metric for
  headline numbers.
* **Sampled@100** — rank the ground truth against 100 fixed negatives
  per user (the SASRec / LLM-ESR / BIGRec convention). Useful for
  apples-to-apples comparison with papers that *only* report sampled
  metrics (e.g. LLM-ESR, NeurIPS 2024).

Why both
--------

Sampled metrics over-state absolute quality (100 negatives is a tiny
slice of a 25K-item catalog) but track relative improvements well, so
they're cheap to run every epoch for early stopping. Full-rank is
expensive (~100-1000x) so we run it once, at the end, on the
best-saved checkpoint. Reporting both at end-of-training lets us drop
SAILRec rows directly into either kind of comparison table.

Reading checkpoints from any trainer
------------------------------------

The script accepts both:

* Bare state-dicts (``torch.save(model.state_dict(), path)``) — the
  P1 baseline format.
* Wrapped dicts (``{"model": state_dict}``) — the SAILRec format.

This lets the same script evaluate either method.

Usage
-----

::

    # Evaluate the P1 baseline checkpoints (single or multi-seed):
    uv run python sailrec/scripts/eval_fullrank.py \\
        --config sailrec/configs/p1_video_games.yaml --seeds 42

    # Evaluate SAILRec checkpoints (write a temp p1-style yaml whose
    # output_dir points at sailrec/outputs/sailrec_video_games/lambda_0.1
    # and pass that as --config):
    uv run python sailrec/scripts/eval_fullrank.py \\
        --config <temp.yaml> --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make ``import sailrec...`` work whether this script is invoked from
# the project root or anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.config import P1Config
from sailrec.data.dataset import FullRankEvalDataset, SASRecEvalDataset
from sailrec.data.loaders import build_user_sequences, load_ratings, preprocess
from sailrec.data.splitting import (
    generate_negative_samples,
    leave_one_out_split,
    load_negative_samples,
    save_negative_samples,
)
from sailrec.evaluation.metrics import full_rank_evaluate, sampled_evaluate
from sailrec.models.sasrec import SASRec


def log(msg: str) -> None:
    """Print with immediate flush."""
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full-rank evaluation of saved SASRec checkpoints. Works "
            "for both P1 baseline and SAILRec checkpoints because "
            "their on-disk shape is identical."
        )
    )
    parser.add_argument(
        "--config",
        default="sailrec/configs/p1_video_games.yaml",
        help=(
            "Config whose ``output_dir`` contains ``seed_<S>/best_model.pt``. "
            "Use a P1-style yaml; for SAILRec, write a small temp yaml "
            "pointing at the lambda subdir you want to evaluate."
        ),
    )
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Subset of seeds to evaluate (defaults to all in the config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Eval batch size. Smaller than training because each "
            "sample carries a (num_items,) exclusion mask which is "
            "memory-heavy on MPS."
        ),
    )
    args = parser.parse_args()

    cfg = P1Config.from_yaml(args.config, args.base_config)
    seeds = args.seeds or cfg.seeds
    device = torch.device(cfg.device)

    # ----------------------------------------------------------------
    # Reproduce the same split the training run used. Critical: the
    # exact ``preprocess`` parameters (dataset name + min_interactions)
    # must match, otherwise the user / item id mappings can drift and
    # the saved item_emb rows won't line up with the catalog.
    # ----------------------------------------------------------------
    log(f"Loading {cfg.dataset_name} data...")
    ratings = load_ratings("data", dataset=cfg.dataset_name)
    df, _, item_map = preprocess(ratings, min_interactions=cfg.min_interactions)
    num_items = len(item_map)
    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"Users: {split.num_users}, Items: {split.num_items}")

    # Build the full-rank test set: encode train + val, predict the
    # test target, exclude every item the user has seen so far.
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ----------------------------------------------------------------
    # Sampled-eval set: 1 positive + N negatives per user. We reuse
    # the ``neg_samples.npz`` saved by training when present so the
    # candidate set is identical to whatever was used for early
    # stopping; if the file is missing (e.g. evaluating an externally
    # provided checkpoint) we regenerate with the same seeded RNG.
    # The negative pool is deterministic in (data, seed=42), so P1 and
    # SAILRec runs that share preprocessing get the same negatives —
    # required for like-for-like sampled comparisons.
    # ----------------------------------------------------------------
    neg_path = Path(cfg.output_dir) / "neg_samples.npz"
    if neg_path.exists():
        log(f"Loading cached sampled-eval negatives from {neg_path}")
        neg_samples = load_negative_samples(str(neg_path))
    else:
        log(f"No cached negatives at {neg_path}; regenerating (seed=42)")
        neg_samples = generate_negative_samples(
            split, num_neg=cfg.evaluation.num_neg_samples
        )
        save_negative_samples(neg_samples, str(neg_path))

    sampled_dataset = SASRecEvalDataset(
        user_seqs=test_seqs,
        targets=split.test_targets,
        neg_samples=neg_samples,
        max_seq_len=cfg.model.max_seq_len,
    )
    # Larger batch is fine here — sampled eval has a (B, 101) candidate
    # tensor instead of full-rank's (B, num_items) mask, so memory is
    # ~250x lighter.
    sampled_loader = torch.utils.data.DataLoader(
        sampled_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    ks = cfg.evaluation.ks
    all_metrics: list[dict[str, float]] = []

    # ----------------------------------------------------------------
    # Load each seed's best checkpoint and run full-rank eval.
    # ----------------------------------------------------------------
    for seed in seeds:
        ckpt_path = Path(cfg.output_dir) / f"seed_{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            log(f"[Seed {seed}] Checkpoint not found at {ckpt_path}, skipping")
            continue

        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        state_dict = (
            ckpt["model"]
            if isinstance(ckpt, dict) and "model" in ckpt
            else ckpt
        )

        # Detect A2 (Input Fusion) by presence of llm_emb keys in the
        # checkpoint. A2 saves llm_emb and llm_proj inside the model
        # state; plain SASRec has neither — reconstruct with the
        # matching fusion_mode so load_state_dict doesn't error.
        has_fusion = any(k.startswith("llm_emb") for k in state_dict)
        if has_fusion:
            import yaml as _yaml
            raw_cfg = _yaml.safe_load(open(args.config))
            llm_path = raw_cfg.get("llm_embeddings_path",
                                   raw_cfg.get("align", {}).get("embeddings_path", ""))
            llm_dim = raw_cfg.get("align", {}).get("llm_dim", 1024)
            llm_emb = torch.load(llm_path, map_location="cpu", weights_only=True)
            model = SASRec(
                num_items=num_items,
                embed_dim=cfg.model.embed_dim,
                num_blocks=cfg.model.num_blocks,
                num_heads=cfg.model.num_heads,
                max_seq_len=cfg.model.max_seq_len,
                dropout=cfg.model.dropout,
                llm_item_emb=llm_emb,
                llm_dim=llm_dim,
                fusion_mode="add",
            ).to(device)
        else:
            model = SASRec(
                num_items=num_items,
                embed_dim=cfg.model.embed_dim,
                num_blocks=cfg.model.num_blocks,
                num_heads=cfg.model.num_heads,
                max_seq_len=cfg.model.max_seq_len,
                dropout=cfg.model.dropout,
            ).to(device)

        model.load_state_dict(state_dict)

        # Both protocols on the same checkpoint. Keys are disjoint —
        # full-rank uses ``full_*`` prefixes, sampled uses bare
        # ``hr@k`` / ``ndcg@k`` / ``recall@k`` — so we can merge the
        # two dicts without collision.
        full_metrics = full_rank_evaluate(model, test_loader, device, ks=ks)
        sampled_metrics = sampled_evaluate(model, sampled_loader, device, ks=ks)
        merged = {**full_metrics, **sampled_metrics}

        log(f"\n[Seed {seed}] Full-rank test metrics:")
        for k, v in sorted(full_metrics.items()):
            log(f"  {k}: {v:.4f}")
        log(f"[Seed {seed}] Sampled@{cfg.evaluation.num_neg_samples} test metrics:")
        for k, v in sorted(sampled_metrics.items()):
            log(f"  {k}: {v:.4f}")
        all_metrics.append(merged)

    if not all_metrics:
        log("No checkpoints found.")
        return

    # ----------------------------------------------------------------
    # Aggregate across seeds: mean ± std on every metric. Sort the
    # output so full-rank rows print first, then sampled — easier to
    # read at a glance.
    # ----------------------------------------------------------------
    bar = "=" * 60
    log(
        f"\n{bar}\nAGGREGATE RESULTS "
        f"(full-rank + sampled@{cfg.evaluation.num_neg_samples})\n{bar}"
    )
    full_keys = sorted(k for k in all_metrics[0] if k.startswith("full_"))
    sampled_keys = sorted(k for k in all_metrics[0] if not k.startswith("full_"))
    metric_keys = full_keys + sampled_keys

    summary: dict = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
        log(f"{key}: {mean:.4f} ± {std:.4f}")

    out_path = Path(cfg.output_dir) / "aggregate_fullrank_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
