"""Train the P1 baseline (vanilla SASRec, no LLM signal).

This is the comparison baseline for SAILRec. Numbers from a single
seed-42 run on Amazon Video_Games:

* Full-rank HR@10  = 0.0584
* Full-rank HR@20  = 0.0913
* Full-rank NDCG@10 = 0.0308

SAILRec must beat HR@10 + 0.005 absolute (≥ 0.0634) to clear the
decision gate.

Usage
-----

::

    # Single seed:
    uv run python sailrec/scripts/train_p1.py \\
        --config sailrec/configs/p1_video_games.yaml --seeds 42

    # Multi-seed (planned next experiment per the verification plan):
    uv run python sailrec/scripts/train_p1.py \\
        --config sailrec/configs/p1_video_games.yaml --seeds 42 123 456

Outputs land under ``cfg.output_dir/``:

* ``neg_samples.npz``                  — shared across seeds
* ``seed_<S>/best_model.pt``           — best-epoch checkpoint per seed
* ``seed_<S>/results.json``            — sampled test metrics
* ``seed_<S>/train_log.jsonl``         — per-epoch records
* ``seed_<S>/train_summary.json``      — final summary
* ``aggregate_results.json``           — mean ± std across seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make ``import sailrec...`` work whether this script is invoked from
# the project root or from ``sailrec/``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.config import P1Config
from sailrec.data.dataset import (
    SASRecEvalDataset,
    SASRecTrainDataset,
    create_eval_loader,
    create_train_loader,
)
from sailrec.data.loaders import build_user_sequences, load_ratings, preprocess
from sailrec.data.splitting import (
    generate_negative_samples,
    leave_one_out_split,
    load_negative_samples,
    save_negative_samples,
)
from sailrec.evaluation.metrics import sampled_evaluate
from sailrec.models.sasrec import SASRec
from sailrec.seeds import set_seed
from sailrec.training.trainer import SASRecTrainer


def log(msg: str) -> None:
    """Print with immediate flush so ``tail -f`` shows progress live."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data prep — shared across seeds within one config
# ---------------------------------------------------------------------------


def prepare_data(cfg: P1Config) -> tuple:
    """Load ratings, run preprocess + split, generate negatives.

    Args:
        cfg: Loaded :class:`P1Config`.

    Returns:
        ``(split, neg_samples, item_map)``:

        * ``split``       — :class:`SplitData` with train/val/test.
        * ``neg_samples`` — sampled negatives per test user (cached
                            on disk so different seeds use the same
                            candidates).
        * ``item_map``    — original → remapped item id mapping
                            (returned for callers that want to load
                            metadata against the same vocabulary).

    Notes:
        Negative samples are *seed-independent*: we generate them
        once per output_dir and reuse across seeds. This means
        per-seed sampled metrics are directly comparable.
    """
    data_dir = "data"
    dataset = cfg.dataset_name
    log(f"Loading {dataset} ratings...")
    ratings = load_ratings(data_dir, dataset=dataset)
    df, user_map, item_map = preprocess(
        ratings, min_interactions=cfg.min_interactions
    )
    num_items = len(item_map)
    log(f"Users: {len(user_map)}, Items: {num_items}")

    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"Split: {split.num_users} users, {split.num_items} items")

    # Cache negatives on disk under the run's output_dir. Keeps every
    # seed using the same negatives so sampled metrics are comparable.
    neg_path = Path(cfg.output_dir) / "neg_samples.npz"
    neg_path.parent.mkdir(parents=True, exist_ok=True)
    if neg_path.exists():
        log(f"Loading cached negative samples from {neg_path}")
        neg_samples = load_negative_samples(str(neg_path))
    else:
        log("Generating negative samples...")
        neg_samples = generate_negative_samples(
            split, num_neg=cfg.evaluation.num_neg_samples
        )
        save_negative_samples(neg_samples, str(neg_path))

    return split, neg_samples, item_map


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def train_single_seed(
    cfg: P1Config,
    seed: int,
    split,
    neg_samples: dict,
    resume: bool = False,
    backbone: str = "sasrec",
) -> dict[str, float]:
    """Train P1 baseline for one seed and return sampled test metrics.

    Args:
        backbone: ``"sasrec"`` (default) or ``"gru4rec"``. Swaps the
            sequence encoder while keeping training loop unchanged.
        resume: If True, look for ``last.pt`` in the seed's output
            directory and continue from that snapshot. Falls through
            to a fresh start if the snapshot is missing.

    Side effects: writes ``best_model.pt``, ``results.json``,
    ``train_log.jsonl``, ``train_summary.json`` to
    ``cfg.output_dir/seed_<seed>/``.
    """
    set_seed(seed)
    seed_dir = Path(cfg.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    # Datasets / loaders.
    train_dataset = SASRecTrainDataset(
        split.train_seqs, split.num_items, max_seq_len=cfg.model.max_seq_len
    )
    val_dataset = SASRecEvalDataset(
        split.train_seqs,                     # Encode using training only…
        split.val_targets,                    # …predict the val target.
        neg_samples,
        max_seq_len=cfg.model.max_seq_len,
    )
    train_loader = create_train_loader(train_dataset, batch_size=cfg.training.batch_size)
    val_loader = create_eval_loader(val_dataset)

    if backbone == "gru4rec":
        from sailrec.models.gru4rec import GRU4Rec
        model = GRU4Rec(
            num_items=split.num_items,
            embed_dim=cfg.model.embed_dim,
            hidden_size=cfg.model.embed_dim,
            num_layers=1,
            dropout=cfg.model.dropout,
        )
    else:
        model = SASRec(
            num_items=split.num_items,
            embed_dim=cfg.model.embed_dim,
            num_blocks=cfg.model.num_blocks,
            num_heads=cfg.model.num_heads,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )

    # ----------------------------------------------------------------
    # Optional W&B tracking. Skip silently when:
    #   * ``WANDB_MODE=disabled`` (explicit opt-out), or
    #   * no ``WANDB_API_KEY`` and no ``~/.netrc`` — otherwise
    #     ``wandb.init`` blocks on an interactive auth prompt, which
    #     deadlocks unattended caffeinate runs.
    # ----------------------------------------------------------------
    import os

    wandb_run = None
    wandb_disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    has_netrc_login = Path.home().joinpath(".netrc").exists()
    if wandb_disabled:
        log("wandb disabled via WANDB_MODE=disabled, skipping tracking")
    elif not (has_api_key or has_netrc_login):
        log("no WANDB_API_KEY / ~/.netrc; skipping wandb to avoid auth prompt")
    else:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"p1_sasrec_seed{seed}",
                config={
                    "seed": seed,
                    "model": cfg.model.__dict__,
                    "training": cfg.training.__dict__,
                },
                reinit=True,
            )
        except Exception:
            log("wandb init failed, skipping tracking")

    trainer = SASRecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        output_dir=str(seed_dir),
        wandb_run=wandb_run,
    )
    best_val_metrics = trainer.train(eval_ks=cfg.evaluation.ks, resume=resume)

    # Test evaluation: encode train + val, predict the test target.
    # Using train + val matches the standard sequential-recommendation
    # evaluation protocol (see SASRec / DLLM2Rec / BIGRec).
    test_seqs = {
        uid: split.train_seqs[uid] + [split.val_targets[uid]]
        for uid in split.test_targets
    }
    test_dataset = SASRecEvalDataset(
        test_seqs,
        split.test_targets,
        neg_samples,
        max_seq_len=cfg.model.max_seq_len,
    )
    test_loader = create_eval_loader(test_dataset)
    test_metrics = sampled_evaluate(model, test_loader, device, ks=cfg.evaluation.ks)

    log(f"\n[Seed {seed}] Sampled test metrics:")
    for k, v in sorted(test_metrics.items()):
        log(f"  {k}: {v:.4f}")

    # Persist a per-seed summary alongside the checkpoint.
    results = {
        "seed": seed,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    with open(seed_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    return test_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train P1 baseline (vanilla SASRec, no LLM signal)."
    )
    parser.add_argument(
        "--config",
        default="sailrec/configs/p1_video_games.yaml",
        help="P1 experiment config.",
    )
    parser.add_argument(
        "--base-config",
        default="sailrec/configs/base.yaml",
        help="Base config merged under --config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override the config's seed list (e.g. --seeds 42 123 456).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "For each seed, resume from <seed_dir>/last.pt if it exists. "
            "Snapshots are written every epoch, so resuming loses at "
            "most one epoch of progress. Falls back to a fresh start "
            "when the snapshot is missing."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="sasrec",
        choices=["sasrec", "gru4rec"],
        help="Sequence encoder backbone. 'sasrec' (default) or 'gru4rec'.",
    )
    args = parser.parse_args()

    cfg = P1Config.from_yaml(args.config, args.base_config)
    if args.seeds:
        cfg.seeds = args.seeds

    log(f"Running P1 baseline with seeds: {cfg.seeds}")
    log(f"Device: {cfg.device}")

    # Data is shared across seeds within one config — so we prep it
    # once and pass it into each seed's training run.
    split, neg_samples, _ = prepare_data(cfg)

    all_metrics: list[dict[str, float]] = []
    for seed in cfg.seeds:
        log(f"\n{'='*60}")
        log(f"Training seed {seed}")
        log(f"{'='*60}")
        all_metrics.append(
            train_single_seed(cfg, seed, split, neg_samples, resume=args.resume, backbone=args.backbone)
        )

    # Aggregate across seeds. Mean ± std on every metric the trainer
    # reported. With a single seed the std is 0 and the report is
    # equivalent to the per-seed numbers.
    log(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    metric_keys = sorted(all_metrics[0].keys())
    summary: dict = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
        log(
            f"{key}: {mean:.4f} ± {std:.4f}  "
            f"(seeds: {[f'{v:.4f}' for v in values]})"
        )

    output_dir = Path(cfg.output_dir)
    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
