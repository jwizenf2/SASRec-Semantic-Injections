"""Train ItemTable — SASRec + InfoNCE alignment to frozen LLM embeddings.

Loss::

    L_total = L_next + lambda_align * L_align

Saved checkpoints are bit-shape-identical to P1 baseline checkpoints
because the AlignProjector lives on the trainer, not the model. That
means the standard ``eval_fullrank.py`` script works directly against
ItemTable checkpoints.

Sweep mode
----------

Pass ``--lambdas 0.05 0.1 0.5`` to run a sweep under one preprocessing
pass (data is loaded only once, the LLM table is loaded only once).
Outputs land at ``{output_dir}/lambda_{value}/seed_{seed}/``.

Usage
-----

::

    # Single seed, single lambda (the headline configuration):
    uv run python sasrec_injection/scripts/train_item_table.py \\
        --config sasrec_injection/configs/sasrec_injection_video_games.yaml \\
        --seeds 42 --lambdas 0.1

    # Multi-seed (next planned experiment):
    uv run python sasrec_injection/scripts/train_item_table.py \\
        --config sasrec_injection/configs/sasrec_injection_video_games.yaml \\
        --seeds 42 123 456 --lambdas 0.1

    # Lambda sweep, single seed:
    uv run python sasrec_injection/scripts/train_item_table.py \\
        --config sasrec_injection/configs/sasrec_injection_video_games.yaml \\
        --seeds 42 --lambdas 0.05 0.1 0.2 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make ``import sasrec_injection...`` work whether this script is run from the
# project root or anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasrec_injection.config import ItemTableConfig
from sasrec_injection.data.dataset import (
    SASRecEvalDataset,
    SASRecTrainDataset,
    create_eval_loader,
    create_train_loader,
)
from sasrec_injection.data.loaders import build_user_sequences, load_ratings, preprocess
from sasrec_injection.data.splitting import (
    generate_negative_samples,
    leave_one_out_split,
    load_negative_samples,
    save_negative_samples,
)
from sasrec_injection.evaluation.metrics import sampled_evaluate
from sasrec_injection.models.sasrec import SASRec
from sasrec_injection.seeds import set_seed
from sasrec_injection.training.item_table_trainer import ItemTableTrainer


def log(msg: str) -> None:
    """Print with immediate flush so ``tail -f`` shows progress live."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data preparation — shared across seeds and λs within one config
# ---------------------------------------------------------------------------


def prepare_data(cfg: ItemTableConfig) -> tuple:
    """Load + split + cache negatives. See ``train_baseline.prepare_data``."""
    data_dir = "data"
    log(f"Loading {cfg.dataset_name} ratings...")
    ratings = load_ratings(data_dir, dataset=cfg.dataset_name)
    df, user_map, item_map = preprocess(
        ratings, min_interactions=cfg.min_interactions
    )
    num_items = len(item_map)
    log(f"Users: {len(user_map)}, Items: {num_items}")

    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"Split: {split.num_users} users, {split.num_items} items")

    # Cache neg samples once per output_dir so every seed/λ sees the
    # same negatives. (Sampled metrics across runs are then directly
    # comparable.)
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


def load_llm_embeddings(
    path: str,
    num_items: int,
    llm_dim: int,
) -> torch.Tensor:
    """Load + shape-check the frozen LLM item-embedding tensor.

    Args:
        path: Filesystem path to the tensor saved by
            ``scripts/extract_llm_embeddings.py``.
        num_items: Expected vocabulary size.
        llm_dim: Expected feature dimension.

    Returns:
        ``(num_items + 1, llm_dim)`` float32 tensor on CPU.

    Raises:
        ValueError: If the tensor's shape doesn't match the expected
            ``(num_items + 1, llm_dim)``.
    """
    emb = torch.load(path, weights_only=True, map_location="cpu")
    if emb.shape != (num_items + 1, llm_dim):
        raise ValueError(
            f"LLM embedding shape {tuple(emb.shape)} does not match "
            f"expected ({num_items + 1}, {llm_dim}). Re-extract with "
            "the right LLM model, or update align.llm_dim in the "
            "ItemTable yaml."
        )
    return emb


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def train_single_seed(
    cfg: ItemTableConfig,
    seed: int,
    split,
    neg_samples: dict,
    llm_item_emb: torch.Tensor,
    seed_dir: Path,
    resume: bool = False,
    init_item_emb: torch.Tensor | None = None,
    item_freq: torch.Tensor | None = None,
    weight_fn: str = "log",
    backbone: str = "sasrec",
    input_fusion: bool = False,
) -> dict[str, float]:
    """Train ItemTable for one (seed, λ) combo. Returns sampled test metrics.

    Args:
        resume: If True, look for ``last.pt`` in ``seed_dir`` and
            continue from that snapshot. Falls through to a fresh
            start if the snapshot is missing.
    """
    set_seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    # Persist experiment identity so resumed runs can detect flag mismatches.
    # If a prior run_config.json exists and disagrees, abort rather than
    # silently train as the wrong variant (e.g. unweighted instead of A7).
    run_config = {
        "freq_weight": item_freq is not None,
        "weight_fn": weight_fn if item_freq is not None else None,
        "llm_init": init_item_emb is not None,
        "backbone": backbone,
        "input_fusion": input_fusion,
        "lambda_align": cfg.align.lambda_align,
        "dataset": cfg.dataset_name,
    }
    run_config_path = seed_dir / "run_config.json"
    if run_config_path.exists():
        import json as _json
        saved = _json.load(open(run_config_path))
        mismatches = {k: (saved.get(k), run_config[k]) for k in run_config if saved.get(k) != run_config[k]}
        if mismatches:
            raise RuntimeError(
                f"Resume flag mismatch in {seed_dir}. "
                f"Saved vs current: {mismatches}. "
                "Pass the same flags as the original run, or delete the checkpoint to start fresh."
            )
    else:
        import json as _json
        with open(run_config_path, "w") as f:
            _json.dump(run_config, f, indent=2)

    train_dataset = SASRecTrainDataset(
        split.train_seqs, split.num_items, max_seq_len=cfg.model.max_seq_len
    )
    val_dataset = SASRecEvalDataset(
        split.train_seqs,
        split.val_targets,
        neg_samples,
        max_seq_len=cfg.model.max_seq_len,
    )
    train_loader = create_train_loader(train_dataset, batch_size=cfg.training.batch_size)
    val_loader = create_eval_loader(val_dataset)

    # Build the sequence encoder. By default this is SASRec (the
    # headline backbone). Pass --backbone gru4rec to swap in GRU4Rec
    # while keeping every other hyperparameter identical — used for the
    # backbone-generalisation ablation. Pass --input-fusion to also
    # inject LLM signal at the encoder input (A2 path), on top of the
    # item-table InfoNCE loss (A9 cross-surface combination test).
    if backbone == "gru4rec":
        from sasrec_injection.models.gru4rec import GRU4Rec
        model = GRU4Rec(
            num_items=split.num_items,
            embed_dim=cfg.model.embed_dim,
            hidden_size=cfg.model.embed_dim,
            num_layers=1,
            dropout=cfg.model.dropout,
            init_item_emb=init_item_emb,
        )
    else:
        model = SASRec(
            num_items=split.num_items,
            embed_dim=cfg.model.embed_dim,
            num_blocks=cfg.model.num_blocks,
            num_heads=cfg.model.num_heads,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
            init_item_emb=init_item_emb,
            llm_item_emb=llm_item_emb if input_fusion else None,
            llm_dim=int(llm_item_emb.shape[1]) if input_fusion else None,
            fusion_mode="add" if input_fusion else "off",
        )

    # ----------------------------------------------------------------
    # Optional W&B tracking. We skip silently in any of these cases:
    #   * ``WANDB_MODE=disabled`` (explicit opt-out for this run);
    #   * no ``WANDB_API_KEY`` and no existing ``~/.netrc`` login —
    #     otherwise ``wandb.init`` blocks on an interactive auth
    #     prompt, which deadlocks unattended caffeinate runs.
    # The training loop itself still works either way; wandb is purely
    # for live curves.
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
                name=f"sasrec_injection_lambda{cfg.align.lambda_align}_seed{seed}",
                config={
                    "seed": seed,
                    "model": cfg.model.__dict__,
                    "training": cfg.training.__dict__,
                    "align": cfg.align.__dict__,
                },
                reinit=True,
            )
        except Exception:
            log("wandb init failed, skipping tracking")

    trainer = ItemTableTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        llm_item_emb=llm_item_emb,
        device=device,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        lambda_align=cfg.align.lambda_align,
        temperature=cfg.align.temperature,
        max_align_ids=cfg.align.max_align_ids,
        projector_hidden_dim=cfg.align.projector_hidden_dim,
        projector_dropout=cfg.align.projector_dropout,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        output_dir=str(seed_dir),
        wandb_run=wandb_run,
        item_freq=item_freq,
        weight_fn=weight_fn,
    )
    best_val_metrics = trainer.train(eval_ks=cfg.evaluation.ks, resume=resume)

    # Test eval against train + val sequences as input.
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

    log(f"\n[Seed {seed} | λ={cfg.align.lambda_align}] Sampled test metrics:")
    for k, v in sorted(test_metrics.items()):
        log(f"  {k}: {v:.4f}")

    results = {
        "seed": seed,
        "lambda_align": cfg.align.lambda_align,
        "temperature": cfg.align.temperature,
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
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    metrics_list: list[dict[str, float]],
    output_path: Path,
) -> dict:
    """Compute and save mean ± std across a list of per-seed metric dicts.

    Args:
        metrics_list: One dict per seed (the test metrics returned by
            :func:`train_single_seed`).
        output_path: Where to write ``aggregate_results.json``.

    Returns:
        The aggregate dict (also written to disk).
    """
    metric_keys = sorted(metrics_list[0].keys())
    summary: dict = {}
    for key in metric_keys:
        values = [m[key] for m in metrics_list]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
        log(
            f"{key}: {mean:.4f} ± {std:.4f}  "
            f"(seeds: {[f'{v:.4f}' for v in values]})"
        )
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train ItemTable (SASRec + item-level InfoNCE alignment to "
            "frozen LLM embeddings)."
        )
    )
    parser.add_argument(
        "--config",
        default="sasrec_injection/configs/sasrec_injection_video_games.yaml",
        help="ItemTable experiment config.",
    )
    parser.add_argument(
        "--base-config",
        default="sasrec_injection/configs/base.yaml",
        help="Base config merged under --config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override the config's seed list.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Sweep these λ_align values. Each runs all configured "
            "seeds and writes to {output_dir}/lambda_{value}/."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override training.max_epochs (smoke convenience).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "For each (λ, seed), resume from <seed_dir>/last.pt if it "
            "exists. Snapshots are written every epoch, so resuming "
            "loses at most one epoch of progress. Falls back to a "
            "fresh start when the snapshot is missing."
        ),
    )
    parser.add_argument(
        "--freq-weight",
        action="store_true",
        help="Enable frequency-weighted InfoNCE alignment (A7/A8). See --weight-fn.",
    )
    parser.add_argument(
        "--weight-fn",
        default="log",
        choices=["log", "sqrt", "linear", "binary"],
        help=(
            "Weight function for --freq-weight. "
            "'log' (default): 1/(1+log(freq+1)); "
            "'sqrt': 1/sqrt(freq+1); "
            "'linear': 1/(freq+1); "
            "'binary': tail(bottom 20%%)=2.0, head=0.5."
        ),
    )
    parser.add_argument(
        "--llm-init",
        action="store_true",
        help=(
            "A6 ablation: initialise the item table from a PCA(embed_dim) "
            "projection of the L2-normed LLM embeddings (same recipe as "
            "A1) before applying the SAILS InfoNCE alignment loss. "
            "Tests the (init: LLM-PCA) × (loss: aligned) cell of the "
            "init × alignment 2x2 factorial."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="sasrec",
        choices=["sasrec", "gru4rec"],
        help="Sequence encoder backbone. 'sasrec' (default) or 'gru4rec'.",
    )
    parser.add_argument(
        "--input-fusion",
        action="store_true",
        help=(
            "A9 cross-surface: also inject LLM signal at the encoder input "
            "(A2 fusion_mode='add') on top of the item-table InfoNCE loss. "
            "Tests whether two injection surfaces compound. SASRec only."
        ),
    )
    args = parser.parse_args()

    cfg = ItemTableConfig.from_yaml(args.config, args.base_config)
    if args.seeds:
        cfg.seeds = args.seeds
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs

    log(f"ItemTable | seeds={cfg.seeds} | device={cfg.device}")
    log(
        f"LLM embeddings: {cfg.align.embeddings_path} "
        f"(llm_dim={cfg.align.llm_dim})"
    )

    # Data and LLM table are loaded once and shared across all
    # (λ, seed) combos in the sweep.
    split, neg_samples, _ = prepare_data(cfg)

    # A7: compute per-item training frequency for frequency-weighted
    # alignment. Building a (num_items+1,) count tensor is a one-liner
    # over the training sequences; item 0 stays zero (padding).
    item_freq: torch.Tensor | None = None
    if args.freq_weight:
        from collections import Counter
        counts = Counter(
            iid
            for seq in split.train_seqs.values()
            for iid in seq
        )
        freq_arr = torch.zeros(split.num_items + 1, dtype=torch.long)
        for iid, c in counts.items():
            freq_arr[iid] = c
        item_freq = freq_arr
        log(
            f"--freq-weight: item freq range "
            f"{item_freq[1:].min().item()}–{item_freq[1:].max().item()}, "
            f"mean {item_freq[1:].float().mean().item():.1f}"
        )

    llm_item_emb = load_llm_embeddings(
        cfg.align.embeddings_path, split.num_items, cfg.align.llm_dim
    )
    log(f"LLM table loaded: {tuple(llm_item_emb.shape)}")

    # A6 ablation: precompute the PCA-projected init tensor once and
    # reuse across all (λ, seed) combos. PCA is deterministic, has no
    # trainable parameters, and matches A1's recipe exactly.
    init_item_emb: torch.Tensor | None = None
    if args.llm_init:
        # Import here to keep the standard path free of an optional
        # dependency. ``train_llm_init.pca_project`` is the same
        # centered-SVD reduction used by A1 — keeping a single
        # implementation source means A6's init is byte-identical to
        # A1's, which is the whole point of the 2x2 factorial.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_llm_init import pca_project

        log(f"--llm-init: fitting PCA({cfg.model.embed_dim}) for A6 init...")
        init_item_emb = pca_project(llm_item_emb, target_dim=cfg.model.embed_dim)
        log(
            f"PCA init shape: {tuple(init_item_emb.shape)}; "
            f"row 1 norm: {init_item_emb[1].norm().item():.4f}"
        )

    lambdas = args.lambdas if args.lambdas else [cfg.align.lambda_align]

    sweep_summary: dict[float, dict] = {}
    for lam in lambdas:
        cfg.align.lambda_align = lam
        log(f"\n{'='*60}\nλ = {lam}\n{'='*60}")

        # When sweeping, each λ gets its own subdir so the per-λ
        # aggregate JSONs sit side by side and are trivial to compare.
        if len(lambdas) == 1:
            lambda_dir = Path(cfg.output_dir)
        else:
            lambda_dir = Path(cfg.output_dir) / f"lambda_{lam}"
        lambda_dir.mkdir(parents=True, exist_ok=True)

        metrics_list: list[dict[str, float]] = []
        for seed in cfg.seeds:
            log(f"\n--- Seed {seed} ---")
            seed_dir = lambda_dir / f"seed_{seed}"
            test_metrics = train_single_seed(
                cfg, seed, split, neg_samples, llm_item_emb, seed_dir,
                resume=args.resume,
                init_item_emb=init_item_emb,
                item_freq=item_freq,
                weight_fn=args.weight_fn,
                backbone=args.backbone,
                input_fusion=args.input_fusion,
            )
            metrics_list.append(test_metrics)

        log(f"\nAGGREGATE for λ={lam}")
        agg = aggregate(metrics_list, lambda_dir / "aggregate_results.json")
        sweep_summary[lam] = agg

    # If we ran a sweep, write a top-level digest summarising the means
    # at each λ. Useful for quick "which λ won?" lookups.
    if len(lambdas) > 1:
        digest = {
            str(lam): {k: v["mean"] for k, v in s.items()}
            for lam, s in sweep_summary.items()
        }
        digest_path = Path(cfg.output_dir) / "sweep_digest.json"
        with open(digest_path, "w") as f:
            json.dump(digest, f, indent=2)
        log(f"\nSweep digest -> {digest_path}")


if __name__ == "__main__":
    main()
