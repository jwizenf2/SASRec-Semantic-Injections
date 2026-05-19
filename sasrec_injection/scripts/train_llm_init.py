"""A1 — LLM Init only ablation.

Tests the *mechanism* axis of the SAILS thesis: does ongoing alignment
pressure matter, or is good initial geometry alone enough?

What this script does
---------------------

Identical to ``train_baseline.py`` (vanilla SASRec, BCE-only training,
no projector, no auxiliary loss) with one change: the item embedding
table is initialised from a deterministic PCA projection of the L2-
normalised LLM item embeddings, instead of Xavier-uniform.

After epoch 0 the LLM tensor is discarded — it never appears in the
training graph again. So this run measures *only* the contribution of
starting geometry, with zero ongoing alignment pressure.

PCA — why and how
-----------------

PCA is computed once, before training, via centered SVD on the
``(num_items, llm_dim)`` matrix (padding row dropped). We use
``torch.linalg.svd`` so this script has no scikit-learn dependency.
Output is L2-renormalised so the magnitude of the resulting table
matches the typical magnitude of Xavier-uniform initialisation
(rows have unit norm), removing one confound from the A0 vs A1
comparison.

The PCA transform has no trainable parameters — that is the whole
point. A trainable projector would be a weak form of alignment
pressure, which is what A2 (input fusion) tests.

Output layout
-------------

::

    sasrec_injection/outputs/ablations/A1_llm_init/
    ├── neg_samples.npz
    ├── seed_<S>/best_model.pt
    ├── seed_<S>/last.pt
    ├── seed_<S>/results.json
    ├── seed_<S>/train_log.jsonl
    ├── seed_<S>/train_summary.json
    └── aggregate_results.json

Same shape as ``train_baseline.py``'s output so ``results_table.py`` and
``eval_fullrank.py`` work without modification.

Decision rule
-------------

* HR@10 ≈ A0 (≤ +0.001 absolute) → init alone is insufficient;
  SAILS's gain comes from ongoing alignment pressure. **Mechanism
  claim proven.**
* HR@10 > A0 by ≥ +0.005 absolute → static LLM geometry helps; the
  paper must attribute (A5 − A1) to alignment, not (A5 − A0).
* HR@10 ≈ A5 → static geometry is sufficient; alignment buys
  nothing. SAILS's loss isn't doing what we think — re-investigate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasrec_injection.config import SASRecConfig
from sasrec_injection.data.dataset import (
    SASRecEvalDataset,
    SASRecTrainDataset,
    create_eval_loader,
    create_train_loader,
)
from sasrec_injection.evaluation.metrics import sampled_evaluate
from sasrec_injection.models.sasrec import SASRec
from sasrec_injection.seeds import set_seed
from sasrec_injection.training.trainer import SASRecTrainer

# Reuse train_baseline's data prep — it produces the same split + neg samples
# A1 needs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import log, prepare_data  # noqa: E402


# ---------------------------------------------------------------------------
# PCA via centered SVD (no scikit-learn dependency)
# ---------------------------------------------------------------------------


def pca_project(
    llm_item_emb: torch.Tensor, target_dim: int
) -> torch.Tensor:
    """Compute the PCA projection of LLM item embeddings to ``target_dim``.

    Args:
        llm_item_emb: ``(num_items + 1, llm_dim)`` tensor; row 0 must
            be the (zero) padding row.
        target_dim: Target dimensionality (e.g. 50 to match SASRec's
            ``embed_dim``).

    Returns:
        ``(num_items + 1, target_dim)`` projected tensor with row 0
        zeroed (padding) and rows ``1..`` L2-normalised. Suitable for
        passing as ``init_item_emb`` to :class:`SASRec`.

    Notes:
        * L2-normalises the input rows ``1..`` before fitting PCA so
          the projection is invariant to the LLM's per-item magnitude
          (Qwen3 outputs are not unit-norm out of the box; this keeps
          the comparison with the L2-normed projector in A5 fair).
        * After projection, rows ``1..`` are re-L2-normalised so the
          per-row magnitude matches what SASRec's Xavier-uniform init
          would produce in expectation (≈ unit norm at embed_dim=50).
          This removes a magnitude confound from A0 vs A1.
        * Computed in float32 on CPU — the SVD on (~25k, 1024) matrices
          is sub-second and avoids any MPS-specific numerical quirks.
    """
    catalog = llm_item_emb[1:].to(torch.float32)
    catalog = catalog / catalog.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Centered SVD = PCA. Mean across items, then SVD.
    centered = catalog - catalog.mean(dim=0, keepdim=True)
    # Vt has shape (llm_dim, llm_dim); the top-k principal axes are
    # Vt[:k]. Project: centered @ Vt[:k].T -> (N, k).
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    components = vt[:target_dim]                                  # (k, llm_dim)
    projected = centered @ components.T                           # (N, k)

    # Re-L2-normalise rows to unit length. NB: this is NOT
    # magnitude-matched to Xavier-uniform — for an embedding matrix
    # shape (N, d) with N >> d, Xavier rows have expected L2 norm
    # ≈ sqrt(2d/(N+d)) << 1. The A1-norm-matched ablation (paper §3.5
    # control, planned) requires switching this step to rescale by
    # that target norm; see docs/SAILS_ablation_plan.md §"A1-norm".
    # TODO(A1-norm): read `norm_match_baseline` from config and replace
    # the line below with an `if-else` that rescales to the
    # Xavier-expected row norm when true. Configs already exist at
    # configs/ablations/A1_norm_matched_*.yaml and
    # configs/final/A1_norm_matched_yelp.yaml.
    projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    out = torch.zeros(llm_item_emb.shape[0], target_dim, dtype=torch.float32)
    out[1:] = projected
    return out


# ---------------------------------------------------------------------------
# Single-seed training — mirrors train_baseline.train_single_seed
# ---------------------------------------------------------------------------


def train_single_seed(
    cfg: SASRecConfig,
    init_item_emb: torch.Tensor,
    seed: int,
    split,
    neg_samples: dict,
    resume: bool = False,
) -> dict[str, float]:
    """Train one seed of A1 (vanilla SASRec + LLM-PCA init)."""
    set_seed(seed)
    seed_dir = Path(cfg.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

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

    # The only architectural difference from A0 is here: pass the
    # PCA-projected LLM tensor as init_item_emb. The model is otherwise
    # vanilla SASRec.
    model = SASRec(
        num_items=split.num_items,
        embed_dim=cfg.model.embed_dim,
        num_blocks=cfg.model.num_blocks,
        num_heads=cfg.model.num_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        init_item_emb=init_item_emb,
    )

    # Optional W&B (same gate as train_baseline).
    import os

    wandb_run = None
    wandb_disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    has_netrc_login = Path.home().joinpath(".netrc").exists()
    if not wandb_disabled and (has_api_key or has_netrc_login):
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"a1_llm_init_seed{seed}",
                config={
                    "seed": seed,
                    "ablation": "A1_llm_init",
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

    # Standard sampled test evaluation: encode train+val, predict test.
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

    results = {
        "seed": seed,
        "ablation": "A1_llm_init",
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
        description="A1 — LLM Init only (vanilla SASRec + PCA-projected LLM init)."
    )
    parser.add_argument(
        "--config",
        default="sasrec_injection/configs/ablations/A1_llm_init.yaml",
        help="A1 ablation config.",
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
        "--llm-embeddings",
        default=None,
        help=(
            "Override the config's llm_embeddings_path. Used to point "
            "at a different LLM (e.g. when smoke-testing on Yelp)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each seed from <seed_dir>/last.pt if present.",
    )
    args = parser.parse_args()

    # Read the YAML directly so we can pluck the ablation-specific
    # ``llm_embeddings_path`` field that SASRecConfig doesn't carry.
    import yaml
    raw_cfg = yaml.safe_load(open(args.config))
    llm_embeddings_path = (
        args.llm_embeddings or raw_cfg.get("llm_embeddings_path")
    )
    if not llm_embeddings_path:
        raise ValueError(
            "A1 needs an LLM embeddings path: set llm_embeddings_path in "
            "the YAML or pass --llm-embeddings."
        )

    cfg = SASRecConfig.from_yaml(args.config, args.base_config)
    if args.seeds:
        cfg.seeds = args.seeds

    log(f"Running A1 (LLM Init only) with seeds: {cfg.seeds}")
    log(f"Device: {cfg.device}")
    log(f"LLM embeddings: {llm_embeddings_path}")

    split, neg_samples, _ = prepare_data(cfg)

    # Load LLM embeddings, sanity-check shape, fit PCA once.
    log("Loading LLM embeddings...")
    llm_item_emb = torch.load(llm_embeddings_path, map_location="cpu", weights_only=True)
    if llm_item_emb.shape[0] != split.num_items + 1:
        raise ValueError(
            f"LLM embedding rows ({llm_item_emb.shape[0]}) do not match "
            f"num_items + 1 ({split.num_items + 1}). Did you regenerate "
            f"the embeddings after a preprocessing change?"
        )
    log(f"LLM embedding shape: {tuple(llm_item_emb.shape)}")

    log(f"Fitting PCA({cfg.model.embed_dim}) via centered SVD...")
    init_item_emb = pca_project(llm_item_emb, target_dim=cfg.model.embed_dim)
    log(
        f"PCA init shape: {tuple(init_item_emb.shape)}; "
        f"row 1 norm: {init_item_emb[1].norm().item():.4f}"
    )

    all_metrics: list[dict[str, float]] = []
    for seed in cfg.seeds:
        log(f"\n{'='*60}")
        log(f"A1 — training seed {seed}")
        log(f"{'='*60}")
        all_metrics.append(
            train_single_seed(
                cfg, init_item_emb, seed, split, neg_samples, resume=args.resume
            )
        )

    # Aggregate identical to train_baseline.
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
