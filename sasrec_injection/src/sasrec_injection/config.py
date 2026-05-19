"""YAML → dataclass configuration loaders for ItemTable experiments.

Two configs in this module:

* :class:`SASRecConfig`       — vanilla SASRec (used by ``train_baseline.py`` and
                            by ``eval_fullrank.py``).
* :class:`ItemTableConfig`  — ItemTable method (used by ``train_item_table.py``).

Both load from a per-experiment YAML, deeply merged onto a shared
``base.yaml``. The merge is dead-simple (recursive dict update) — we
intentionally avoid Hydra/OmegaConf because every experiment in this
folder fits in <100 lines of YAML and the indirection isn't worth it.

Adding a new config
-------------------
1. Define a new ``@dataclass`` for the method's hyperparameters
   (e.g. ``ContrastiveAlignConfig``).
2. Define the top-level ``@dataclass`` that composes it (e.g.
   ``ItemTableConfig``).
3. Implement ``from_dict`` and ``from_yaml`` classmethods. Use
   ``load_config`` for the merge.
4. Add a per-experiment yaml under ``sasrec_injection/configs/``.

Tensor / shape conventions
--------------------------
None of the dataclasses here own tensors — they only carry numbers /
paths / strings. Tensor shape conventions are documented in the
modules that actually allocate them (``models/sasrec.py``,
``alignment/contrastive.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file into a Python dict.

    Args:
        path: Filesystem path to a ``.yaml`` file.

    Returns:
        The parsed top-level mapping. May contain nested dicts/lists.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base``; ``override`` wins on conflicts.

    Used to layer a per-experiment yaml on top of ``configs/base.yaml``.
    Lists are *not* merged element-wise: if both sides define a list at
    the same key, ``override`` replaces ``base`` wholesale. This matches
    user expectations (e.g. an experiment's ``seeds: [42]`` should fully
    override the base ``seeds: [42, 123, 456, 789, 1024]``).
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path,
    base_path: str | Path | None = None,
) -> dict:
    """Load an experiment YAML, optionally merged with a base YAML.

    Args:
        config_path: Path to the per-experiment yaml.
        base_path: Optional path to a base yaml to merge under
            ``config_path``. Standard pattern: pass
            ``configs/base.yaml`` so device / wandb defaults flow
            through.

    Returns:
        The merged configuration as a plain dict, ready to feed into
        the appropriate ``ConfigClass.from_dict``.
    """
    config = load_yaml(config_path)
    if base_path is not None:
        base = load_yaml(base_path)
        config = merge_configs(base, config)
    return config


# ---------------------------------------------------------------------------
# Sub-configs (shared across P1 and ItemTable)
# ---------------------------------------------------------------------------


@dataclass
class SASRecConfig:
    """SASRec architecture hyperparameters.

    These are intentionally identical between P1 and ItemTable — ItemTable
    does not change the model, only the training objective. Defaults
    match the LLM4Rec literature (Kang & McAuley 2018; DLLM2Rec; etc.)
    so external numbers are directly comparable.

    Attributes:
        embed_dim: Item embedding dimension. Also the dimension of the
            transformer hidden state and of every per-position output.
            50 is the canonical SASRec default.
        num_blocks: Transformer-block depth. 2 blocks is enough at
            embed_dim=50 to match published SASRec numbers.
        num_heads: Attention heads. 1 is canonical; multi-head doesn't
            buy much at this width.
        max_seq_len: Maximum context length. Sequences longer than
            this are *left-truncated* in the dataset class; the most
            recent ``max_seq_len`` items are kept.
        dropout: Applied to embeddings, attention output, and FFN
            output. 0.2 is the canonical default for ML-1M / Amazon.
    """

    embed_dim: int = 50
    num_blocks: int = 2
    num_heads: int = 1
    max_seq_len: int = 200
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Optimisation hyperparameters.

    Attributes:
        batch_size: Number of users per training batch. Larger
            batches give better InfoNCE negative pools (each batch
            user provides ~1 negative for the alignment loss),
            so 512 is preferred over the ML-1M default of 128.
        lr: Adam learning rate. 1e-3 is the canonical SASRec default.
        max_epochs: Hard upper bound. Early stopping almost always
            terminates earlier (typically epoch 30-150).
        early_stopping_patience: Number of consecutive epochs without
            improvement before stopping. Tuned to be tight enough that
            we don't waste compute on plateaus, loose enough that
            short val-metric oscillations don't stop training prematurely.
        early_stopping_metric: Which val metric to monitor. NDCG@10 is
            standard in the LLM4Rec literature.
        early_stopping_min_delta: Minimum improvement to reset the
            patience counter. 1e-4 ≈ "0.0001 NDCG@10" — small enough
            to track real progress, large enough to ignore numerical
            noise.
    """

    batch_size: int = 128
    lr: float = 0.001
    max_epochs: int = 200
    early_stopping_patience: int = 10
    early_stopping_metric: str = "ndcg@10"
    early_stopping_min_delta: float = 1e-4


@dataclass
class EvalConfig:
    """Evaluation knobs.

    Attributes:
        num_neg_samples: Number of negatives sampled per user for the
            "sampled" evaluation protocol used during validation.
            100 is standard. Sampled metrics tend to over-estimate
            absolute ranking quality vs full-rank but track relative
            improvements well, so we use them for early stopping and
            run full-rank only at the end.
        ks: Cutoffs for HR@K / NDCG@K / Recall@K reporting.
    """

    num_neg_samples: int = 100
    ks: list[int] = field(default_factory=lambda: [5, 10, 20])


# ---------------------------------------------------------------------------
# P1: vanilla SASRec
# ---------------------------------------------------------------------------


@dataclass
class SASRecConfig:
    """Top-level config for vanilla SASRec training (no LLM signal).

    The same dataclass also drives full-rank evaluation
    (``eval_fullrank.py``) — we re-use ``output_dir`` to find saved
    checkpoints and ``model`` to construct the right shape on load.

    Attributes:
        model: Architecture knobs (SASRecConfig).
        training: Optimisation knobs (TrainingConfig).
        evaluation: Eval knobs (EvalConfig).
        seeds: List of integer seeds for this experiment. Every seed
            produces a separate ``seed_<S>/best_model.pt``; aggregate
            metrics are mean ± std across seeds.
        device: Torch device string. ``"mps"``/``"cuda"``/``"cpu"``.
        output_dir: Filesystem root for this experiment's artefacts.
        dataset_name: Dataset identifier (e.g. ``"amazon-Video_Games"``).
        min_interactions: 5-core filter threshold.
        wandb_project / wandb_entity: Weights & Biases tracking, opt-in.
    """

    model: SASRecConfig
    training: TrainingConfig
    evaluation: EvalConfig
    seeds: list[int]
    device: str
    output_dir: str
    dataset_name: str = "amazon-Video_Games"
    min_interactions: int = 5
    wandb_project: str = "sasrec_injection"
    wandb_entity: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SASRecConfig":
        """Build a SASRecConfig from a plain dict (typically from YAML)."""
        model_cfg = SASRecConfig(**d.get("model", {}))
        train_cfg = TrainingConfig(**d.get("training", {}))
        eval_cfg = EvalConfig(**d.get("evaluation", {}))
        dataset = d.get("dataset", {})
        tracking = d.get("tracking", {})
        return cls(
            model=model_cfg,
            training=train_cfg,
            evaluation=eval_cfg,
            seeds=d.get("seeds", [42]),
            device=d.get("device", "mps"),
            output_dir=d.get("output_dir", "sasrec_injection/outputs/p1"),
            dataset_name=dataset.get("name", "amazon-Video_Games"),
            min_interactions=dataset.get("min_interactions", 5),
            wandb_project=tracking.get("wandb_project", "sasrec_injection"),
            wandb_entity=tracking.get("wandb_entity"),
        )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        base_path: str | Path | None = None,
    ) -> "SASRecConfig":
        """Convenience: read YAML, merge base, return populated config."""
        return cls.from_dict(load_config(config_path, base_path))


# ---------------------------------------------------------------------------
# ItemTable: vanilla SASRec + InfoNCE alignment
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveAlignConfig:
    """ItemTable auxiliary-loss hyperparameters.

    Attributes:
        embeddings_path: Path to the frozen LLM item-embedding tensor
            ``(num_items + 1, llm_dim)`` produced by
            ``scripts/extract_llm_embeddings.py``. Row 0 must be padding
            (zero row).
        lambda_align: Weight on ``L_align`` in the total objective
            ``L_total = L_next + lambda_align * L_align``. Empirically
            the best on Video_Games is 0.1; values above 0.5 start to
            overpower the BCE objective.
        temperature: InfoNCE softmax temperature. 0.1 is our default
            for L2-normalised Qwen3 embeddings; CLIP uses 0.07.
        projector_hidden_dim: ``0`` for a single bias-free
            ``Linear(llm_dim, embed_dim)`` (current default); ``> 0``
            for a 2-layer MLP with ReLU+Dropout in between.
        projector_dropout: Dropout in the MLP projector. Ignored when
            ``projector_hidden_dim == 0``.
        llm_dim: Feature dimension of the LLM embeddings on disk. Must
            match the tensor shape exactly. 1024 for
            Qwen3-Embedding-0.6B; 4096 for the 8B variant.
        max_align_ids: Optional cap on the number of unique non-padding
            item ids passed through the InfoNCE matmul each batch. The
            loss cost is dominated by an ``(N, N)`` similarity matrix;
            ``N`` is the unique-id count which scales with both batch
            size and catalog density. On Video_Games (25K items)
            ``N`` settles at ~10-15K and no cap is needed (default
            ``None`` = no cap). On Yelp (148K items) ``N`` can spike
            to 25-40K, with the matmul peaking at 5-12 GB during the
            backward pass — set ``max_align_ids = 8000`` (or similar)
            to bound peak memory to ~500 MB. Random-subsampling
            preserves the InfoNCE objective in expectation.
    """

    embeddings_path: str = "sasrec_injection/outputs/embeddings/video_games.pt"
    lambda_align: float = 0.1
    temperature: float = 0.1
    projector_hidden_dim: int = 0
    projector_dropout: float = 0.1
    llm_dim: int = 1024
    max_align_ids: int | None = None


@dataclass
class ItemTableConfig:
    """Top-level config for ItemTable training.

    Identical schema to :class:`SASRecConfig` plus an ``align`` block that
    holds the auxiliary-loss hyperparameters. The architecture (``model``)
    is unchanged: ItemTable does not modify SASRec.
    """

    model: SASRecConfig
    training: TrainingConfig
    evaluation: EvalConfig
    align: ContrastiveAlignConfig
    seeds: list[int]
    device: str
    output_dir: str
    dataset_name: str = "amazon-Video_Games"
    min_interactions: int = 5
    wandb_project: str = "sasrec_injection"
    wandb_entity: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ItemTableConfig":
        """Build a ItemTableConfig from a plain dict (typically from YAML)."""
        model_cfg = SASRecConfig(**d.get("model", {}))
        train_cfg = TrainingConfig(**d.get("training", {}))
        eval_cfg = EvalConfig(**d.get("evaluation", {}))
        align_cfg = ContrastiveAlignConfig(**d.get("align", {}))
        dataset = d.get("dataset", {})
        tracking = d.get("tracking", {})
        return cls(
            model=model_cfg,
            training=train_cfg,
            evaluation=eval_cfg,
            align=align_cfg,
            seeds=d.get("seeds", [42]),
            device=d.get("device", "mps"),
            output_dir=d.get("output_dir", "sasrec_injection/outputs/sasrec_injection"),
            dataset_name=dataset.get("name", "amazon-Video_Games"),
            min_interactions=dataset.get("min_interactions", 5),
            wandb_project=tracking.get("wandb_project", "sasrec_injection"),
            wandb_entity=tracking.get("wandb_entity"),
        )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        base_path: str | Path | None = None,
    ) -> "ItemTableConfig":
        """Convenience: read YAML, merge base, return populated config."""
        return cls.from_dict(load_config(config_path, base_path))
