"""ItemTableTrainer — SASRec + auxiliary InfoNCE alignment.

Trainer for the ItemTable method. Subclasses :class:`BaseTrainer` and
adds:

1. A frozen LLM item-embedding table held on-device.
2. A trainable :class:`AlignProjector` (lives only on the trainer).
3. A modified :meth:`train_epoch` that adds the InfoNCE alignment loss
   to BCE every batch:

::

    L_total = L_next + lambda_align * L_align

The model itself is unchanged from the P1 baseline. At inference, the
saved ``best_model.pt`` is loaded as a plain SASRec — no projector, no
LLM lookup. See ``../../docs/B1_method.md`` §3 for the full method
description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sasrec_injection.alignment.contrastive import (
    AlignProjector,
    infonce_align_loss,
    weighted_infonce_align_loss,
)
from sasrec_injection.training.losses import bce_loss
from sasrec_injection.training.trainer import BaseTrainer


class ItemTableTrainer(BaseTrainer):
    """SASRec + item-level InfoNCE alignment to frozen LLM embeddings.

    Differs from :class:`BaseTrainer` in two places:

    1. Holds the frozen LLM item-embedding table and an
       :class:`AlignProjector` (``llm_dim → embed_dim``).
    2. :meth:`train_epoch` adds ``lambda_align * infonce_align_loss``
       on the unique non-padding item ids touched by the batch
       (positives ∪ negatives — exactly the rows BCE is updating).

    Args:
        model: SASRec instance (any architecture, but the headline
            numbers were measured on the standard ``embed_dim=50``
            shape).
        train_loader: Yields ``(seq, pos, neg)`` triples, padded.
        val_loader: Yields ``(seq, candidates)`` pairs for sampled eval.
        llm_item_emb: Frozen LLM item-embedding tensor, shape
            ``(num_items + 1, llm_dim)``. Row 0 must be padding (zero).
        device: Compute device.
        lr: Adam learning rate. Same as P1; we use one optimiser over
            both the SASRec parameters and the AlignProjector.
        max_epochs: Hard upper bound; early stopping usually fires
            sooner.
        lambda_align: Weight on ``L_align`` in the total objective.
            0.1 is the empirical sweet spot on Video_Games.
        temperature: InfoNCE temperature. 0.1 default.
        projector_hidden_dim: 0 for linear; > 0 for MLP. See
            :class:`AlignProjector`.
        projector_dropout: Dropout in the MLP projector.
        early_stopping_patience / metric / min_delta: standard early
            stopping knobs (see :class:`EarlyStopping`).
        output_dir: Filesystem root for this run's artefacts.
        wandb_run: Optional W&B run object; opt-in tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        llm_item_emb: torch.Tensor,
        device: torch.device,
        lr: float = 0.001,
        max_epochs: int = 200,
        lambda_align: float = 0.1,
        temperature: float = 0.1,
        projector_hidden_dim: int = 0,
        projector_dropout: float = 0.1,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "ndcg@10",
        early_stopping_min_delta: float = 1e-4,
        output_dir: str | Path = "outputs",
        wandb_run: Any = None,
        max_align_ids: int | None = None,
        item_freq: torch.Tensor | None = None,
        weight_fn: str = "log",
    ):
        """
        Additional argument:
            max_align_ids: Optional cap on the number of unique non-
                padding item ids passed to the InfoNCE auxiliary loss
                each batch. The cost of the loss is dominated by an
                ``(N, N)`` matmul where ``N`` = unique ids in the
                batch; on small catalogs (Video_Games, 25K items)
                ``N`` settles at ~10-15K and no cap is needed, but on
                a much larger catalog (Yelp, 148K items) ``N`` can
                spike to 25-40K and the matmul peaks at 5-12 GB. When
                ``max_align_ids`` is set and the batch's unique-id
                count exceeds it, we uniformly random-subsample down
                to the cap. The InfoNCE loss is unbiased under uniform
                subsampling — every retained id still sees every other
                retained id as a negative — so this is a clean memory
                bound, not an algorithm change. Set to ``None`` to
                disable (the cap is a no-op on Video_Games anyway).
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=lr,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_min_delta=early_stopping_min_delta,
            output_dir=output_dir,
            wandb_run=wandb_run,
        )

        self.lambda_align = lambda_align
        self.temperature = temperature
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_dropout = projector_dropout
        self.max_align_ids = max_align_ids

        # Frequency-weighted alignment. Pre-compute per-item weights from
        # raw training counts using the selected weight_fn, then normalise
        # to mean 1 so overall loss magnitude is comparable to unweighted.
        # Stored on CPU; moved to device per-batch in the training loop.
        if item_freq is not None:
            f = item_freq.float()
            if weight_fn == "log":
                raw = 1.0 / (1.0 + torch.log(f + 1.0))
            elif weight_fn == "sqrt":
                raw = 1.0 / (torch.sqrt(f + 1.0))
            elif weight_fn == "linear":
                raw = 1.0 / (f + 1.0)
            elif weight_fn == "binary":
                # Items in the bottom 20% (tail) get weight 2, rest get 0.5.
                # Uses the same 20th-percentile threshold as stratified eval.
                threshold = torch.quantile(f[f > 0], 0.20)
                raw = torch.where(f <= threshold, torch.full_like(f, 2.0), torch.full_like(f, 0.5))
            else:
                raise ValueError(f"Unknown weight_fn {weight_fn!r}. Choose: log, sqrt, linear, binary")
            raw[0] = 0.0  # padding row
            self.item_weights: torch.Tensor | None = raw / raw[1:].mean()
        else:
            self.item_weights = None
        # The LLM table dimension is implied by its shape — capturing
        # it here makes ``_summary_config`` recordable without the
        # tensor itself.
        self.llm_dim = int(llm_item_emb.size(1))

        # Move the frozen LLM table to the training device once.
        # ``requires_grad_(False)`` ensures Adam never touches it.
        self.llm_item_emb = llm_item_emb.to(device)
        self.llm_item_emb.requires_grad_(False)

        # The trainable projection from llm_dim → embed_dim. Lives on
        # the trainer (NOT the model) so saved checkpoints stay
        # bit-shape-identical to P1 baselines.
        embed_dim = getattr(self.model, "embed_dim", 50)
        self.projector = AlignProjector(
            llm_dim=self.llm_dim,
            student_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        ).to(device)

        # Replace the BaseTrainer's optimiser with one that includes
        # the projector parameters. Single Adam over the union keeps
        # learning rates / momentum consistent across the two
        # parameter groups.
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.projector.train()

        total_rec = total_align = total_combined = 0.0
        n_batches = 0

        for seq, pos, neg in self.train_loader:
            seq = seq.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            # ----------------------------------------------------------
            # 1. Standard SASRec next-item BCE — identical to P1.
            # ----------------------------------------------------------
            seq_repr = self.model(seq)                              # (B, L, D)
            pos_emb = self.model.item_emb(pos)
            neg_emb = self.model.item_emb(neg)
            pos_logits = (seq_repr * pos_emb).sum(dim=-1)
            neg_logits = (seq_repr * neg_emb).sum(dim=-1)
            mask = (pos > 0).float()
            loss_rec = bce_loss(pos_logits, neg_logits, mask)

            # ----------------------------------------------------------
            # 2. ItemTable auxiliary loss.
            #
            # Use the unique non-padding ids touched by this batch
            # (positives ∪ negatives). These are exactly the rows of
            # ``item_emb`` that BCE is updating, so aligning them
            # against their LLM counterparts costs no extra rows.
            # ----------------------------------------------------------
            ids = torch.cat([pos.flatten(), neg.flatten()])
            ids = ids[ids > 0].unique()

            # Bound the InfoNCE (N, N) matmul peak when the batch
            # touches a large number of unique items (Yelp / large
            # catalogs). Uniform random subsampling preserves the
            # InfoNCE objective in expectation — every retained id
            # still sees every other retained id as a negative.
            if (
                self.max_align_ids is not None
                and ids.numel() > self.max_align_ids
            ):
                perm = torch.randperm(ids.numel(), device=ids.device)
                ids = ids[perm[: self.max_align_ids]]

            if ids.numel() < 2:
                # Degenerate batch (everything padded). Returning a
                # 0 with ``requires_grad=True`` keeps ``backward()``
                # happy without contributing to any update.
                loss_align = torch.zeros(
                    (), device=self.device, requires_grad=True
                )
            else:
                # Look up the student and LLM rows, project, compute
                # InfoNCE. See ``alignment/contrastive.py`` for the
                # math.
                student_embs = self.model.item_emb(ids)             # (N, D)
                llm_raw = self.llm_item_emb.index_select(0, ids)    # (N, llm_dim)
                projected_llm = self.projector(llm_raw)             # (N, D)
                if self.item_weights is not None:
                    w = self.item_weights[ids.cpu()].to(self.device)
                    loss_align = weighted_infonce_align_loss(
                        student_embs, projected_llm, w,
                        temperature=self.temperature,
                    )
                else:
                    loss_align = infonce_align_loss(
                        student_embs, projected_llm,
                        temperature=self.temperature,
                    )

            # ----------------------------------------------------------
            # 3. Combined loss + Adam step.
            # ----------------------------------------------------------
            loss = loss_rec + self.lambda_align * loss_align

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_rec += loss_rec.item()
            total_align += loss_align.item()
            total_combined += loss.item()
            n_batches += 1

        return {
            "loss_rec": total_rec / n_batches,
            "loss_align": total_align / n_batches,
            "loss_combined": total_combined / n_batches,
        }

    # ------------------------------------------------------------------
    # Cosmetic / bookkeeping overrides
    # ------------------------------------------------------------------

    def _format_loss_log_line(self, losses: dict[str, float]) -> str:
        """Pretty-print the three loss components in a fixed order.

        Format: ``rec=X.XXXX align=Y.YYYY total=Z.ZZZZ``. ``align``
        starts near ``log(N) ≈ 5.5–6.5`` for random init and should
        drop into the 1.0–1.5 range as the projector fits.
        """
        return (
            f"rec={losses['loss_rec']:.4f} "
            f"align={losses['loss_align']:.4f} "
            f"total={losses['loss_combined']:.4f}"
        )

    def _summary_config(self) -> dict[str, Any]:
        """Add ItemTable-specific knobs to the train_summary.json config."""
        cfg = super()._summary_config()
        cfg.update({
            "lambda_align": self.lambda_align,
            "temperature": self.temperature,
            "llm_dim": self.llm_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projector_dropout": self.projector_dropout,
        })
        return cfg

    # ------------------------------------------------------------------
    # Resume hooks
    # ------------------------------------------------------------------
    #
    # The base trainer already snapshots model + optimizer + RNG +
    # early-stopping bookkeeping. ItemTable also owns a trainable
    # ``AlignProjector`` that lives on the trainer (not the model);
    # without snapshotting it, resume would silently re-randomise the
    # projector and lose all of its training. We persist its
    # ``state_dict`` under the ``extra`` key.

    def _extra_resume_state(self) -> dict[str, Any]:
        return {"projector": self.projector.state_dict()}

    def _load_extra_resume_state(self, state: dict[str, Any]) -> None:
        if "projector" in state:
            self.projector.load_state_dict(state["projector"])
