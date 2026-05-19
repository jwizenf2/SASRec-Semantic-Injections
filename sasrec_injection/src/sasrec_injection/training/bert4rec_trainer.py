"""BERT4Rec trainers — Cloze-task baseline and ItemTable variant.

Two trainers:

* :class:`BERT4RecTrainer`      — P1 equivalent for BERT4Rec. Trains with
                                  masked Cloze cross-entropy only.
* :class:`BERT4RecSAILTrainer`  — A8 equivalent for BERT4Rec. Adds the
                                  ItemTable InfoNCE alignment loss on top of
                                  the Cloze loss.

Both subclass :class:`sasrec_injection.training.trainer.BaseTrainer` and override
only :meth:`train_epoch`, so checkpointing, early stopping, JSONL logging,
and resume behaviour are identical to SASRec / GRU4Rec trainers.

Training data format
--------------------

Unlike SASRec trainers (which consume ``(seq, pos, neg)`` triples from
:class:`~sasrec_injection.data.dataset.SASRecTrainDataset`), BERT4Rec trainers
expect ``(masked_seq, labels)`` pairs from
:class:`~sasrec_injection.data.dataset.BERT4RecTrainDataset`:

* ``masked_seq``: ``(B, L)`` — input sequence with some item positions
  replaced by ``model.mask_token_id``.
* ``labels``: ``(B, L)`` — original item id at masked positions, ``0``
  everywhere else. Loss is computed only where ``labels > 0``.

Cloze loss
----------

At masked positions we score all ``num_items`` items with a single matmul
and compute categorical cross-entropy:

::

    masked_repr = seq_repr[labels > 0]          # (M, D)
    logits = masked_repr @ item_emb[1:N+1].T    # (M, N)  skip pad + MASK rows
    loss = cross_entropy(logits, labels[labels > 0] - 1)   # 0-indexed targets

ItemTable alignment (BERT4RecSAILTrainer only)
--------------------------------------------

The InfoNCE alignment loss is identical to ItemTableTrainer: look up the unique
non-padding items that appear as MASK targets in this batch, fetch their
student and LLM embeddings, and compute symmetric InfoNCE. The combined
objective is:

::

    L_total = L_cloze + lambda_align * L_align
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sasrec_injection.alignment.contrastive import (
    AlignProjector,
    infonce_align_loss,
    weighted_infonce_align_loss,
)
from sasrec_injection.training.trainer import BaseTrainer


class BERT4RecTrainer(BaseTrainer):
    """P1-equivalent trainer for BERT4Rec — Cloze loss only.

    Args:
        model: :class:`~sasrec_injection.models.bert4rec.BERT4Rec` instance.
        train_loader: Yields ``(masked_seq, labels)`` pairs from
            :class:`~sasrec_injection.data.dataset.BERT4RecTrainDataset`.
        val_loader: Yields ``(seq, candidates)`` pairs for sampled eval
            (:class:`~sasrec_injection.data.dataset.SASRecEvalDataset` — same as
            SASRec; no masking during evaluation).
        device / lr / max_epochs / early_stopping_* / output_dir / wandb_run:
            Identical semantics to
            :class:`~sasrec_injection.training.trainer.BaseTrainer`.
    """

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for masked_seq, labels in self.train_loader:
            masked_seq = masked_seq.to(self.device)
            labels     = labels.to(self.device)

            seq_repr = self.model(masked_seq)          # (B, L, D)

            # Gather only the masked positions.
            mask = labels > 0                          # (B, L) bool
            if not mask.any():
                continue

            pred    = seq_repr[mask]                   # (M, D)
            targets = labels[mask]                     # (M,) — 1-indexed

            # Score against all real items (skip padding row 0 and MASK
            # row num_items+1).
            item_embs = self.model.item_emb.weight[
                1 : self.model.num_items + 1
            ]                                          # (N, D)
            logits = pred @ item_embs.T                # (M, N)

            # Cross-entropy wants 0-indexed class labels.
            loss = F.cross_entropy(logits, targets - 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def _format_loss_log_line(self, losses: dict[str, float]) -> str:
        return f"loss={losses['train_loss']:.4f}"


class BERT4RecSAILTrainer(BaseTrainer):
    """A8-equivalent trainer for BERT4Rec — Cloze + InfoNCE alignment.

    Combines the BERT4Rec Cloze objective with the ItemTable item-table
    InfoNCE alignment loss. The alignment component is identical to
    :class:`~sasrec_injection.training.item_table_trainer.ItemTableTrainer`: a frozen
    LLM embedding table is projected into the student embedding space and
    the InfoNCE loss pulls matched pairs together.

    The combined loss is:

    ::

        L_total = L_cloze + lambda_align * L_align

    Args:
        model: :class:`~sasrec_injection.models.bert4rec.BERT4Rec` instance.
        train_loader: Yields ``(masked_seq, labels)`` pairs.
        val_loader: Yields ``(seq, candidates)`` for sampled eval.
        llm_item_emb: Frozen LLM embedding tensor ``(num_items+1, llm_dim)``.
            Row 0 must be zero (padding). Note: the LLM table has
            ``num_items+1`` rows (no MASK row); only real items are aligned.
        device / lr / max_epochs / early_stopping_* / output_dir / wandb_run:
            Standard trainer arguments.
        lambda_align: Weight on ``L_align``.
        temperature: InfoNCE temperature.
        projector_hidden_dim: 0 for linear projector; > 0 for MLP.
        projector_dropout: Dropout inside MLP projector.
        max_align_ids: Optional cap on unique item ids per batch for the
            InfoNCE matmul. Keeps peak memory bounded on large catalogs.
        item_freq: Optional ``(num_items+1,)`` training frequency tensor
            for frequency-weighted alignment.
        weight_fn: Weighting scheme for ``item_freq``:
            ``"log"`` | ``"sqrt"`` | ``"linear"`` | ``"binary"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
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

        self.lambda_align    = lambda_align
        self.temperature     = temperature
        self.max_align_ids   = max_align_ids
        self.llm_dim         = int(llm_item_emb.size(1))
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_dropout    = projector_dropout

        # Frequency-weighted item weights — same formula as ItemTableTrainer.
        if item_freq is not None:
            f = item_freq.float()
            if weight_fn == "log":
                raw = 1.0 / (1.0 + torch.log(f + 1.0))
            elif weight_fn == "sqrt":
                raw = 1.0 / (torch.sqrt(f + 1.0))
            elif weight_fn == "linear":
                raw = 1.0 / (f + 1.0)
            elif weight_fn == "binary":
                threshold = torch.quantile(f[f > 0], 0.20)
                raw = torch.where(
                    f <= threshold,
                    torch.full_like(f, 2.0),
                    torch.full_like(f, 0.5),
                )
            else:
                raise ValueError(f"Unknown weight_fn {weight_fn!r}")
            raw[0] = 0.0
            self.item_weights: torch.Tensor | None = raw / raw[1:].mean()
        else:
            self.item_weights = None

        self.llm_item_emb = llm_item_emb.to(device)
        self.llm_item_emb.requires_grad_(False)

        embed_dim = getattr(model, "embed_dim", 64)
        self.projector = AlignProjector(
            llm_dim=self.llm_dim,
            student_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr,
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.projector.train()

        total_cloze = total_align = total_combined = 0.0
        n_batches = 0

        for masked_seq, labels in self.train_loader:
            masked_seq = masked_seq.to(self.device)
            labels     = labels.to(self.device)

            # ----------------------------------------------------------
            # 1. Cloze loss — cross-entropy at masked positions.
            # ----------------------------------------------------------
            seq_repr = self.model(masked_seq)       # (B, L, D)
            mask = labels > 0
            if mask.any():
                pred    = seq_repr[mask]            # (M, D)
                targets = labels[mask]              # (M,) 1-indexed
                item_embs = self.model.item_emb.weight[
                    1 : self.model.num_items + 1
                ]
                logits     = pred @ item_embs.T     # (M, N)
                loss_cloze = F.cross_entropy(logits, targets - 1)
            else:
                loss_cloze = torch.zeros((), device=self.device, requires_grad=True)

            # ----------------------------------------------------------
            # 2. InfoNCE alignment — on unique masked item ids.
            # Using items that appear as MASK targets is consistent with
            # the Cloze loss: we only align embeddings the model is
            # actively being trained to predict.
            # ----------------------------------------------------------
            ids = labels[mask] if mask.any() else torch.tensor([], dtype=torch.long, device=self.device)
            ids = ids[ids > 0].unique()

            if self.max_align_ids is not None and ids.numel() > self.max_align_ids:
                perm = torch.randperm(ids.numel(), device=ids.device)
                ids  = ids[perm[: self.max_align_ids]]

            if ids.numel() < 2:
                loss_align = torch.zeros((), device=self.device, requires_grad=True)
            else:
                student_embs = self.model.item_emb(ids)
                llm_raw      = self.llm_item_emb.index_select(0, ids)
                projected    = self.projector(llm_raw)
                if self.item_weights is not None:
                    w = self.item_weights[ids.cpu()].to(self.device)
                    loss_align = weighted_infonce_align_loss(
                        student_embs, projected, w, temperature=self.temperature
                    )
                else:
                    loss_align = infonce_align_loss(
                        student_embs, projected, temperature=self.temperature
                    )

            # ----------------------------------------------------------
            # 3. Combined loss + step.
            # ----------------------------------------------------------
            loss = loss_cloze + self.lambda_align * loss_align

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_cloze    += loss_cloze.item()
            total_align    += loss_align.item()
            total_combined += loss.item()
            n_batches      += 1

        denom = max(n_batches, 1)
        return {
            "loss_cloze":    total_cloze    / denom,
            "loss_align":    total_align    / denom,
            "loss_combined": total_combined / denom,
        }

    def _format_loss_log_line(self, losses: dict[str, float]) -> str:
        return (
            f"cloze={losses['loss_cloze']:.4f} "
            f"align={losses['loss_align']:.4f} "
            f"total={losses['loss_combined']:.4f}"
        )

    def _summary_config(self) -> dict[str, Any]:
        cfg = super()._summary_config()
        cfg.update({
            "lambda_align":        self.lambda_align,
            "temperature":         self.temperature,
            "llm_dim":             self.llm_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projector_dropout":   self.projector_dropout,
        })
        return cfg

    def _extra_resume_state(self) -> dict[str, Any]:
        return {"projector": self.projector.state_dict()}

    def _load_extra_resume_state(self, state: dict[str, Any]) -> None:
        if "projector" in state:
            self.projector.load_state_dict(state["projector"])
