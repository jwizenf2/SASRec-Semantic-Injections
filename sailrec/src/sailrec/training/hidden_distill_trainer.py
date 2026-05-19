"""A3 — HiddenStateDistillTrainer: InfoNCE on encoder output."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sailrec.alignment.contrastive import AlignProjector, infonce_align_loss
from sailrec.training.losses import bce_loss
from sailrec.training.trainer import BaseTrainer


class HiddenStateDistillTrainer(BaseTrainer):
    """A3 ablation: InfoNCE between encoder hidden states and projected LLM.

    At position t, the encoder output seq_repr[:, t, :] is contrasted
    against proj(llm_emb[pos[:, t]]) using symmetric InfoNCE. Padding
    positions are excluded. Temperature and lambda match SAILS exactly
    so location is the only variable vs A5.
    """

    def __init__(self, model, train_loader, val_loader, llm_item_emb, device,
                 lr=0.001, max_epochs=200, lambda_align=0.1, temperature=0.1,
                 projector_hidden_dim=0, projector_dropout=0.1,
                 early_stopping_patience=10, early_stopping_metric="ndcg@10",
                 early_stopping_min_delta=1e-4, output_dir="outputs",
                 wandb_run=None, max_align_ids=None):
        super().__init__(model=model, train_loader=train_loader, val_loader=val_loader,
                         device=device, lr=lr, max_epochs=max_epochs,
                         early_stopping_patience=early_stopping_patience,
                         early_stopping_metric=early_stopping_metric,
                         early_stopping_min_delta=early_stopping_min_delta,
                         output_dir=output_dir, wandb_run=wandb_run)
        self.lambda_align = lambda_align
        self.temperature = temperature
        self.max_align_ids = max_align_ids
        self.llm_item_emb = llm_item_emb.to(device)
        self.llm_item_emb.requires_grad_(False)
        embed_dim = getattr(model, "embed_dim", 50)
        self.llm_dim = int(llm_item_emb.size(1))
        self.projector = AlignProjector(self.llm_dim, embed_dim, projector_hidden_dim, projector_dropout).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()), lr=lr)

    def train_epoch(self):
        self.model.train(); self.projector.train()
        total_rec = total_align = total = 0.0; n = 0
        for seq, pos, neg in self.train_loader:
            seq, pos, neg = seq.to(self.device), pos.to(self.device), neg.to(self.device)
            seq_repr = self.model(seq)                         # (B, L, D)
            pos_emb = self.model.item_emb(pos)
            neg_emb = self.model.item_emb(neg)
            mask = (pos > 0).float()
            loss_rec = bce_loss((seq_repr * pos_emb).sum(-1), (seq_repr * neg_emb).sum(-1), mask)

            # Flatten non-padding positions; align encoder output at t
            # against projected LLM embedding of the target item at t.
            pos_flat = pos.flatten()                           # (B*L,)
            valid = pos_flat > 0
            if valid.sum() >= 2:
                anchor = seq_repr.flatten(0, 1)[valid]        # (N, D)
                llm_tgt = self.projector(
                    self.llm_item_emb.index_select(0, pos_flat[valid]))  # (N, D)
                if self.max_align_ids and anchor.size(0) > self.max_align_ids:
                    perm = torch.randperm(anchor.size(0), device=self.device)[:self.max_align_ids]
                    anchor, llm_tgt = anchor[perm], llm_tgt[perm]
                loss_align = infonce_align_loss(anchor, llm_tgt, self.temperature)
            else:
                loss_align = torch.zeros((), device=self.device, requires_grad=True)

            loss = loss_rec + self.lambda_align * loss_align
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_rec += loss_rec.item(); total_align += loss_align.item()
            total += loss.item(); n += 1
        return {"loss_rec": total_rec/n, "loss_align": total_align/n, "loss_combined": total/n}

    def _format_loss_log_line(self, losses):
        return (f"rec={losses['loss_rec']:.4f} align={losses['loss_align']:.4f} "
                f"total={losses['loss_combined']:.4f}")
