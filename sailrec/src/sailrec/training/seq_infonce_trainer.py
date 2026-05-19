"""A4 — SeqInfoNCETrainer: InfoNCE on sequence summary vs user-LLM mean."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sailrec.alignment.contrastive import AlignProjector, infonce_align_loss
from sailrec.training.losses import bce_loss
from sailrec.training.trainer import BaseTrainer


class SeqInfoNCETrainer(BaseTrainer):
    """A4 ablation: InfoNCE between encoder sequence summary and mean-pooled LLM.

    Encoder summary = last non-padding position's output (B, D).
    User LLM emb   = mean of llm_emb over training sequence (B, llm_dim).
    Same loss/temperature as SAILS; only the surface differs.
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
            B, L = seq.shape
            seq_repr = self.model(seq)                          # (B, L, D)
            pos_emb = self.model.item_emb(pos)
            neg_emb = self.model.item_emb(neg)
            mask_bce = (pos > 0).float()
            loss_rec = bce_loss((seq_repr * pos_emb).sum(-1), (seq_repr * neg_emb).sum(-1), mask_bce)

            # Encoder summary: final position (= most recent real item
            # under left-padding; positions 0..pad_len-1 are pad zeros).
            enc_summary = seq_repr[:, -1, :]                         # (B, D)

            # User LLM embedding: mean-pool over training-seq items.
            seq_mask = (seq != 0).float().unsqueeze(-1)              # (B, L, 1)
            llm_seq = self.llm_item_emb.index_select(
                0, seq.flatten()).view(B, L, -1)                     # (B, L, llm_dim)
            user_llm = (llm_seq * seq_mask).sum(1) / seq_mask.sum(1).clamp(min=1)  # (B, llm_dim)
            user_llm_proj = self.projector(user_llm)                # (B, D)

            if B >= 2:
                if self.max_align_ids and B > self.max_align_ids:
                    perm = torch.randperm(B, device=self.device)[:self.max_align_ids]
                    enc_summary, user_llm_proj = enc_summary[perm], user_llm_proj[perm]
                loss_align = infonce_align_loss(enc_summary, user_llm_proj, self.temperature)
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
