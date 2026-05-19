"""GRU4Rec — Gated Recurrent Unit Sequential Recommendation.

Reference: Hidasi et al., "Session-based Recommendations with Recurrent
Neural Networks", ICLR 2016.

This implementation adapts GRU4Rec to the standard leave-one-out sequential
recommendation setting (rather than session-based), matching the exact
interface exported by :class:`sasrec_injection.models.sasrec.SASRec` so it can be
dropped into every existing trainer, eval script, and ablation (A1 LLM-init,
A8 ItemTable) without modification.

Architecture
------------

::

    seq (B, L)
      │
      ├── item_emb : Embedding(num_items+1, embed_dim)  [trainable, padding_idx=0]
      │
      ▼
    x = item_emb(seq)          # (B, L, embed_dim)
    x = emb_dropout(x)
    x[padding] = 0
      │
    ┌─┘
    │   GRU(embed_dim → hidden_size, num_layers)        # (B, L, hidden_size)
    │   [if hidden_size != embed_dim]:
    │       proj(hidden_size → embed_dim)               # (B, L, embed_dim)
    │   x[padding] = 0
    └─►
    x = LayerNorm(x)                                    # (B, L, embed_dim)

Inference uses only the last-position output ``x[:, -1, :]``, identical to
SASRec. Training uses every non-padding position via BCE next-item loss,
identical to :class:`sasrec_injection.training.trainer.SASRecTrainer`.

Padding with left-padded sequences
-----------------------------------

Sequences are left-padded with zeros (``padding_idx=0``). The GRU therefore
processes a prefix of zero-embedding vectors before the real items begin.
Because ``item_emb`` zeroes the row at ``padding_idx=0`` and the GRU's reset
gate keeps the hidden state near zero while input is zero, this has negligible
practical effect — the hidden state at the first real item is effectively the
initial state. Outputs at padding positions are explicitly zeroed post-GRU to
ensure the BCE mask and the alignment loss see clean zeros there.

No positional embedding
-----------------------

Unlike SASRec, GRU4Rec omits a learned positional embedding. The GRU's
recurrent state encodes position implicitly; adding an explicit positional
bias would double-count it. This is the only structural difference from SASRec
that affects the forward pass; both produce ``(B, L, embed_dim)`` outputs
compatible with the same trainer and scoring pipeline.

Interface contract (shared with SASRec)
----------------------------------------

* ``item_emb``            — public ``nn.Embedding(num_items+1, embed_dim)``.
* ``embed_dim``           — public int attribute.
* ``forward(seq)``        — ``(B, L)`` → ``(B, L, embed_dim)``.
* ``predict(seq, cands)`` — ``(B, L), (B, K)`` → ``(B, K)`` logits.
* ``score_all_items(seq)``— ``(B, L)`` → ``(B, num_items)`` logits.
"""

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """GRU-based sequential recommendation model.

    Args:
        num_items: Item vocabulary size. Items are ``1..num_items``; id
            ``0`` is reserved for padding.
        embed_dim: Item embedding dimension. Also the output dimension of
            ``forward()``, regardless of ``hidden_size``.
        hidden_size: GRU hidden state width. Defaults to ``embed_dim``.
            When ``hidden_size != embed_dim``, a trainable linear
            projection maps GRU output → ``embed_dim`` so that the
            scoring dot-products (``seq_repr @ item_emb.T``) remain
            dimensionally consistent.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout applied to the item embedding. When
            ``num_layers > 1``, also applied between GRU layers via
            ``nn.GRU(dropout=...)``.
        init_item_emb: Optional ``(num_items+1, embed_dim)`` tensor to
            copy into ``item_emb`` in place of Xavier-uniform. Used by
            the A1 LLM-Init ablation. The table remains fully trainable.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 50,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.2,
        init_item_emb: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size if hidden_size is not None else embed_dim

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        # GRU: takes embed_dim input, outputs hidden_size per position.
        # inter-layer dropout only applies when num_layers > 1.
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Optional projection when the GRU hidden width differs from the
        # item embedding dim. Keeps scoring consistent: both seq_repr and
        # item_emb live in embed_dim space.
        if self.hidden_size != embed_dim:
            self.proj = nn.Linear(self.hidden_size, embed_dim, bias=False)
        else:
            self.proj = None

        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights(init_item_emb)

    def _init_weights(self, init_item_emb: torch.Tensor | None) -> None:
        if init_item_emb is None:
            nn.init.xavier_uniform_(self.item_emb.weight[1:])
        else:
            if init_item_emb.shape != self.item_emb.weight.shape:
                raise ValueError(
                    f"init_item_emb must be {tuple(self.item_emb.weight.shape)}; "
                    f"got {tuple(init_item_emb.shape)}"
                )
            with torch.no_grad():
                self.item_emb.weight.copy_(init_item_emb)
                self.item_emb.weight[0].zero_()

        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of left-padded item sequences.

        Args:
            seq: ``(batch, seq_len)`` long tensor; item ids
                ``1..num_items``, with ``0`` for padding (left-padded).

        Returns:
            ``(batch, seq_len, embed_dim)`` per-position representations.
            Padding positions are explicitly zeroed.
        """
        padding_mask = (seq == 0).unsqueeze(-1)  # (B, L, 1)

        x = self.item_emb(seq)       # (B, L, embed_dim); padding rows are 0
        x = self.emb_dropout(x)
        x = x.masked_fill(padding_mask, 0.0)

        # GRU processes the full padded sequence. Because padding inputs
        # are zero vectors and the GRU reset gate suppresses accumulated
        # state near zero input, the effective computation starts at the
        # first real item.
        x, _ = self.gru(x)           # (B, L, hidden_size)

        if self.proj is not None:
            x = self.proj(x)         # (B, L, embed_dim)

        # Re-zero padding positions: GRU can produce small non-zero
        # values at padding positions due to bias terms.
        x = x.masked_fill(padding_mask, 0.0)

        return self.final_norm(x)    # (B, L, embed_dim)

    # ------------------------------------------------------------------
    # Scoring helpers (identical interface to SASRec)
    # ------------------------------------------------------------------

    def score(self, seq_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score a small set of candidate items per user.

        Args:
            seq_repr: ``(batch, embed_dim)`` per-user representations
                (typically the last-position output of :meth:`forward`).
            items: ``(batch, num_candidates)`` candidate item ids.

        Returns:
            ``(batch, num_candidates)`` dot-product logits.
        """
        item_embs = self.item_emb(items)                           # (B, K, D)
        return (seq_repr.unsqueeze(1) * item_embs).sum(dim=-1)     # (B, K)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidates for sampled evaluation.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences.
            candidates: ``(batch, num_candidates)`` candidate item ids.
                Positive sits at index 0 by convention.

        Returns:
            ``(batch, num_candidates)`` dot-product logits.
        """
        seq_repr = self.forward(seq)[:, -1, :]   # (B, D) — last position
        return self.score(seq_repr, candidates)

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item in the catalog for full-rank evaluation.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences.

        Returns:
            ``(batch, num_items)`` dot-product logits over items
            ``1..num_items`` (padding row skipped).
        """
        seq_repr = self.forward(seq)[:, -1, :]   # (B, D)
        all_item_embs = self.item_emb.weight[1:]  # (N, D) — skip padding row
        return seq_repr @ all_item_embs.T         # (B, N)
