"""BERT4Rec — Bidirectional Encoder Representations for Sequential Recommendation.

Reference: Sun et al., "BERT4Rec: Sequential Recommendation with
Bidirectional Encoder Representations from Transformer", CIKM 2019.

Architecture
------------

BERT4Rec replaces SASRec's causal (left-to-right) transformer with a
**bidirectional** one trained via a Cloze (masked item prediction) objective:
a random subset of items in each training sequence is replaced with a special
[MASK] token; the model must predict the original items at those positions
using full left-and-right context.

At inference, a [MASK] token is appended to the end of the input sequence
(replacing the oldest item to stay within ``max_seq_len``). The representation
at the MASK position is used to score all candidate items — identical to how
SASRec uses its last-position representation.

Key differences from SASRec
----------------------------

* **Bidirectional attention** — no causal mask; every position can attend to
  every other non-padding position.
* **Cloze training loss** — cross-entropy at masked positions instead of BCE
  at all positions. Requires a separate dataset class
  (:class:`sasrec_injection.data.dataset.BERT4RecTrainDataset`) and trainer
  (:class:`sasrec_injection.training.bert4rec_trainer.BERT4RecTrainer`).
* **MASK special token** — embedding table is ``num_items + 2`` wide:
  index 0 = padding, 1..num_items = items, num_items+1 = [MASK].
* **Inference shift** — ``predict()`` and ``score_all_items()`` shift the
  input left by one and append [MASK] at the end, placing the query token at
  the last sequence position where ``forward()`` output is taken.

Interface contract (shared with SASRec and GRU4Rec)
----------------------------------------------------

* ``item_emb``             — ``nn.Embedding(num_items+2, embed_dim)``.
* ``embed_dim``            — public int.
* ``mask_token_id``        — ``num_items + 1`` (public int).
* ``forward(seq)``         — ``(B, L)`` → ``(B, L, embed_dim)``.
* ``predict(seq, cands)``  — ``(B, L), (B, K)`` → ``(B, K)`` logits.
* ``score_all_items(seq)`` — ``(B, L)`` → ``(B, num_items)`` logits.
"""

import torch
import torch.nn as nn


class BERT4RecBlock(nn.Module):
    """Bidirectional transformer block with pre-norm.

    Identical structure to ``SASRecBlock`` except:

    * No ``attn_mask`` argument — attention is bidirectional (full).
    * Accepts ``key_padding_mask`` so the encoder can prevent queries
      from attending to padding positions (index 0 in the input).

    Args:
        embed_dim: Hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout applied inside the FFN and on attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one bidirectional block.

        Args:
            x: ``(B, L, D)`` per-position representations.
            key_padding_mask: ``(B, L)`` bool tensor; True at padding
                positions. Prevents queries from attending to padding.

        Returns:
            ``(B, L, D)`` updated representations.
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        normed = self.norm2(x)
        return x + self.ffn(normed)


class BERT4Rec(nn.Module):
    """Bidirectional Encoder for Sequential Recommendation.

    Args:
        num_items: Item vocabulary size. Items are ``1..num_items``;
            ``0`` is padding, ``num_items + 1`` is the [MASK] token.
        embed_dim: Hidden dimension (transformer width and item embedding
            dim).
        num_blocks: Number of bidirectional transformer blocks.
        num_heads: Attention heads per block.
        max_seq_len: Maximum context length; positional embeddings are
            sized to this.
        dropout: Dropout on input embeddings and inside each block.
        init_item_emb: Optional ``(num_items+2, embed_dim)`` tensor to
            copy into ``item_emb`` in place of Xavier-uniform. Used by
            the A1 LLM-Init ablation. Rows 0 (padding) and
            ``num_items+1`` ([MASK]) are zeroed after copying. The
            table remains fully trainable.

    Notes:
        The embedding table has ``num_items + 2`` rows (one extra for the
        [MASK] token vs SASRec's ``num_items + 1``). The ``score_all_items``
        and ``score`` methods only use rows ``1..num_items``, matching
        the SASRec interface exactly.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        init_item_emb: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.mask_token_id = num_items + 1  # public — dataset needs this

        # Embedding table: +2 for padding (0) and MASK (num_items+1).
        self.item_emb = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BERT4RecBlock(embed_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights(init_item_emb)

    def _init_weights(self, init_item_emb: torch.Tensor | None) -> None:
        if init_item_emb is None:
            # Xavier-uniform on real item rows only; leave padding (0) and
            # MASK (num_items+1) as zeros.
            nn.init.xavier_uniform_(self.item_emb.weight[1:self.num_items + 1])
        else:
            expected = (self.num_items + 2, self.embed_dim)
            if tuple(init_item_emb.shape) != expected:
                raise ValueError(
                    f"init_item_emb must be {expected}; got {tuple(init_item_emb.shape)}"
                )
            with torch.no_grad():
                self.item_emb.weight.copy_(init_item_emb)
                self.item_emb.weight[0].zero_()                   # padding
                self.item_emb.weight[self.num_items + 1].zero_()  # MASK
        nn.init.xavier_uniform_(self.pos_emb.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences (may contain MASK tokens).

        Args:
            seq: ``(batch, seq_len)`` long tensor. Values:
                ``0`` = padding (left-padded), ``1..num_items`` = real
                items, ``num_items+1`` = [MASK] token.

        Returns:
            ``(batch, seq_len, embed_dim)`` per-position representations.
            Padding positions are explicitly zeroed.
        """
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0)  # (1, L)

        x = self.item_emb(seq) + self.pos_emb(positions)  # (B, L, D)
        x = self.emb_dropout(x)

        padding_mask = (seq == 0)          # (B, L) — True at padding
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return self.final_norm(x)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _append_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Build the inference sequence: drop oldest item, append [MASK].

        At inference the query is the [MASK] token placed at the last
        position. We keep total length at ``max_seq_len`` by dropping
        the leftmost (oldest) item: ``seq[:, 1:]`` + [MASK].

        This is the standard BERT4Rec evaluation construction: the model
        sees all of the user's recent history plus a mask token at the
        end, and the representation at the mask position is used to rank
        candidate items.
        """
        mask_col = torch.full(
            (seq.size(0), 1), self.mask_token_id,
            dtype=torch.long, device=seq.device,
        )
        return torch.cat([seq[:, 1:], mask_col], dim=1)  # (B, L)

    def score(self, seq_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score a small set of candidate items per user.

        Args:
            seq_repr: ``(batch, embed_dim)`` per-user representations.
            items: ``(batch, num_candidates)`` candidate item ids.

        Returns:
            ``(batch, num_candidates)`` dot-product logits.
        """
        item_embs = self.item_emb(items)                           # (B, K, D)
        return (seq_repr.unsqueeze(1) * item_embs).sum(dim=-1)     # (B, K)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidates for sampled evaluation.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences
                (real items only; no MASK tokens).
            candidates: ``(batch, num_candidates)`` candidate item ids.

        Returns:
            ``(batch, num_candidates)`` logits.
        """
        eval_seq = self._append_mask(seq)
        seq_repr = self.forward(eval_seq)[:, -1, :]  # MASK position
        return self.score(seq_repr, candidates)

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item in the catalog for full-rank evaluation.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences.

        Returns:
            ``(batch, num_items)`` logits over items ``1..num_items``.
        """
        eval_seq = self._append_mask(seq)
        seq_repr = self.forward(eval_seq)[:, -1, :]         # (B, D)
        # Use only item rows 1..num_items — skip padding (0) and MASK
        # (num_items+1).
        item_embs = self.item_emb.weight[1:self.num_items + 1]  # (N, D)
        return seq_repr @ item_embs.T                           # (B, N)
