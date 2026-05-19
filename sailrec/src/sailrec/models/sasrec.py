"""SASRec — Self-Attentive Sequential Recommendation.

Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation",
ICDM 2018.

This is a **clean** SASRec — no LLM-fusion path, no perturbation hooks.
SAILRec uses this exact architecture; the auxiliary alignment loss
lives entirely on the trainer (see
:class:`sailrec.training.sailrec_trainer.SAILRecTrainer`). Saved
checkpoints are bit-shape-identical to baseline SASRec.

Architecture overview
---------------------

::

    seq (B, L)
      │
      ├── item_emb     : Embedding(num_items+1, embed_dim)   [trainable, padding_idx=0]
      ├── pos_emb      : Embedding(max_seq_len, embed_dim)   [trainable]
      │
      ▼
    x = item_emb(seq) + pos_emb(positions)                   # (B, L, D)
    x = dropout(x)
    x[padding] = 0
      │
    ┌─┘
    │   for each block in [SASRecBlock x num_blocks]:
    │       x = block(x, causal_mask)
    │       x[padding] = 0
    └─►
    x = LayerNorm(x)                                         # (B, L, D)

The output ``x`` is the per-position sequence representation. SASRec
training uses every position via BCE; SAILRec inference uses only the
last position (``x[:, -1, :]``).

Padding convention
------------------

* Item id ``0`` is padding (matches ``nn.Embedding(padding_idx=0)``).
* Sequences are **left-padded** to ``max_seq_len`` by the dataset
  classes, so the user's most recent interaction sits at position
  ``-1`` regardless of history length. ``score_all_items`` and
  ``predict`` rely on this when slicing ``[:, -1, :]``.

Causal attention
----------------

The transformer uses a triangular attention mask so position ``i``
only attends to positions ``≤ i``. This makes training and inference
consistent: at training every position has a valid "next item"
target; at inference we predict from the last-position rep without
needing a separate decoder.
"""

import torch
import torch.nn as nn


class SASRecBlock(nn.Module):
    """Single transformer block with pre-norm (LayerNorm before attn / FFN).

    Pre-norm rather than post-norm because pre-norm is more stable at
    small embed_dim and shorter training schedules (Xiong et al. 2020).
    The original SASRec paper used post-norm; this implementation
    follows the more modern convention used by DLLM2Rec and others.

    Tensor shapes throughout: ``(batch, seq_len, embed_dim)``.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # Standard transformer FFN with embed_dim hidden width. Some
        # papers use 4× embed_dim; SASRec on these benchmarks doesn't
        # benefit from the extra capacity at embed_dim=50.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Run one block.

        Args:
            x: ``(B, L, D)`` per-position representations from the
                previous layer (or input embedding for the first block).
            attn_mask: ``(L, L)`` causal mask with ``-inf`` above the
                diagonal.

        Returns:
            ``(B, L, D)`` updated representations.
        """
        # Pre-norm self-attention with residual.
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-norm FFN with residual.
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation.

    Args:
        num_items: Item vocabulary size. Items are 1..num_items;
            id 0 is reserved for padding.
        embed_dim: Item embedding dim and transformer hidden dim.
        num_blocks: Number of transformer blocks.
        num_heads: Attention heads per block.
        max_seq_len: Maximum context length.
        dropout: Dropout rate, applied to embedding and inside each
            block.

    Notes:
        Item embedding is initialised via Xavier-uniform on rows
        ``1..`` (the padding row at index 0 stays zero so it can't leak
        into the sequence representation through a non-zero bias).

    Ablation extensions (off by default — A0/A5 paths are unaffected):

    * ``init_item_emb`` — A1 LLM-Init ablation. When provided, copies a
      pre-computed (e.g. PCA-projected LLM) tensor into the item table
      at construction in place of Xavier-uniform. The table remains
      fully trainable.
    * ``llm_item_emb`` + ``fusion_mode="add"`` — A2 Input-Fusion
      ablation. Attaches a frozen LLM embedding table and a trainable
      ``llm_dim → embed_dim`` projector; the forward pass adds the
      projected LLM embeddings to the trainable id embeddings before
      the encoder. Single-axis change vs A0: only the encoder *input*
      gets LLM signal; everything downstream is identical.

    Both kwargs default to inactive, so the SAILS headline path is
    byte-identical to the pre-ablation implementation.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        init_item_emb: torch.Tensor | None = None,
        llm_item_emb: torch.Tensor | None = None,
        llm_dim: int | None = None,
        fusion_mode: str = "off",
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.fusion_mode = fusion_mode

        # Item embedding table. ``padding_idx=0`` ensures the gradient
        # for the padding row is suppressed (PyTorch zeros it after
        # every step). The trainable rows 1..num_items are what the
        # SAILRec alignment loss reshapes.
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # Standard learned absolute positional embedding. Indexed by
        # position 0..max_seq_len-1; rows are added to the
        # corresponding item embeddings.
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Single shared dropout on the input embedding sum.
        self.emb_dropout = nn.Dropout(dropout)

        # Stack of transformer blocks.
        self.blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        # Final LayerNorm before scoring — gives a clean output
        # geometry and stabilises training.
        self.final_norm = nn.LayerNorm(embed_dim)

        # A2 (Input Fusion): frozen LLM table + trainable projector that
        # injects LLM signal at the encoder input. No-op when fusion_mode
        # == "off". Built before _init_weights so the projector picks up
        # PyTorch defaults (which are fine — the projector is what the
        # ablation tests).
        if fusion_mode == "add":
            if llm_item_emb is None or llm_dim is None:
                raise ValueError(
                    "fusion_mode='add' requires llm_item_emb and llm_dim"
                )
            if llm_item_emb.shape != (num_items + 1, llm_dim):
                raise ValueError(
                    f"llm_item_emb must be ({num_items + 1}, {llm_dim}); "
                    f"got {tuple(llm_item_emb.shape)}"
                )
            self.llm_emb = nn.Embedding.from_pretrained(
                llm_item_emb, freeze=True, padding_idx=0
            )
            self.llm_proj = nn.Linear(llm_dim, embed_dim, bias=False)
        elif fusion_mode != "off":
            raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

        self._init_weights(init_item_emb)

    def _init_weights(self, init_item_emb: torch.Tensor | None) -> None:
        """Initialise trainable parameters.

        Default path (init_item_emb=None): Xavier-uniform on item rows
        ≥ 1 and on positional embeddings. Row 0 of ``item_emb`` stays
        at zero (padding). Other layers use PyTorch defaults.

        A1 path (init_item_emb provided): copy the pre-computed init
        tensor (e.g. PCA(50) of L2-normed LLM embeddings) into the
        item table in place of Xavier-uniform. Padding row still
        zeroed. The table remains fully trainable.
        """
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
                self.item_emb.weight[0].zero_()  # enforce padding row
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the upper-triangular ``-inf`` causal attention mask.

        Returns a ``(seq_len, seq_len)`` tensor. PyTorch's
        ``MultiheadAttention`` adds this to the attention logits, so
        positions strictly above the diagonal get -inf and softmax
        zeros them out.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of left-padded item sequences.

        Args:
            seq: ``(batch, seq_len)`` long tensor; item ids in
                ``1..num_items``, with ``0`` for padding (left-padded).

        Returns:
            ``(batch, seq_len, embed_dim)`` per-position representations.

        Notes:
            Padding positions are explicitly zeroed after every block
            (rather than relying on the attention mask alone) so a
            downstream slice like ``out[:, -1, :]`` gets the *real*
            last-position rep even when followed by no padding.
        """
        _, seq_len = seq.shape

        # Position ids 0..seq_len-1, broadcast across the batch.
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)

        # Sum item + positional embeddings, dropout, then zero padding.
        # When fusion_mode == "add" (A2 ablation), also inject the
        # projected frozen-LLM embedding at the input. The encoder
        # downstream is identical to A0 — only its input differs.
        x = self.item_emb(seq) + self.pos_emb(positions)
        if self.fusion_mode == "add":
            x = x + self.llm_proj(self.llm_emb(seq))
        x = self.emb_dropout(x)
        padding_mask = (seq == 0).unsqueeze(-1)  # (B, L, 1) — True at padding
        x = x.masked_fill(padding_mask, 0.0)

        # Stack of causal-attended transformer blocks. We rebuild the
        # causal mask once and reuse it across blocks.
        attn_mask = self._causal_mask(seq_len, seq.device)
        for block in self.blocks:
            x = block(x, attn_mask)
            # Re-zero padding after each block; non-zero residuals can
            # leak through LayerNorm if we don't.
            x = x.masked_fill(padding_mask, 0.0)

        x = self.final_norm(x)
        return x

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score(self, seq_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score a small set of candidate items per user.

        Used by :meth:`predict` (which is used by the sampled
        evaluation protocol). For full-rank scoring use
        :meth:`score_all_items`.

        Args:
            seq_repr: ``(batch, embed_dim)`` per-user reps (typically
                the last-position rep from :meth:`forward`).
            items: ``(batch, num_candidates)`` candidate item ids.

        Returns:
            ``(batch, num_candidates)`` dot-product logits.
        """
        # Look up the candidate items' embeddings then dot-product
        # against the user reps.
        item_embs = self.item_emb(items)                          # (B, K, D)
        scores = (seq_repr.unsqueeze(1) * item_embs).sum(dim=-1)  # (B, K)
        return scores

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Score ``candidates`` for sampled evaluation.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences.
            candidates: ``(batch, num_candidates)`` candidate item ids.
                By convention the positive sits at index 0; metrics
                use this to compute ranks.

        Returns:
            ``(batch, num_candidates)`` dot-product logits.
        """
        # Encode the sequence, take the last-position rep, score the
        # candidates against it.
        seq_repr = self.forward(seq)[:, -1, :]
        return self.score(seq_repr, candidates)

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item in the catalog for full-rank evaluation.

        We skip the padding row (index 0) so the returned tensor is
        zero-indexed over items ``1..num_items``.

        Args:
            seq: ``(batch, seq_len)`` left-padded input sequences.

        Returns:
            ``(batch, num_items)`` dot-product logits over items
            ``1..num_items``.

        Notes:
            The full-rank eval datasets pass an exclusion mask that
            sets logits for seen items to ``-inf`` before computing
            the rank of the ground truth. This module only computes
            the raw logits.
        """
        seq_repr = self.forward(seq)[:, -1, :]              # (B, D)
        all_item_embs = self.item_emb.weight[1:]            # (N, D), skip padding
        return seq_repr @ all_item_embs.T                   # (B, N)
