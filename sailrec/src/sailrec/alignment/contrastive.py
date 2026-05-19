"""SAILRec — InfoNCE alignment between SASRec and frozen LLM item embeddings.

This is the *one* novel piece of code in SAILRec. Everything else
(dataset, model, baseline trainer, eval) is standard SASRec.

What it does
------------

Given:

* SASRec's trainable item embedding table ``item_emb`` of shape
  ``(num_items + 1, embed_dim)``, and
* a frozen LLM item embedding table ``LLM_emb`` of shape
  ``(num_items + 1, llm_dim)`` (e.g. Qwen3-Embedding-0.6B, 1024-dim),

we add an **auxiliary symmetric InfoNCE loss** that pulls each item's
trainable row toward a projected version of its LLM row, with other
items in the batch as negatives:

::

    student   = item_emb[ids]                       # (N, embed_dim)
    projected = AlignProjector(LLM_emb[ids])        # (N, embed_dim)

    s_norm    = ℓ2_normalise(student)
    t_norm    = ℓ2_normalise(projected)
    logits    = s_norm @ t_norm.T / temperature      # (N, N)
    labels    = arange(N)
    L_align   = ½ · CE(logits, labels) + ½ · CE(logitsᵀ, labels)

The total training objective is ``L_total = L_next + λ_align · L_align``,
with ``λ_align = 0.1`` and ``temperature = 0.1`` by default. ``ids``
are the unique non-padding item ids touched by the current batch
(positives ∪ sampled negatives), so the loss only updates the
embedding rows that BCE is updating anyway.

Why this works (when other LLM-injection paths don't)
-----------------------------------------------------

The SASRec sequence encoder is a finely-tuned behavioural predictor.
Auxiliary objectives that constrain *its* representations (input
fusion, perturbation-delta matching, sequence-level invariance) all
introduce gradient conflict with BCE and hurt performance. SAILRec's
loss only touches the static item embedding lookup table, leaving the
sequence encoder untouched. The behavioural BCE signal stays
dominant; the LLM signal acts as a *geometric prior* over the
embedding space.

See ``../../docs/B1_method.md`` §6 for the full ablation that supports
this claim.

Compute cost
------------

Per batch: one ``index_select`` on the LLM table, one matmul of size
``(N, llm_dim) @ (llm_dim, embed_dim)`` for the projector, one
``(N, N)`` matmul for the InfoNCE logits. ``N`` is the batch's unique
non-padding item count — typically ~5K-15K on Video_Games at
batch_size=512. The dominant cost is the ``(N, N)`` matmul.

References
----------

* Radford et al., 2021 — CLIP, the original symmetric-InfoNCE
  cross-modal alignment.
* Kang & McAuley, 2018 — SASRec.
* The paper writeup at ``../../docs/B1_method.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Trainable projection: LLM space → student space
# ---------------------------------------------------------------------------


class AlignProjector(nn.Module):
    """Trainable projection ``llm_dim → student_dim``.

    Two variants:

    * ``hidden_dim == 0`` (default) — a single bias-free
      :class:`torch.nn.Linear`. Cheap, does most of the work in
      practice on Video_Games.
    * ``hidden_dim > 0`` — a 2-layer MLP (Linear → ReLU → Dropout →
      Linear) with the given hidden width. Available for ablation;
      the linear path is sufficient for the +16% headline result.

    The projector is owned by :class:`~sailrec.training.sailrec_trainer.SAILRecTrainer`
    rather than the SASRec model, because:

    1. It's only used at training time — at inference, SAILRec falls
       back to plain SASRec (no LLM, no projector).
    2. Keeping it off the model means saved checkpoints have the same
       state-dict shape as P1 baseline checkpoints, so any P1
       evaluation script works on SAILRec checkpoints unchanged.

    Args:
        llm_dim: Feature dimension of the frozen LLM embeddings.
        student_dim: Target dimension (``embed_dim`` of SASRec; 50 by
            default).
        hidden_dim: 0 for a linear projector; > 0 for the MLP variant.
        dropout: Dropout used inside the MLP variant. Ignored when
            ``hidden_dim == 0``.
    """

    def __init__(
        self,
        llm_dim: int,
        student_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim > 0:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(llm_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, student_dim),
            )
        else:
            # Bias-free Linear: a pure change-of-basis. The InfoNCE
            # loss normalises both sides anyway, so a bias is
            # mathematically redundant.
            self.proj = nn.Linear(llm_dim, student_dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform on every Linear layer; biases zeroed."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection.

        Args:
            x: ``(N, llm_dim)`` LLM embeddings (one row per item id).

        Returns:
            ``(N, student_dim)`` projected representations, in the
            same space as ``SASRec.item_emb``.
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Symmetric InfoNCE loss
# ---------------------------------------------------------------------------


def infonce_align_loss(
    student_embs: torch.Tensor,
    projected_llm_embs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE between matched (student, LLM) item-embedding pairs.

    Identical math to CLIP's symmetric InfoNCE, just applied between
    two embedding spaces of the same item rather than between an
    image and a caption.

    Args:
        student_embs: ``(N, embed_dim)`` — row ``i`` is the student's
            item embedding for the ``i``-th item id in the batch.
        projected_llm_embs: ``(N, embed_dim)`` — row ``i`` is the
            *projected* LLM embedding for the same item id, i.e.
            ``AlignProjector(LLM_emb[id_i])``.
        temperature: Softmax temperature. Smaller = sharper attention
            to the positive pair, more sensitive to negatives. CLIP
            uses 0.07; we use 0.1 by default because Qwen3 embeddings
            are L2-normalised at extraction time, so the dot products
            already live in ``[-1, 1]``.

    Returns:
        Scalar loss. The mean of the two CE directions (student → LLM
        and LLM → student) so the gradient is symmetric.

    Notes:
        Returns ``0`` (with ``requires_grad=True``) for ``N < 2``,
        which can only happen on degenerate batches (everything padded
        out). InfoNCE with one row has no negatives and the gradient
        is trivially zero.
    """
    if student_embs.size(0) < 2:
        return torch.zeros((), device=student_embs.device, requires_grad=True)

    # ℓ2-normalise so the dot products in ``logits`` are cosine
    # similarities. Required for the temperature to be meaningful
    # across runs / dimensions.
    s = F.normalize(student_embs, dim=-1)
    t = F.normalize(projected_llm_embs, dim=-1)

    # All-pairs cosine similarity in float16 (MPS/CUDA matmul is 2-4×
    # faster in half-precision; the (N, N) matmul is the sole bottleneck
    # for large catalogs). We cast back to float32 before cross-entropy
    # because CE numerics are sensitive to half-precision accumulation
    # errors at small temperatures (logits / 0.1 can saturate fp16).
    device_type = s.device.type if s.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(s.device.type in ("cuda", "mps"))):
        logits = (s @ t.T) / temperature             # (N, N) in float16
    logits = logits.float()                          # back to float32 for CE
    labels = torch.arange(s.size(0), device=s.device)

    # Two CE directions: row-wise (student → LLM) and column-wise
    # (LLM → student). Each says "for each anchor, the diagonal
    # entry should be the largest". Symmetric mean keeps gradients
    # balanced across the two views.
    loss_st = F.cross_entropy(logits, labels)
    loss_ts = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_st + loss_ts)


def weighted_infonce_align_loss(
    student_embs: torch.Tensor,
    projected_llm_embs: torch.Tensor,
    item_weights: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Frequency-weighted symmetric InfoNCE (A7 ablation).

    Identical to :func:`infonce_align_loss` except each item's
    contribution to the loss is scaled by ``item_weights[i]``.

    Weights are inverse-log-frequency scores normalised to mean 1,
    so the total loss magnitude stays comparable to the unweighted
    version — tail items pull harder, head items pull softer, but the
    overall λ_align scale is unchanged.

    Args:
        student_embs: ``(N, embed_dim)``.
        projected_llm_embs: ``(N, embed_dim)``.
        item_weights: ``(N,)`` non-negative per-item weights.
            Typically ``1 / (1 + log(freq + 1))`` normalised to mean 1.
        temperature: Same as :func:`infonce_align_loss`.

    Returns:
        Scalar weighted mean loss.
    """
    if student_embs.size(0) < 2:
        return torch.zeros((), device=student_embs.device, requires_grad=True)

    s = F.normalize(student_embs, dim=-1)
    t = F.normalize(projected_llm_embs, dim=-1)

    # float16 matmul — same optimisation as infonce_align_loss.
    device_type = s.device.type if s.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(s.device.type in ("cuda", "mps"))):
        logits_raw = (s @ t.T) / temperature
    logits = logits_raw.float()
    labels = torch.arange(s.size(0), device=s.device)

    loss_st = F.cross_entropy(logits,   labels, reduction="none")  # (N,)
    loss_ts = F.cross_entropy(logits.T, labels, reduction="none")  # (N,)

    w = item_weights.to(student_embs.device)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError(
            f"item_weights contains invalid values (NaN, inf, or negative). "
            f"min={w.min().item():.4f}, max={w.max().item():.4f}. "
            "Check the weight_fn formula and frequency tensor."
        )
    w_sum = w.sum()
    if not torch.isfinite(w_sum) or w_sum <= 0:
        # Degenerate weights (all-zero) — fall back to unweighted mean.
        return 0.5 * (loss_st.mean() + loss_ts.mean())
    return 0.5 * ((w * loss_st).sum() + (w * loss_ts).sum()) / w_sum


__all__ = [
    "AlignProjector",
    "infonce_align_loss",
    "weighted_infonce_align_loss",
]
