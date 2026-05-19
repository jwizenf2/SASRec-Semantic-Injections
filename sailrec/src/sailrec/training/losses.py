"""Loss functions for SASRec training.

Currently only :func:`bce_loss` — the standard next-item binary
cross-entropy used by both P1 baseline and SAILRec. The auxiliary
SAILRec InfoNCE loss lives in
:mod:`sailrec.alignment.contrastive`.
"""

import torch
import torch.nn.functional as F


def bce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-position binary cross-entropy for SASRec next-item training.

    SASRec's training objective scores both a positive (the next item)
    and a sampled negative at every position, then pushes the
    positive's logit up (toward 1) and the negative's logit down
    (toward 0) via independent BCE. This function does both halves
    masked over valid (non-padding) positions.

    Args:
        pos_logits: ``(batch, seq_len)`` raw scores for positive
            items (typically ``(seq_repr * pos_emb).sum(-1)``).
        neg_logits: ``(batch, seq_len)`` raw scores for negative
            items.
        mask: ``(batch, seq_len)`` 0/1 mask, ``1`` at valid positions,
            ``0`` at padding. Loss contributions at padding positions
            are zeroed out and don't enter the denominator.

    Returns:
        Scalar mean loss over valid positions.

    Notes:
        ``F.binary_cross_entropy_with_logits`` applies the sigmoid
        internally and is numerically more stable than computing
        ``BCE(sigmoid(x))`` by hand.
    """
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="none"
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="none"
    )

    # Sum the two halves, mask out padding, divide by the number of
    # valid positions. ``mask.sum()`` is the user's effective sequence
    # length summed across the batch.
    loss = (pos_loss + neg_loss) * mask
    return loss.sum() / mask.sum()
