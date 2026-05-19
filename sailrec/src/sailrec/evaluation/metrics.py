"""Ranking metrics + sampled/full-rank evaluation drivers.

Two evaluation protocols, both reported in the LLM4Rec literature:

* **Sampled**: rank the ground-truth item against ``num_neg`` random
  negatives. Cheap (we use it during validation for early stopping).
* **Full-rank**: rank the ground-truth against *every* item in the
  catalog, with seen items masked. Honest. Run at the end for the
  headline numbers.

Sampled metrics tend to *over-estimate* absolute ranking quality vs
full-rank — with 100 negatives they only test a tiny slice of the
distribution. They track relative improvements well, which is why
they're the right metric for early stopping. They are *not* a
substitute for full-rank when reporting headline numbers.

Metric definitions
------------------

For each user, given the *rank* (0-indexed) of the ground-truth item:

* ``HR@K``    = ``1 if rank < K else 0``     (averaged across users)
* ``NDCG@K``  = ``1 / log2(rank + 2) if rank < K else 0``
* ``Recall@K`` = same as ``HR@K`` for single-target evaluation.

We report ``HR``, ``NDCG``, and ``Recall`` to mirror the LLM4Rec
literature even though, with one ground-truth item per user, ``HR``
and ``Recall`` collapse to the same value.
"""

import torch

# ---------------------------------------------------------------------------
# Per-user metric primitives (operate on a tensor of ranks)
# ---------------------------------------------------------------------------


def hit_rate_at_k(rankings: torch.Tensor, k: int) -> float:
    """Hit Rate @ K.

    Args:
        rankings: ``(num_users,)`` 0-indexed ranks of each user's
            ground-truth item.
        k: Cutoff.

    Returns:
        Fraction of users with ``rank < k``.
    """
    return (rankings < k).float().mean().item()


def ndcg_at_k(rankings: torch.Tensor, k: int) -> float:
    """NDCG @ K for single-ground-truth evaluation.

    With one positive per user, ideal DCG = 1 (positive at rank 0
    contributes ``1 / log2(0 + 2) = 1``). The formula reduces to:

        NDCG = mean_user [ 1 / log2(rank + 2)  if rank < k else 0 ]
    """
    hits = rankings < k
    log2 = torch.log2(rankings.float() + 2)
    dcg = torch.where(hits, 1.0 / log2, torch.zeros_like(rankings.float()))
    return dcg.mean().item()


def recall_at_k(rankings: torch.Tensor, k: int) -> float:
    """Recall @ K. Single-ground-truth ⇒ identical to HR@K."""
    return hit_rate_at_k(rankings, k)


# ---------------------------------------------------------------------------
# Helper: derive ranks from a (batch, num_candidates) score tensor
# ---------------------------------------------------------------------------


def compute_rankings(scores: torch.Tensor) -> torch.Tensor:
    """0-indexed rank of the *first* candidate (i.e. ground truth).

    Args:
        scores: ``(num_users, num_candidates)`` where ``scores[:, 0]``
            is the ground-truth item's score and ``scores[:, 1:]`` are
            the negatives.

    Returns:
        ``(num_users,)`` long tensor; smaller is better.

    Notes:
        We use *strictly greater* (``>``), so ties between the GT and
        a negative count as a *correct* hit (rank tie goes to the GT).
        This matches the SASRec / DLLM2Rec convention.
    """
    gt_scores = scores[:, 0].unsqueeze(1)          # (U, 1)
    rankings = (scores[:, 1:] > gt_scores).sum(dim=1)
    return rankings


# ---------------------------------------------------------------------------
# Sampled evaluation: 1 positive + N negatives per user
# ---------------------------------------------------------------------------


@torch.no_grad()
def sampled_evaluate(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ks: list[int] = [5, 10, 20],
) -> dict[str, float]:
    """Run the sampled (1-vs-N) evaluation protocol.

    Args:
        model: SASRec instance with ``predict(seq, candidates)``.
        eval_loader: DataLoader over
            :class:`sailrec.data.dataset.SASRecEvalDataset`. Yields
            ``(seq, candidates)`` batches.
        device: Tensors are moved here before the forward pass.
        ks: Cutoffs for HR/NDCG/Recall reporting.

    Returns:
        ``{"hr@k": ..., "ndcg@k": ..., "recall@k": ...}`` for each k.
    """
    model.eval()
    all_rankings: list[torch.Tensor] = []

    for seq, candidates in eval_loader:
        seq = seq.to(device)
        candidates = candidates.to(device)

        # ``predict`` runs forward + scores the K candidates.
        scores = model.predict(seq, candidates)     # (B, K)
        rankings = compute_rankings(scores)
        all_rankings.append(rankings.cpu())

    all_rankings_cat = torch.cat(all_rankings, dim=0)

    results: dict[str, float] = {}
    for k in ks:
        results[f"hr@{k}"] = hit_rate_at_k(all_rankings_cat, k)
        results[f"ndcg@{k}"] = ndcg_at_k(all_rankings_cat, k)
        results[f"recall@{k}"] = recall_at_k(all_rankings_cat, k)

    return results


# ---------------------------------------------------------------------------
# Full-rank evaluation: rank against the entire item catalog
# ---------------------------------------------------------------------------


@torch.no_grad()
def full_rank_evaluate(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ks: list[int] = [5, 10, 20],
) -> dict[str, float]:
    """Run the full-rank evaluation protocol.

    Scores every item per user, masks out previously-seen items, then
    computes the ground-truth rank against the unmasked candidates.

    Args:
        model: SASRec instance with ``score_all_items(seq)``.
        eval_loader: DataLoader over
            :class:`sailrec.data.dataset.FullRankEvalDataset`. Yields
            ``(seq, gt_item, exclude_mask)`` batches.
        device: Compute device.
        ks: Cutoffs.

    Returns:
        ``{"full_hr@k": ..., "full_ndcg@k": ..., "full_recall@k": ...}``.
        Metric keys are prefixed with ``full_`` to make them
        impossible to confuse with sampled metrics in downstream
        aggregation / logging.
    """
    model.eval()
    all_rankings: list[torch.Tensor] = []

    for seq, gt_items, exclude_masks in eval_loader:
        seq = seq.to(device)
        exclude_masks = exclude_masks.to(device)

        # Score every item; this returns a (B, num_items) tensor.
        scores = model.score_all_items(seq)

        # Mask out items the user has already interacted with by
        # setting their logits to -inf. Softmax-equivalent ranking
        # ignores them.
        scores[exclude_masks] = float("-inf")

        # Pull the GT score per user. ``gt_items`` are 1-indexed; the
        # ``scores`` tensor is 0-indexed over items 1..num_items, so
        # we subtract 1.
        gt_indices = (gt_items - 1).to(device)
        gt_scores = scores[torch.arange(len(gt_indices), device=device), gt_indices]

        # Rank = number of items strictly outscoring the GT.
        rankings = (scores > gt_scores.unsqueeze(1)).sum(dim=1)
        all_rankings.append(rankings.cpu())

    all_rankings_cat = torch.cat(all_rankings, dim=0)

    results: dict[str, float] = {}
    for k in ks:
        results[f"full_hr@{k}"] = hit_rate_at_k(all_rankings_cat, k)
        results[f"full_ndcg@{k}"] = ndcg_at_k(all_rankings_cat, k)
        results[f"full_recall@{k}"] = recall_at_k(all_rankings_cat, k)

    return results
