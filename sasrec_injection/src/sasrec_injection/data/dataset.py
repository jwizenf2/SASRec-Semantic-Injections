"""PyTorch ``Dataset`` classes for SASRec / BERT4Rec training and evaluation.

Four dataset variants:

* :class:`SASRecTrainDataset`     — Yields ``(input_seq, pos, neg)``
                                    triples for next-item BCE training
                                    (SASRec / GRU4Rec).
* :class:`BERT4RecTrainDataset`   — Yields ``(masked_seq, labels)``
                                    pairs for masked Cloze training
                                    (BERT4Rec).
* :class:`SASRecEvalDataset`      — Yields ``(input_seq, candidates)``
                                    for *sampled* eval (1 positive +
                                    N negatives per user).
* :class:`FullRankEvalDataset`    — Yields ``(input_seq, target,
                                    exclude_mask)`` for *full-rank* eval
                                    (rank against all items, mask out
                                    seen).

Padding convention
------------------

All sequences are **left-padded with zeros to ``max_seq_len``**. This
means the user's most recent interaction always sits at position
``-1``, which makes "take the last position" identical to "take
position -1" regardless of the user's history length.

The item-id vocabulary is ``1..num_items``, with ``0`` reserved for
padding (matching ``nn.Embedding(padding_idx=0)`` in the model).
For BERT4Rec, ``num_items+1`` is additionally reserved for the [MASK]
special token.
"""

import random

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------


class SASRecTrainDataset(Dataset):
    """Training dataset for SASRec next-item BCE.

    For each user, produces:

    * ``input_seq``: padded sequence of past item ids
      (length ``max_seq_len``).
    * ``pos_items``: per-position positive targets (the next item).
    * ``neg_items``: per-position random negatives (not in the user's
      history).

    The model scores both positives and negatives at every position;
    BCE pushes the positive score up and the negative score down.

    Tensor shapes: every ``__getitem__`` returns three tensors of shape
    ``(max_seq_len,)``.
    """

    def __init__(
        self,
        user_seqs: dict[int, list[int]],
        num_items: int,
        max_seq_len: int = 200,
    ):
        """
        Args:
            user_seqs: ``user_id`` → training-sequence (the
                ``train_seqs`` member of :class:`SplitData`).
            num_items: Total item vocabulary size (used to bound
                negative sampling).
            max_seq_len: Maximum context length. Sequences longer than
                this are left-truncated; the most recent
                ``max_seq_len`` items are kept.
        """
        # ``sorted`` gives a stable order across runs, which is useful
        # when checkpointing or resuming.
        self.user_ids = sorted(user_seqs.keys())
        self.user_seqs = user_seqs
        self.num_items = num_items
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.user_ids[idx]
        seq = self.user_seqs[uid]
        # Set used to verify negatives don't accidentally land on a
        # positive item the user has actually interacted with.
        item_set = set(seq)

        # Left-truncate to max_seq_len.
        seq = seq[-self.max_seq_len :]

        # Build (input, target) pairs at every position:
        #   input[i]  = item the user saw at step i
        #   target[i] = item the user saw at step i+1
        # We then left-pad with zeros so input/target sequences have
        # length ``max_seq_len``. This shift-by-one is the canonical
        # SASRec next-item formulation.
        n = len(seq)
        pad_len = self.max_seq_len - n + 1

        input_seq = [0] * pad_len + seq[:-1]
        pos_items = [0] * pad_len + seq[1:]

        # Sample one negative per position. Standard SASRec; matches
        # DLLM2Rec / BIGRec / etc. for direct number comparability.
        neg_items: list[int] = []
        for _ in range(self.max_seq_len):
            neg = random.randint(1, self.num_items)
            while neg in item_set:
                neg = random.randint(1, self.num_items)
            neg_items.append(neg)

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# BERT4Rec masked Cloze training dataset
# ---------------------------------------------------------------------------


class BERT4RecTrainDataset(Dataset):
    """Masked Cloze training dataset for BERT4Rec.

    At each call to ``__getitem__``, a random subset of the user's
    non-padding items (controlled by ``mask_ratio``) are replaced with
    ``mask_token_id``. The ``labels`` tensor records the original item id
    at each masked position and zero everywhere else; the BERT4Rec trainer
    computes cross-entropy loss only where ``labels > 0``.

    At least one item is always masked — for very short sequences where
    no position is selected by chance, one random position is forcibly
    masked. This prevents degenerate batches with no training signal.

    Args:
        user_seqs: ``user_id`` → training-sequence list.
        num_items: Catalog size (items are ``1..num_items``).
        mask_token_id: Special MASK token id (typically ``num_items+1``).
        mask_ratio: Fraction of non-padding items to replace with MASK.
            BERT4Rec paper uses 0.2.
        max_seq_len: Sequences longer than this are left-truncated.
    """

    def __init__(
        self,
        user_seqs: dict[int, list[int]],
        num_items: int,
        mask_token_id: int,
        mask_ratio: float = 0.2,
        max_seq_len: int = 200,
    ):
        self.user_ids = sorted(user_seqs.keys())
        self.user_seqs = user_seqs
        self.num_items = num_items
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        uid = self.user_ids[idx]
        seq = self.user_seqs[uid][-self.max_seq_len:]

        pad_len = self.max_seq_len - len(seq)
        padded = [0] * pad_len + list(seq)

        masked = list(padded)
        labels = [0] * self.max_seq_len

        # Candidate positions are non-padding (indices pad_len..max_seq_len-1).
        real_positions = list(range(pad_len, self.max_seq_len))
        mask_positions = [
            i for i in real_positions if random.random() < self.mask_ratio
        ]
        # Guarantee at least one masked position per sequence.
        if not mask_positions:
            mask_positions = [random.choice(real_positions)]

        for i in mask_positions:
            labels[i] = padded[i]
            masked[i] = self.mask_token_id

        return (
            torch.tensor(masked, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Sampled evaluation
# ---------------------------------------------------------------------------


class SASRecEvalDataset(Dataset):
    """Sampled-evaluation dataset.

    For each user, yields:

    * The user's full input sequence (left-padded to ``max_seq_len``).
    * A ``(num_candidates,)`` candidate tensor: ``[positive,
      neg_1, neg_2, ..., neg_N]``.

    The model scores all candidates; ranking metrics compute the rank
    of the positive (index 0) against the negatives.

    This is **not** a substitute for full-rank evaluation — sampled
    metrics over-state absolute quality when the true catalog has
    25K items and we sample only 100 negatives per user. We use it
    during training (cheap, runs every epoch for early stopping) and
    report full-rank numbers separately at the end.
    """

    def __init__(
        self,
        user_seqs: dict[int, list[int]],
        targets: dict[int, int],
        neg_samples: dict[int, list[int]],
        max_seq_len: int = 200,
    ):
        """
        Args:
            user_seqs: ``user_id`` → sequence to encode (training only
                during validation; train+val during test).
            targets: ``user_id`` → ground-truth target item.
            neg_samples: ``user_id`` → fixed list of negatives.
            max_seq_len: Padding length.
        """
        self.user_ids = sorted(targets.keys())
        self.user_seqs = user_seqs
        self.targets = targets
        self.neg_samples = neg_samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        uid = self.user_ids[idx]
        seq = self.user_seqs[uid][-self.max_seq_len :]

        # Left-pad.
        pad_len = self.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq

        # Candidates: positive first (index 0), then negatives.
        # The metrics module relies on this ordering when computing
        # ranks.
        candidates = [self.targets[uid]] + self.neg_samples[uid]

        return (
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Full-rank evaluation
# ---------------------------------------------------------------------------


class FullRankEvalDataset(Dataset):
    """Full-rank evaluation dataset.

    For each user, yields:

    * ``input_seq``: padded encoded sequence.
    * ``target``: ground-truth item id (1-indexed).
    * ``exclude_mask``: ``(num_items,)`` bool tensor; True at positions
      to mask from ranking (typically train + val items).

    The downstream metrics function scores all items, applies the
    mask (sets masked logits to ``-inf``), then computes the rank of
    the ground truth among the *unmasked* items.

    Why exclude seen items
    ----------------------

    A trained recommender will often score the items the user has
    *already interacted with* highly — those are the best evidence of
    the user's taste. But recommending an item the user already owns
    is useless. Standard practice in seq-rec evaluation is to mask
    seen items so we measure ranking quality on the *novel* items the
    user hasn't yet bought / watched.

    The ground-truth test item is the only "seen" item we don't mask
    (we want to know its rank).
    """

    def __init__(
        self,
        user_seqs: dict[int, list[int]],
        targets: dict[int, int],
        num_items: int,
        exclude_items: dict[int, set[int]] | None = None,
        max_seq_len: int = 200,
    ):
        """
        Args:
            user_seqs: ``user_id`` → sequence to encode (typically
                train+val for test evaluation).
            targets: ``user_id`` → ground-truth test item.
            num_items: Vocabulary size; the mask will be a
                ``(num_items,)`` bool tensor.
            exclude_items: Optional ``user_id`` → set of items to mask.
                If None, masks the items in ``user_seqs[uid]`` (i.e.
                everything the user has been shown so far).
            max_seq_len: Padding length.
        """
        self.user_ids = sorted(targets.keys())
        self.user_seqs = user_seqs
        self.targets = targets
        self.num_items = num_items
        self.max_seq_len = max_seq_len

        if exclude_items is not None:
            self.exclude_items = exclude_items
        else:
            self.exclude_items = {uid: set(seq) for uid, seq in user_seqs.items()}

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        uid = self.user_ids[idx]
        seq = self.user_seqs[uid][-self.max_seq_len :]

        pad_len = self.max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq

        # Build the (num_items,)-shaped exclusion mask. Item ids are
        # 1-indexed (0 = padding); the mask is 0-indexed over items
        # 1..num_items, so item id ``k`` lives at mask position
        # ``k - 1``.
        mask = torch.zeros(self.num_items, dtype=torch.bool)
        for item_id in self.exclude_items.get(uid, set()):
            mask[item_id - 1] = True

        # Defensive: never exclude the ground truth — its rank is the
        # whole point of the evaluation.
        gt = self.targets[uid]
        mask[gt - 1] = False

        return (
            torch.tensor(padded_seq, dtype=torch.long),
            gt,
            mask,
        )


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------


def create_train_loader(
    dataset: SASRecTrainDataset,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """Standard training DataLoader — shuffle on, no pin-memory.

    ``pin_memory=False`` because MPS doesn't benefit from pinning the
    way CUDA does, and on CPU it's pointless. ``num_workers=0`` because
    ``__getitem__`` is fast (negative sampling is the only loop) and
    spawning workers has its own latency cost on macOS.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


def create_eval_loader(
    dataset: SASRecEvalDataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """Evaluation DataLoader — no shuffle (deterministic ordering)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
