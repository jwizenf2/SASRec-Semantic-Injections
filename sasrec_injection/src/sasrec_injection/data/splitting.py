"""Leave-one-out splitting and negative-sample generation.

The standard sequential-recommendation evaluation protocol:

* For each user, the **last** item in their chronological sequence is
  the test target.
* The **second-to-last** item is the validation target.
* Everything before that is the training sequence.

This is "leave-one-out" because each user contributes exactly one
test example. ItemTable uses this protocol so its numbers are directly
comparable to SASRec, DLLM2Rec, BIGRec, and the wider LLM4Rec
literature.

Negative sampling
-----------------

For "sampled" evaluation (used during validation for early stopping),
each user gets a fixed set of ``num_neg`` negative items that are not
in their interaction history. The same set is reused at every
evaluation epoch (cached on disk under ``output_dir/neg_samples.npz``)
so different runs are directly comparable.

For "full-rank" evaluation (used at the end for headline numbers),
*every* item not in the user's seen set is a candidate. This is the
honest evaluation; sampled metrics tend to over-state absolute ranking
quality but track relative improvements well.
"""

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SplitData:
    """Result of a leave-one-out split.

    Attributes:
        train_seqs: ``user_id`` → list of training items (everything
            except the final two interactions). Used to build SASRec
            training batches.
        val_targets: ``user_id`` → second-to-last item id. The
            validation target.
        test_targets: ``user_id`` → last item id. The test target.
        num_users: Number of users with at least 3 interactions
            (i.e. enough to support train + val + test).
        num_items: Total item vocabulary size (1..num_items, 0 reserved
            for padding).
    """

    train_seqs: dict[int, list[int]]
    val_targets: dict[int, int]
    test_targets: dict[int, int]
    num_users: int
    num_items: int


def leave_one_out_split(
    user_sequences: dict[int, list[int]],
    num_items: int,
) -> SplitData:
    """Perform the leave-one-out split on per-user sequences.

    Args:
        user_sequences: ``user_id`` → chronologically-ordered list of
            item ids. Output of
            :func:`sasrec_injection.data.movielens.build_user_sequences`.
        num_items: Total item vocabulary size, captured into the
            return value.

    Returns:
        A :class:`SplitData` with ``train_seqs``, ``val_targets``,
        ``test_targets``, and the population statistics.

    Notes:
        Users with fewer than 3 interactions are silently dropped (they
        can't supply both a train sequence *and* val/test targets).
        ``preprocess`` already filters with ``min_interactions=5`` so
        in practice this branch fires zero times.
    """
    train_seqs: dict[int, list[int]] = {}
    val_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}

    for uid, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train_seqs[uid] = seq[:-2]
        val_targets[uid] = seq[-2]
        test_targets[uid] = seq[-1]

    return SplitData(
        train_seqs=train_seqs,
        val_targets=val_targets,
        test_targets=test_targets,
        num_users=len(train_seqs),
        num_items=num_items,
    )


def generate_negative_samples(
    split: SplitData,
    num_neg: int = 100,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Sample ``num_neg`` items per user that aren't in their history.

    Used for the "sampled" evaluation protocol. The returned dict maps
    every test user to a fixed list of negatives; the same list is
    reused at every validation step so the metric is comparable across
    epochs.

    Args:
        split: Output of :func:`leave_one_out_split`.
        num_neg: Negatives per user. 100 is the convention in
            SASRec / DLLM2Rec / BIGRec.
        seed: RNG seed for reproducibility. Not the *training* seed —
            this controls only the sampled-eval candidate set.

    Returns:
        ``user_id`` → list of negative item ids of length ``num_neg``
        (or fewer if the user's catalog complement is smaller).
    """
    rng = random.Random(seed)
    all_items = set(range(1, split.num_items + 1))
    neg_samples: dict[int, list[int]] = {}

    for uid in split.test_targets:
        # Exclude every item the user has interacted with: train + val + test.
        # Sampling against this complement guarantees the negatives are
        # genuinely unseen.
        user_items = set(split.train_seqs[uid])
        user_items.add(split.val_targets[uid])
        user_items.add(split.test_targets[uid])
        candidates = list(all_items - user_items)
        negs = rng.sample(candidates, min(num_neg, len(candidates)))
        neg_samples[uid] = negs

    return neg_samples


def save_negative_samples(neg_samples: dict[int, list[int]], path: str) -> None:
    """Save the negative-sample dict to a compressed ``.npz`` file.

    Format: one int-array per user, keyed by str(uid). NumPy ``.npz``
    is used because it's a single self-contained file that lives next
    to ``best_model.pt`` and travels with the rest of the run's
    artefacts.
    """
    arrays = {str(uid): np.array(negs) for uid, negs in neg_samples.items()}
    np.savez(path, **arrays)


def load_negative_samples(path: str) -> dict[int, list[int]]:
    """Load negatives saved by :func:`save_negative_samples`."""
    data = np.load(path)
    return {int(uid): data[uid].tolist() for uid in data.files}
