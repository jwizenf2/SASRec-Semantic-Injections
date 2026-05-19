"""Dataset-agnostic dispatcher: ``ml-*`` → MovieLens, ``amazon-*`` → Amazon, ``yelp`` → Yelp.

This module exists so scripts can take a ``--config`` with any of:

.. code-block:: yaml

    dataset:
      name: "ml-1m"             # MovieLens-1M
      name: "ml-100k"           # MovieLens-100K
      name: "amazon-Video_Games"  # Amazon Reviews 2023, Video_Games subset
      name: "amazon-Beauty"
      name: "amazon-Toys_and_Games"
      name: "amazon-Sports_and_Outdoors"
      name: "amazon-Books"
      name: "yelp"              # Yelp Open Dataset (single dataset, no subset)
      ...

…without the script knowing how to load each one. The dispatcher reads
the name prefix and forwards to the appropriate loader (Yelp matches
the bare string ``"yelp"`` since the Yelp Open Dataset has no
sub-categories, unlike Amazon).

Re-exports
----------

For convenience the dispatcher also re-exports
:func:`sasrec_injection.data.movielens.preprocess` and
:func:`sasrec_injection.data.movielens.build_user_sequences`, which are
dataset-agnostic but live in the MovieLens module for historical
reasons. Most callers can ``from sasrec_injection.data.loaders import …`` and
not touch the inner modules at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Re-export the dataset-agnostic helpers so scripts only need this
# module. ``preprocess`` and ``build_user_sequences`` work on any
# DataFrame matching the four-column contract.
from sasrec_injection.data.movielens import build_user_sequences, preprocess

__all__ = [
    "build_user_sequences",
    "is_amazon",
    "is_movielens",
    "is_yelp",
    "load_ratings",
    "preprocess",
    "resolve_amazon_category",
]


def is_movielens(dataset: str) -> bool:
    """True if ``dataset`` names a MovieLens variant (``ml-1m``, ``ml-100k``, etc.)."""
    return dataset.startswith("ml-")


def is_amazon(dataset: str) -> bool:
    """True if ``dataset`` names an Amazon Reviews 2023 category subset."""
    return dataset.startswith("amazon-")


def is_yelp(dataset: str) -> bool:
    """True if ``dataset`` names the Yelp Open Dataset.

    Yelp ships as a single un-subdivided dataset, so we match the bare
    name ``"yelp"`` rather than a prefix. Anything starting with
    ``"yelp-"`` is also accepted for forward-compatibility (e.g. if we
    later split by city).
    """
    return dataset == "yelp" or dataset.startswith("yelp-")


def resolve_amazon_category(dataset: str) -> str:
    """Strip the ``amazon-`` prefix to get the HF subset name.

    Args:
        dataset: Full prefixed dataset name, e.g. ``"amazon-Video_Games"``.

    Returns:
        Bare category name accepted by :func:`sasrec_injection.data.amazon.load_ratings`,
        e.g. ``"Video_Games"``.

    Raises:
        ValueError: If ``dataset`` doesn't start with ``"amazon-"``.
    """
    if not is_amazon(dataset):
        raise ValueError(f"Not an Amazon dataset: {dataset!r}")
    return dataset.removeprefix("amazon-")


def is_amazon2018(dataset: str) -> bool:
    """True if ``dataset`` names an Amazon 2018 dataset (LLMEmb preprocessing)."""
    return dataset.startswith("amazon2018-")


def load_ratings(data_dir: str | Path, dataset: str) -> pd.DataFrame:
    """Dispatch to the right ratings loader based on the dataset prefix.

    Args:
        data_dir: Root data directory (cache directory for Amazon;
            interactions directory for MovieLens).
        dataset: Dataset name with a recognised prefix
            (``ml-*`` or ``amazon-*``).

    Returns:
        DataFrame with the canonical four columns:
        ``[user_id, item_id, rating, timestamp]``. Ready for
        :func:`preprocess`.

    Raises:
        ValueError: For unknown dataset prefixes.
    """
    if is_movielens(dataset):
        from sasrec_injection.data.movielens import load_ratings as _ml_load
        return _ml_load(data_dir, dataset=dataset)

    if is_amazon(dataset):
        from sasrec_injection.data.amazon import load_ratings as _amz_load
        return _amz_load(data_dir, dataset=resolve_amazon_category(dataset))

    if is_yelp(dataset):
        from sasrec_injection.data.yelp import load_ratings as _yelp_load
        return _yelp_load(data_dir, dataset=dataset)

    if is_amazon2018(dataset):
        from sasrec_injection.data.amazon2018 import load_amazon2018
        slug = dataset.removeprefix("amazon2018-")
        df, _, _ = load_amazon2018(slug, data_dir)
        return df

    raise ValueError(
        f"Unknown dataset {dataset!r}; expected an 'ml-*', "
        f"'amazon-*', or 'yelp' name."
    )
