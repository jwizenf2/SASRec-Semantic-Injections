"""MovieLens dataset loader + canonical preprocessing helpers.

Even though ItemTable's headline experiment runs on Amazon Video_Games,
this module is still the home of two dataset-agnostic helpers
(:func:`preprocess` and :func:`build_user_sequences`) used by every
loader in the project. They live here for historical reasons and
because they were *first* written against the MovieLens schema.

Supported MovieLens datasets:

* ``ml-1m``    — MovieLens 1M, ``ratings.dat`` and ``movies.dat``
* ``ml-100k``  — MovieLens 100K, ``u.data`` and ``u.item``

Schema produced by :func:`load_ratings`
---------------------------------------

DataFrame with exactly four columns:

* ``user_id``    (int)
* ``item_id``    (int — for MovieLens; str ``parent_asin`` for Amazon)
* ``rating``     (numeric)
* ``timestamp``  (Unix seconds)

This is the contract :func:`preprocess` consumes; any dataset loader
that produces a DataFrame with these four columns works with the rest
of the pipeline unchanged.
"""

from pathlib import Path

import pandas as pd

# ML-100K's u.item file encodes genres as 19 binary columns. We expand
# them into a single pipe-separated genre string for parity with
# ml-1m's "Genre1|Genre2|..." format.
ML100K_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# Raw file loaders
# ---------------------------------------------------------------------------


def load_ratings(data_dir: str | Path, dataset: str = "ml-1m") -> pd.DataFrame:
    """Load MovieLens ratings into the project's standard 4-column DataFrame.

    Args:
        data_dir: Root directory containing ``ml-1m/`` or ``ml-100k/``.
            Expected to be the project's ``data/`` folder.
        dataset: Dataset name. Currently only ``"ml-1m"`` and
            ``"ml-100k"`` are accepted.

    Returns:
        DataFrame with columns ``[user_id, item_id, rating, timestamp]``.

    Raises:
        ValueError: For unknown dataset names.
        FileNotFoundError: If the expected file is missing under
            ``data_dir``.
    """
    data_dir = Path(data_dir)

    if dataset == "ml-1m":
        # ml-1m uses ``::`` as the field separator and Latin-1 encoding
        # because the original release predates UTF-8's ubiquity.
        path = data_dir / "ml-1m" / "ratings.dat"
        df = pd.read_csv(
            path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",  # ``::`` requires the Python engine
            encoding="latin-1",
        )
    elif dataset == "ml-100k":
        # ml-100k uses tab-separated; no header row, no BOM.
        path = data_dir / "ml-100k" / "u.data"
        df = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported: ml-100k, ml-1m"
        )

    return df


def load_movies(data_dir: str | Path, dataset: str = "ml-1m") -> pd.DataFrame:
    """Load MovieLens item metadata (title + genres).

    Returns a DataFrame with three columns:
    ``[item_id, title, genres]``. ``genres`` is a pipe-separated string
    matching ml-1m's native format (e.g. ``"Animation|Children's|Comedy"``).
    """
    data_dir = Path(data_dir)

    if dataset == "ml-1m":
        path = data_dir / "ml-1m" / "movies.dat"
        df = pd.read_csv(
            path,
            sep="::",
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
    elif dataset == "ml-100k":
        # 100K's u.item encodes genres as 19 binary columns; we collapse
        # them into the ml-1m-style pipe-separated string for uniformity.
        path = data_dir / "ml-100k" / "u.item"
        col_names = [
            "item_id", "title", "release_date", "video_release_date", "imdb_url",
        ] + ML100K_GENRES
        df = pd.read_csv(
            path,
            sep="|",
            names=col_names,
            engine="python",
            encoding="latin-1",
        )

        def _genres_from_row(row):
            genres = [g for g in ML100K_GENRES if row.get(g, 0) == 1]
            return "|".join(genres) if genres else "unknown"

        df["genres"] = df.apply(_genres_from_row, axis=1)
        df = df[["item_id", "title", "genres"]]
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported: ml-100k, ml-1m"
        )

    return df


# ---------------------------------------------------------------------------
# Dataset-agnostic preprocessing
# ---------------------------------------------------------------------------


def preprocess(
    ratings: pd.DataFrame,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, dict, dict]:
    """Iterative bipartite 5-core filter + chronological sort + ID remap.

    This is the canonical preprocess function used by every dataset
    pipeline (MovieLens, Amazon, Yelp).

    Steps:

    1. **Iterative bipartite k-core filter.** Repeatedly drop users
       *and* items with fewer than ``min_interactions`` interactions
       until the surviving table is stable. This is the convention used
       by SASRec / BERT4Rec / LLM-ESR / LLMEmb. A single one-sided pass
       (e.g. only filtering users) leaves a heavy item-side long tail
       on datasets like Yelp where the raw catalog ships unfiltered;
       that tail then inflates sampled@K evaluation metrics because
       uniform random negatives are mostly poorly-trained tail items.

       For datasets that ship pre-filtered (Amazon Reviews 2023's
       ``benchmark/5core/`` subsets, MovieLens canonical splits) this
       loop terminates in a single iteration with no rows dropped — so
       it's a no-op there and a real fix on Yelp.
    2. **Chronological sort** within each user (so
       :func:`build_user_sequences` produces correctly-ordered sequences).
    3. **ID remap.** Replace original (potentially non-contiguous,
       potentially string) ids with contiguous ints starting at 1.
       0 is reserved for padding.

    Args:
        ratings: DataFrame with columns
            ``[user_id, item_id, rating, timestamp]``.
        min_interactions: Minimum per-user *and* per-item interaction
            count after the iterative loop converges.

    Returns:
        Tuple ``(processed_df, user_map, item_map)`` where:

        * ``processed_df`` has the same four columns, with remapped int
          ids and rows sorted by ``(user_id, timestamp)``.
        * ``user_map`` maps original user id → contiguous int (≥ 1).
        * ``item_map`` maps original item id → contiguous int (≥ 1).

    Notes:
        * The ``rating`` column is preserved but unused downstream
          (SASRec's BCE objective uses positives vs negatives, not
          explicit ratings).
        * Item ids in ``item_map`` are *strings* for Amazon
          (``parent_asin``), *strings* for Yelp (``business_id``), and
          *ints* for MovieLens. The mapped values are always ints.
          Downstream code (PyTorch Datasets, SASRec) only ever sees the
          mapped ints.
    """
    # Iterative bipartite k-core: alternate dropping low-activity users
    # and items until the table is stable. Converges in 3-5 rounds on
    # Yelp; in 1 round (no-op) on the pre-filtered Amazon / MovieLens
    # subsets. Each round is two grouped value_counts, so cost stays
    # linear in surviving rows.
    df = ratings
    while True:
        n_before = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_interactions].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_interactions].index)]
        if len(df) == n_before:
            break
    df = df.copy()

    # Sort ascending by user, then by timestamp within each user.
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Remap users: original id → 1..|users|.
    unique_users = sorted(df["user_id"].unique())
    user_map = {orig: idx + 1 for idx, orig in enumerate(unique_users)}

    # Remap items: original id → 1..|items|. 0 is reserved for padding
    # in every downstream tensor (SASRec's nn.Embedding has padding_idx=0).
    unique_items = sorted(df["item_id"].unique())
    item_map = {orig: idx + 1 for idx, orig in enumerate(unique_items)}

    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df, user_map, item_map


def build_user_sequences(df: pd.DataFrame) -> dict[int, list[int]]:
    """Group items by user, in chronological order.

    Args:
        df: Output of :func:`preprocess` (sorted by
            ``(user_id, timestamp)``).

    Returns:
        ``user_id`` → ``[item_id_1, item_id_2, ...]`` mapping with
        items in chronological order.

    Notes:
        * Assumes ``df`` is already sorted; we don't re-sort here for
          performance. If you call this on un-sorted input you will
          get ill-ordered sequences and SASRec will produce garbage.
        * The dict's keys are remapped int user ids (i.e. the values
          in the ``user_map`` returned by :func:`preprocess`).
    """
    return {
        uid: group["item_id"].tolist()
        for uid, group in df.groupby("user_id")
    }
