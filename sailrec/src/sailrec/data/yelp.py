"""Yelp Open Dataset loader for SAILRec.

Counterpart to :mod:`sailrec.data.amazon` and
:mod:`sailrec.data.movielens`. Ingests the public Yelp Open Dataset
(https://www.yelp.com/dataset) as a sequential-recommendation problem:
each user's chronological list of reviewed businesses is the
interaction sequence, businesses are items.

Why Yelp
--------

Yelp is the canonical "non-Amazon, text-rich" dataset for
LLM-augmented sequential recommendation. Both LLM-ESR (NeurIPS 2024)
and LLMEmb (AAAI 2025) report results on Yelp with a SASRec backbone,
which makes it the only dataset where SAILRec can be put in a
direct head-to-head with both papers.

We do **not** match their preprocessing exactly (their public code is
a separate dependency to integrate; doing so cleanly is a follow-up).
What this loader produces is *5-core SAILRec-style* preprocessing,
mirroring what we already do for Amazon:

* Drop businesses / users with fewer than 5 interactions.
* Sort each user's interactions by review timestamp.
* Leave-one-out split (handled downstream in
  :func:`sailrec.data.splitting.leave_one_out_split`).

The numbers we report on Yelp are therefore self-consistent with our
Video_Games numbers but **not** bit-equivalent to LLM-ESR's / LLMEmb's
Yelp numbers; we document this explicitly in the writeup.

Expected on-disk layout
-----------------------

::

    data/yelp/
      yelp_academic_dataset_business.json   # JSONL, one business per line
      yelp_academic_dataset_review.json     # JSONL, one review per line

These are the raw files from the Yelp Open Dataset tarball (current
release at the time of writing: ``yelp_dataset.tgz``,
~3.7 GB compressed, ~10 GB extracted). Only the ``business`` and
``review`` files are needed for SRS — ``user``, ``tip``, ``checkin``
are ignored.

Why JSONL not Parquet
---------------------

The Yelp dataset ships as JSONL by default; we read it via
:func:`pandas.read_json` with ``lines=True`` which streams chunked
internally and is fast enough on M-series hardware (~30s for the full
review file). Converting to Parquet would speed re-loads but adds a
dependency / cache step we don't need for one-shot preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Default identifier for the Yelp dataset in
# :func:`sailrec.data.loaders.load_ratings`. The bare string ``"yelp"``
# (no prefix) matches the convention used by the Yelp Open Dataset
# itself; the dispatcher in ``loaders.py`` recognises it directly.
DATASET_NAME = "yelp"

# Filenames inside ``data_dir / "yelp" /``. These match the names the
# Yelp Open Dataset tarball uses out of the box — no renaming needed
# after unpacking.
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"


def _yelp_dir(data_dir: str | Path) -> Path:
    """Resolve the Yelp data subdirectory under the project ``data/`` root."""
    return Path(data_dir) / "yelp"


def load_ratings(data_dir: str | Path, dataset: str = DATASET_NAME) -> pd.DataFrame:
    """Load Yelp reviews as a SASRec-ready ratings DataFrame.

    Args:
        data_dir: Project ``data/`` directory. The Yelp files are
            expected at ``<data_dir>/yelp/yelp_academic_dataset_*.json``.
        dataset: Always ``"yelp"`` here. Accepted as a kwarg for API
            symmetry with :func:`sailrec.data.amazon.load_ratings` and
            :func:`sailrec.data.movielens.load_ratings`; ignored.

    Returns:
        DataFrame with the canonical four columns:
        ``[user_id, item_id, rating, timestamp]``. ``user_id`` and
        ``item_id`` are kept as strings (Yelp uses 22-char base64
        IDs); :func:`sailrec.data.movielens.preprocess` will remap
        them to contiguous integers downstream.

    Raises:
        FileNotFoundError: If the expected JSONL file is missing —
            usually because the Yelp Open Dataset hasn't been
            downloaded yet. The error message points the caller at the
            download instructions.
    """
    review_path = _yelp_dir(data_dir) / REVIEW_FILE
    if not review_path.exists():
        raise FileNotFoundError(
            f"Yelp reviews not found at {review_path}.\n"
            "Download the Yelp Open Dataset from "
            "https://www.yelp.com/dataset (terms acceptance required), "
            "extract the tarball, and place the JSONL files at "
            f"{_yelp_dir(data_dir)}/.\n"
            f"Required files: {BUSINESS_FILE}, {REVIEW_FILE}."
        )

    # Why a streaming line-by-line parse instead of pd.read_json
    # ----------------------------------------------------------
    # The review JSONL is ~5 GB on disk. ``pd.read_json(lines=True)``
    # internally builds the full DataFrame in one shot, peaking at
    # ~8-10 GB of RAM during parse (text buffer + tokenisation +
    # object materialisation + dtype inference all live in memory at
    # once). On a 32 GB machine that's most of the budget gone before
    # training even starts.
    #
    # We instead stream one record at a time, keep ONLY the four
    # columns we use, and accumulate them into pre-typed Python lists.
    # Peak RAM tracks the size of those four projected lists (~600 MB
    # for ~7M Yelp reviews), not the size of the parsed JSONL. End
    # result is identical to the pd.read_json path; cost is ~30s parse
    # vs ~25s for pandas, which is a tiny price for ~10x lower peak.
    user_ids: list[str] = []
    item_ids: list[str] = []
    ratings: list[float] = []
    dates: list[str] = []

    with open(review_path, encoding="utf-8") as f:
        for line in f:
            # Each line is a complete JSON object. We project to the
            # four fields we need; the review text / vote counts /
            # review_id are discarded immediately.
            obj = json.loads(line)
            user_ids.append(obj["user_id"])
            item_ids.append(obj["business_id"])
            ratings.append(float(obj["stars"]))
            dates.append(obj["date"])

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": pd.array(ratings, dtype="float32"),
            "date": dates,
        }
    )

    # Free the line-buffer lists eagerly — DataFrame now owns the
    # memory, and on a 32 GB machine these duplicates would otherwise
    # linger until next GC pass.
    del user_ids, item_ids, ratings, dates

    # ``date`` arrives as ISO-like strings ("2018-01-01 00:00:00").
    # Coerce to int seconds since epoch to match the Amazon / MovieLens
    # schema where ``timestamp`` is an int.
    df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
    df = df[["user_id", "item_id", "rating", "timestamp"]]
    return df


UNKNOWN = "unknown"


def _coerce_str(value: object) -> str:
    """Normalise a Yelp JSONL field to a clean string.

    Yelp records have a mix of null encodings: ``None``, JSON ``null``
    (which pandas reads as NaN-float), or empty strings. ``value or ""``
    doesn't catch NaN because NaN is truthy in Python. This helper
    returns ``""`` for any non-string / null / NaN input, leaving the
    caller free to substitute ``UNKNOWN`` where the prompt needs it.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _primary_type(categories: str) -> str:
    """Return the *parent* / most-general category from Yelp's category
    string.

    Yelp's category taxonomy lists subcategories first and the parent
    last (e.g. ``"Pizza, Italian, Restaurants"`` → ``"Restaurants"``).
    Taking the tail keeps ``type`` semantically distinct from
    ``category``: ``category`` carries the specific descriptor, ``type``
    carries the broader bucket.
    """
    if not categories:
        return UNKNOWN
    parts = [p.strip() for p in categories.split(",") if p.strip()]
    return parts[-1] if parts else UNKNOWN


def load_metadata(
    data_dir: str | Path,
    dataset: str = DATASET_NAME,
) -> pd.DataFrame:
    """Load Yelp business metadata for LLM prompt construction.

    Returns the seven raw fields needed by the LLM-ESR Yelp template
    (assembled in :meth:`ItemMetadata.from_yelp`):
    name, categories, primary type, is_open, review_count, city, stars.

    Args:
        data_dir: Project ``data/`` directory.
        dataset: Always ``"yelp"`` here; ignored. Kept for API symmetry.

    Returns:
        DataFrame with columns
        ``[item_id, name, category, type, is_open, review_count, city, stars]``.
        Missing values are rendered as the literal string ``"unknown"``
        so the downstream prompt template stays positionally consistent
        across items — every business gets a 7-field record, even if
        some fields were null in the source JSONL.

    Raises:
        FileNotFoundError: If the business JSONL is missing.
    """
    business_path = _yelp_dir(data_dir) / BUSINESS_FILE
    if not business_path.exists():
        raise FileNotFoundError(
            f"Yelp businesses not found at {business_path}. "
            "Run :func:`load_ratings` first to see the full "
            "download instructions."
        )

    # The business file is ~150K rows — small enough for pandas to
    # parse in one shot (unlike the review file).
    df = pd.read_json(business_path, lines=True)

    def _open_status(value: object) -> str:
        """``is_open`` is 0/1 in the JSONL; render as ``open``/``closed``."""
        if value is None:
            return UNKNOWN
        if isinstance(value, float) and pd.isna(value):
            return UNKNOWN
        try:
            return "open" if int(value) == 1 else "closed"
        except (TypeError, ValueError):
            return UNKNOWN

    def _int_or_unknown(value: object) -> str:
        if value is None:
            return UNKNOWN
        if isinstance(value, float) and pd.isna(value):
            return UNKNOWN
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return UNKNOWN

    def _stars_or_unknown(value: object) -> str:
        if value is None:
            return UNKNOWN
        if isinstance(value, float) and pd.isna(value):
            return UNKNOWN
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return UNKNOWN

    def _str_or_unknown(value: object) -> str:
        s = _coerce_str(value)
        return s if s else UNKNOWN

    out = pd.DataFrame(
        {
            "item_id": df["business_id"].astype(str),
            "name": df["name"].apply(_str_or_unknown),
            "category": df["categories"].apply(_str_or_unknown),
            "type": df["categories"].apply(
                lambda v: _primary_type(_coerce_str(v))
            ),
            "is_open": df["is_open"].apply(_open_status),
            "review_count": df["review_count"].apply(_int_or_unknown),
            "city": df["city"].apply(_str_or_unknown),
            "stars": df["stars"].apply(_stars_or_unknown),
        }
    )
    return out
