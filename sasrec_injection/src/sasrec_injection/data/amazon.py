"""Amazon Reviews 2023 loader (McAuley-Lab/Amazon-Reviews-2023 on HF).

Counterpart to :mod:`sasrec_injection.data.movielens` for the standard LLM4Rec
benchmark family (Beauty, Toys, Sports, Books, Video Games, ...). The
2023 release ships pre-filtered ``5core_*`` subsets with timestamps,
plus rich item metadata (title + multi-paragraph description +
categories + images).

Why the 2023 release vs 2018
----------------------------
DLLM2Rec, P5, TIGER, BIGRec and others use the 2018 McAuley dump. The
2023 release fixes timestamp granularity, adds ~5× more reviews, and
provides clean per-category 5-core subsets — all of which simplify
SASRec preprocessing. Numbers reported on 2023-Beauty will not be
apples-to-apples with 2018-Beauty papers, but our internal P1 → ItemTable
comparison is self-consistent because both use this loader.

What this module returns
------------------------

* :func:`load_ratings`  — DataFrame with the four canonical columns
                          (``[user_id, item_id, rating, timestamp]``)
                          ready for :func:`sasrec_injection.data.movielens.preprocess`.
* :func:`load_metadata` — DataFrame with
                          ``[item_id, title, description]``, used by
                          :class:`sasrec_injection.data.item_metadata.ItemMetadata`
                          to build LLM prompts.

Cache layout
------------

``hf_hub_download`` lives at ``<data_dir>/amazon-reviews-2023/`` (or
wherever the caller's HF cache is configured to). Subsequent calls
hit the cache, so the network round-trip happens exactly once per
category per machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default Amazon "category" within Amazon-Reviews-2023. Beauty is the
# smallest standard subset; matches the DLLM2Rec/P5/TIGER family of
# papers. ItemTable's headline experiment uses Video_Games instead — see
# `sasrec_injection/configs/sasrec_injection_video_games.yaml`.
DEFAULT_AMAZON_CATEGORY = "All_Beauty"

# HuggingFace dataset ID. Public repo, Apache 2.0 license.
HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


def _hf_download(repo_path: str, cache_dir: Path) -> Path:
    """Download one file from the McAuley HF repo (cached, idempotent).

    Why we don't use the ``datasets`` library
    -----------------------------------------
    The 2023 release publishes Python-script-based ``datasets`` loaders
    which the new ``datasets`` library refuses (``trust_remote_code``
    is no longer supported as a top-level kwarg). We sidestep by
    pulling raw files via :func:`huggingface_hub.hf_hub_download`,
    which is the dependency the old loader used internally anyway.

    Args:
        repo_path: Path within the HF repo, e.g.
            ``"benchmark/5core/rating_only/Video_Games.csv"``.
        cache_dir: Where to cache the file. The HF library organises
            its own subdir layout under this directory.

    Returns:
        Local filesystem path of the downloaded (or cached) file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover -- import-time guard
        raise ImportError(
            "Loading Amazon Reviews 2023 requires `huggingface_hub` "
            "(usually a transitive dep of `datasets`). "
            "Install with `pip install huggingface_hub` or rerun "
            "`uv sync` from the project root."
        ) from e
    local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local)


def load_ratings(
    data_dir: str | Path,
    dataset: str = DEFAULT_AMAZON_CATEGORY,
) -> pd.DataFrame:
    """Load 5-core ratings + timestamps for one Amazon category.

    Args:
        data_dir: Root cache directory. Files are stored under
            ``<data_dir>/amazon-reviews-2023/`` (forwarded to HF as
            ``cache_dir``).
        dataset: Amazon category (e.g. ``"All_Beauty"``,
            ``"Toys_and_Games"``, ``"Sports_and_Outdoors"``,
            ``"Video_Games"``, ``"Books"``).

    Returns:
        DataFrame with columns ``[user_id, item_id, rating, timestamp]``,
        ready for :func:`sasrec_injection.data.movielens.preprocess`.

    Notes:
        The McAuley benchmark splits live under
        ``benchmark/5core/timestamp/``, but those are *already* split
        leave-one-out (one item per user in val/test). For our pipeline
        we want the *full* per-user sequence and apply our own LOO
        split, so we use ``benchmark/5core/rating_only/<cat>.csv`` —
        the "rating_only" name is misleading; it does include
        timestamps.
    """
    cache_dir = Path(data_dir) / "amazon-reviews-2023"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _hf_download(
        f"benchmark/5core/rating_only/{dataset}.csv", cache_dir
    )

    df = pd.read_csv(csv_path)
    # The CSV uses ``parent_asin`` as the item identifier (a string
    # SKU). Rename for parity with MovieLens.
    df = df.rename(columns={"parent_asin": "item_id"})

    keep = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Amazon CSV {csv_path} missing expected columns {missing}; "
            f"got {df.columns.tolist()}"
        )
    return df[keep]


def load_metadata(
    data_dir: str | Path,
    dataset: str = DEFAULT_AMAZON_CATEGORY,
) -> pd.DataFrame:
    """Load per-item metadata (title + description) for one category.

    Args:
        data_dir: Root cache directory.
        dataset: Amazon category name.

    Returns:
        DataFrame with three columns:
        ``[item_id, title, description]``. ``description`` is a single
        string (joined with newlines if the source had multiple
        paragraphs).
    """
    import json as _json

    cache_dir = Path(data_dir) / "amazon-reviews-2023"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = _hf_download(
        f"raw/meta_categories/meta_{dataset}.jsonl", cache_dir
    )

    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                # The metadata file is large and occasionally has
                # truncated lines at the tail. Skip them rather than
                # failing the whole load.
                continue
            rows.append({
                "item_id": obj.get("parent_asin"),
                "title": obj.get("title") or "",
                "description": _join_text_field(obj.get("description")),
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["item_id"])
    return df[["item_id", "title", "description"]]


def _join_text_field(value) -> str:
    """Robustly turn a list-of-strings (or odd value) into one string.

    The Amazon metadata's ``description`` field is sometimes a list of
    paragraph strings, sometimes a single string, sometimes None.
    Normalise to a single string with newline joins.
    """
    if isinstance(value, list):
        return "\n".join(str(p) for p in value if p)
    if value is None:
        return ""
    return str(value)
