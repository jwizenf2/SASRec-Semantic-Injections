"""Amazon Reviews 2018 loader with LLMEmb/LLM-ESR exact preprocessing.

Downloads and parses the Amazon 2018 5-core review files from Stanford
SNAP, applies the exact filter_common (single-pass 3-core) and
filter_minmum (min_len=3) used by LLMEmb (arXiv:2409.19925) and
LLM-ESR (NeurIPS 2024), and produces:

  1. A 4-column DataFrame (user_id, item_id, rating, timestamp) in the
     same format our preprocess() and leave-one-out split expect.
  2. A dict mapping item asin → prompt string using LLMEmb's exact
     Beauty/Sports prompt template.

Why single-pass filtering here (vs our iterative bipartite):
  LLMEmb/LLM-ESR both use filter_common (one-pass). To enable direct
  numeric comparison to their published tables we replicate their
  preprocessing exactly on these datasets. Our other datasets (VG,
  Yelp) still use iterative bipartite k-core.

Datasets supported:
  "beauty"  → Amazon Beauty 2018 5-core (22K users, 12K items)
  "sports"  → Amazon Sports and Outdoors 2018 5-core (35K users, 18K items)
"""

from __future__ import annotations

import ast
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Snap Stanford URLs for Amazon 2018 5-core reviews + full meta.
_SNAP_BASE = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

_DATASET_FILES = {
    "beauty": {
        "reviews": f"{_SNAP_BASE}/reviews_Beauty_5.json.gz",
        "meta":    f"{_SNAP_BASE}/meta_Beauty.json.gz",
        "slug":    "Beauty",
    },
    "sports": {
        "reviews": f"{_SNAP_BASE}/reviews_Sports_and_Outdoors_5.json.gz",
        "meta":    f"{_SNAP_BASE}/meta_Sports_and_Outdoors.json.gz",
        "slug":    "Sports_and_Outdoors",
    },
}

# LLMEmb's exact prompt templates per dataset.
_PROMPT_TEMPLATES = {
    "beauty": (
        "The beauty item has following attributes: \n"
        "name is {title}; brand is {brand}; price is {price}. \n"
        "The item has following features: {categories}. \n"
        "The item has following descriptions: {description}. \n"
    ),
    "sports": (
        "The sports item has following attributes: \n"
        "name is {title}; brand is {brand}; price is {price}. \n"
        "The item has following features: {categories}. \n"
        "The item has following descriptions: {description}. \n"
    ),
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, verbose: bool = True) -> None:
    if dest.exists():
        if verbose:
            print(f"  Already cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  Downloading {url} → {dest} ...")
    urllib.request.urlretrieve(url, dest)
    if verbose:
        print(f"  Done ({dest.stat().st_size // 1024 // 1024} MB)")


# ---------------------------------------------------------------------------
# Parse Amazon 2018 gzip files
# ---------------------------------------------------------------------------


def _parse_reviews(path: Path) -> list[tuple[str, str, int]]:
    """Read (user, item, timestamp) triples from a reviews .json.gz file."""
    triples = []
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            # Rating filter: LLMEmb uses rating_score=0.0 (keep all ≥ 1★).
            if float(obj.get("overall", 5.0)) <= 0.0:
                continue
            user = obj.get("reviewerID", "")
            item = obj.get("asin", "")
            ts   = int(obj.get("unixReviewTime", 0))
            if user and item:
                triples.append((user, item, ts))
    return triples


def _parse_meta(path: Path) -> dict[str, dict]:
    """Read item metadata from a meta .json.gz file.

    Amazon 2018 meta files use Python dict syntax (eval-able), not
    strict JSON — hence ast.literal_eval instead of json.loads.
    """
    meta: dict[str, dict] = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                obj = ast.literal_eval(line.decode("utf-8").strip())
            except Exception:
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
            asin = obj.get("asin", "")
            if asin:
                meta[asin] = obj
    return meta


# ---------------------------------------------------------------------------
# LLMEmb/LLM-ESR exact single-pass filter
# ---------------------------------------------------------------------------


def _filter_common(
    triples: list[tuple[str, str, int]],
    user_t: int = 3,
    item_t: int = 3,
) -> dict[str, list[str]]:
    """Single-pass k-core filter + chronological sort — exact LLMEmb recipe.

    Counts user and item frequencies once, drops tuples where either
    user < user_t or item < item_t, then sorts each user's sequence by
    timestamp. This is NOT iterative bipartite k-core; it's the same
    filter_common used in LLMEmb's data_process.py.
    """
    user_count: dict[str, int] = defaultdict(int)
    item_count: dict[str, int] = defaultdict(int)
    for user, item, _ in triples:
        user_count[user] += 1
        item_count[item] += 1

    user_seqs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for user, item, ts in triples:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        user_seqs[user].append((item, ts))

    # Sort each user's history by timestamp.
    result: dict[str, list[str]] = {}
    for user, items_ts in user_seqs.items():
        items_ts.sort(key=lambda x: x[1])
        result[user] = [i for i, _ in items_ts]
    return result


def _filter_minmum(
    user_items: dict[str, list[str]], min_len: int = 3
) -> dict[str, list[str]]:
    """Drop users with fewer than min_len interactions (LLMEmb exact)."""
    return {u: items for u, items in user_items.items() if len(items) >= min_len}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_amazon2018(
    dataset: str,
    data_dir: str | Path,
    user_core: int = 3,
    item_core: int = 3,
    min_len: int = 3,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    """Download, preprocess, and return a 4-column DataFrame.

    Applies LLMEmb/LLM-ESR's exact single-pass filtering. The returned
    DataFrame has the same schema as our other loaders:
    (user_id: int, item_id: str, rating: float, timestamp: int).

    Args:
        dataset: "beauty" or "sports".
        data_dir: Root data directory (raw files go under
            ``data_dir/amazon2018/<dataset>/``).
        user_core / item_core: K-core thresholds (default 3 = LLMEmb).
        min_len: Minimum sequence length after filtering (default 3).
        verbose: Print progress.

    Returns:
        (df, user_map, item_map) — same contract as movielens.preprocess().
        user_map: original str id → contiguous int (≥1).
        item_map: original asin str → contiguous int (≥1).
    """
    dataset = dataset.lower()
    if dataset not in _DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(_DATASET_FILES)}")

    info = _DATASET_FILES[dataset]
    raw_dir = Path(data_dir) / "amazon2018" / dataset / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / f"reviews_{info['slug']}_5.json.gz"
    meta_path    = raw_dir / f"meta_{info['slug']}.json.gz"

    _download(info["reviews"], reviews_path, verbose=verbose)
    _download(info["meta"],    meta_path,    verbose=verbose)

    if verbose:
        print(f"  Parsing reviews...")
    triples = _parse_reviews(reviews_path)
    if verbose:
        print(f"  Raw interactions: {len(triples):,}")

    # Apply LLMEmb's exact preprocessing.
    user_items = _filter_common(triples, user_t=user_core, item_t=item_core)
    user_items = _filter_minmum(user_items, min_len=min_len)

    if verbose:
        n_users = len(user_items)
        n_items = len({i for seq in user_items.values() for i in seq})
        n_inters = sum(len(seq) for seq in user_items.values())
        print(f"  After {user_core}-core + min_len={min_len}: "
              f"{n_users:,} users, {n_items:,} items, {n_inters:,} interactions")

    # Build item_map (asin → int) for metadata lookup. user_map is
    # intentionally NOT used for df IDs — we keep string IDs in the
    # DataFrame so our standard preprocess() handles the remapping.
    # This ensures the item embedding tensor index is consistent with
    # the item_map produced by preprocess().
    unique_items = sorted({i for seq in user_items.values() for i in seq})
    item_map = {i: idx + 1 for idx, i in enumerate(unique_items)}  # for metadata use only
    unique_users = sorted(user_items.keys())
    user_map = {u: idx + 1 for idx, u in enumerate(unique_users)}  # for metadata use only

    # Flatten to 4-column DataFrame with ORIGINAL string IDs.
    # preprocess() downstream will assign contiguous int IDs.
    rows = []
    for user, seq in user_items.items():
        for pos, asin in enumerate(seq):
            rows.append({
                "user_id":   user,
                "item_id":   asin,
                "rating":    5.0,
                "timestamp": pos,   # chronological position (already sorted)
            })
    df = pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df, user_map, item_map


def load_amazon2018_meta(
    dataset: str,
    data_dir: str | Path,
    item_map: dict,
    verbose: bool = True,
) -> dict[int, str]:
    """Load item metadata and build LLMEmb's exact prompt strings.

    Args:
        item_map: asin → int item id (from load_amazon2018).

    Returns:
        Dict mapping int item id → prompt string.
    """
    dataset = dataset.lower()
    info = _DATASET_FILES[dataset]
    meta_path = Path(data_dir) / "amazon2018" / dataset / "raw" / f"meta_{info['slug']}.json.gz"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}. Run load_amazon2018 first.")

    if verbose:
        print(f"  Parsing metadata for {dataset}...")
    raw_meta = _parse_meta(meta_path)

    template = _PROMPT_TEMPLATES[dataset]
    prompts: dict[int, str] = {}
    for asin, item_id in item_map.items():
        info_dict = raw_meta.get(asin, {})
        title = str(info_dict.get("title", "unknown"))[:100]
        brand = str(info_dict.get("brand", "unknown"))[:100]
        price = str(info_dict.get("price", "unknown"))[:100]
        # categories is a list of lists; take the first list, join with "; "
        cats = info_dict.get("categories", [[]])
        cat_str = "; ".join(str(c) for c in (cats[0] if cats else []))[:100]
        desc = str(info_dict.get("description", "unknown"))[:200]
        prompt = template.format(
            title=title, brand=brand, price=price,
            categories=cat_str, description=desc,
        )
        prompts[item_id] = prompt
    return prompts
