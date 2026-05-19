"""Prepare the Yelp Open Dataset for SAILRec.

The Yelp Open Dataset cannot be downloaded automatically — it sits
behind a clickwrap terms-of-use form at https://www.yelp.com/dataset.
This script does *not* download for you. What it does:

1. Verify ``data/yelp/`` exists and contains the two files SAILRec
   actually needs (``yelp_academic_dataset_business.json`` and
   ``yelp_academic_dataset_review.json``).
2. If they're missing, print clear download / extract instructions.
3. If they're present, run a one-shot preprocessing pass and print the
   dataset's resulting size (users / items / interactions / sparsity)
   so you can size the training run before kicking it off.

Why a separate script
---------------------

Putting the size check here means the training scripts
(``train_p1.py``, ``train_sailrec.py``, ``extract_llm_embeddings.py``)
don't have to special-case "missing data" with bespoke error
messages — they call the loader, which raises a precise
``FileNotFoundError`` with the same instructions.

Usage
-----

::

    uv run python sailrec/scripts/prepare_yelp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``import sailrec...`` work when running this script from the
# project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.data.loaders import (
    build_user_sequences,
    load_ratings,
    preprocess,
)
from sailrec.data.splitting import leave_one_out_split
from sailrec.data.yelp import BUSINESS_FILE, REVIEW_FILE, _yelp_dir


def log(msg: str) -> None:
    """Print with immediate flush."""
    print(msg, flush=True)


def _print_download_instructions(yelp_dir: Path) -> None:
    """Print the steps the user must run by hand to make Yelp available."""
    log("")
    log("=" * 70)
    log("Yelp Open Dataset is not available. To install it:")
    log("=" * 70)
    log("")
    log("  1. Visit https://www.yelp.com/dataset")
    log("  2. Click 'Download Dataset' and accept the terms of service.")
    log("     (The download is ~3.7 GB compressed; ~10 GB extracted.)")
    log(f"  3. Extract the tarball into:  {yelp_dir}/")
    log("     The relevant files inside the tarball are:")
    log(f"        - {BUSINESS_FILE}")
    log(f"        - {REVIEW_FILE}")
    log("     The ``user``, ``tip``, and ``checkin`` files are NOT needed.")
    log("  4. Re-run this script to verify and size the dataset.")
    log("")


def main() -> int:
    yelp_dir = _yelp_dir("data")
    business_path = yelp_dir / BUSINESS_FILE
    review_path = yelp_dir / REVIEW_FILE

    log(f"Looking for Yelp data at: {yelp_dir}")
    missing = [
        p.name for p in (business_path, review_path) if not p.exists()
    ]
    if missing:
        log(f"  MISSING: {', '.join(missing)}")
        _print_download_instructions(yelp_dir)
        return 1

    log("  Found business + review files.")
    log("")

    # Run the loader to confirm the JSONL files parse correctly, then
    # the standard 5-core preprocessing to report final size.
    log("Loading reviews (this can take ~30s on first run)...")
    ratings = load_ratings("data", dataset="yelp")
    log(f"  Raw reviews: {len(ratings):,}")

    log("Running 5-core filtering and id remapping...")
    df, user_map, item_map = preprocess(ratings, min_interactions=5)
    seqs = build_user_sequences(df)
    split = leave_one_out_split(seqs, len(item_map))

    sparsity = (
        len(df) / (split.num_users * split.num_items) * 100
        if split.num_users and split.num_items
        else 0.0
    )
    avg_seq = len(df) / split.num_users if split.num_users else 0.0

    log("")
    log("=" * 70)
    log("Yelp dataset stats (5-core preprocessing):")
    log("=" * 70)
    log(f"  Users:             {split.num_users:>10,}")
    log(f"  Items:             {split.num_items:>10,}")
    log(f"  Interactions:      {len(df):>10,}")
    log(f"  Avg seq length:    {avg_seq:>10.1f}")
    log(f"  Sparsity:          {sparsity:>10.4f}%")
    log("")
    log("Next steps:")
    log("  1. Extract Qwen3 item embeddings:")
    log("       uv run python sailrec/scripts/extract_llm_embeddings.py \\")
    log("           --config sailrec/configs/sailrec_yelp.yaml \\")
    log("           --output sailrec/outputs/embeddings/yelp.pt")
    log("  2. Train P1 baseline (3 seeds):")
    log("       uv run python sailrec/scripts/train_p1.py \\")
    log("           --config sailrec/configs/p1_yelp.yaml \\")
    log("           --seeds 42 7 18")
    log("  3. Train SAILRec (3 seeds):")
    log("       uv run python sailrec/scripts/train_sailrec.py \\")
    log("           --config sailrec/configs/sailrec_yelp.yaml \\")
    log("           --seeds 42 7 18 --lambdas 0.1")
    log("  4. Eval both:")
    log("       uv run python sailrec/scripts/eval_fullrank.py \\")
    log("           --config sailrec/configs/sailrec_yelp.yaml \\")
    log("           --seeds 42 7 18")
    log("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
