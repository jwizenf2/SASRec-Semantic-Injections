"""Extract per-item LLM embeddings using Qwen3-Embedding via MLX.

What this script does
---------------------

Given a SAILRec experiment config (which names a dataset), this
script:

1. Loads the dataset's ratings to build the item id → contiguous int
   map (so the saved tensor's row indices line up with what SAILRec
   sees during training).
2. Loads the dataset's metadata (item titles + side info).
3. For each item, encodes ``"<prefix>{title} ({side_info})"`` (the
   exact format depends on the dataset family) through Qwen3-Embedding
   in MLX, takes the last-token hidden state as the item embedding.
4. ℓ2-normalises each row.
5. Saves the resulting ``(num_items + 1, hidden_dim)`` tensor to disk.
   Row 0 is zeros (padding).

The saved tensor is consumed by the SAILRec training script via the
``align.embeddings_path`` field in the YAML config.

Why MLX
-------

Apple Silicon's MLX is the fastest path to running 4-bit quantised
LLMs on M-series GPUs. The 0.6B-4bit-DWQ Qwen3 variant fits the entire
25,612-item Video_Games catalog in approximately 10 minutes on M2
Max. The 8B variant takes ~40 minutes and gives 4096-dim embeddings;
SAILRec uses 0.6B/1024-dim by default.

Qwen3-Embedding extraction recipe
---------------------------------

* **Documents have no instruction prefix** (only queries do). Items
  are documents, so the raw text goes in directly.
* **EOS token must be present** at sequence end — its hidden state is
  the pooled embedding. The MLX tokenizer doesn't auto-append it, so
  we do.
* **L2-normalise the output** — the model was trained for cosine
  similarity. Downstream the alignment loss applies its own
  normalisation, but pre-normalising keeps rows comparable for
  diagnostics.

Usage
-----

::

    uv run python sailrec/scripts/extract_llm_embeddings.py \\
        --config sailrec/configs/sailrec_video_games.yaml \\
        --model-name mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \\
        --output sailrec/outputs/embeddings/video_games.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make ``import sailrec...`` work whether this script is run from the
# project root or from anywhere else. Adds ``sailrec/src/`` to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.config import load_config
from sailrec.data.item_metadata import ItemMetadata
from sailrec.data.loaders import is_amazon, is_yelp, load_ratings, preprocess


def log(msg: str) -> None:
    """Print with immediate flush so ``tail -f`` shows progress live."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Core extraction loop
# ---------------------------------------------------------------------------


def extract_embeddings(
    metadata: ItemMetadata,
    num_items: int,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    normalize: bool = True,
    progress_every: int = 200,
    item_prefix: str = "Movie: ",
) -> torch.Tensor:
    """Extract one Qwen3 embedding per item.

    Args:
        metadata: Per-item title + side-info (already keyed by remapped
            item ids so this loop's ``item_id`` matches the SAILRec
            training pipeline's ids exactly).
        num_items: Number of items (ids 1..num_items). The output
            tensor has ``num_items + 1`` rows (row 0 = padding).
        model_name: HuggingFace MLX model id. Defaults to the 0.6B
            4-bit-DWQ Qwen3 variant; the 8B version
            (``mlx-community/Qwen3-Embedding-8B-4bit-DWQ``) gives
            4096-dim embeddings.
        normalize: ℓ2-normalise output rows. Should be True; only set
            False for debugging.
        progress_every: Emit a throughput line every N items.
        item_prefix: String prepended to every item's text. Standard
            choices: ``"Movie: "`` for MovieLens, ``"Product: "`` for
            Amazon. Picked automatically by the CLI when not set
            explicitly.

    Returns:
        ``(num_items + 1, hidden_dim)`` float32 tensor; row 0 is zero.
    """
    # Local imports — MLX is a heavy dependency we only want at
    # extraction time, not when the rest of the pipeline imports
    # this module.
    import mlx.core as mx
    from mlx_lm import load

    log(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)

    # mlx_lm's ``load`` wraps the base transformer under ``.model``
    # when a LM head is attached. Strip it so forward returns hidden
    # states (what we want for embedding extraction), not token
    # logits.
    base_model = model.model if hasattr(model, "model") else model

    # Probe hidden_dim with a single dummy token. Cheaper than
    # pre-declaring and failing late if someone points at a different
    # Qwen variant (e.g. swapping 0.6B for 8B without updating llm_dim
    # in the SAILRec config).
    probe_tokens = mx.array([[tokenizer.encode("test")[0]]])
    probe_out = base_model(probe_tokens)
    hidden_dim = probe_out.shape[-1]
    log(f"Hidden dim (probed): {hidden_dim}")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Defensive — if the tokenizer config has drifted, the last
        # content token will be used as the pooled embedding, which
        # degrades quality but doesn't crash.
        log(
            "WARNING: tokenizer has no eos_token_id; "
            "last content token will be used for pooling."
        )

    # Pre-allocate the output tensor. Row 0 is padding (zeros) and
    # SASRec's nn.Embedding(padding_idx=0) ensures it stays that way.
    embeddings = torch.zeros(num_items + 1, hidden_dim, dtype=torch.float32)

    log(f"Extracting embeddings for {num_items} items...")
    t_start = time.time()

    for item_id in range(1, num_items + 1):
        # Build the prompt: keep this format identical across runs so
        # any number drift is attributable to the encoder, not the
        # prompt wording.
        text = f"{item_prefix}{metadata.format_item(item_id)}"

        # Tokenise. The Qwen3 recipe requires an explicit EOS at the
        # end since its last-token-pooling reads that position.
        tokens = tokenizer.encode(text)
        if eos_id is not None and (not tokens or tokens[-1] != eos_id):
            tokens = tokens + [eos_id]

        input_ids = mx.array([tokens])
        hidden_states = base_model(input_ids)            # (1, seq_len, hidden_dim)

        # Last-token pool. With batch_size=1 there's no padding so
        # ``[0, -1, :]`` is the valid pooled embedding.
        last_hidden = hidden_states[0, -1, :]
        row = torch.tensor(last_hidden.tolist(), dtype=torch.float32)

        if normalize:
            row = F.normalize(row, p=2, dim=0)

        embeddings[item_id] = row

        # Throughput / ETA logging. Useful when extracting on a 25K-
        # item catalog where the run takes ~10 minutes.
        if item_id % progress_every == 0 or item_id == num_items:
            elapsed = time.time() - t_start
            rate = item_id / elapsed
            eta = (num_items - item_id) / rate if rate > 0 else 0
            log(f"  [{item_id}/{num_items}] {rate:.1f} items/s, ETA: {eta:.0f}s")

    total_time = time.time() - t_start
    log(
        f"Extraction complete in {total_time:.1f}s "
        f"({num_items / total_time:.1f} items/s)"
    )

    return embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Qwen3-Embedding item embeddings for SAILRec "
            "training. Saves a (num_items+1, hidden_dim) tensor to "
            "the path given by --output."
        )
    )
    parser.add_argument(
        "--config",
        default="sailrec/configs/sailrec_video_games.yaml",
        help="SAILRec config (used only to resolve dataset name).",
    )
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument(
        "--model-name",
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        help=(
            "MLX model ID on HuggingFace. 0.6B-4bit-DWQ (1024-dim) "
            "is SAILRec's default — small and fast on M2 Max."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to save the resulting tensor. If omitted, defaults to "
            "the config's ``align.embeddings_path`` so the file ends up "
            "exactly where the SAILRec trainer will look for it. Pass "
            "explicitly only when you intentionally want to write "
            "elsewhere."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Smoke-test convenience: extract only the first N items.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip ℓ2 normalisation (debugging only — quality drops).",
    )
    parser.add_argument(
        "--item-prefix",
        default=None,
        help=(
            "Prefix prepended to every item's text. Defaults to "
            "'Movie: ' for MovieLens, 'Product: ' for Amazon, "
            "and '' (empty) for Yelp — the Yelp template is already a "
            "complete LLM-ESR-style sentence."
        ),
    )
    args = parser.parse_args()

    # Resolve dataset from the SAILRec config so the row indices in
    # the output tensor line up with what training will see.
    cfg = load_config(args.config, args.base_config)
    dataset_cfg = cfg.get("dataset", {})
    dataset = dataset_cfg.get("name", "amazon-Video_Games")
    min_interactions = dataset_cfg.get("min_interactions", 5)

    log(f"Loading {dataset} data...")
    ratings = load_ratings("data", dataset=dataset)
    _, _, item_map = preprocess(ratings, min_interactions=min_interactions)
    num_items = len(item_map)
    log(f"Items: {num_items}")

    # Item metadata (titles + side info) keyed by remapped item id.
    metadata = ItemMetadata.from_dataset("data", item_map, dataset=dataset)

    # Default item-prefix is dataset-dependent. Each prefix anchors
    # the LLM in the right domain (a "Product:" prefix on a movie item
    # would push the embedding toward retail-product semantics).
    item_prefix = args.item_prefix
    if item_prefix is None:
        if is_amazon(dataset):
            item_prefix = "Product: "
        elif is_yelp(dataset):
            # The Yelp template ("The point of interest has following
            # attributes: …") is already a complete sentence — adding
            # a "Business: " prefix would break its grammar.
            item_prefix = ""
        else:
            item_prefix = "Movie: "

    # ``--max-items`` is for sanity-check runs only — it produces a
    # padded output tensor with only the first N items filled.
    effective_num_items = (
        min(args.max_items, num_items) if args.max_items else num_items
    )
    if args.max_items:
        log(
            f"SMOKE MODE: extracting only first {effective_num_items} items; "
            "remaining rows zero-padded."
        )

    extracted = extract_embeddings(
        metadata,
        effective_num_items,
        model_name=args.model_name,
        normalize=not args.no_normalize,
        item_prefix=item_prefix,
    )

    # Pad up to (num_items + 1, hidden_dim) when smoke-testing so the
    # SAILRec config's shape assertion still passes.
    if effective_num_items < num_items:
        hidden_dim = extracted.shape[1]
        full = torch.zeros(num_items + 1, hidden_dim, dtype=torch.float32)
        full[: effective_num_items + 1] = extracted
        extracted = full

    # Default output path comes from the config so each dataset writes
    # to its own file. Hard-coding a Video_Games default here previously
    # caused a Yelp run to silently overwrite the locked headline file.
    if args.output is not None:
        output_path = Path(args.output)
    else:
        align_cfg = cfg.get("align", {}) or {}
        cfg_output = align_cfg.get("embeddings_path")
        if not cfg_output:
            raise SystemExit(
                "No --output given and config has no align.embeddings_path. "
                "Either pass --output explicitly or add align.embeddings_path "
                "to the YAML."
            )
        output_path = Path(cfg_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, output_path)
    log(f"Saved embeddings: {tuple(extracted.shape)} -> {output_path}")


if __name__ == "__main__":
    main()
