"""Download + preprocess + extract embeddings for Amazon 2018 datasets.

Replicates LLMEmb/LLM-ESR's exact preprocessing (single-pass 3-core,
min_len=3) on Amazon Beauty and Sports 2018 5-core data, then extracts
Qwen3-Embedding-0.6B-4bit-DWQ embeddings using LLMEmb's exact item
prompt template. Outputs everything the training configs expect.

Usage:
    uv run python sailrec/scripts/prepare_amazon2018.py --datasets beauty sports
    uv run python sailrec/scripts/prepare_amazon2018.py --datasets beauty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sailrec.data.amazon2018 import load_amazon2018, load_amazon2018_meta
from sailrec.data.splitting import (
    generate_negative_samples,
    leave_one_out_split,
    save_negative_samples,
)
from sailrec.data.loaders import build_user_sequences


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_embeddings(
    prompts: dict[int, str],
    num_items: int,
    model_name: str,
    output_path: Path,
    progress_every: int = 200,
) -> None:
    """Extract Qwen3 embeddings for all items and save as a (N+1, D) tensor.

    Mirrors extract_llm_embeddings.py exactly:
    - Uses model.model (transformer body) not model (LM head → logits).
    - Appends EOS token — required for Qwen3-Embedding last-token pooling.
    - Converts via .tolist() to avoid bfloat16 PEP-3118 buffer mismatch.
    - L2-normalises each row.
    """
    import time
    import torch.nn.functional as F

    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        raise ImportError("mlx-lm required: uv pip install mlx-lm")

    log(f"  Loading {model_name}...")
    model, tokenizer = load(model_name)

    # Strip the LM head — we want hidden states, not token logits.
    base_model = model.model if hasattr(model, "model") else model

    # Probe hidden_dim once so we can pre-allocate.
    probe = mx.array([[tokenizer.encode("test")[0]]])
    hidden_dim = base_model(probe).shape[-1]
    log(f"  Hidden dim: {hidden_dim}")

    eos_id = tokenizer.eos_token_id

    embeddings = torch.zeros(num_items + 1, hidden_dim, dtype=torch.float32)

    log(f"  Extracting embeddings for {num_items} items...")
    t_start = time.time()

    for item_id in range(1, num_items + 1):
        text = prompts.get(item_id, "")
        tokens = tokenizer.encode(text)[:511]  # leave room for EOS
        if eos_id is not None and (not tokens or tokens[-1] != eos_id):
            tokens = tokens + [eos_id]

        input_ids = mx.array([tokens])
        hidden_states = base_model(input_ids)       # (1, seq_len, hidden_dim)
        last_hidden = hidden_states[0, -1, :]
        # .tolist() avoids bfloat16 PEP-3118 buffer mismatch with numpy/torch.
        row = torch.tensor(last_hidden.tolist(), dtype=torch.float32)
        row = F.normalize(row, p=2, dim=0)
        embeddings[item_id] = row

        if item_id % progress_every == 0 or item_id == num_items:
            elapsed = time.time() - t_start
            rate = item_id / elapsed
            eta = (num_items - item_id) / rate if rate > 0 else 0
            log(f"    [{item_id}/{num_items}] {rate:.1f} items/s  ETA {eta:.0f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    log(f"  Saved embeddings: {tuple(embeddings.shape)} → {output_path}")


def prepare_dataset(dataset: str, data_dir: str, model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Preparing Amazon 2018 — {dataset}")
    log(f"{'='*60}")

    # 1. Load + preprocess with LLMEmb's exact settings.
    df, user_map, item_map = load_amazon2018(
        dataset=dataset, data_dir=data_dir, user_core=3, item_core=3, min_len=3
    )
    num_items = len(item_map)
    log(f"  num_users={len(user_map):,}, num_items={num_items:,}")

    # Save item_map for reproducibility.
    out_dir = Path(data_dir) / "amazon2018" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "item_map.json", "w") as f:
        json.dump({v: k for k, v in item_map.items()}, f)  # int → asin

    # 2. Build sequences + leave-one-out split.
    user_seqs = build_user_sequences(df)
    split = leave_one_out_split(user_seqs, num_items)
    log(f"  Split: {split.num_users:,} users, {split.num_items:,} items")

    # 3. Generate and cache negative samples.
    neg_path = out_dir / "neg_samples.npz"
    if not neg_path.exists():
        log("  Generating negative samples...")
        neg_samples = generate_negative_samples(split, num_neg=100)
        save_negative_samples(neg_samples, str(neg_path))
        log(f"  Saved neg_samples → {neg_path}")
    else:
        log(f"  Cached neg_samples found: {neg_path}")

    # 4. Build item prompts + extract embeddings.
    emb_path = Path("sailrec/outputs/embeddings") / f"{dataset}2018.pt"
    if emb_path.exists():
        existing = torch.load(emb_path, map_location="cpu", weights_only=True)
        if existing.shape[0] == num_items + 1:
            log(f"  Cached embeddings found: {emb_path} {tuple(existing.shape)}")
            return
        log(f"  Cached embeddings shape mismatch — re-extracting.")

    log("  Building item prompts (LLMEmb exact template)...")
    prompts = load_amazon2018_meta(dataset, data_dir, item_map)
    log(f"  Example prompt for item 1:\n    {prompts.get(1, '?')[:120]}")

    extract_embeddings(prompts, num_items, model_name, emb_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["beauty", "sports"],
        choices=["beauty", "sports"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--model-name",
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        help="MLX model for embedding extraction.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        prepare_dataset(dataset, args.data_dir, args.model_name)

    log("\nAll datasets prepared. Training configs can now be run.")


if __name__ == "__main__":
    main()
