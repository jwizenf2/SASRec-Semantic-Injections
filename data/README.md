# Data

This directory is intentionally empty. Dataset files are large and not tracked in git.

## Amazon Datasets (Video Games, Beauty, Sports & Outdoors)

These datasets are downloaded automatically from HuggingFace on first run via the `datasets` library:

- `McAuley-Lab/Amazon-Reviews-2023` — used for Video Games, Beauty, and Sports & Outdoors
- No manual download required; the data loader handles caching

## Yelp Open Dataset

Yelp data requires a manual download due to licensing:

1. Register and download from https://www.yelp.com/dataset
2. Extract to `data/yelp/` so the following files exist:
   - `data/yelp/yelp_academic_dataset_review.json`
   - `data/yelp/yelp_academic_dataset_business.json`
3. Run the preparation script:
   ```bash
   uv run python sailrec/scripts/prepare_yelp.py
   ```

## LLM Embeddings

Pre-computed Qwen3-Embedding-0.6B embeddings are stored under `sailrec/outputs/embeddings/` and are NOT tracked in git (large binary files). To regenerate them for a dataset, run:

```bash
uv run python sailrec/scripts/extract_llm_embeddings.py \
    --config sailrec/configs/cross_dataset/beauty.yaml
```

This requires `mlx-lm` (Apple Silicon) or a compatible LLM inference backend. The extraction must be run once per dataset before training SAILRec models.
