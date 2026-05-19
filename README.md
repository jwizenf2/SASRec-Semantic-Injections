# Where LLM Knowledge Belongs

Code for the NeurIPS 2026 paper:

> **Where LLM Knowledge Belongs: A Controlled Study of Semantic Injection Surfaces in SASRec-Style Sequential Recommendation**

We compare four surfaces for injecting LLM semantic knowledge into a fixed SASRec backbone — encoder input fusion, per-position distillation, sequence-level alignment, and item-table integration — across Video Games (development benchmark), Sports, Beauty, and Yelp (held-out).

The paper PDF is at `paper/where_llm_knowledge_belongs.pdf`.

---

## Repository Layout

```
SASRec-Semantic-Injections/
├── paper/
│   ├── where_llm_knowledge_belongs.pdf
│   ├── where_llm_knowledge_belongs.tex
│   ├── neurips_2026.sty
│   ├── checklist.tex
│   └── figures/
│
├── sailrec/                              # code (internal package name)
│   ├── src/sailrec/
│   │   ├── alignment/contrastive.py      # InfoNCE alignment loss + projector
│   │   ├── data/                         # dataset loaders (Amazon, Yelp, MovieLens)
│   │   ├── models/                       # SASRec, BERT4Rec, GRU4Rec
│   │   ├── training/                     # trainer for each injection surface
│   │   ├── evaluation/metrics.py         # HR@K / NDCG@K (sampled + full-rank)
│   │   ├── config.py
│   │   └── seeds.py
│   │
│   ├── scripts/
│   │   ├── extract_llm_embeddings.py     # one-time LLM embedding extraction
│   │   ├── train_p1.py                   # SASRec baseline
│   │   ├── train_sailrec.py              # item-table integration (main method)
│   │   ├── train_a1_llm_init.py          # ablation: LLM-PCA init, no alignment
│   │   ├── train_a2_input_fusion.py      # surface: encoder input fusion
│   │   ├── train_a3_hidden_distill.py    # surface: per-position distillation
│   │   ├── train_a4_seq_infonce.py       # surface: sequence-level InfoNCE
│   │   ├── eval_fullrank.py
│   │   ├── eval_stratified.py
│   │   ├── eval_stratified_sampled.py
│   │   ├── prepare_yelp.py
│   │   └── prepare_amazon2018.py
│   │
│   └── configs/
│       ├── base.yaml                     # shared defaults
│       ├── baseline/                     # SASRec (no LLM) on each dataset
│       ├── cross_surface/                # Table 3: four surfaces on Video Games
│       ├── cross_dataset/                # Table 4: item-table on Sports/Beauty/Yelp
│       ├── decomposition/                # Table 5 + Figure 3: item-table ablation
│       └── appendix/                     # Appendix C (weight-fn) + D (λ sweep)
│
├── data/                                 # not tracked — see data/README.md
├── pyproject.toml
└── uv.lock
```

Every config file's top comment lists the exact command used to reproduce it.

---

## Setup

Requires Python 3.12+. We use [uv](https://github.com/astral-sh/uv).

```bash
git clone git@github.com:jwizenf2/SASRec-Semantic-Injections.git
cd SASRec-Semantic-Injections
uv sync
```

Or with pip: `pip install -e .`

**Apple Silicon note:** Training defaults to `mps`. Set `device: cpu` in `sailrec/configs/base.yaml` for Linux, or `device: cuda` for NVIDIA. The embedding extraction script (`extract_llm_embeddings.py`) uses `mlx-lm` and requires Apple Silicon; on other hardware swap in a compatible inference backend for that step.

---

## Datasets

See `data/README.md` for full instructions.

| Dataset | Source | Download |
|---------|--------|----------|
| Amazon Video Games | HuggingFace `McAuley-Lab/Amazon-Reviews-2023` | Automatic |
| Amazon Beauty | HuggingFace `McAuley-Lab/Amazon-Reviews-2023` | Automatic |
| Amazon Sports & Outdoors | HuggingFace `McAuley-Lab/Amazon-Reviews-2023` | Automatic |
| Yelp | Yelp Open Dataset (requires free registration) | Manual → `data/yelp/` |

All datasets use a 5-core filter and leave-one-out split.

---

## Reproducing Paper Results

### 1. Extract LLM embeddings (once per dataset)

Embeddings come from `Qwen3-Embedding-0.6B` via `mlx-lm`, cached to `sailrec/outputs/embeddings/<dataset>.pt`.

```bash
uv run python sailrec/scripts/extract_llm_embeddings.py \
    --config sailrec/configs/cross_surface/surface_d_item_table.yaml
```

Repeat with a config from each dataset (the dataset name is read from the config).

### 2. SASRec baselines (Table 3 row 1, Table 4 baselines)

```bash
uv run python sailrec/scripts/train_p1.py \
    --config sailrec/configs/baseline/video_games.yaml --seeds 42 7 18

uv run python sailrec/scripts/train_p1.py \
    --config sailrec/configs/baseline/sports.yaml --seeds 42 7 18

uv run python sailrec/scripts/train_p1.py \
    --config sailrec/configs/baseline/beauty.yaml --seeds 42 7 18

uv run python sailrec/scripts/train_p1.py \
    --config sailrec/configs/baseline/yelp.yaml --seeds 42 7 18
```

### 3. Cross-surface ablation on Video Games (Table 3)

```bash
# Surface A: encoder input fusion
uv run python sailrec/scripts/train_a2_input_fusion.py \
    --config sailrec/configs/cross_surface/surface_a_input_fusion.yaml --seeds 42 7 18

# Surface B: per-position distillation
uv run python sailrec/scripts/train_a3_hidden_distill.py \
    --config sailrec/configs/cross_surface/surface_b_per_position_distill.yaml --seeds 42 7 18

# Surface C: sequence-level InfoNCE alignment
uv run python sailrec/scripts/train_a4_seq_infonce.py \
    --config sailrec/configs/cross_surface/surface_c_seq_alignment.yaml --seeds 42 7 18

# Surface D: item-table integration (our method)
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/cross_surface/surface_d_item_table.yaml \
    --seeds 42 7 18 --llm-init --freq-weight --weight-fn binary
```

### 4. Cross-dataset generalization of item-table integration (Table 4)

```bash
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/cross_dataset/sports.yaml \
    --seeds 42 7 18 --llm-init --freq-weight --weight-fn binary

uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/cross_dataset/beauty.yaml \
    --seeds 42 7 18 --llm-init --freq-weight --weight-fn binary

uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/cross_dataset/yelp.yaml \
    --seeds 42 7 18 --llm-init --freq-weight --weight-fn binary
```

### 5. Item-table decomposition (Table 5, Figure 3)

```bash
# LLM-PCA initialization only
uv run python sailrec/scripts/train_a1_llm_init.py \
    --config sailrec/configs/decomposition/llm_pca_init_only.yaml --seeds 42 7 18

# Uniform InfoNCE alignment (no frequency weighting)
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/decomposition/uniform_alignment.yaml --seeds 42 7 18

# Frequency-weighted InfoNCE alignment (full method)
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/decomposition/freq_weighted_alignment.yaml \
    --seeds 42 7 18 --llm-init --freq-weight --weight-fn binary
```

### 6. Appendix sweeps

```bash
# Appendix C — weight-function sweep
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/appendix/weight_fn_sweep.yaml --seeds 42 7 18

# Appendix D — λ sweep
uv run python sailrec/scripts/train_sailrec.py \
    --config sailrec/configs/appendix/lambda_sweep.yaml \
    --seeds 42 7 18 --llm-init --freq-weight \
    --lambdas 0.01 0.05 0.1 0.5 1.0
```

### 7. Evaluation

```bash
uv run python sailrec/scripts/eval_fullrank.py \
    --config sailrec/configs/cross_dataset/beauty.yaml

uv run python sailrec/scripts/eval_stratified_sampled.py \
    --config sailrec/configs/cross_dataset/beauty.yaml
```

---

## Configs

Each config folder maps to a paper section:

| Folder | Paper section | Contents |
|--------|--------------|----------|
| `baseline/` | Comparison baseline | SASRec (no LLM) on Video Games, Sports, Beauty, Yelp |
| `cross_surface/` | Table 3 + Appendix B | All four injection surfaces on Video Games |
| `cross_dataset/` | Table 4 | Item-table integration on held-out Sports, Beauty, Yelp |
| `decomposition/` | Table 5 + Figure 3 | Item-table ablation: init-only, uniform align, freq-weighted align |
| `appendix/` | Appendix C + D | Weight-function sweep, λ sweep |

Key fields shared across configs:

```yaml
model:
  embed_dim: 50        # fixed across all paper experiments
  num_blocks: 2
  num_heads: 1
  max_seq_len: 200
  dropout: 0.2

training:
  batch_size: 512
  lr: 0.001
  max_epochs: 200
  early_stopping_patience: 10
  early_stopping_metric: "ndcg@10"

evaluation:
  num_neg_samples: 100   # sampled@100 protocol
  ks: [5, 10, 20]

align:                   # present only in item-table configs
  lambda_align: 0.5
  temperature: 0.1
  llm_dim: 1024
```

---

## Reproducibility Notes

- Results reported as mean ± std across seeds {7, 18, 42}.
- Sampled@100: 100 randomly sampled negatives per test positive. Full-rank: ranked against all items.
- Popularity stratification: 20% head / 60% torso / 20% tail by training-set interaction count.
- SASRec backbone fixed at d=50, 2 blocks, 1 head, L=200, leave-one-out split across all surfaces.
- Experiments run on Apple M2 Max (MPS). Rankings between methods should be preserved on other hardware.
