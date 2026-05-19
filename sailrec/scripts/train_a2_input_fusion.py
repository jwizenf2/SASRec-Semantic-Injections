"""A2 — Input Fusion ablation.

Location ablation: LLM signal injected at the encoder INPUT every
forward pass. No auxiliary loss. Tests whether the encoder can
usefully consume LLM features as part of its input (expected: no).

Uses SASRec with fusion_mode='add' and the standard P1 training loop.
"""
from __future__ import annotations
import argparse, json, sys, yaml
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sailrec.config import P1Config
from sailrec.data.dataset import SASRecEvalDataset, SASRecTrainDataset, create_eval_loader, create_train_loader
from sailrec.evaluation.metrics import sampled_evaluate
from sailrec.models.sasrec import SASRec
from sailrec.seeds import set_seed
from sailrec.training.trainer import SASRecTrainer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_p1 import log, prepare_data


def train_single_seed(cfg, llm_item_emb, seed, split, neg_samples, resume=False):
    set_seed(seed)
    seed_dir = Path(cfg.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    train_ds = SASRecTrainDataset(split.train_seqs, split.num_items, max_seq_len=cfg.model.max_seq_len)
    val_ds = SASRecEvalDataset(split.train_seqs, split.val_targets, neg_samples, max_seq_len=cfg.model.max_seq_len)
    train_loader = create_train_loader(train_ds, batch_size=cfg.training.batch_size)
    val_loader = create_eval_loader(val_ds)

    # A2: SASRec with fusion_mode='add' — LLM signal at encoder input, no aux loss.
    model = SASRec(
        num_items=split.num_items,
        embed_dim=cfg.model.embed_dim,
        num_blocks=cfg.model.num_blocks,
        num_heads=cfg.model.num_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        llm_item_emb=llm_item_emb,
        llm_dim=llm_item_emb.shape[1],
        fusion_mode="add",
    )

    trainer = SASRecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        output_dir=str(seed_dir),
    )
    trainer.train(eval_ks=cfg.evaluation.ks, resume=resume)

    test_seqs = {uid: split.train_seqs[uid]+[split.val_targets[uid]] for uid in split.test_targets}
    test_ds = SASRecEvalDataset(test_seqs, split.test_targets, neg_samples, max_seq_len=cfg.model.max_seq_len)
    test_metrics = sampled_evaluate(model, create_eval_loader(test_ds), device, ks=cfg.evaluation.ks)

    log(f"\n[Seed {seed}] A2 sampled test metrics:")
    for k, v in sorted(test_metrics.items()):
        log(f"  {k}: {v:.4f}")
    results = {"seed": seed, "ablation": "A2_input_fusion", "test_metrics": test_metrics}
    with open(seed_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sailrec/configs/ablations/A2_input_fusion.yaml")
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(open(args.config))
    llm_path = raw.get("llm_embeddings_path", "sailrec/outputs/embeddings/video_games.pt")

    cfg = P1Config.from_yaml(args.config, args.base_config)
    if args.seeds:
        cfg.seeds = args.seeds

    log(f"A2 Input Fusion | seeds={cfg.seeds}")
    split, neg_samples, _ = prepare_data(cfg)

    log(f"Loading LLM embeddings from {llm_path}...")
    llm_item_emb = torch.load(llm_path, map_location="cpu", weights_only=True)
    if llm_item_emb.shape[0] != split.num_items + 1:
        raise ValueError(f"LLM rows {llm_item_emb.shape[0]} != num_items+1 ({split.num_items+1})")

    all_metrics = []
    for seed in cfg.seeds:
        log(f"\n{'='*60}\nA2 — seed {seed}\n{'='*60}")
        all_metrics.append(train_single_seed(cfg, llm_item_emb, seed, split, neg_samples, resume=args.resume))

    log(f"\n{'='*60}\nAGGREGATE\n{'='*60}")
    summary = {}
    for key in sorted(all_metrics[0]):
        values = [m[key] for m in all_metrics]
        mean = sum(values) / len(values)
        std = (sum((v-mean)**2 for v in values)/len(values))**0.5
        summary[key] = {"mean": mean, "std": std, "values": values}
        log(f"{key}: {mean:.4f} ± {std:.4f}")
    with open(Path(cfg.output_dir) / "aggregate_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    main()
