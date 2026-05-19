"""A4 — Seq-level InfoNCE (location ablation)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sailrec.config import SAILRecConfig
from sailrec.data.dataset import SASRecEvalDataset, SASRecTrainDataset, create_eval_loader, create_train_loader
from sailrec.evaluation.metrics import sampled_evaluate
from sailrec.models.sasrec import SASRec
from sailrec.seeds import set_seed
from sailrec.training.seq_infonce_trainer import SeqInfoNCETrainer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sailrec import log, prepare_data, load_llm_embeddings
from sailrec.data.splitting import load_negative_samples


def train_seed(cfg, seed, split, neg_samples, llm_item_emb, resume=False):
    set_seed(seed)
    seed_dir = Path(cfg.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    neg_path = Path(cfg.output_dir) / "neg_samples.npz"
    ns = load_negative_samples(str(neg_path)) if neg_path.exists() else neg_samples

    train_ds = SASRecTrainDataset(split.train_seqs, split.num_items, max_seq_len=cfg.model.max_seq_len)
    val_ds = SASRecEvalDataset(split.train_seqs, split.val_targets, ns, max_seq_len=cfg.model.max_seq_len)
    train_loader = create_train_loader(train_ds, batch_size=cfg.training.batch_size)
    val_loader = create_eval_loader(val_ds)

    model = SASRec(num_items=split.num_items, embed_dim=cfg.model.embed_dim,
                   num_blocks=cfg.model.num_blocks, num_heads=cfg.model.num_heads,
                   max_seq_len=cfg.model.max_seq_len, dropout=cfg.model.dropout).to(device)

    trainer = SeqInfoNCETrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        llm_item_emb=llm_item_emb, device=device, lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs, lambda_align=cfg.align.lambda_align,
        temperature=cfg.align.temperature, max_align_ids=cfg.align.max_align_ids,
        projector_hidden_dim=cfg.align.projector_hidden_dim,
        projector_dropout=cfg.align.projector_dropout,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_metric=cfg.training.early_stopping_metric,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        output_dir=str(seed_dir))
    trainer.train(eval_ks=cfg.evaluation.ks, resume=resume)

    test_seqs = {uid: split.train_seqs[uid]+[split.val_targets[uid]] for uid in split.test_targets}
    test_ds = SASRecEvalDataset(test_seqs, split.test_targets, ns, max_seq_len=cfg.model.max_seq_len)
    metrics = sampled_evaluate(model, create_eval_loader(test_ds), device, ks=cfg.evaluation.ks)
    log(f"\n[Seed {seed}] A4 test:"); [log(f"  {k}: {v:.4f}") for k, v in sorted(metrics.items())]
    with open(seed_dir / "results.json", "w") as f:
        json.dump({"seed": seed, "ablation": "A4_seq_infonce", "test_metrics": metrics}, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sailrec/configs/ablations/A4_seq_infonce.yaml")
    parser.add_argument("--base-config", default="sailrec/configs/base.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help="Lambda sweep; each gets its own lambda_<v>/ subdir.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = SAILRecConfig.from_yaml(args.config, args.base_config)
    if args.seeds: cfg.seeds = args.seeds
    lambdas = args.lambdas or [cfg.align.lambda_align]
    split, neg_samples, _ = prepare_data(cfg)
    llm = load_llm_embeddings(cfg.align.embeddings_path, split.num_items, cfg.align.llm_dim)
    base_output_dir = Path(cfg.output_dir)
    for lam in lambdas:
        cfg.align.lambda_align = lam
        lam_dir = base_output_dir / f"lambda_{lam}" if len(lambdas) > 1 else base_output_dir
        lam_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(lam_dir)
        log(f"A4 | seeds={cfg.seeds} | λ={lam}")
        all_m = [train_seed(cfg, s, split, neg_samples, llm, resume=args.resume) for s in cfg.seeds]
        summary = {}
        for k in sorted(all_m[0]):
            vals = [m[k] for m in all_m]; mean = sum(vals)/len(vals)
            std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
            summary[k] = {"mean": mean, "std": std, "values": vals}
            log(f"  {k}: {mean:.4f} ± {std:.4f}")
        with open(lam_dir / "aggregate_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    main()
