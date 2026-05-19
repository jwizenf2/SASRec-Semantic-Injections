"""Training primitives shared across P1 and ItemTable.

Three classes here:

* :class:`EarlyStopping`   — monitors a metric across epochs and signals
                              when training should stop.
* :class:`TrainingLogger`  — append-only JSONL logger for per-epoch records.
* :class:`BaseTrainer`     — concrete epoch-loop, checkpointing, summary
                              writing, and early stopping. Subclasses
                              override :meth:`BaseTrainer.train_epoch`
                              with the loss they care about.
* :class:`SASRecTrainer`   — P1 baseline (BCE next-item only).

The corresponding ItemTable trainer
(:class:`sasrec_injection.training.item_table_trainer.ItemTableTrainer`) is in a
sibling module to keep the alignment-specific imports out of this file.

Why a base class
----------------

Every trainer in this codebase shares the same skeleton:

::

    for each epoch:
        train_epoch()       <- subclass-specific loss
        validate()          <- sampled HR/NDCG
        early-stopping check
        checkpoint if best

Putting that skeleton in :class:`BaseTrainer` and only overriding
:meth:`train_epoch` keeps subclasses small and ensures every variant
has identical checkpointing and JSONL logging semantics.
"""

from __future__ import annotations

import json
import random
import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sasrec_injection.evaluation.metrics import sampled_evaluate
from sasrec_injection.training.losses import bce_loss

if TYPE_CHECKING:
    # Narrow ``self.model`` for IDE / type-checker purposes only —
    # every trainer in this package wraps a SASRec, but we don't want
    # to hard-import it at runtime.
    from sasrec_injection.models.sasrec import SASRec


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitor a validation metric across epochs and signal when to stop.

    Standard "patience" early stopping: if the monitored metric hasn't
    improved by at least ``min_delta`` for ``patience`` consecutive
    epochs, return True.

    Attributes:
        patience: Epochs of no improvement before stopping.
        metric: Key into the per-epoch metrics dict (e.g. ``"ndcg@10"``).
        min_delta: Minimum improvement to reset the patience counter.
            Useful for ignoring numerical noise in the val metric.
        best_value: Best value seen so far.
        counter: Epochs since the best value was last set.
        best_epoch: Epoch number at which best_value was achieved.
    """

    def __init__(
        self,
        patience: int,
        metric: str = "ndcg@10",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.best_value = -float("inf")
        self.counter = 0
        self.best_epoch = -1

    def step(self, metrics: dict[str, float], epoch: int) -> bool:
        """Update bookkeeping for one epoch's metrics.

        Returns:
            True if training should stop (patience exhausted).
        """
        value = metrics[self.metric]
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# JSONL training logger
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Per-epoch records as JSON Lines.

    One file per training run, written next to the checkpoint. Useful
    for post-hoc analysis (e.g. "what was the rec/align loss curve at
    λ=0.1?") and for the latency benchmark in the verification plan
    (epoch wall-clock from existing logs).
    """

    def __init__(self, log_path: Path, append: bool = False):
        """
        Args:
            log_path: Where to write JSONL records.
            append: If True, open in append mode (used when resuming
                training so the prior epochs' records aren't truncated).
        """
        self.log_path = log_path
        # Open immediately so a crashing run still leaves a partial
        # log on disk. ``flush()`` after every write keeps the file
        # current under ``tail -f``.
        self._file = open(log_path, "a" if append else "w")

    def log(self, record: dict) -> None:
        """Write one record as a JSON line and flush."""
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class BaseTrainer:
    """Concrete training loop shared by every ItemTable / P1 trainer.

    Subclasses must override :meth:`train_epoch`, which returns a
    ``dict[str, float]`` of per-epoch average losses. They may
    override:

    * :meth:`_summary_config`     — fields written to ``train_summary.json``.
    * :meth:`_format_loss_log_line` — pretty-print loss components.
    * :meth:`save_checkpoint` /
      :meth:`load_checkpoint`     — when extra trainer state needs to
                                    be persisted (the ItemTable trainer
                                    keeps its projector live but does
                                    NOT save it, since inference is
                                    plain SASRec).

    The training loop:

    1. ``train_epoch()`` — subclass-specific loss(es).
    2. ``validate()``    — sampled HR/NDCG on the validation set.
    3. Update early stopping; record a JSONL line.
    4. If this epoch is a new best, ``save_checkpoint()``.
    5. If patience is exhausted, break.

    On exit, ``train()`` returns the best validation metrics and the
    ``train_summary.json`` file contains the full run summary.
    """

    def __init__(
        self,
        model: "SASRec",
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.001,
        max_epochs: int = 200,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "ndcg@10",
        early_stopping_min_delta: float = 1e-4,
        output_dir: str | Path = "outputs",
        wandb_run: Any = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

        # Adam over only the model parameters by default. Subclasses
        # that own extra trainable modules (e.g. ItemTable's
        # AlignProjector) should overwrite ``self.optimizer`` after
        # calling ``super().__init__``.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.early_stopping = EarlyStopping(
            early_stopping_patience,
            early_stopping_metric,
            min_delta=early_stopping_min_delta,
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Per-epoch average losses (e.g. ``{"loss_rec": 0.31}`` or
            ``{"loss_rec": 0.31, "loss_align": 1.4, "loss_combined": 0.45}``).
        """
        raise NotImplementedError

    def _summary_config(self) -> dict[str, Any]:
        """Fields written to ``train_summary.json`` under ``"config"``.

        Base class only captures generic training hyperparameters.
        Subclasses extend this to record method-specific knobs (e.g.
        ``lambda_align`` and ``temperature`` for ItemTable).
        """
        return {
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping.patience,
            "early_stopping_metric": self.early_stopping.metric,
            "device": str(self.device),
        }

    def _format_loss_log_line(self, losses: dict[str, float]) -> str:
        """Format the loss portion of the per-epoch console line."""
        return " ".join(f"{k}={v:.4f}" for k, v in losses.items())

    # ------------------------------------------------------------------
    # Validation + checkpoint helpers
    # ------------------------------------------------------------------

    def validate(self, ks: list[int] | None = None) -> dict[str, float]:
        """Sampled evaluation on ``self.val_loader``."""
        if ks is None:
            ks = [5, 10, 20]
        return sampled_evaluate(self.model, self.val_loader, self.device, ks=ks)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the model state-dict in the wrapped ``{"model": ...}`` format."""
        torch.save({"model": self.model.state_dict()}, path)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    #
    # ``best_model.pt`` only carries the model weights at the best
    # validation epoch — that's all you need for inference / eval. To
    # resume *training* we need much more: optimizer state, the
    # AlignProjector (in ItemTable), early-stopping bookkeeping, every
    # RNG, and the epoch counter. We persist all of that to
    # ``last.pt`` at the end of every epoch and atomic-rename it into
    # place so a kill mid-write never leaves a corrupt file.

    def _extra_resume_state(self) -> dict[str, Any]:
        """Subclass hook: extra trainer state to persist in ``last.pt``.

        ItemTableTrainer overrides this to save its AlignProjector. The
        base trainer has nothing extra to save.
        """
        return {}

    def _load_extra_resume_state(self, state: dict[str, Any]) -> None:
        """Subclass hook: restore extra state loaded from ``last.pt``."""
        return None

    def save_resume_state(
        self,
        path: str | Path,
        epoch: int,
        best_metrics: dict[str, float],
    ) -> None:
        """Atomically write a complete snapshot of trainer state.

        Args:
            path: Destination file (typically ``<output_dir>/last.pt``).
            epoch: 1-indexed epoch number that just completed.
            best_metrics: Best val metrics seen so far (for restoring
                the early-stopping baseline at resume time).

        Notes:
            We capture every RNG (``torch``, NumPy, Python ``random``)
            so resuming reproduces the same negative-sampling and
            dropout sequence the killed run would have produced.
            CUDA RNG is captured only when CUDA is the active backend;
            MPS has no separate RNG. Atomic rename avoids leaving a
            half-written ``last.pt`` if the process is killed during
            ``torch.save``.
        """
        snapshot: dict[str, Any] = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "early_stopping": {
                "best_value": self.early_stopping.best_value,
                "counter": self.early_stopping.counter,
                "best_epoch": self.early_stopping.best_epoch,
            },
            "best_metrics": best_metrics,
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": self._extra_resume_state(),
        }
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(snapshot, tmp_path)
        tmp_path.replace(path)

    def load_resume_state(
        self, path: str | Path
    ) -> tuple[int, dict[str, float]]:
        """Restore trainer state from a snapshot written by ``save_resume_state``.

        Args:
            path: Snapshot file written previously.

        Returns:
            ``(last_completed_epoch, best_metrics)``. Resume training
            from ``last_completed_epoch + 1``.

        Notes:
            We use ``weights_only=False`` because the snapshot
            contains non-tensor state (RNG tuples, optimizer state).
            This is safe here because we wrote the file ourselves;
            never point this at an untrusted file.
        """
        # weights_only=False is required to deserialise the optimizer
        # state and Python-tuple RNG state. Trusted file (we wrote it).
        snapshot = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])

        es = snapshot["early_stopping"]
        self.early_stopping.best_value = es["best_value"]
        self.early_stopping.counter = es["counter"]
        self.early_stopping.best_epoch = es["best_epoch"]

        rng = snapshot["rng"]
        # ``torch.load(map_location=device)`` above moves every tensor
        # in the snapshot to the training device — including the RNG
        # ByteTensor — and then ``torch.set_rng_state`` rejects it
        # because the CPU generator state must live on CPU. Copy back
        # to CPU and re-cast to uint8 ByteTensor before restoring.
        torch_rng = rng["torch"]
        if hasattr(torch_rng, "cpu"):
            torch_rng = torch_rng.cpu()
        if hasattr(torch_rng, "dtype") and torch_rng.dtype != torch.uint8:
            torch_rng = torch_rng.byte()
        torch.set_rng_state(torch_rng)
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

        self._load_extra_resume_state(snapshot.get("extra", {}))
        return int(snapshot["epoch"]), dict(snapshot.get("best_metrics", {}))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint. Accepts both wrapped and bare formats.

        The base class saves wrapped (``{"model": sd}``); the P1
        trainer subclass saves bare (``sd``). Accepting both keeps
        external evaluation scripts from caring which trainer wrote
        the checkpoint.
        """
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

    # ------------------------------------------------------------------
    # The actual training loop
    # ------------------------------------------------------------------

    def train(
        self,
        eval_ks: list[int] | None = None,
        resume: bool = False,
    ) -> dict[str, float]:
        """Run full training with early stopping.

        Side effects (under ``self.output_dir``):

        * ``best_model.pt``       — checkpoint at the best validation epoch.
        * ``last.pt``             — full resume snapshot (every epoch).
        * ``train_log.jsonl``     — per-epoch records.
        * ``train_summary.json``  — final summary written on exit.

        Args:
            eval_ks: Cutoffs for sampled validation HR/NDCG.
            resume: If True and ``<output_dir>/last.pt`` exists, restore
                from that snapshot and continue. If False (or no
                snapshot exists), start training from epoch 1 and
                truncate any prior log.

        Returns:
            ``best_val_metrics`` — the validation metrics at the
            best epoch.
        """
        if eval_ks is None:
            eval_ks = [5, 10, 20]
        best_metrics: dict[str, float] = {}
        best_model_path = self.output_dir / "best_model.pt"
        last_path = self.output_dir / "last.pt"

        # ----------------------------------------------------------------
        # Resume bookkeeping. ``start_epoch`` is the first epoch number
        # to actually run (so ``last_completed + 1``). When resuming we
        # open the JSONL log in append mode to preserve the prior
        # records; on a fresh start we truncate.
        # ----------------------------------------------------------------
        start_epoch = 1
        log_append = False
        if resume and last_path.exists():
            last_completed, best_metrics = self.load_resume_state(last_path)
            start_epoch = last_completed + 1
            log_append = True
            print(
                f"Resuming from epoch {start_epoch} "
                f"(best so far: epoch {self.early_stopping.best_epoch}, "
                f"{self.early_stopping.metric}={self.early_stopping.best_value:.4f})",
                flush=True,
            )
        elif resume:
            print(
                f"--resume passed but {last_path} not found; starting fresh.",
                flush=True,
            )

        logger = TrainingLogger(
            self.output_dir / "train_log.jsonl", append=log_append
        )
        t_start = time.time()
        epoch = start_epoch - 1  # so the post-loop summary is correct if start_epoch > max_epochs
        should_stop = False

        for epoch in range(start_epoch, self.max_epochs + 1):
            t0 = time.time()
            losses = self.train_epoch()
            val_metrics = self.validate(ks=eval_ks)
            elapsed = time.time() - t0

            # JSONL record: every key the user might want post-hoc.
            record = {
                "epoch": epoch,
                **losses,
                "elapsed_s": round(elapsed, 2),
                "wall_clock_s": round(time.time() - t_start, 2),
                "es_counter": self.early_stopping.counter,
                "is_best": False,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            # Pretty-print for console / log files. Keep this readable
            # under ``tail -f`` — that's how remote experiments are
            # monitored.
            log_str = (
                f"Epoch {epoch:3d} | "
                + self._format_loss_log_line(losses)
                + " | "
                + " | ".join(
                    f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())
                )
                + f" | {elapsed:.1f}s"
            )
            print(log_str, flush=True)

            if self.wandb_run is not None:
                self.wandb_run.log({"epoch": epoch, **losses, **val_metrics})

            should_stop = self.early_stopping.step(val_metrics, epoch)

            # If this is a new best, save and update best_metrics.
            # ``early_stopping.counter == 0`` after ``step`` exactly
            # when the metric improved.
            if self.early_stopping.counter == 0:
                self.save_checkpoint(best_model_path)
                best_metrics = val_metrics.copy()
                record["is_best"] = True

            logger.log(record)

            # Persist a complete resume snapshot at the end of every
            # epoch so a kill at any point loses at most one epoch of
            # progress. Atomic rename inside ``save_resume_state``
            # protects against partial writes.
            self.save_resume_state(last_path, epoch, best_metrics)

            if should_stop:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.early_stopping.best_epoch}",
                    flush=True,
                )
                break

        total_time = time.time() - t_start
        logger.close()

        # Final summary, written next to the JSONL log.
        summary = {
            "best_epoch": self.early_stopping.best_epoch,
            "total_epochs": epoch,
            "total_time_s": round(total_time, 2),
            "early_stopped": should_stop,
            "best_val_metrics": best_metrics,
            "config": self._summary_config(),
        }
        with open(self.output_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Restore best model on the way out so the caller can run test
        # evaluation against the best checkpoint without an extra load.
        self.load_checkpoint(best_model_path)
        return best_metrics


# ---------------------------------------------------------------------------
# P1 baseline trainer (BCE next-item only)
# ---------------------------------------------------------------------------


class SASRecTrainer(BaseTrainer):
    """P1 baseline trainer — plain BCE next-item training.

    No alignment, no auxiliary loss, no projector. The loop is
    intentionally minimal so it serves as a clean reference point for
    the ItemTable trainer in the sibling module.
    """

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for seq, pos, neg in self.train_loader:
            seq = seq.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            # Standard SASRec forward: per-position rep at every time
            # step. We score positive and negative against every
            # position; padding gets masked out by the loss.
            seq_repr = self.model(seq)                              # (B, L, D)
            pos_emb = self.model.item_emb(pos)
            neg_emb = self.model.item_emb(neg)
            pos_logits = (seq_repr * pos_emb).sum(dim=-1)           # (B, L)
            neg_logits = (seq_repr * neg_emb).sum(dim=-1)           # (B, L)

            # Mask padding positions: only valid positions (where
            # ``pos > 0``) contribute to the loss.
            mask = (pos > 0).float()
            loss = bce_loss(pos_logits, neg_logits, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"train_loss": total_loss / num_batches}

    def _format_loss_log_line(self, losses: dict[str, float]) -> str:
        return f"loss={losses['train_loss']:.4f}"

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the bare state-dict for compatibility with prior P1 runs.

        The wider TSDRec ecosystem treats a bare state-dict as the P1
        on-disk format; the wrapped ``{"model": …}`` format is the
        modern default used by ItemTable. ``BaseTrainer.load_checkpoint``
        accepts both.
        """
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        # Accept both wrapped and bare formats so this trainer can
        # also load checkpoints written by other trainers.
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
