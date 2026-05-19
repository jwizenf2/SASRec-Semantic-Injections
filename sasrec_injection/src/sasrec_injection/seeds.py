"""Reproducibility utilities — seed every RNG that affects training.

Why this is its own module
--------------------------

Every training script needs to call ``set_seed`` once before constructing
data loaders, the model, or the optimiser. Centralising it means we
never have a "I forgot to seed numpy" bug across scripts.

The function seeds **every** RNG that any other code in this package
uses, so calling it once is sufficient. If a downstream caller needs
reproducible behaviour from a *new* RNG (e.g. an inference-time random
sampler), seed that RNG explicitly — don't rely on this function having
already done it.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed. Conventional values across this codebase:
            42 for the headline run, 123 / 456 for additional seeds.

    Notes:
        * ``torch.cuda.manual_seed_all`` is called only when CUDA is
          available — it raises on machines without an NVIDIA GPU.
        * Apple Silicon (MPS) does not have a separate seed API;
          ``torch.manual_seed`` covers it.
        * This function does **not** set ``torch.backends.cudnn.deterministic``
          because doing so meaningfully slows training and the
          remaining non-determinism (asynchronous CUDA kernels) is small
          enough to be in the noise band of seed-to-seed variance we're
          measuring.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
