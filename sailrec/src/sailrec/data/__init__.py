"""Data loading, preprocessing, splitting, and PyTorch Datasets.

Module map
----------

* :mod:`sailrec.data.movielens`     — MovieLens (ml-1m, ml-100k) loader.
                                      Also home of the *canonical*
                                      :func:`preprocess` and
                                      :func:`build_user_sequences` helpers
                                      that work on any dataset that emits
                                      the four-column DataFrame.
* :mod:`sailrec.data.amazon`        — Amazon Reviews 2023 loader (HF).
* :mod:`sailrec.data.loaders`       — Dispatcher: ``ml-*`` → movielens,
                                      ``amazon-*`` → amazon.
* :mod:`sailrec.data.splitting`     — Leave-one-out split + negative
                                      sampling.
* :mod:`sailrec.data.dataset`       — PyTorch ``Dataset`` classes for
                                      training (BCE), sampled eval, and
                                      full-rank eval.
* :mod:`sailrec.data.item_metadata` — Title + side-info loader for LLM
                                      prompt construction.

Most callers only need :mod:`sailrec.data.loaders` (for the dispatcher
and the re-exported helpers).
"""

__all__: list[str] = []
