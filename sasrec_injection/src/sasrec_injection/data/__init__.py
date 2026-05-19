"""Data loading, preprocessing, splitting, and PyTorch Datasets.

Module map
----------

* :mod:`sasrec_injection.data.movielens`     — MovieLens (ml-1m, ml-100k) loader.
                                      Also home of the *canonical*
                                      :func:`preprocess` and
                                      :func:`build_user_sequences` helpers
                                      that work on any dataset that emits
                                      the four-column DataFrame.
* :mod:`sasrec_injection.data.amazon`        — Amazon Reviews 2023 loader (HF).
* :mod:`sasrec_injection.data.loaders`       — Dispatcher: ``ml-*`` → movielens,
                                      ``amazon-*`` → amazon.
* :mod:`sasrec_injection.data.splitting`     — Leave-one-out split + negative
                                      sampling.
* :mod:`sasrec_injection.data.dataset`       — PyTorch ``Dataset`` classes for
                                      training (BCE), sampled eval, and
                                      full-rank eval.
* :mod:`sasrec_injection.data.item_metadata` — Title + side-info loader for LLM
                                      prompt construction.

Most callers only need :mod:`sasrec_injection.data.loaders` (for the dispatcher
and the re-exported helpers).
"""

__all__: list[str] = []
