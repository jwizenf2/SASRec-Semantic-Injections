"""Model architectures for sequential recommendation.

* :class:`~sasrec_injection.models.sasrec.SASRec`   — transformer-based (headline model).
* :class:`~sasrec_injection.models.gru4rec.GRU4Rec` — GRU-based backbone for
  generalisation ablations (A1/A8 with an alternative encoder).

Both models expose the same interface (``forward``, ``predict``,
``score_all_items``, ``item_emb``) so they drop into every trainer and
eval script unchanged.
"""

__all__: list[str] = []
