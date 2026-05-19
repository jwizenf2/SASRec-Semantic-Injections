"""Item metadata (titles, descriptions, genres) for LLM prompt construction.

Used by ``scripts/extract_llm_embeddings.py`` to feed the right text
through Qwen3 when extracting item embeddings. The class itself is
dataset-aware:

* **MovieLens** — items are movies; side info is a pipe-separated
  genre string. ``format_item`` returns ``"Title (Genres)"`` matching
  the prompts in the LLM4Rec literature.
* **Amazon Reviews 2023** — items are products; side info is a
  free-text description. ``format_item`` returns
  ``"Title — Description"``.
* **Yelp** — items are businesses. The full LLM-ESR Yelp template is
  precomputed at load time and stored in ``self.titles``;
  ``self.side_info`` is empty. ``format_item`` returns the precomputed
  template verbatim. The template is::

      The point of interest has following attributes:
       name is <NAME>; category is <CATEGORY>; type is <TYPE>; open
       status is <OPEN>; review count is <COUNT>; city is <CITY>;
       average score is <STARS>.

  Missing fields render as the literal string ``"unknown"`` so every
  item produces a 7-field record (positional consistency matters for
  the embedding model — a transformer encodes position, so dropping
  fields would inject noise unrelated to item semantics).

The class stores everything in ``self.titles`` and ``self.side_info``
keyed by the *remapped* item_id (1..num_items). Callers must apply
the same ``item_map`` produced by
:func:`sailrec.data.movielens.preprocess` so titles and behavioural
data align.
"""

from pathlib import Path
from typing import Literal

from sailrec.data.movielens import load_movies


class ItemMetadata:
    """Per-item title + side info for LLM prompt construction.

    Attributes:
        titles: ``mapped_item_id`` → display title string.
        side_info: ``mapped_item_id`` → genre string (MovieLens),
            description (Amazon), or "Categories: ...; Located in
            ..." (Yelp).
        dataset_family: ``"movielens"``, ``"amazon"``, or ``"yelp"`` —
            controls how :meth:`format_item` joins title and side info.
    """

    def __init__(
        self,
        titles: dict[int, str],
        side_info: dict[int, str],
        dataset_family: Literal["movielens", "amazon", "yelp"] = "movielens",
    ):
        self.titles = titles
        self.side_info = side_info
        self.dataset_family = dataset_family

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_movielens(
        cls,
        data_dir: str | Path,
        item_map: dict[int, int],
        dataset: str = "ml-1m",
    ) -> "ItemMetadata":
        """Load metadata from ml-1m / ml-100k and remap IDs.

        Args:
            data_dir: Project ``data/`` directory.
            item_map: Original int item id → remapped int item id.
                Output of
                :func:`sailrec.data.movielens.preprocess`.
            dataset: ``"ml-1m"`` or ``"ml-100k"``.

        Returns:
            An ``ItemMetadata`` keyed by remapped int item ids.
        """
        movies_df = load_movies(data_dir, dataset=dataset)

        titles: dict[int, str] = {}
        side: dict[int, str] = {}
        for _, row in movies_df.iterrows():
            orig_id = row["item_id"]
            if orig_id in item_map:
                mapped_id = item_map[orig_id]
                titles[mapped_id] = row["title"]
                side[mapped_id] = row["genres"]

        return cls(titles=titles, side_info=side, dataset_family="movielens")

    @classmethod
    def from_amazon(
        cls,
        data_dir: str | Path,
        item_map: dict[str, int],
        dataset: str = "Video_Games",
        max_description_chars: int = 600,
    ) -> "ItemMetadata":
        """Load metadata from Amazon Reviews 2023 and remap IDs.

        Args:
            data_dir: Cache directory passed to the Amazon loader.
            item_map: Maps raw ``parent_asin`` (str) → remapped int
                item id. Note the keys are *strings* here — Amazon
                item IDs are SKUs, not numbers.
            dataset: Amazon category name (e.g. ``"Video_Games"``).
            max_description_chars: Truncate descriptions longer than
                this so a single sequence prompt of 10 items doesn't
                blow past Qwen3's 8K-token context. 600 chars ≈ 150
                tokens; a 10-item sequence stays comfortably under
                2K tokens.

        Returns:
            An ``ItemMetadata`` keyed by remapped int item ids.
        """
        # Local import to avoid a circular module import at top of
        # file. The Amazon loader is heavyweight (huggingface_hub
        # network calls) and we don't want to import it just for
        # MovieLens runs.
        from sailrec.data.amazon import load_metadata as _amazon_load_metadata

        meta_df = _amazon_load_metadata(data_dir, dataset=dataset)

        titles: dict[int, str] = {}
        side: dict[int, str] = {}
        for _, row in meta_df.iterrows():
            orig_id = row["item_id"]
            if orig_id not in item_map:
                continue
            mapped_id = item_map[orig_id]

            # Some Amazon items have empty titles; fall back to
            # "Untitled" so the LLM has something to encode.
            titles[mapped_id] = (row.get("title") or "").strip() or "Untitled"

            desc = (row.get("description") or "").strip()
            # Truncate at the last whitespace before the cut so we
            # don't slice mid-word.
            if max_description_chars and len(desc) > max_description_chars:
                desc = desc[: max_description_chars].rsplit(" ", 1)[0] + "…"
            side[mapped_id] = desc

        return cls(titles=titles, side_info=side, dataset_family="amazon")

    @classmethod
    def from_yelp(
        cls,
        data_dir: str | Path,
        item_map: dict[str, int],
    ) -> "ItemMetadata":
        """Load metadata from the Yelp Open Dataset and remap IDs.

        Builds the full LLM-ESR Yelp prompt for every item and stores
        it as the item's "title" (with empty ``side_info``); the
        template is then returned verbatim by :meth:`format_item`. See
        the module docstring for the template shape.

        Args:
            data_dir: Project ``data/`` directory. Yelp JSONL files
                expected at ``<data_dir>/yelp/``.
            item_map: Maps raw 22-char Yelp ``business_id`` (str) →
                remapped int item id.

        Returns:
            An ``ItemMetadata`` keyed by remapped int item ids.
        """
        # Local import to avoid pulling in the Yelp data dependency
        # for unrelated runs.
        from sailrec.data.yelp import load_metadata as _yelp_load_metadata

        meta_df = _yelp_load_metadata(data_dir)

        titles: dict[int, str] = {}
        side: dict[int, str] = {}
        for _, row in meta_df.iterrows():
            orig_id = row["item_id"]
            if orig_id not in item_map:
                continue
            mapped_id = item_map[orig_id]

            # The leading newline + space before "name is …" matches
            # the LLM-ESR formatting verbatim.
            template = (
                "The point of interest has following attributes: \n"
                f" name is {row['name']};"
                f" category is {row['category']};"
                f" type is {row['type']};"
                f" open status is {row['is_open']};"
                f" review count is {row['review_count']};"
                f" city is {row['city']};"
                f" average score is {row['stars']}."
            )
            titles[mapped_id] = template
            side[mapped_id] = ""

        return cls(titles=titles, side_info=side, dataset_family="yelp")

    @classmethod
    def from_dataset(
        cls,
        data_dir: str | Path,
        item_map: dict,
        dataset: str,
    ) -> "ItemMetadata":
        """Dispatch to the right loader based on the dataset prefix.

        Convention matches :mod:`sailrec.data.loaders`:

        * ``ml-*``     → :meth:`from_movielens` (item_map keys are int)
        * ``amazon-*`` → :meth:`from_amazon` (item_map keys are str)
        * ``yelp``     → :meth:`from_yelp` (item_map keys are str)
        """
        if dataset.startswith("ml-"):
            return cls.from_movielens(data_dir, item_map, dataset=dataset)
        if dataset.startswith("amazon-"):
            category = dataset.removeprefix("amazon-")
            return cls.from_amazon(data_dir, item_map, dataset=category)
        if dataset == "yelp" or dataset.startswith("yelp-"):
            return cls.from_yelp(data_dir, item_map)
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'ml-*', "
            f"'amazon-*', or 'yelp'."
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_title(self, item_id: int) -> str:
        """Title for a remapped item id; fallback for unknown ids."""
        return self.titles.get(item_id, f"Unknown Item {item_id}")

    def get_side_info(self, item_id: int) -> str:
        """Genre (MovieLens) or description (Amazon); empty if unknown."""
        return self.side_info.get(item_id, "")

    def format_item(self, item_id: int) -> str:
        """Format an item as a single string suitable for an LLM prompt.

        * MovieLens: ``"Title (Genres)"``
        * Amazon:    ``"Title — Description"`` (em dash; description
          omitted if empty so the prompt stays clean for items with
          no metadata).
        * Yelp:      the full LLM-ESR template, precomputed in
          :meth:`from_yelp` and stored in ``self.titles``. Returned
          verbatim — the template is already a complete sentence and
          shouldn't be wrapped further.
        """
        title = self.get_title(item_id)
        side = self.get_side_info(item_id)
        if self.dataset_family == "yelp":
            return title
        if self.dataset_family == "amazon":
            return f"{title} — {side}" if side else title
        return f"{title} ({side})"
