"""Derived results: one returned item that stands for chunks instead of being one.

Most retrievers return chunks, and a returned row *is* the chunk it names. Some
return something else — a parent document standing for the chunks inside it, a
summary written from several passages, a curated answer, or a verified "the
corpus does not answer this". A host still has to know which corpus chunks such
a result stands for, what text was actually served, and what kind of thing it
was. This module is the single definition of how a connector says so, through
``RetrievedChunk.metadata``; ``docs/contract.md`` ("Derived results") is the
normative text.

The generic model is four fields:

=====================  ========================================================
covered chunk ids      ``metadata["item_covered_ids"]`` — the corpus chunks the
                       result stands for. Id-keyed scoring credits the result
                       through these, never through its own row id.
text                   ``RetrievedChunk.text`` — what was served: the chunk's
                       text, or a summary representing the covered chunks.
kind                   ``metadata["item_kind"]`` — ``chunk``, ``summary`` or
                       ``gap``. ``group`` is accepted as a synonym of
                       ``summary`` for compatibility.
curated (optional)     ``metadata["item_curated"]`` — curator notes, kept
                       separate from ``text`` and never part of it.
=====================  ========================================================

Two grouping keys that already exist in use are documented here too:
``item_index`` (rows sharing an index are one returned item) and ``item_label``
(the connector's own free-form name for the item, never interpreted).

These key names were first defined by RAGauge (``src/scoring/item_groups.py``)
and copied by Pelorus Query. They live here now, as the one source both import.
Vendor-specific extras stay in ``<vendor>_*`` metadata keys, which this library
never defines or reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DerivedResult",
    "ITEM_COVERED_IDS_KEY",
    "ITEM_CURATED_KEY",
    "ITEM_INDEX_KEY",
    "ITEM_KEYS",
    "ITEM_KINDS",
    "ITEM_KIND_CHUNK",
    "ITEM_KIND_GAP",
    "ITEM_KIND_KEY",
    "ITEM_KIND_SUMMARY",
    "ITEM_KIND_SYNONYMS",
    "ITEM_LABEL_KEY",
    "derived_result_metadata",
    "derived_result_problems",
    "normalize_item_kind",
    "read_derived_result",
]

#: Which returned item this row belongs to, 0-based. Rows sharing an index are
#: one item occupying one ranked slot. Optional.
ITEM_INDEX_KEY = "item_index"
#: The corpus chunk ids the result stands for (a list of strings). Optional.
ITEM_COVERED_IDS_KEY = "item_covered_ids"
#: ``chunk`` | ``summary`` | ``gap`` (``group`` = ``summary``). Optional.
ITEM_KIND_KEY = "item_kind"
#: The connector's free-form name for the item. Optional; never interpreted.
ITEM_LABEL_KEY = "item_label"
#: Curator notes, separate from the served text. Optional.
ITEM_CURATED_KEY = "item_curated"

ITEM_KEYS = (
    ITEM_INDEX_KEY, ITEM_COVERED_IDS_KEY, ITEM_KIND_KEY, ITEM_LABEL_KEY, ITEM_CURATED_KEY,
)

#: The row is the corpus chunk it names.
ITEM_KIND_CHUNK = "chunk"
#: The row stands for its covered chunks; its text represents them.
ITEM_KIND_SUMMARY = "summary"
#: A verified "the corpus does not answer this". Covers no chunks.
ITEM_KIND_GAP = "gap"
ITEM_KINDS = (ITEM_KIND_CHUNK, ITEM_KIND_SUMMARY, ITEM_KIND_GAP)

#: Older spellings still accepted, mapped to their current kind. ``group`` is
#: what RAGauge and Pelorus emit today for a result covering several chunks.
ITEM_KIND_SYNONYMS = {"group": ITEM_KIND_SUMMARY}


def normalize_item_kind(value: Any) -> str:
    """Return the current spelling of a declared kind, or raise ``ValueError``.

    An unknown kind is refused rather than guessed: a host that silently read
    ``"sumary"`` as a chunk would grade a summary as corpus text.
    """
    if isinstance(value, str):
        if value in ITEM_KINDS:
            return value
        if value in ITEM_KIND_SYNONYMS:
            return ITEM_KIND_SYNONYMS[value]
    raise ValueError(
        f"item_kind {value!r} is not one of {ITEM_KINDS} "
        f"(or the synonym{'s' if len(ITEM_KIND_SYNONYMS) > 1 else ''} "
        f"{tuple(ITEM_KIND_SYNONYMS)})"
    )


@dataclass(frozen=True)
class DerivedResult:
    """One returned row read through the generic model.

    ``covered_ids`` is ``None`` when the row declared none — a host then
    credits the row through its own ``chunk_id`` — and a tuple (possibly
    empty) when it declared some. An empty tuple is a declared zero coverage,
    which is **not** an abstention unless ``kind`` is ``gap``.

    ``kind_declared`` records whether ``kind`` came from the row or from the
    default (``summary`` when covered ids are declared, ``chunk`` otherwise).
    """

    chunk_id: str | None
    text: str
    kind: str
    covered_ids: tuple[str, ...] | None
    curated: str | None = None
    index: int | None = None
    label: str | None = None
    kind_declared: bool = False

    @property
    def credited_ids(self) -> tuple[str, ...]:
        """The corpus ids id-keyed scoring credits this row through."""
        if self.covered_ids is not None:
            return self.covered_ids
        return (self.chunk_id,) if self.chunk_id is not None else ()

    @property
    def is_gap(self) -> bool:
        return self.kind == ITEM_KIND_GAP


def _metadata(hit: Any) -> Mapping[str, Any]:
    meta = getattr(hit, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def derived_result_problems(hit: Any) -> list[str]:
    """Every way one row's derived-result declaration breaks the documented rules.

    Structural only: whether covered ids exist in the corpus needs the corpus,
    and is the validator's job. An empty list means the declaration is
    well-formed (including the case where it declares nothing at all).
    """
    meta = _metadata(hit)
    problems: list[str] = []

    covered = meta.get(ITEM_COVERED_IDS_KEY)
    if ITEM_COVERED_IDS_KEY in meta:
        if not isinstance(covered, (list, tuple)):
            problems.append(
                f"{ITEM_COVERED_IDS_KEY} must be a list of chunk ids, not "
                f"{type(covered).__name__}"
            )
            covered = None
        elif not all(isinstance(c, str) and c for c in covered):
            problems.append(f"{ITEM_COVERED_IDS_KEY} must hold non-empty strings only")
        elif len(set(covered)) != len(covered):
            problems.append(f"{ITEM_COVERED_IDS_KEY} lists the same chunk id twice")

    kind = None
    if ITEM_KIND_KEY in meta:
        try:
            kind = normalize_item_kind(meta.get(ITEM_KIND_KEY))
        except ValueError as exc:
            problems.append(str(exc))

    if kind == ITEM_KIND_GAP and covered:
        problems.append(
            "a gap covers no chunks, but this one declares "
            f"{ITEM_COVERED_IDS_KEY}={list(covered)!r}"
        )
    if kind == ITEM_KIND_CHUNK:
        chunk_id = getattr(hit, "chunk_id", None)
        if chunk_id is None:
            problems.append("a chunk result must carry the chunk's own chunk_id")
        elif isinstance(covered, (list, tuple)) and list(covered) != [chunk_id]:
            problems.append(
                f"a chunk result covers exactly itself ({[chunk_id]!r}), but "
                f"declares {ITEM_COVERED_IDS_KEY}={list(covered)!r}"
            )

    if ITEM_CURATED_KEY in meta and not isinstance(meta.get(ITEM_CURATED_KEY), str):
        problems.append(f"{ITEM_CURATED_KEY} must be a string")
    if ITEM_LABEL_KEY in meta and meta.get(ITEM_LABEL_KEY) is not None and not isinstance(
        meta.get(ITEM_LABEL_KEY), str
    ):
        problems.append(f"{ITEM_LABEL_KEY} must be a string")
    if ITEM_INDEX_KEY in meta:
        index = meta.get(ITEM_INDEX_KEY)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            problems.append(f"{ITEM_INDEX_KEY} must be a non-negative integer")
    return problems


def read_derived_result(hit: Any) -> DerivedResult:
    """Read a returned row through the generic model, applying the defaults.

    Raises ``ValueError`` naming every problem when the declaration is
    malformed (see :func:`derived_result_problems`); a host should not score a
    row whose meaning it had to guess.
    """
    problems = derived_result_problems(hit)
    if problems:
        raise ValueError("malformed derived-result declaration: " + "; ".join(problems))
    meta = _metadata(hit)
    covered = meta.get(ITEM_COVERED_IDS_KEY)
    covered_ids = tuple(covered) if ITEM_COVERED_IDS_KEY in meta else None
    declared = meta.get(ITEM_KIND_KEY) is not None
    if declared:
        kind = normalize_item_kind(meta[ITEM_KIND_KEY])
    else:
        kind = ITEM_KIND_SUMMARY if covered_ids is not None else ITEM_KIND_CHUNK
    return DerivedResult(
        chunk_id=getattr(hit, "chunk_id", None),
        text=str(getattr(hit, "text", "") or ""),
        kind=kind,
        covered_ids=covered_ids,
        curated=meta.get(ITEM_CURATED_KEY),
        index=meta.get(ITEM_INDEX_KEY),
        label=meta.get(ITEM_LABEL_KEY),
        kind_declared=declared,
    )


def derived_result_metadata(
    *,
    kind: str,
    covered_ids: Sequence[str] | None = None,
    curated: str | None = None,
    index: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Build the ``metadata`` keys for one returned row, checked.

    Merge the result into the row's metadata alongside any ``<vendor>_*`` keys
    of your own::

        RetrievedChunk(
            chunk_id="answer:42",            # not a corpus id
            text="A summary of the refund rules...",
            score=0.91, rank=0,
            metadata={
                **derived_result_metadata(
                    kind="summary",
                    covered_ids=["refunds.md:chunk-0", "refunds.md:chunk-3"],
                    curated="Checked against the 2026 policy.",
                ),
                "acme_answer_id": 42,
            },
        )

    Only the keys given are written, so a plain chunk needs nothing but its
    kind. A ``gap`` declares an empty ``item_covered_ids``, so a host never
    mistakes it for a row that simply declared nothing. The same structural
    rules :func:`derived_result_problems` applies are enforced here; the
    ``chunk`` rule needs the row's ``chunk_id``, which this builder does not
    see, so it is checked by the validator instead.
    """
    kind = normalize_item_kind(kind)
    meta: dict[str, Any] = {ITEM_KIND_KEY: kind}
    if kind == ITEM_KIND_GAP:
        if covered_ids:
            raise ValueError("a gap covers no chunks; pass no covered_ids")
        meta[ITEM_COVERED_IDS_KEY] = []
    elif covered_ids is not None:
        meta[ITEM_COVERED_IDS_KEY] = list(covered_ids)
    elif kind == ITEM_KIND_SUMMARY:
        raise ValueError("a summary must declare the chunk ids it covers (possibly none)")
    if curated is not None:
        meta[ITEM_CURATED_KEY] = curated
    if index is not None:
        meta[ITEM_INDEX_KEY] = index
    if label is not None:
        meta[ITEM_LABEL_KEY] = label

    class _Row:  # the structural check reads attributes off a row
        chunk_id = None
        metadata = meta

    problems = [p for p in derived_result_problems(_Row()) if "chunk result" not in p]
    if problems:
        raise ValueError("; ".join(problems))
    return meta
