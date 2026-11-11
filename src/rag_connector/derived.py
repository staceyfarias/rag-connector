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
kind                   ``metadata["item_kind"]`` — ``chunk``, ``passage``,
                       ``summary`` or ``gap``. ``group`` is accepted as a
                       synonym of ``summary`` for compatibility.
curated (optional)     ``metadata["item_curated"]`` — curator notes, kept
                       separate from ``text`` and never part of it.
=====================  ========================================================

Two grouping keys that already exist in use are documented here too:
``item_index`` (rows sharing an index are one returned item) and ``item_label``
(the connector's own free-form name for the item, never interpreted).

**Passages and row roles (2026-11-10).** Some systems return a matched chunk
together with chunks they add around it (neighbour expansion, a parent
section). Each of those rows is real corpus text that can be graded, but only
the matched row was scored against the query. A ``passage`` item is several
verbatim corpus chunks served as one unit: its rows share an ``item_index``,
each is credited through its own ``chunk_id``, and the item covers the union
of its rows. ``metadata["item_role"]`` says what each row is: ``match`` (scored
against the query; ``score`` is that score, and ``None`` means the system did
not expose it) or ``context`` (added by the system around a match, never
scored: its ``score`` is ``None`` by design). The item ranks by its match
row(s). :func:`read_items` groups rows into items with these rules applied.

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
    "ITEM_KIND_PASSAGE",
    "ITEM_ROLES",
    "ITEM_ROLE_CONTEXT",
    "ITEM_ROLE_KEY",
    "ITEM_ROLE_MATCH",
    "ReturnedItem",
    "item_group_problems",
    "read_items",
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
#: ``match`` | ``context``: whether the row was scored against the query or
#: added around a match by the system. Optional.
ITEM_ROLE_KEY = "item_role"

ITEM_KEYS = (
    ITEM_INDEX_KEY, ITEM_COVERED_IDS_KEY, ITEM_KIND_KEY, ITEM_LABEL_KEY, ITEM_CURATED_KEY,
    ITEM_ROLE_KEY,
)

#: The row was scored against the query (its score, or None if not exposed).
ITEM_ROLE_MATCH = "match"
#: The row was added by the system around a match and never scored.
ITEM_ROLE_CONTEXT = "context"
ITEM_ROLES = (ITEM_ROLE_MATCH, ITEM_ROLE_CONTEXT)

#: The row is the corpus chunk it names.
ITEM_KIND_CHUNK = "chunk"
#: The row is the corpus chunk it names, served with other chunks as one item
#: (rows sharing ``item_index``); the item covers the union of its rows.
ITEM_KIND_PASSAGE = "passage"
#: The row stands for its covered chunks; its text represents them.
ITEM_KIND_SUMMARY = "summary"
#: A verified "the corpus does not answer this". Covers no chunks.
ITEM_KIND_GAP = "gap"
ITEM_KINDS = (ITEM_KIND_CHUNK, ITEM_KIND_PASSAGE, ITEM_KIND_SUMMARY, ITEM_KIND_GAP)

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
    role: str | None = None

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

    if kind == ITEM_KIND_PASSAGE:
        chunk_id = getattr(hit, "chunk_id", None)
        if chunk_id is None:
            problems.append("a passage row must carry its corpus chunk's own chunk_id")
        elif isinstance(covered, (list, tuple)) and list(covered) != [chunk_id]:
            problems.append(
                f"a passage row covers exactly itself ({[chunk_id]!r}); the item "
                f"covers the union of its rows, but this row declares "
                f"{ITEM_COVERED_IDS_KEY}={list(covered)!r}")
        if ITEM_INDEX_KEY not in meta:
            problems.append(f"a passage row must declare {ITEM_INDEX_KEY}: its "
                            "rows are one item")

    if ITEM_ROLE_KEY in meta:
        role = meta.get(ITEM_ROLE_KEY)
        if role not in ITEM_ROLES:
            problems.append(f"{ITEM_ROLE_KEY} {role!r} is not one of {ITEM_ROLES}")
        elif role == ITEM_ROLE_CONTEXT and getattr(hit, "score", None) is not None:
            problems.append("a context row was never scored against the query, "
                            "so its score must be None")
        if kind == ITEM_KIND_GAP:
            problems.append(f"a gap has no {ITEM_ROLE_KEY}")

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
        role=meta.get(ITEM_ROLE_KEY),
    )


def derived_result_metadata(
    *,
    kind: str,
    covered_ids: Sequence[str] | None = None,
    curated: str | None = None,
    index: int | None = None,
    label: str | None = None,
    role: str | None = None,
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
    if role is not None:
        meta[ITEM_ROLE_KEY] = role

    class _Row:  # the structural check reads attributes off a row
        chunk_id = None
        metadata = meta

    problems = [p for p in derived_result_problems(_Row())
                if "chunk result" not in p and "passage row must carry" not in p]
    if problems:
        raise ValueError("; ".join(problems))
    return meta


# --- items: rows grouped by item_index ---------------------------------------


@dataclass(frozen=True)
class ReturnedItem:
    """One returned item: the rows sharing an ``item_index`` (or one row).

    ``covered_ids`` is what id-keyed scoring credits: the union of the rows'
    own ids for a ``passage``, the declared covered ids otherwise. ``score`` is
    the best score among the item's ``match`` rows (a row with no role counts
    as a match), or ``None`` when none carries one. ``roles`` gives each row's
    declared role in row order (``None`` where undeclared).
    """

    index: int | None
    kind: str
    covered_ids: tuple[str, ...]
    row_ranks: tuple[int, ...]
    score: float | None
    roles: tuple[str | None, ...]
    label: str | None = None

    @property
    def has_match(self) -> bool:
        return any(role in (None, ITEM_ROLE_MATCH) for role in self.roles)


def item_group_problems(hits: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Rules across rows sharing an ``item_index``: ``(failures, warnings)``.

    Rows of one item declare the same kind. For every kind but ``passage`` they
    also declare the same covered ids (a host reads them once); passage rows
    each cover themselves. A passage item with no ``match`` row warns: nothing
    in it was scored against the query.
    """
    failures: list[str] = []
    warnings: list[str] = []
    groups: dict[int, list[Any]] = {}
    for hit in hits:
        index = _metadata(hit).get(ITEM_INDEX_KEY)
        if isinstance(index, int) and not isinstance(index, bool):
            groups.setdefault(index, []).append(hit)
    for index, rows in groups.items():
        kinds = {_metadata(r).get(ITEM_KIND_KEY) for r in rows}
        first = getattr(rows[0], "rank", "?")
        if len(kinds) > 1:
            failures.append(
                f"rows of item_index {index} (first at rank {first}) declare "
                f"different item_kind values {sorted(map(str, kinds))}")
            continue
        kind = next(iter(kinds))
        if kind == ITEM_KIND_PASSAGE:
            if not any(_metadata(r).get(ITEM_ROLE_KEY, ITEM_ROLE_MATCH) == ITEM_ROLE_MATCH
                       for r in rows):
                warnings.append(
                    f"passage item_index {index} (first at rank {first}) has no "
                    "match row: nothing in it was scored against the query")
            continue
        covered = {repr(_metadata(r).get(ITEM_COVERED_IDS_KEY)) for r in rows}
        if len(covered) > 1:
            failures.append(
                f"rows of item_index {index} (first at rank {first}) declare "
                "different item_covered_ids")
    return failures, warnings


def read_items(hits: Sequence[Any]) -> list[ReturnedItem]:
    """Group returned rows into items, in first-seen order, rules applied.

    A row without ``item_index`` is its own item. Raises ``ValueError`` when a
    row or a group is malformed (see :func:`derived_result_problems` and
    :func:`item_group_problems`).
    """
    problems = [p for hit in hits for p in derived_result_problems(hit)]
    problems += item_group_problems(hits)[0]
    if problems:
        raise ValueError("malformed item declarations: " + "; ".join(problems))
    order: list[Any] = []
    rows_by: dict[Any, list[tuple[int, Any]]] = {}
    for position, hit in enumerate(hits):
        index = _metadata(hit).get(ITEM_INDEX_KEY)
        key = ("i", index) if index is not None else ("r", position)
        if key not in rows_by:
            rows_by[key] = []
            order.append(key)
        rows_by[key].append((position, hit))
    items: list[ReturnedItem] = []
    for key in order:
        rows = rows_by[key]
        results = [read_derived_result(hit) for _, hit in rows]
        kind = results[0].kind
        if kind == ITEM_KIND_PASSAGE:
            covered: list[str] = []
            for result in results:
                for cid in result.credited_ids:
                    if cid not in covered:
                        covered.append(cid)
        else:
            covered = list(results[0].credited_ids)
        scores = [getattr(hit, "score", None)
                  for (_, hit), result in zip(rows, results)
                  if result.role in (None, ITEM_ROLE_MATCH)]
        scores = [x for x in scores if x is not None]
        items.append(ReturnedItem(
            index=results[0].index,
            kind=kind,
            covered_ids=tuple(covered),
            row_ranks=tuple(getattr(hit, "rank", position) for position, hit in rows),
            score=max(scores) if scores else None,
            roles=tuple(result.role for result in results),
            label=results[0].label,
        ))
    return items
