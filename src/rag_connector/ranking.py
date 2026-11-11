"""Ranking basis: what in a returned result may be ranked, and how.

``docs/contract.md`` ("Ranking basis") is the normative text. A host never
assumes that row order is a ranking. It ranks only what was **matched and
scored** against the query, and uses everything else only to decide whether
the answer was returned:

=================  ==========================================================
row                used for
=================  ==========================================================
scored match       ranking: the host sorts these by score itself (reciprocal
                   rank, first item, top-k)
context            answer returned only: added by the system beside the core
                   (vector) search -- chunk expansion around a match, or a
                   separate keyword search -- never ranked. ``item_origin``
                   (``neighbour`` / ``keyword``) says which; encouraged, so a
                   host can show which part of the pipeline delivered an
                   answer the core search did not rank
unscored match     answer returned only: matched, but with no score
=================  ==========================================================

A row with no ``item_role`` is a match. The declared ``retrieval_mode`` decides
what may be assumed:

``scored``         Every match row must carry a score; a missing one is an
                   **error** -- the host does not fall back to row order.
``ordered``        The connector vouches that its returned order is a
                   ranking: match rows are ranked in returned order (context
                   rows still never are).
``complete_set``   No order is assumed. Match rows that carry a score are
                   ranked by it; with none, rank-based measures are **not
                   applicable**.

So decoupling holds here too: what matched decides the rank, and what a system
adds around a match (its neighbour configuration) only decides what else is
returned. ``info()["result_order"]`` is informational; it never makes a host
trust row order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RANKING_ERROR",
    "RANKING_NOT_APPLICABLE",
    "RANKING_RETURNED_ORDER",
    "RANKING_SCORES",
    "RESULT_ORDER_RANKED",
    "RESULT_ORDER_SET",
    "RankingBasis",
    "ranking_basis",
]

#: Match rows ranked by their scores.
RANKING_SCORES = "scores"
#: Match rows ranked in returned order (``ordered``: the connector vouches).
RANKING_RETURNED_ORDER = "returned_order"
#: Nothing to rank: no scored match row and no order vouched for.
RANKING_NOT_APPLICABLE = "not_applicable"
#: The result contradicts its declared mode: rank-based measures are refused.
RANKING_ERROR = "error"

#: Informational order semantics a connector may declare in ``info()``.
RESULT_ORDER_RANKED = "ranked"
RESULT_ORDER_SET = "set"


@dataclass(frozen=True)
class RankingBasis:
    """What one result contributes to ranking.

    ``ranked_ids`` are the match rows' chunk ids in rank order (first
    occurrence wins); ``context_rows`` and ``unscored_match_rows`` count the
    rows that count only toward "answer returned". ``rankable`` is True when
    rank-based measures may be computed from ``ranked_ids``.
    """

    status: str
    reason: str
    ranked_ids: tuple[str, ...] = ()
    context_rows: int = 0
    unscored_match_rows: int = 0
    scores: tuple[float, ...] = field(default=(), repr=False)
    #: Every returned chunk id (what "answer returned" reads), and the context
    #: rows' ids by ``item_origin`` (``"unlabelled"`` when none was declared).
    returned_ids: tuple[str, ...] = ()
    context_by_origin: dict = field(default_factory=dict)

    @property
    def rankable(self) -> bool:
        return self.status in (RANKING_SCORES, RANKING_RETURNED_ORDER)


def _get(row: Any, key: str, meta_key: str | None = None) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if meta_key is not None:
        meta = getattr(row, "metadata", None)
        return meta.get(meta_key) if isinstance(meta, dict) else None
    return getattr(row, key, None)


def ranking_basis(retrieval_mode: str, rows: Sequence[Any],
                  result_order: str | None = None) -> RankingBasis:  # noqa: ARG001
    """What one result may rank, and in what order.

    ``rows`` are the returned rows in returned order: ``RetrievedChunk``
    objects, or dicts with ``chunk_id``, ``score`` and ``role`` (as a host's
    stored run sheet holds them). ``result_order`` is accepted for callers that
    pass a connector's declaration; it does not change the basis.
    """
    mode = str(retrieval_mode or "scored")
    if mode not in ("scored", "ordered", "complete_set"):
        return RankingBasis(RANKING_ERROR, f"unknown retrieval_mode {mode!r}")
    match, context = [], 0
    returned: list[str] = []
    by_origin: dict[str, list[str]] = {}
    for row in rows:
        cid = _get(row, "chunk_id")
        if cid is not None and str(cid) not in returned:
            returned.append(str(cid))
        role = _get(row, "role", "item_role")
        if role == "context":
            context += 1
            origin = _get(row, "origin", "item_origin") or "unlabelled"
            if cid is not None:
                by_origin.setdefault(str(origin), []).append(str(cid))
            continue
        match.append((_get(row, "chunk_id"), _get(row, "score")))
    scored = [(cid, float(score)) for cid, score in match if score is not None]
    unscored = len(match) - len(scored)

    def ids(pairs):
        seen, out = set(), []
        for cid, _ in pairs:
            if cid is not None and str(cid) not in seen:
                seen.add(str(cid))
                out.append(str(cid))
        return tuple(out)

    if mode == "scored" and unscored:
        return RankingBasis(
            RANKING_ERROR,
            f"declared scored, but {unscored} of {len(match)} matched rows carry no "
            "score; row order is not assumed to be a ranking",
            context_rows=context, unscored_match_rows=unscored,
            returned_ids=tuple(returned),
            context_by_origin={k: tuple(v) for k, v in by_origin.items()})
    if mode == "ordered":
        return RankingBasis(
            RANKING_RETURNED_ORDER,
            "match rows ranked in returned order (retrieval_mode ordered: the "
            "connector vouches for it); context rows never ranked",
            ranked_ids=ids(match), context_rows=context,
            returned_ids=tuple(returned),
            context_by_origin={k: tuple(v) for k, v in by_origin.items()})
    if not scored:
        return RankingBasis(
            RANKING_NOT_APPLICABLE,
            "no matched row carries a score, and no order is vouched for "
            f"(retrieval_mode {mode}): rank-based measures do not apply",
            context_rows=context, unscored_match_rows=unscored,
            returned_ids=tuple(returned),
            context_by_origin={k: tuple(v) for k, v in by_origin.items()})
    ordered = sorted(scored, key=lambda pair: -pair[1])      # stable on ties
    reason = "match rows ranked by score"
    if unscored:
        reason += (f"; {unscored} matched row(s) with no comparable score count "
                   "only toward answer returned")
    if context:
        reason += f"; {context} context row(s) count only toward answer returned"
    return RankingBasis(RANKING_SCORES, reason, ranked_ids=ids(ordered),
                        context_rows=context, unscored_match_rows=unscored,
                        scores=tuple(score for _, score in ordered),
                        returned_ids=tuple(returned),
                        context_by_origin={k: tuple(v) for k, v in by_origin.items()})
