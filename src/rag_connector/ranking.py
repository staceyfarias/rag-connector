"""Ranking basis: whether a returned result can be read as a ranking, and how.

``docs/contract.md`` ("Ranking basis") is the normative text. A host never
assumes that row order is a ranking. It reads the connector's declared
``retrieval_mode`` and the rows' scores and roles (``rag_connector.derived``):

=================  ==========================================================
``scored``         The ranking is the scores, best first. Every row that was
                   matched against the query (``item_role`` ``match``, or no
                   role) must carry a score, and the rows must be in
                   descending score order. Anything else is an **error**: the
                   host does not fall back to row order.
``ordered``        The ranking is the returned order: the connector's own,
                   possibly fusing channels whose scores are not comparable
                   (vector similarity and keyword relevance). Scores are
                   evidence only.
``complete_set``   The system decides cardinality. Unless the connector
                   declares ``result_order: ranked`` in ``info()`` (then the
                   returned order is its ranking), the result is a set, not a
                   ranking. Rank-based measures
                   (reciprocal rank, nDCG, top-k, first item) are **not
                   applicable**; order-free measures (answer returned,
                   payload, set precision) are.
=================  ==========================================================

Context rows (``item_role`` ``context``) are never scored, by design, and are
ignored by these checks. Scores are compared only within one result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

#: Declared order semantics (``info()["result_order"]``). ``ranked``: the
#: returned order is the connector's ranking. ``set``: it is not. Matters for
#: ``complete_set``, where cardinality is the system's and order is otherwise
#: not a ranking.
RESULT_ORDER_RANKED = "ranked"
RESULT_ORDER_SET = "set"

#: Ranked by the scores (``scored``).
RANKING_SCORES = "scores"
#: Ranked by the connector's returned order (``ordered``).
RANKING_RETURNED_ORDER = "returned_order"
#: Not a ranking (``complete_set``): rank-based measures do not apply.
RANKING_NOT_APPLICABLE = "not_applicable"
#: The result contradicts its declared mode: rank-based measures are refused.
RANKING_ERROR = "error"


@dataclass(frozen=True)
class RankingBasis:
    """How one result may be read as a ranking.

    ``status`` is one of the ``RANKING_*`` values; ``reason`` says why in a
    sentence a person can read; ``rankable`` is True when rank-based measures
    may be computed (``scores`` or ``returned_order``).
    """

    status: str
    reason: str

    @property
    def rankable(self) -> bool:
        return self.status in (RANKING_SCORES, RANKING_RETURNED_ORDER)


def _role(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("role")
    meta = getattr(row, "metadata", None)
    return meta.get("item_role") if isinstance(meta, dict) else None


def _score(row: Any) -> Any:
    return row.get("score") if isinstance(row, dict) else getattr(row, "score", None)


def ranking_basis(retrieval_mode: str, rows: Sequence[Any],
                  result_order: str | None = None) -> RankingBasis:
    """The ranking basis of one result.

    ``rows`` are the returned rows in returned order: ``RetrievedChunk``
    objects, or dicts with ``score`` and ``role`` (as a host's stored run sheet
    holds them). An empty result is ``returned_order`` (nothing to rank, and
    nothing contradicts the mode).
    """
    mode = str(retrieval_mode or "scored")
    if mode == "complete_set" and result_order == RESULT_ORDER_RANKED:
        return RankingBasis(RANKING_RETURNED_ORDER,
                            "the system decides how many results, and returns "
                            "them in its own ranked order (result_order ranked)")
    if mode == "complete_set":
        return RankingBasis(RANKING_NOT_APPLICABLE,
                            "the system returns a set, not a ranking "
                            "(retrieval_mode complete_set)")
    if mode == "ordered":
        return RankingBasis(RANKING_RETURNED_ORDER,
                            "the connector's returned order is its ranking "
                            "(retrieval_mode ordered)")
    if mode != "scored":
        return RankingBasis(RANKING_ERROR, f"unknown retrieval_mode {mode!r}")
    matched = [r for r in rows if _role(r) in (None, "match")]
    missing = sum(1 for r in matched if _score(r) is None)
    if missing:
        return RankingBasis(
            RANKING_ERROR,
            f"declared scored, but {missing} of {len(matched)} matched rows "
            "carry no score; row order is not assumed to be a ranking")
    scores = [float(_score(r)) for r in matched]
    if any(b > a for a, b in zip(scores, scores[1:])):
        return RankingBasis(
            RANKING_ERROR,
            "declared scored, but the rows are not in descending score order")
    return RankingBasis(RANKING_SCORES, "ranked by score (retrieval_mode scored)")
