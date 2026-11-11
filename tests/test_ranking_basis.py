"""Ranking basis (docs/contract.md, "Ranking basis")."""
from rag_connector import RetrievedChunk, ranking_basis


def _r(score, role=None, rank=0):
    meta = {"item_role": role} if role else {}
    return RetrievedChunk(chunk_id=f"c{rank}", text="t", score=score, rank=rank, metadata=meta)


def test_complete_set_is_not_a_ranking():
    b = ranking_basis("complete_set", [_r(0.9), _r(None, rank=1)])
    assert b.status == "not_applicable" and not b.rankable


def test_ordered_uses_returned_order_whatever_the_scores():
    b = ranking_basis("ordered", [_r(0.4), _r(7.1, rank=1), _r(None, rank=2)])
    assert b.status == "returned_order" and b.rankable


def test_scored_needs_a_score_on_every_matched_row():
    b = ranking_basis("scored", [_r(0.9), _r(None, rank=1)])
    assert b.status == "error" and "1 of 2" in b.reason


def test_scored_ignores_context_rows():
    b = ranking_basis("scored", [_r(0.9, "match"), _r(None, "context", 1), _r(0.5, "match", 2)])
    assert b.status == "scores"


def test_scored_out_of_order_is_an_error():
    b = ranking_basis("scored", [_r(0.4), _r(0.9, rank=1)])
    assert b.status == "error" and "descending" in b.reason


def test_rows_may_be_stored_dicts():
    assert ranking_basis("scored", [{"score": 0.8}, {"score": 0.2}]).status == "scores"


def test_complete_set_declared_ranked_uses_returned_order():
    b = ranking_basis("complete_set", [_r(0.4), _r(7.1, rank=1)], "ranked")
    assert b.status == "returned_order" and b.rankable
    assert ranking_basis("complete_set", [_r(0.4)], "set").status == "not_applicable"
