"""Ranking basis (docs/contract.md, "Ranking basis"): rank what matched and
was scored; everything else counts only toward answer returned."""
from rag_connector import RetrievedChunk, ranking_basis


def _r(cid, score, role=None):
    meta = {"item_role": role} if role else {}
    return RetrievedChunk(chunk_id=cid, text="t", score=score, rank=0, metadata=meta)


def test_scored_match_rows_are_sorted_by_score_not_row_order():
    b = ranking_basis("complete_set", [_r("a", 0.3), _r("b", 0.9), _r("c", 0.5)])
    assert b.status == "scores" and b.ranked_ids == ("b", "c", "a")


def test_context_rows_are_never_ranked():
    b = ranking_basis("complete_set", [_r("n1", None, "context"), _r("hit", 0.4, "match"),
                                       _r("n2", None, "context")])
    assert b.ranked_ids == ("hit",) and b.context_rows == 2


def test_unscored_matches_count_only_toward_answer_returned():
    b = ranking_basis("complete_set", [_r("kw", None, "match"), _r("v", 0.6, "match")])
    assert b.ranked_ids == ("v",) and b.unscored_match_rows == 1
    assert "answer returned" in b.reason


def test_a_set_with_no_scores_is_not_rankable():
    b = ranking_basis("complete_set", [_r("a", None), _r("b", None)])
    assert b.status == "not_applicable" and not b.rankable


def test_result_order_never_makes_row_order_trusted():
    b = ranking_basis("complete_set", [_r("a", None), _r("b", None)], "ranked")
    assert b.status == "not_applicable"


def test_ordered_ranks_match_rows_in_returned_order():
    b = ranking_basis("ordered", [_r("a", 0.2), _r("n", None, "context"), _r("b", 0.9)])
    assert b.status == "returned_order" and b.ranked_ids == ("a", "b")


def test_scored_needs_a_score_on_every_match_row():
    b = ranking_basis("scored", [_r("a", 0.9), _r("b", None)])
    assert b.status == "error" and "1 of 2" in b.reason


def test_rows_may_be_stored_dicts():
    b = ranking_basis("scored", [{"chunk_id": "a", "score": 0.2}, {"chunk_id": "b", "score": 0.8}])
    assert b.ranked_ids == ("b", "a")


def test_context_rows_are_grouped_by_origin_and_every_row_counts_as_returned():
    rows = [_r("hit", 0.7, "match"),
            RetrievedChunk(chunk_id="n1", text="t", score=None, rank=1,
                           metadata={"item_role": "context", "item_origin": "neighbour"}),
            RetrievedChunk(chunk_id="kw", text="t", score=None, rank=2,
                           metadata={"item_role": "context", "item_origin": "keyword"}),
            _r("x", None, "context")]
    b = ranking_basis("complete_set", rows)
    assert b.ranked_ids == ("hit",)
    assert b.returned_ids == ("hit", "n1", "kw", "x")
    assert b.context_by_origin == {"neighbour": ("n1",), "keyword": ("kw",), "unlabelled": ("x",)}


def test_origin_labels_context_rows_only():
    from rag_connector import derived_result_problems
    ok = RetrievedChunk(chunk_id="kw", text="t", score=None, rank=0,
                        metadata={"item_role": "context", "item_origin": "keyword"})
    bad = RetrievedChunk(chunk_id="v", text="t", score=0.5, rank=0,
                         metadata={"item_role": "match", "item_origin": "keyword"})
    assert derived_result_problems(ok) == []
    assert any("item_origin" in p for p in derived_result_problems(bad))
