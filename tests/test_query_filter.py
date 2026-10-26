"""query()'s optional metadata ``filter`` keyword (docs/contract.md)."""

import json

import pytest

from rag_connector import (
    FILTER_OPERATORS,
    ChunkRecord,
    InvalidFilterError,
    RagPipeline,
    RetrievedChunk,
    UnsupportedCapability,
    UnsupportedFilterError,
    and_filters,
    filter_fields,
    filter_operators,
    query_accepts_filter,
    require_filter_operators,
    supported_filter_operators,
    validate_filter,
)
from rag_connector.validate import FAIL, PASS, SKIP, Report, _check_query_filter

# -- the neutral syntax ------------------------------------------------------

@pytest.mark.parametrize("filt", [
    {"type": "general_info"},
    {"type": {"$eq": "general_info"}},
    {"type": {"$ne": "cm"}, "sponsor_id": "s1"},
    {"task_id": {"$in": ["a", "b"]}},
    {"task_id": {"$nin": ["a"]}},
    {"chunk_number": {"$gte": 1, "$lt": 10}},
    {"is_system_file": {"$exists": True}},
    {"is_system_file": True},
    {"$or": [{"type": "faq"}, {"$and": [{"type": "content"},
                                        {"step_id": "s"}]}]},
    {"type": "general_info", "$or": [{"is_system_file": True},
                                     {"sponsor_id": "s1"}]},
])
def test_valid_filters_pass_unchanged(filt):
    assert validate_filter(filt) == filt


def test_none_and_empty_mean_no_constraint():
    assert validate_filter(None) == {}
    assert validate_filter({}) == {}


def test_tuples_normalize_to_lists_and_the_input_is_not_shared():
    original = {"task_id": {"$in": ("a", "b")}}
    out = validate_filter(original)
    assert out == {"task_id": {"$in": ["a", "b"]}}
    json.dumps(out)
    out["task_id"]["$in"].append("c")
    assert original["task_id"]["$in"] == ("a", "b")


@pytest.mark.parametrize("filt, fragment", [
    ("type = cm", "is an object"),
    ({"type": ["a", "b"]}, "use {\"$in\""),
    ({"type": None}, "null is not supported"),
    ({"type": {}}, "is empty"),
    ({"type": {"$regex": "x"}}, "unknown operator"),
    ({"type": {"$in": []}}, "non-empty list"),
    ({"type": {"$in": "abc"}}, "non-empty list"),
    ({"type": {"$in": [{"a": 1}]}}, "non-empty list"),
    ({"n": {"$gt": "3"}}, "finite number"),
    ({"n": {"$gt": True}}, "finite number"),
    ({"n": {"$lt": float("nan")}}, "finite number"),
    ({"f": {"$exists": 1}}, "true or false"),
    ({"f": {"$eq": [1]}}, "string, number or bool"),
    ({"f": {"nested": 1}}, "not an operator"),
    ({"f": {"$and": []}}, "belongs at filter level"),
    ({"$and": []}, "non-empty list of filters"),
    ({"$or": {"a": 1}}, "non-empty list of filters"),
    ({"$and": [{}]}, "empty filter"),
    ({"$and": ["x"]}, "is an object"),
    ({"$nor": [{"a": 1}]}, "not allowed at filter level"),
    ({"$eq": 1}, "not allowed at filter level"),
    ({"": 1}, "non-empty strings"),
])
def test_malformed_filters_are_rejected_with_the_path(filt, fragment):
    with pytest.raises(InvalidFilterError) as info:
        validate_filter(filt)
    assert fragment in str(info.value)
    assert str(info.value).startswith("filter")


def test_invalid_filter_error_is_a_value_error():
    with pytest.raises(ValueError):
        validate_filter({"a": None})


def test_operators_and_fields_are_listed():
    filt = {"type": "faq", "$or": [{"task_id": {"$in": ["a"]}},
                                   {"n": {"$gt": 1, "$lte": 3}}]}
    assert filter_operators(filt) == {"$eq", "$or", "$in", "$gt", "$lte"}
    assert filter_fields(filt) == {"type", "task_id", "n"}
    assert filter_operators({}) == frozenset()


def test_and_filters_drops_empties_and_never_merges_keys():
    assert and_filters(None, {}) == {}
    assert and_filters({"a": 1}, None) == {"a": 1}
    assert and_filters({"a": 1}, {"a": 2}) == {"$and": [{"a": 1}, {"a": 2}]}
    with pytest.raises(InvalidFilterError):
        and_filters({"a": 1}, {"a": None})


def test_the_operator_set_is_the_documented_subset():
    assert FILTER_OPERATORS == {
        "$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte",
        "$exists", "$and", "$or"}


# -- unsupported -> error ----------------------------------------------------

def test_no_support_refuses_any_non_empty_filter():
    with pytest.raises(UnsupportedFilterError) as info:
        require_filter_operators({"a": 1}, frozenset(), connector="X")
    assert "X does not support query filters" in str(info.value)
    assert info.value.operators == {"$eq"}
    assert require_filter_operators(None, frozenset()) == {}
    assert require_filter_operators({}, frozenset()) == {}


def test_an_unsupported_operator_is_named():
    with pytest.raises(UnsupportedFilterError) as info:
        require_filter_operators(
            {"a": {"$exists": True}, "$or": [{"b": 1}, {"c": 2}]},
            {"$eq", "$and"})
    assert info.value.operators == {"$exists", "$or"}
    assert "$exists" in str(info.value) and "$or" in str(info.value)


def test_unsupported_filter_is_an_unsupported_capability():
    assert issubclass(UnsupportedFilterError, UnsupportedCapability)


def test_malformed_beats_unsupported():
    with pytest.raises(InvalidFilterError):
        require_filter_operators({"a": None}, frozenset())


# -- capability declaration and backward compatibility ----------------------

class _Old:
    def query(self, text, top_k=5):
        return []


class _OldKwargs:
    """Predates the keyword: **kwargs would swallow a filter silently."""

    def query(self, text, **kwargs):
        return []


class _Declares:
    def query(self, text, top_k=5, *, answer_metadata=None, filter=None):
        require_filter_operators(filter, self.query_filter_operators())
        return []

    def query_filter_operators(self):
        return frozenset({"$eq", "$in", "$and", "$bogus"})


def _chunk(cid="c1"):
    return ChunkRecord(chunk_id=cid, doc_id="d", source_file="f.txt",
                       doc_chunk_index=0, text="some text about filters")


class _Pipeline(RagPipeline):
    """A RagPipeline written before the filter keyword existed."""

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id="c1", text="t", score=1.0, rank=0)]

    def pull_all_chunks(self):
        return [_chunk()]


def test_a_connector_written_before_the_keyword_still_works():
    pipe = _Pipeline()
    assert not query_accepts_filter(pipe)
    assert supported_filter_operators(pipe) == frozenset()
    assert pipe.query_filter_operators() == frozenset()
    assert [c.chunk_id for c in pipe.query("q")] == ["c1"]


def test_kwargs_accepts_the_keyword_but_declares_no_support():
    assert query_accepts_filter(_OldKwargs())
    assert supported_filter_operators(_OldKwargs()) == frozenset()
    assert supported_filter_operators(_Old()) == frozenset()
    assert supported_filter_operators(object()) == frozenset()


def test_a_declaring_connector_reports_its_operators_within_the_syntax():
    assert supported_filter_operators(_Declares()) == {"$eq", "$in", "$and"}


def test_a_broken_declaration_counts_as_none():
    class Broken(_Declares):
        def query_filter_operators(self):
            raise RuntimeError("boom")

    class Stringy(_Declares):
        def query_filter_operators(self):
            return "$eq"

    assert supported_filter_operators(Broken()) == frozenset()
    assert supported_filter_operators(Stringy()) == frozenset()


def test_the_reference_rag_accepts_the_keyword_and_refuses_a_filter(tmp_path):
    from rag_connector.reference import ReferenceRagConnector

    assert query_accepts_filter(ReferenceRagConnector)
    pipe = ReferenceRagConnector.__new__(ReferenceRagConnector)
    assert pipe.query_filter_operators() == frozenset()
    with pytest.raises(UnsupportedFilterError):
        pipe.query("q", filter={"type": "x"})


# -- validator ---------------------------------------------------------------

def _status(pipeline):
    report = Report(target="t")
    _check_query_filter(report, pipeline, "probe", 5)
    (check,) = report.checks
    return check.status, check


def test_validator_skips_a_connector_without_the_keyword():
    assert _status(_Pipeline())[0] == SKIP


def test_validator_passes_a_keyword_connector_that_refuses():
    class Refuses(_Pipeline):
        def query(self, text, top_k=5, *, filter=None):
            require_filter_operators(filter, self.query_filter_operators())
            return super().query(text, top_k)

    assert _status(Refuses())[0] == PASS


def test_validator_fails_a_connector_that_ignores_a_filter():
    class Ignores(_Pipeline):
        def query(self, text, top_k=5, *, filter=None):
            return super().query(text, top_k)

    assert _status(Ignores())[0] == FAIL


def test_validator_checks_a_declaration():
    class Declares(_Pipeline):
        def query(self, text, top_k=5, *, filter=None):
            return super().query(text, top_k)

        def query_filter_operators(self):
            return frozenset({"$eq", "$or"})

    class DeclaresWithoutKeyword(_Pipeline):
        def query_filter_operators(self):
            return frozenset({"$eq"})

    class DeclaresBogus(Declares):
        def query_filter_operators(self):
            return frozenset({"$eq", "$regex"})

    assert _status(Declares())[0] == PASS
    assert _status(DeclaresWithoutKeyword())[0] == FAIL
    assert _status(DeclaresBogus())[0] == FAIL
