"""The generic derived-result model (docs/contract.md, "Derived results").

Expected readings are taken from the documented rules, case by case; the
connectors are synthetic.
"""

from __future__ import annotations

import pytest

from rag_connector.base import ChunkRecord, RagPipeline, RetrievedChunk
from rag_connector.derived import (
    ITEM_COVERED_IDS_KEY,
    ITEM_CURATED_KEY,
    ITEM_INDEX_KEY,
    ITEM_KIND_KEY,
    ITEM_LABEL_KEY,
    derived_result_metadata,
    derived_result_problems,
    normalize_item_kind,
    read_derived_result,
)
from rag_connector.validate import FAIL, PASS, SKIP, WARN, validate_pipeline


class GoodConnector(RagPipeline):
    """A compliant plain chunk retriever over the corpus a:0, a:1, b:0."""

    name = "good"

    def __init__(self):
        self._chunks = [
            ChunkRecord(chunk_id=cid, doc_id=doc, source_file=f"{doc}.md",
                        doc_chunk_index=idx, global_index=g, text=text,
                        metadata={"source_fingerprint": "fp"})
            for g, (cid, doc, idx, text) in enumerate([
                ("a:0", "a", 0, "alpha beta gamma delta"),
                ("a:1", "a", 1, "epsilon zeta eta theta"),
                ("b:0", "b", 0, "iota kappa lambda mu"),
            ])
        ]

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id=c.chunk_id, text=c.text,
                               score=round(1.0 - 0.1 * i, 4), rank=i)
                for i, c in enumerate(self._chunks[:top_k])]

    def pull_all_chunks(self):
        return list(self._chunks)


def _hit(chunk_id="unit:1", text="served text", rank=0, **meta):
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=1.0 - rank / 10,
                          rank=rank, metadata=meta)


# --- the key names are the ones RAGauge and Pelorus already emit -------------

def test_key_names_are_the_existing_spellings():
    assert (ITEM_INDEX_KEY, ITEM_COVERED_IDS_KEY, ITEM_KIND_KEY, ITEM_LABEL_KEY) == (
        "item_index", "item_covered_ids", "item_kind", "item_label")
    assert ITEM_CURATED_KEY == "item_curated"


def test_group_is_accepted_as_a_synonym_of_summary():
    assert normalize_item_kind("group") == "summary"
    assert normalize_item_kind("summary") == "summary"
    assert normalize_item_kind("gap") == "gap"
    with pytest.raises(ValueError):
        normalize_item_kind("sumary")
    with pytest.raises(ValueError):
        normalize_item_kind(None)


# --- reading -----------------------------------------------------------------

def test_a_row_declaring_nothing_is_a_chunk_credited_through_its_own_id():
    result = read_derived_result(_hit(chunk_id="a:0"))
    assert result.kind == "chunk" and not result.kind_declared
    assert result.covered_ids is None
    assert result.credited_ids == ("a:0",)


def test_declared_covered_ids_without_a_kind_default_to_summary():
    result = read_derived_result(_hit(item_covered_ids=["a:0", "a:1"]))
    assert result.kind == "summary" and not result.kind_declared
    assert result.credited_ids == ("a:0", "a:1")


def test_a_summary_covering_one_chunk_is_still_a_summary():
    # RAGauge's grouping defaults a <=1-id unit to "chunk"; the documented rule
    # is that a declared summary is a summary however many ids it covers.
    result = read_derived_result(_hit(item_kind="summary", item_covered_ids=["a:0"]))
    assert result.kind == "summary"


def test_zero_coverage_is_not_a_gap_unless_it_says_so():
    lost = read_derived_result(_hit(item_kind="summary", item_covered_ids=[]))
    gap = read_derived_result(_hit(item_kind="gap", item_covered_ids=[]))
    assert lost.covered_ids == () and not lost.is_gap
    assert gap.is_gap and gap.credited_ids == ()


def test_curated_notes_are_kept_apart_from_the_served_text():
    result = read_derived_result(_hit(text="the answer", item_kind="summary",
                                      item_covered_ids=["a:0"],
                                      item_curated="checked by a curator"))
    assert result.text == "the answer"
    assert result.curated == "checked by a curator"


def test_vendor_keys_are_ignored():
    result = read_derived_result(_hit(chunk_id="a:0", acme_extract_id=7, acme_gap=True))
    assert result.kind == "chunk" and not result.is_gap


@pytest.mark.parametrize("meta, fragment", [
    ({"item_covered_ids": "a:0"}, "must be a list"),
    ({"item_covered_ids": ["a:0", 3]}, "non-empty strings"),
    ({"item_covered_ids": ["a:0", "a:0"]}, "same chunk id twice"),
    ({"item_kind": "parent"}, "is not one of"),
    ({"item_kind": "gap", "item_covered_ids": ["a:0"]}, "a gap covers no chunks"),
    ({"item_kind": "chunk", "item_covered_ids": ["a:0", "a:1"]}, "covers exactly itself"),
    ({"item_curated": ["note"]}, "item_curated must be a string"),
    ({"item_label": 5}, "item_label must be a string"),
    ({"item_index": -1}, "non-negative integer"),
    ({"item_index": True}, "non-negative integer"),
])
def test_malformed_declarations_are_named_and_refused(meta, fragment):
    hit = _hit(chunk_id="a:0", **meta)
    assert any(fragment in p for p in derived_result_problems(hit))
    with pytest.raises(ValueError, match="malformed"):
        read_derived_result(hit)


def test_a_chunk_result_covering_exactly_itself_is_well_formed():
    hit = _hit(chunk_id="b:0", item_kind="chunk", item_covered_ids=["b:0"])
    assert derived_result_problems(hit) == []


# --- building ------------------------------------------------------------------

def test_the_builder_writes_only_the_keys_given():
    assert derived_result_metadata(kind="chunk") == {"item_kind": "chunk"}
    assert derived_result_metadata(
        kind="group", covered_ids=("a:0", "a:1"), curated="n", index=2, label="Extract"
    ) == {"item_kind": "summary", "item_covered_ids": ["a:0", "a:1"],
          "item_curated": "n", "item_index": 2, "item_label": "Extract"}


def test_the_builder_gives_a_gap_an_explicit_empty_coverage():
    assert derived_result_metadata(kind="gap") == {"item_kind": "gap", "item_covered_ids": []}
    with pytest.raises(ValueError):
        derived_result_metadata(kind="gap", covered_ids=["a:0"])


def test_the_builder_refuses_a_summary_that_does_not_say_what_it_covers():
    with pytest.raises(ValueError):
        derived_result_metadata(kind="summary")
    assert derived_result_metadata(kind="summary", covered_ids=[])[ITEM_COVERED_IDS_KEY] == []


def test_the_builder_output_reads_back_as_declared():
    meta = derived_result_metadata(kind="summary", covered_ids=["a:0"], curated="c")
    result = read_derived_result(_hit(**meta))
    assert (result.kind, result.covered_ids, result.curated, result.kind_declared) == (
        "summary", ("a:0",), "c", True)


# --- the validator -------------------------------------------------------------

def _status(report, fragment):
    return next(c.status for c in report.checks if fragment in c.name)


class _Serving(GoodConnector):
    """Serves whatever rows the test sets; corpus is a:0, a:1, b:0."""

    rows: list = []

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id=r.chunk_id, text=r.text, score=r.score,
                               rank=r.rank, metadata=dict(r.metadata))
                for r in self.rows][:top_k]


def _validate(rows):
    connector = _Serving()
    connector.rows = rows
    return validate_pipeline(connector, target="t", sample_query="alpha")


def test_a_plain_chunk_retriever_skips_the_derived_result_check():
    report = validate_pipeline(GoodConnector(), target="t", sample_query="alpha")
    assert _status(report, "Derived-result") == SKIP


def test_well_formed_declarations_pass():
    report = _validate([
        _hit("answer:1", rank=0, **derived_result_metadata(
            kind="summary", covered_ids=["a:0", "a:1"], curated="note", index=0)),
        _hit("b:0", rank=1, **derived_result_metadata(
            kind="chunk", covered_ids=["b:0"], index=1)),
        _hit("gap:1", rank=2, **derived_result_metadata(kind="gap", index=2)),
    ])
    assert _status(report, "Derived-result") == PASS
    assert _status(report, "classic bug") == PASS
    assert report.ready


def test_covered_ids_without_item_index_are_accepted():
    report = _validate([_hit("answer:1", item_kind="summary",
                             item_covered_ids=["a:0"])])
    assert _status(report, "Derived-result") == PASS
    assert _status(report, "classic bug") == PASS


def test_a_gap_declaring_coverage_fails():
    report = _validate([_hit("gap:1", item_kind="gap", item_covered_ids=["a:0"])])
    assert _status(report, "Derived-result") == FAIL
    assert not report.ready


def test_covered_ids_given_as_a_string_fail():
    # The classic-bug check falls back to the row id for a non-list and would
    # miss this; a host iterating the string would credit single characters.
    report = _validate([_hit("a:0", item_covered_ids="a:1")])
    assert _status(report, "Derived-result") == FAIL


def test_rows_of_one_item_that_disagree_fail():
    report = _validate([
        _hit("a:0", rank=0, item_index=0, item_kind="summary", item_covered_ids=["a:0"]),
        _hit("a:1", rank=1, item_index=0, item_kind="summary",
             item_covered_ids=["a:0", "a:1"]),
    ])
    assert _status(report, "Derived-result") == FAIL


def test_zero_coverage_that_is_not_a_gap_warns():
    report = _validate([_hit("answer:1", item_kind="group", item_covered_ids=[])])
    assert _status(report, "Derived-result") == WARN
    detail = " ".join(next(c for c in report.checks if "Derived" in c.name).details)
    assert "item_kind='gap'" in detail
    assert report.ready  # legal; only a warning


def test_a_summary_borrowing_a_real_chunk_id_warns():
    report = _validate([_hit("a:0", item_kind="summary", item_covered_ids=["a:0", "a:1"])])
    assert _status(report, "Derived-result") == WARN


def test_an_undeclared_kind_warns_with_the_default_it_was_read_as():
    report = _validate([_hit("answer:1", item_covered_ids=["a:0"])])
    check = next(c for c in report.checks if "Derived" in c.name)
    assert check.status == WARN
    assert any("read as 'summary'" in d for d in check.details)
