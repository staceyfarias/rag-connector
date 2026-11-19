"""Per-call filters narrow the open scope (never widen it); a missing index is one typed
error naming one build command; index_status says what exists and what builds it."""

from __future__ import annotations

import json

import pytest
from test_search_index import MOD_A, MOD_B, _copies_records, _records, _write

import rag_connector.search_index as si
from rag_connector.cli import main
from rag_connector.index_status import index_status, signal_buildable

A_OR_B = {"task_id": {"$in": [MOD_A, MOD_B]}}
COUNT = "SELECT COUNT(*) FROM chunks"


@pytest.fixture
def dataset(tmp_path):
    return _write(tmp_path / "ds", _records() + _copies_records())


@pytest.fixture
def built(dataset):
    si.build(dataset)
    return dataset


def _count(index, **kw):
    return index.sql(COUNT, **kw)["rows"][0][0]


def _modules(index, **kw):
    return {row["task_id"] for row in index.modules(**kw)}


# -- per-call filter ----------------------------------------------------------

def test_a_call_filter_narrows_the_open_scope_for_that_call_only(built):
    with si.open_index(built, filter=A_OR_B) as index:
        whole = _count(index)
        assert _modules(index) == {MOD_A, MOD_B}
        assert _modules(index, filter={"task_id": MOD_A}) == {MOD_A}
        assert _count(index, filter={"task_id": MOD_A}) < whole
        assert _count(index) == whole                      # restored after the call
        assert index.filter == A_OR_B


def test_a_call_filter_never_widens_the_open_scope(built):
    with si.open_index(built, filter={"task_id": MOD_A}) as index:
        assert _count(index, filter={"task_id": MOD_B}) == 0
        assert _count(index, filter={}) == _count(index)   # {} adds nothing


def test_a_call_filter_scopes_an_unscoped_index_and_its_sql(built):
    with si.open_index(built) as index:
        assert not index.scoped
        everything = _count(index)
        assert _modules(index, filter={"task_id": MOD_A}) == {MOD_A}
        with pytest.raises(si.SearchIndexError):            # main.* refused while narrowed
            index.sql("SELECT COUNT(*) FROM main.chunks", filter={"task_id": MOD_A})
        assert not index.scoped and _count(index) == everything


def test_narrowing_works_after_sql_left_the_connection_query_only(built):
    with si.open_index(built, filter=A_OR_B) as index:
        _count(index)                                       # sets query_only
        assert _modules(index, filter={"task_id": MOD_B}) == {MOD_B}
        assert _modules(index) == {MOD_A, MOD_B}


def test_search_and_lookup_take_a_call_filter(built):
    with si.open_index(built) as index:
        hits = index.search(any=["the"], filter={"task_id": MOD_B}).hits
        assert hits and all(hit["task_id"] == MOD_B for hit in hits)
        assert isinstance(index.lookup("a", filter={"task_id": MOD_A}), dict)


def test_a_bad_call_filter_is_an_error_and_leaves_the_scope_as_it_was(built):
    from rag_connector.errors import UnsupportedFilterError

    with si.open_index(built, filter=A_OR_B) as index:
        whole = _count(index)
        with pytest.raises(UnsupportedFilterError):
            index.modules(filter={"no_such_field": "x"})
        assert _count(index) == whole


# -- missing index ------------------------------------------------------------

def test_a_missing_index_is_one_typed_error_naming_index_prepare(dataset):
    with pytest.raises(si.SearchIndexMissing) as err:
        si.open_index(dataset)
    assert "index prepare" in str(err.value) and "no model" in str(err.value)


@pytest.mark.parametrize("command", [["search", "--term", "x"], ["lookup", "x"],
                                     ["sql", "SELECT 1"]])
def test_every_index_command_fails_the_same_way_without_an_index(dataset, command, capsys):
    assert main(["index", command[0], "--dataset", str(dataset), *command[1:]]) != 0
    err = capsys.readouterr().err
    assert "SearchIndexMissing" in err and "index prepare" in err


# -- status -------------------------------------------------------------------

def test_status_before_and_after_building(dataset, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    before = index_status(dataset)
    assert before["search_index"]["state"] == "missing"
    assert before["book_index"]["state"] == "missing"
    assert not before["ready_to_search"]
    assert "index prepare" in before["build"] and before["build_uses_model"] is False
    assert before["signal"]["buildable"] is False
    assert "ANTHROPIC_API_KEY is not set" in before["signal"]["reason"]

    assert main(["index", "prepare", "--dataset", str(dataset)]) == 0
    after = index_status(dataset)
    assert after["search_index"]["state"] == "current"
    assert after["book_index"]["state"] == "current"
    assert after["ready_to_search"] and after["build"] is None


def test_status_reports_a_stale_search_index(built):
    meta_rows = si.SearchIndex(si.search_db_path(built))
    meta_rows.close()
    chunks = built / "data" / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] += " changed"
    chunks.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    status = index_status(built)
    assert status["search_index"]["state"] == "stale" and status["search_index"]["reasons"]
    assert not status["ready_to_search"] and status["build"]


def test_signal_buildable_never_returns_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    found = signal_buildable()
    assert "sk-test" not in json.dumps(found)


def test_status_command_prints_and_emits_json(dataset, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["index", "status", "--dataset", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "search index: missing" in out and "index prepare" in out
    assert "cannot be built here" in out
    assert main(["index", "status", "--dataset", str(dataset), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["search_index"]["state"] == "missing"
