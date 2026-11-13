"""Scoped lookups: one neutral metadata filter applied to every search-index method,
free-form SQL included, and the book index stored in the search index."""

from __future__ import annotations

import json
import sqlite3

import pytest
from test_search_index import MOD_A, MOD_B, _copies_records, _records, _write

import rag_connector.corpus_index as corpus_index
import rag_connector.search_index as si
from rag_connector.base import ChunkRecord
from rag_connector.cli import main
from rag_connector.errors import InvalidFilterError, UnsupportedFilterError
from rag_connector.filters import SQL_INTEGER, SQL_TEXT, to_sql_where

A_OR_GENERAL = {"$or": [{"task_id": MOD_A}, {"task_id": {"$exists": False}}]}


@pytest.fixture
def dataset(tmp_path):
    return _write(tmp_path / "ds", _records() + _copies_records())


@pytest.fixture
def built(dataset):
    si.build(dataset)
    return dataset


def _open(folder, filt):
    return si.open_index(folder, filter=filt)


def _ids(result):
    return {hit["chunk_id"] for hit in result.hits}


# ---------------------------------------------------------------------------
# The filter as SQL
# ---------------------------------------------------------------------------

_COLUMNS = {"k": SQL_TEXT, "n": SQL_INTEGER}


def _matching(filt):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id TEXT, k TEXT, n INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?,?,?)",
                     [("a", "x", 1), ("b", "y", 2), ("c", None, 3), ("d", "it's", None)])
    where = to_sql_where(filt, _COLUMNS)
    return sorted(r[0] for r in conn.execute(f"SELECT id FROM t WHERE {where}"))


def test_filter_translates_with_mongo_semantics_for_missing_values():
    assert _matching(None) == ["a", "b", "c", "d"]
    assert _matching({"k": "x"}) == ["a"]
    assert _matching({"k": {"$ne": "x"}}) == ["b", "c", "d"]        # missing counts as "not x"
    assert _matching({"k": {"$in": ["x", "y"]}}) == ["a", "b"]
    assert _matching({"k": {"$nin": ["x"]}}) == ["b", "c", "d"]
    assert _matching({"k": {"$exists": False}}) == ["c"]
    assert _matching({"k": {"$exists": True}}) == ["a", "b", "d"]
    assert _matching({"k": "it's"}) == ["d"]                        # quoted, not interpolated
    assert _matching({"n": {"$gte": 2, "$lt": 3}}) == ["b"]
    assert _matching({"$or": [{"k": "x"}, {"n": 3}]}) == ["a", "c"]
    assert _matching({"$and": [{"k": {"$exists": True}}, {"n": {"$gt": 1}}]}) == ["b"]
    assert _matching({"k": "x", "n": 2}) == []


def test_filter_refuses_what_it_cannot_honour():
    with pytest.raises(UnsupportedFilterError) as unknown:
        to_sql_where({"colour": "red"}, _COLUMNS)
    assert unknown.value.fields == {"colour"}
    with pytest.raises(UnsupportedFilterError):
        to_sql_where({"k": {"$gt": 3}}, _COLUMNS)                   # range on text
    with pytest.raises(UnsupportedFilterError):
        to_sql_where({"n": "3"}, _COLUMNS)                          # text against a number
    with pytest.raises(InvalidFilterError):
        to_sql_where({"k": None}, _COLUMNS)
    with pytest.raises(InvalidFilterError):
        to_sql_where({"k": "a\x00b"}, _COLUMNS)


# ---------------------------------------------------------------------------
# Every method under a filter
# ---------------------------------------------------------------------------

def test_an_unscoped_index_reports_an_empty_filter(built):
    with si.open_index(built) as index:
        assert index.filter == {} and not index.scoped
        result = index.search(all="face")
        assert result.filter == {} and result.to_dict()["filter"] == {}


def test_unknown_fields_and_bad_filters_are_refused_at_open(built):
    with pytest.raises(UnsupportedFilterError):
        _open(built, {"metadata.region": "south"})
    with pytest.raises(UnsupportedFilterError):
        _open(built, {"task_id": {"$gt": 3}})
    with pytest.raises(InvalidFilterError):
        _open(built, {"task_id": ["a", "b"]})


def test_search_sees_only_the_scope_and_records_it(built):
    with _open(built, {"task_id": MOD_A}) as index:
        assert index.scoped and index.filter == {"task_id": MOD_A}
        result = index.search(any=["face", "parking", "members"])
        assert _ids(result) == {"c-guide-0", "c-guide-1"}
        assert result.filter == {"task_id": MOD_A}
        assert result.to_dict()["filter"] == {"task_id": MOD_A}
        assert {m["task_id"] for m in result.by_module} == {MOD_A}
        # Substring, regex and filter-only listings go through the same rows.
        assert _ids(index.search(substring="$18")) == {"c-guide-1"}
        assert _ids(index.search(regex=r"opens at nine")) == set()
        assert {h["chunk_id"] for h in index.search(types="pdf").hits} == {
            "c-dose-0", "c-dose-1"}
        assert [m["task_id"] for m in index.modules()] == [MOD_A]
    with _open(built, A_OR_GENERAL) as index:
        result = index.search(any=["face", "parking", "members"])
        assert _ids(result) == {"c-guide-0", "c-guide-1", "c-gen-0"}
        summary = index.scope_summary()
        assert summary["chunks"] == 5 + 2 + 7 and summary["dataset_chunks"] == 15
        assert summary["general_chunks"] == 9 and summary["filter"] == A_OR_GENERAL


def test_ids_outside_the_scope_say_so(built):
    with _open(built, {"task_id": MOD_A}) as index:
        with pytest.raises(si.OutOfScopeError, match="outside the scope"):
            index.chunk("c-faq-0")
        with pytest.raises(si.AmbiguousIdError):                     # a subclass
            index.neighbors("c-faq")
        with pytest.raises(si.OutOfScopeError):
            index.search(all="parking", modules=MOD_B)
        with pytest.raises(si.OutOfScopeError):
            index.outline("FAQ")                                    # a title
        with pytest.raises(si.OutOfScopeError):
            index.outline("faq")                                    # a doc id
        with pytest.raises(si.AmbiguousIdError, match="no chunk id"):
            index.chunk("no-such-chunk")
        assert index.resolve_module("aaaa") == MOD_A                # no longer ambiguous


def test_a_filter_splitting_a_document_skips_its_out_of_scope_chunks(built):
    with _open(built, {"doc_chunk_index": {"$ne": 1}}) as index:
        around = index.neighbors("c-guide-0", radius=1)
        assert [item["chunk_id"] for item in around] == ["c-guide-0", "c-guide-2"]
        assert [item["doc_chunk_index"] for item in index.outline("guide")] == [0, 2]
    with _open(built, {"doc_id": {"$in": ["guide", "dosage"]}}) as index:
        assert len(index.outline("guide")) == 3                     # whole documents


def test_copies_list_only_copies_in_scope(built):
    with si.open_index(built) as index:
        assert [c["chunk_id"] for c in index.copies("x-one-0")["identical"]] == ["x-two-0"]
        assert index.copies("x-pa-0")["overlap"]
    with _open(built, {"doc_id": {"$nin": ["two", "pb"]}}) as index:
        found = index.copies("x-one-0")
        assert found["identical"] == []
        assert [c["chunk_id"] for c in found["near"]] == ["x-three-0"]
        assert {c["chunk_id"] for c in index.copies("x-pa-0")["overlap"]} <= {"x-pa-1"}
        hit = next(h for h in index.search(substring="Visitors must").hits
                   if h["chunk_id"] == "x-one-0")
        assert "identical" not in hit["copies"]                     # its group is out of scope


# ---------------------------------------------------------------------------
# Free-form SQL under a filter
# ---------------------------------------------------------------------------

def test_sql_reads_the_scoped_views(built):
    with _open(built, {"task_id": MOD_A}) as index:
        assert index.sql("SELECT count(*) FROM chunks")["rows"] == [[5]]
        assert index.sql("SELECT count(*) FROM chunks_plain")["rows"] == [[5]]
        assert index.sql("SELECT count(*) FROM temp.chunks")["rows"] == [[5]]
        assert index.sql("SELECT count(*) FROM copy_groups")["rows"] == [[0]]
        fts = index.sql("SELECT c.chunk_id FROM chunks_fts JOIN chunks c "
                        "ON c.n = chunks_fts.rowid WHERE chunks_fts MATCH 'members'")
        assert fts["rows"] == []
        found = index.sql("SELECT chunk_id FROM chunks_plain WHERE contains(plain, '$18')")
        assert found["rows"] == [["c-guide-1"]]
        assert "SCOPED" in index.schema() and MOD_A in index.schema()


def test_sql_refuses_the_files_own_tables_and_schema_changes(built):
    with _open(built, {"task_id": MOD_A}) as index:
        for statement in ("SELECT count(*) FROM main.chunks",
                          "SELECT 1 FROM main.chunks",
                          "SELECT chunk_id FROM main.chunks_plain",
                          "SELECT * FROM main.copy_groups",
                          "SELECT * FROM main.index_locations",
                          "SELECT * FROM chunks c JOIN main.chunks m ON m.n = c.n",
                          "DROP VIEW chunks", "DROP VIEW temp.chunks",
                          "CREATE TEMP VIEW x AS SELECT * FROM main.chunks",
                          "PRAGMA query_only = OFF",
                          "ATTACH 'other.db' AS other"):
            with pytest.raises(si.SearchIndexError):
                index.sql(statement)
        # Still scoped and usable afterwards.
        assert index.sql("SELECT count(*) FROM chunks")["rows"] == [[5]]
        assert index.search(all="parking").total == 0


def test_unscoped_sql_still_reads_main(built):
    with si.open_index(built) as index:
        assert index.sql("SELECT count(*) FROM main.chunks")["rows"] == [[15]]


# ---------------------------------------------------------------------------
# The book index in the search index
# ---------------------------------------------------------------------------

_FERRY = {
    "fares.md": ["# Fares\n\nTidewater Ferries sails from Portmere to Skerry Isle.",
                 "Children under 12 travel at half fare from Portmere."],
    "baggage.md": ["# Baggage\n\nOne bag is free.",
                   "Bicycles are not permitted on the Night Crossing service."],
    "faq.md": ["# Frequently asked questions\n\nHow much luggage can I take?",
               "Bicycles are allowed on the Night Crossing for a fee."],
}


def _ferry(tmp_path, extra=()):
    records = [ChunkRecord(chunk_id=f"{doc}:chunk-{i}", doc_id=doc, source_file=doc,
                           doc_chunk_index=i, global_index=None, text=text,
                           metadata={"task_id": "boats" if doc != "faq.md" else "help"})
               for doc, texts in _FERRY.items() for i, text in enumerate(texts)]
    return _write(tmp_path / "ferry", records + list(extra), name="Ferry")


def test_lookup_reads_the_same_book_index_as_index_json(tmp_path):
    folder = _ferry(tmp_path)
    corpus_index.prepare(folder)
    si.build(folder)
    book = corpus_index.load(folder)
    with si.open_index(folder) as index:
        assert index.has_book_index
        for query in ("Night Crossing", "Portmere", "fares", "bicycles", "a", "zzz"):
            assert index.lookup(query) == corpus_index.find(book, query), query
    assert corpus_index.lookup(folder, "Portmere") == corpus_index.find(book, "Portmere")


def test_lookup_is_scoped(tmp_path):
    folder = _ferry(tmp_path)
    si.build(folder)
    with _open(folder, {"task_id": "help"}) as index:
        night = index.lookup("Night Crossing")["entries"]
        assert [loc["chunk_id"] for loc in night[0]["locations"]] == ["faq.md:chunk-1"]
        assert index.lookup("Portmere") == {"entries": [], "threads": []}
        assert [t["title"] for t in index.lookup("a")["threads"]] == [
            "Frequently asked questions"]
        # The SQL views agree.
        terms = {row[0] for row in index.sql("SELECT term FROM index_entries")["rows"]}
        assert "Night Crossing" in terms and "Portmere" not in terms
        assert index.sql("SELECT count(*) FROM index_thread_chunks")["rows"] == [[2]]
    found = corpus_index.lookup(folder, "Night Crossing", filter={"doc_id": "baggage.md"})
    loc = found["entries"][0]["locations"][0]
    assert loc["chunk_id"] == "baggage.md:chunk-1" and loc["prev"] == "baggage.md:chunk-0"


def test_a_split_document_gives_in_scope_neighbours(tmp_path):
    folder = _ferry(tmp_path)
    si.build(folder)
    with _open(folder, {"chunk_id": {"$ne": "fares.md:chunk-0"}}) as index:
        loc = index.lookup("Portmere")["entries"][0]["locations"]
        assert [(x["chunk_id"], x["prev"], x["next"], x["doc_chunks"]) for x in loc] == [
            ("fares.md:chunk-1", "", "", 1)]
        assert index.lookup("Fares")["threads"][0]["chunk_ids"] == ["fares.md:chunk-1"]


def test_an_out_of_scope_location_gives_way_to_an_in_scope_copy(tmp_path):
    step = "0460ec31-2d9f-4150-8381-80a73d0e12fa::b547239e-3269-474e-a22c-03038698a4ce"
    same = "# Port Rules\n\nMeet at Port St. Joe on Monday."
    extra = [
        ChunkRecord(chunk_id="copy:0", doc_id="copy", source_file=step, doc_chunk_index=0,
                    global_index=None, text=same, metadata={"task_id": "m2"}),
        ChunkRecord(chunk_id="dist:0", doc_id="dist", source_file="District 17",
                    doc_chunk_index=0, global_index=None, text=same,
                    metadata={"task_id": "m1"}),
    ]
    folder = _ferry(tmp_path, extra)
    si.build(folder)
    with si.open_index(folder) as index:
        loc = index.lookup("Port St. Joe")["entries"][0]["locations"][0]
        assert loc["chunk_id"] == "dist:0" and loc["copies"][0]["chunk_id"] == "copy:0"
        thread = index.lookup("Port Rules")["threads"][0]
        assert thread["doc_id"] == "dist" and thread["copies"][0]["doc_id"] == "copy"
    with _open(folder, {"task_id": "m2"}) as index:
        loc = index.lookup("Port St. Joe")["entries"][0]["locations"][0]
        assert loc["chunk_id"] == "copy:0" and loc["copies"] == []
        assert loc["section"] == "Port Rules"
        thread = index.lookup("Port Rules")["threads"][0]
        assert thread["doc_id"] == "copy" and thread["copies"] == []
        terms = {r[0] for r in index.sql("SELECT term FROM index_entries")["rows"]}
        assert {"Port Rules", "Port St. Joe"} <= terms and "Portmere" not in terms


def test_search_db_uses_the_stopwords_index_json_records(tmp_path):
    folder = _ferry(tmp_path)
    corpus_index.prepare(folder, stopwords=["Portmere"])
    si.build(folder)
    with si.open_index(folder) as index:
        assert json.loads(index.meta["book_index_stopwords"]) == ["portmere"]
        assert index.lookup("Portmere")["entries"] == []
    si.build(folder, stopwords=())
    with si.open_index(folder) as index:
        assert index.lookup("Portmere")["entries"]


def _make_old(path):
    """Turn a search index into what an older rag-connector wrote: format 1,
    no book-index tables."""
    conn = sqlite3.connect(path)
    for table in ("index_entries", "index_locations", "index_copies", "index_sections",
                  "index_threads", "index_thread_chunks"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'format_version'")
    conn.commit()
    conn.close()


def test_an_old_search_index_is_reported_stale(built):
    _make_old(si.search_db_path(built))
    with pytest.raises(si.StaleSearchIndex, match="book-index tables"):
        si.open_index(built)
    with si.open_index(built, allow_stale=True) as index:
        assert not index.has_book_index
        problems = index.stale_reasons(built)
        assert any("format version '1'" in p for p in problems)
        with pytest.raises(si.SearchIndexError, match="no book-index tables"):
            index.lookup("Stroke")
    # Scoping an old file still scopes what it holds.
    with si.open_index(built, allow_stale=True, filter={"task_id": MOD_B}) as index:
        assert _ids(index.search(any=["parking", "face"])) == {"c-faq-0"}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_filter_on_every_lookup(built, capsys):
    folder = str(built)
    scope = json.dumps({"task_id": MOD_A})
    assert main(["index", "search", "--dataset", folder, "--any", "face", "--any",
                 "parking", "--filter", scope, "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["filter"] == {"task_id": MOD_A} and payload["total"] == 2
    assert "scope:" in captured.err and "5 of 15 chunks" in captured.err

    assert main(["index", "sql", "--dataset", folder, "--filter", scope,
                 "SELECT count(*) FROM chunks"]) == 0
    assert "\n5\n" in capsys.readouterr().out
    assert main(["index", "sql", "--dataset", folder, "--filter", scope,
                 "SELECT count(*) FROM main.chunks"]) == 1
    capsys.readouterr()
    assert main(["index", "outline", "--dataset", folder, "--filter", scope, "faq"]) == 1
    assert "outside the scope" in capsys.readouterr().err
    assert main(["index", "neighbors", "--dataset", folder, "--filter", scope,
                 "c-guide-1"]) == 0
    assert main(["index", "copies", "--dataset", folder, "--filter", scope,
                 "c-guide-1", "--json"]) == 0
    capsys.readouterr()

    assert main(["index", "lookup", "--dataset", folder, "--filter", scope, "Stroke"]) == 0
    assert "c-guide-0" in capsys.readouterr().out
    assert main(["index", "lookup", "--dataset", folder, "--filter", scope, "Opening",
                 "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"entries": [], "threads": []}

    for bad in ('{"colour": "red"}', "not json", "[1]", '{"task_id": {"$gt": 1}}'):
        assert main(["index", "search", "--dataset", folder, "--term", "face",
                     "--filter", bad]) == 1
        assert "rag-connector index:" in capsys.readouterr().err


def test_cli_lookup_falls_back_to_index_json_only_unscoped(dataset, capsys):
    folder = str(dataset)
    assert main(["index", "prepare", "--dataset", folder, "--no-search-db"]) == 0
    capsys.readouterr()
    assert main(["index", "lookup", "--dataset", folder, "Stroke"]) == 0
    captured = capsys.readouterr()
    assert "c-guide-0" in captured.out and "index.json" in captured.err
    assert main(["index", "lookup", "--dataset", folder, "Stroke",
                 "--filter", json.dumps({"task_id": MOD_A})]) == 1
    assert "search-db" in capsys.readouterr().err
