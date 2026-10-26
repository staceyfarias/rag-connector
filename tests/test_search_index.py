"""The search index: one SQLite file of full-text indexes over a Dataset's chunks."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import rag_connector.search_index as si
from rag_connector.base import ChunkRecord
from rag_connector.cli import main
from rag_connector.dataset import list_analyses, write_dataset_folder

MOD_A = "aaaa1111-0000-4000-8000-000000000001"
MOD_B = "aaaa2222-0000-4000-8000-000000000002"

# A passage long enough to be an overlap copy: 120 distinct words.
PASSAGE = [f"lex{i}" for i in range(120)]


def _record(cid, doc, index, text, *, task=None, source=None, rtype=None, kind="content"):
    meta = {"type": kind}
    if task:
        meta["task_id"] = task
        meta["step_id"] = f"step-{doc}"
    if rtype:
        meta["resource_type"] = rtype
    meta["chunk_label"] = f"{index + 1} / x"
    return ChunkRecord(chunk_id=cid, doc_id=doc, source_file=source or doc,
                       doc_chunk_index=index, global_index=None, text=text,
                       content_module_id=task, metadata=meta)


def _records():
    return [
        # Module A: a stroke guide, written out of order on purpose.
        _record("c-guide-2", "guide", 2, "The study average was 41.5 percent of patients.",
                task=MOD_A, source="Stroke Guide", rtype="url"),
        _record("c-guide-0", "guide", 0,
                "# Stroke signs\n\nKnow the BE-FAST signs: balance, eyes, face, arm, "
                "speech, time. Call for help.", task=MOD_A, source="Stroke Guide",
                rtype="url"),
        _record("c-guide-1", "guide", 1,
                "Face drooping and arm weakness often come first.\n"
                "The clinic fee is \\$18 per visit.", task=MOD_A, source="Stroke Guide",
                rtype="url"),
        # Module A: a heading whose section continues into the next chunk.
        _record("c-dose-0", "dosage", 0, "## Dosage\n\nTake two tablets",
                task=MOD_A, source="Medicine Sheet", rtype="pdf"),
        _record("c-dose-1", "dosage", 1, "with water after meals, twice a day.",
                task=MOD_A, source="Medicine Sheet", rtype="pdf"),
        # Module B: a FAQ with a question line and a bold heading.
        _record("c-faq-0", "faq", 0,
                "How much does parking cost?\nParking is free for members.\n\n"
                "**Opening hours**\nThe office opens at nine.", task=MOD_B, source="FAQ",
                rtype="video"),
        # General info: no module.
        _record("c-gen-0", "news", 0, "Members summarized the meeting and carried "
                "the motion forward to the next session.", source="Newsletter",
                kind="general_info"),
        _record("c-gen-1", "news", 1, "Prior conversations are summarized and carried "
                "forward between sessions.", source="Newsletter", kind="general_info"),
    ]


def _copies_records():
    """Identical, near and overlapping copies across documents."""
    same = "Visitors must sign in at the front desk before entering the ward."
    near = "VISITORS must sign in, at the front desk -- before entering the ward!"
    a0 = " ".join(PASSAGE[:100])
    a1 = " ".join(PASSAGE[80:120])
    b0 = "intro words here " + " ".join(PASSAGE[:90])
    b1 = " ".join(PASSAGE[90:120]) + " closing remarks follow"
    return [
        _record("x-one-0", "one", 0, same, source="Ward Rules"),
        _record("x-two-0", "two", 0, same, source="Ward Rules Copy"),
        _record("x-three-0", "three", 0, near, source="Ward Rules Old"),
        _record("x-pa-0", "pa", 0, a0, source="Passage A"),
        _record("x-pa-1", "pa", 1, a1, source="Passage A"),
        _record("x-pb-0", "pb", 0, b0, source="Passage B"),
        _record("x-pb-1", "pb", 1, b1, source="Passage B"),
    ]


def _write(path, records, name="Synthetic"):
    write_dataset_folder(path, records, name=name, chunking={"method": "hand-written-fixture"},
                         created_at="2026-10-26T00:00:00Z")
    return path


@pytest.fixture
def dataset(tmp_path):
    return _write(tmp_path / "ds", _records() + _copies_records())


@pytest.fixture
def index(dataset):
    si.build(dataset)
    with si.open_index(dataset) as opened:
        yield opened


def _ids(result):
    return {hit["chunk_id"] for hit in result.hits}


def _core_digest(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("dataset.json", "data/chunks.jsonl", "data/sources.json")}


# ---------------------------------------------------------------------------
# Building, location, staleness
# ---------------------------------------------------------------------------

def test_build_writes_the_analysis_area_and_never_the_core(dataset):
    before = _core_digest(dataset)
    path = si.build(dataset)
    assert path == dataset / "analysis" / "rag-connector" / "search.sqlite"
    assert _core_digest(dataset) == before
    [envelope] = list_analyses(dataset)
    assert envelope.problems == ()
    assert {"path": "search.sqlite", "format": "sqlite"}.items() <= next(
        a for a in envelope.artifacts if a["path"] == "search.sqlite").items()
    with si.open_index(dataset) as index:
        manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
        assert index.meta["core_sha256"] == manifest["core_sha256"]
        assert index.meta["chunk_inventory_sha256"] == manifest["chunk_inventory_sha256"]
        assert index.meta["chunk_count"] == "15"


def test_out_builds_elsewhere_and_writes_nothing_in_the_dataset(dataset, tmp_path):
    out_dir = tmp_path / "elsewhere"
    out_dir.mkdir()
    path = si.build(dataset, out=out_dir)
    assert path == out_dir / "search.sqlite"
    assert not (dataset / "analysis").exists()
    with pytest.raises(si.SearchIndexMissing):
        si.open_index(dataset)
    with si.open_index(dataset, path=path) as index:
        assert index.search(all=["dosage"]).total == 1


def test_an_index_from_another_core_is_stale(dataset, tmp_path):
    path = si.build(dataset, out=tmp_path / "a.sqlite")
    other = _write(tmp_path / "other", _records()[:3], name="Other")
    with pytest.raises(si.StaleSearchIndex, match="core"):
        si.open_index(other, path=path)
    with si.open_index(other, path=path, allow_stale=True) as index:
        assert index.stale_reasons(other)

    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = ? WHERE key = 'core_sha256'", ("0" * 64,))
    conn.commit()
    conn.close()
    with pytest.raises(si.StaleSearchIndex):
        si.open_index(dataset, path=path)


def test_rebuilding_is_idempotent(dataset):
    def dump():
        conn = sqlite3.connect(si.search_db_path(dataset))
        try:
            return (conn.execute("SELECT * FROM chunks ORDER BY n").fetchall(),
                    conn.execute("SELECT * FROM copy_groups ORDER BY group_id, chunk_id")
                    .fetchall(),
                    conn.execute("SELECT * FROM overlap_pairs").fetchall())
        finally:
            conn.close()
    si.build(dataset)
    first = dump()
    si.build(dataset)
    assert dump() == first


def test_verify_rereads_the_dataset(index, dataset):
    index.verify(dataset)


# ---------------------------------------------------------------------------
# Word search
# ---------------------------------------------------------------------------

def test_and_or_not(index):
    assert _ids(index.search(all=["face", "arm"])) == {"c-guide-0", "c-guide-1"}
    assert _ids(index.search(all=["face"], not_=["drooping"])) == {"c-guide-0"}
    assert _ids(index.search(any=["parking", "dosage"])) == {"c-faq-0", "c-dose-0"}
    assert _ids(index.search(any=["parking", "dosage"], not_=["free"])) == {"c-dose-0"}
    assert _ids(index.search(not_=["face"], modules=[MOD_A])) == {
        "c-guide-2", "c-dose-0", "c-dose-1"}


def test_phrase_and_prefix(index):
    assert _ids(index.search(phrase="summarized and carried forward")) == {"c-gen-1"}
    assert _ids(index.search(all=["summarized and carried forward"])) == {"c-gen-1"}
    assert _ids(index.search(all=["droop*"])) == {"c-guide-1"}
    assert index.search(all=["droop"]).total == 0


def test_near(index):
    assert _ids(index.search(near=("summarized", "forward", 3))) == {"c-gen-1"}
    assert _ids(index.search(near=("summarized", "forward", 10))) == {"c-gen-0", "c-gen-1"}
    with pytest.raises(si.SearchIndexError):
        index.search(near=("summarized", 3))


def test_titles_are_searched_only_when_asked(index):
    assert index.search(all=["newsletter"]).total == 0
    assert _ids(index.search(all=["newsletter"], titles=True)) == {"c-gen-0", "c-gen-1"}


def test_user_text_is_never_fts_syntax(index):
    for term in ['face" OR "arm', "NEAR(face arm)", "face AND", "plain: face", "a*b"]:
        index.search(all=[term])
    with pytest.raises(si.SearchIndexError):
        index.search(all=["$$"])


def test_hits_are_ranked_and_carry_context(index):
    result = index.search(all=["arm"])
    assert [hit["score"] for hit in result.hits] == sorted(
        (hit["score"] for hit in result.hits), reverse=True)
    hit = next(h for h in result.hits if h["chunk_id"] == "c-guide-1")
    assert hit["doc_id"] == "guide" and hit["doc_chunk_index"] == 1
    assert hit["task_id"] == MOD_A and hit["resource_type"] == "url"
    assert hit["source"] == "Stroke Guide" and "arm weakness" in hit["snippet"]


# ---------------------------------------------------------------------------
# Regex and substrings
# ---------------------------------------------------------------------------

def test_regex(index):
    assert _ids(index.search(regex=r"\b\d+\.\d\b")) == {"c-guide-2"}
    assert _ids(index.search(regex=r"be-fast")) == {"c-guide-0"}
    assert index.search(regex=r"(?-i:be-fast)").total == 0
    with pytest.raises(si.SearchIndexError, match="regular expression"):
        index.search(regex="(")


def test_substring_finds_what_word_search_splits(index):
    assert _ids(index.search(substring="$18")) == {"c-guide-1"}
    assert _ids(index.search(substring="41.5")) == {"c-guide-2"}
    assert _ids(index.search(substring="be-fast")) == {"c-guide-0"}
    assert _ids(index.search(substring="arkin")) == {"c-faq-0"}
    # Shorter than a trigram: checked on every chunk instead.
    assert _ids(index.search(substring="$1")) == {"c-guide-1"}


def test_terms_as_substrings(index):
    assert _ids(index.search(all=["droop"], word=False)) == {"c-guide-1"}
    assert _ids(index.search(any=["BE-FAST", "$18"], word=False)) == {
        "c-guide-0", "c-guide-1"}
    assert _ids(index.search(all=["face"], not_=["$18"], word=False)) == {"c-guide-0"}


def test_text_is_kept_as_written_and_searched_unescaped(index):
    row = index.chunk("c-guide-1")
    assert "\\$18" in row["text"] and "$18" in row["plain"] and "\\$" not in row["plain"]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_module_prefix_and_ambiguity(index):
    assert _ids(index.search(all=["parking"], modules=["aaaa2222"])) == {"c-faq-0"}
    assert index.search(all=["parking"], modules=["aaaa1111"]).total == 0
    with pytest.raises(si.AmbiguousIdError, match="ambiguous"):
        index.search(all=["face"], modules=["aaaa"])
    with pytest.raises(si.AmbiguousIdError, match="no module"):
        index.search(all=["face"], modules=["zzzz"])


def test_general_only(index):
    assert _ids(index.search(all=["summarized"], general_only=True)) == {"c-gen-0", "c-gen-1"}
    assert index.search(all=["face"], general_only=True).total == 0
    both = index.search(any=["summarized", "parking"], modules=["aaaa2222"],
                        general_only=True)
    assert _ids(both) == {"c-gen-0", "c-gen-1", "c-faq-0"}


def test_doc_and_type_filters(index):
    assert _ids(index.search(all=["face"], docs=["guide"])) == {"c-guide-0", "c-guide-1"}
    assert _ids(index.search(all=["face"], docs=["Stroke Guide"])) == {
        "c-guide-0", "c-guide-1"}
    assert index.search(all=["face"], exclude_docs=["guide"]).total == 0
    assert _ids(index.search(any=["tablets", "face"], types=["pdf"])) == {"c-dose-0"}
    assert _ids(index.search(all=["summarized"], types=["general_info"])) == {
        "c-gen-0", "c-gen-1"}


def test_no_silent_truncation(tmp_path):
    records = [_record(f"many-{i:02d}", "many", i, f"Shared token alpha, item {i}.")
               for i in range(30)]
    folder = _write(tmp_path / "many", records)
    si.build(folder)
    with si.open_index(folder) as index:
        everything = index.search(all=["alpha"])
        assert everything.total == 30 and len(everything.hits) == 30
        assert not everything.truncated
        some = index.search(all=["alpha"], limit=5)
        assert some.total == 30 and len(some.hits) == 5 and some.truncated
        assert some.by_doc == [{"doc_id": "many", "source": "many", "task_id": None,
                                "hits": 30}]


def test_filters_alone_list_chunks_in_document_order(index):
    result = index.search(docs=["guide"])
    assert [hit["chunk_id"] for hit in result.hits] == ["c-guide-0", "c-guide-1", "c-guide-2"]
    with pytest.raises(si.SearchIndexError):
        index.search()


# ---------------------------------------------------------------------------
# Copies, neighbours, outline, headings
# ---------------------------------------------------------------------------

def test_identical_and_near_copies(index):
    found = index.copies("x-one-0")
    assert [c["chunk_id"] for c in found["identical"]] == ["x-two-0"]
    assert [c["chunk_id"] for c in found["near"]] == ["x-two-0", "x-three-0"] or \
        {c["chunk_id"] for c in found["near"]} == {"x-two-0", "x-three-0"}
    near_only = index.copies("x-three-0")
    assert near_only["identical"] == []
    assert {c["chunk_id"] for c in near_only["near"]} == {"x-one-0", "x-two-0"}
    hit = next(h for h in index.search(all=["visitors"]).hits if h["chunk_id"] == "x-one-0")
    assert hit["copies"]["identical"]["size"] == 2 and hit["copies"]["near"]["size"] == 3


def test_overlap_copies_with_shifted_boundaries(index):
    found = index.copies("x-pa-0")
    overlap = {c["chunk_id"]: c["containment"] for c in found["overlap"]}
    assert "x-pb-0" in overlap and overlap["x-pb-0"] >= si.OVERLAP_CONTAINMENT
    assert found["identical"] == [] and found["near"] == []
    # The second halves overlap too (both hold lex90..lex119).
    assert "x-pb-1" in {c["chunk_id"] for c in index.copies("x-pa-1")["overlap"]}
    # Short chunks are never overlap copies.
    assert index.copies("c-dose-1")["overlap"] == []
    hit = next(h for h in index.search(all=["lex5"]).hits if h["chunk_id"] == "x-pa-0")
    assert hit["copies"]["overlap"]["size"] == 2


def test_boilerplate_inside_long_chunks_does_not_chain_them(tmp_path):
    footer = " ".join(f"foot{i}" for i in range(40))
    long_a = " ".join(f"alpha{i}" for i in range(160)) + " " + footer
    long_b = " ".join(f"beta{i}" for i in range(160)) + " " + footer
    rows = [{"chunk_id": cid, "doc_id": cid, "source_file": cid, "doc_chunk_index": 0,
             "text": text} for cid, text in (("foot", footer), ("a", long_a), ("b", long_b))]
    with si.SearchIndex(si.write_search_db(rows, tmp_path / "f.sqlite")) as index:
        found = index.copies("foot")
        assert {c["chunk_id"]: (c["grouped"], c["containment"]) for c in found["overlap"]} == {
            "a": (False, 1.0), "b": (False, 1.0)}
        assert all(c["containment_larger"] < si.OVERLAP_GROUP_LARGER
                   for c in found["overlap"])
        assert [c["chunk_id"] for c in index.copies("a")["overlap"]] == ["foot"]
        assert index.conn.execute(
            "SELECT COUNT(*) FROM copy_groups WHERE kind = 'overlap'").fetchone()[0] == 0


def test_neighbours_and_outline_are_in_document_order(index):
    around = index.neighbours("c-guide-1", radius=1)
    assert [item["chunk_id"] for item in around] == ["c-guide-0", "c-guide-1", "c-guide-2"]
    assert [item["target"] for item in around] == [False, True, False]
    assert [item["chunk_id"] for item in index.neighbors("c-guide-0", radius=5)] == [
        "c-guide-0", "c-guide-1", "c-guide-2"]
    outline = index.outline("guide")
    assert [item["doc_chunk_index"] for item in outline] == [0, 1, 2]
    assert outline[0]["heading"] == "Stroke signs"
    assert outline[1]["heading"] is None and outline[1]["heading_before"] == "Stroke signs"
    assert [item["chunk_id"] for item in index.outline("c-guide-2")] == [
        "c-guide-0", "c-guide-1", "c-guide-2"]


def test_chunk_prefixes_resolve_or_refuse(index):
    assert index.chunk("c-dose-1")["chunk_id"] == "c-dose-1"
    assert index.resolve_chunk("c-faq") == "c-faq-0"
    with pytest.raises(si.AmbiguousIdError):
        index.resolve_chunk("c-guide")


def test_heading_from_the_previous_chunk(index):
    [hit] = index.search(all=["water"]).hits
    assert hit["chunk_id"] == "c-dose-1" and hit["heading"] == "Dosage"


def test_question_and_bold_lines_are_headings(index):
    [parking] = index.search(all=["free"]).hits
    assert parking["heading"] == "How much does parking cost?"
    [hours] = index.search(all=["nine"]).hits
    assert hours["heading"] == "Opening hours"


def test_wrapped_transcript_lines_are_not_headings():
    text = ("[00:12] spk_1: and so we asked them what they were\nspeaking?\n"
            "Then it ended. Why?\n[00:15] spk_2: Is this a question?")
    assert si.headings_in(text) == []


# ---------------------------------------------------------------------------
# Metadata shapes
# ---------------------------------------------------------------------------

def test_metadata_as_a_python_literal_string(tmp_path):
    rows = [
        ChunkRecord(chunk_id="s-0", doc_id="d", source_file="Sheet", doc_chunk_index=0,
                    global_index=None, text="Literal metadata row about tablets.",
                    metadata=f"{{'task_id': '{MOD_A}', 'resource_type': 'pdf', "
                             "'reachable': True, 'chunk_label': '1 / 2'}"),
        {"chunk_id": "s-1", "doc_id": "d", "source_file": "Sheet", "doc_chunk_index": 1,
         "text": "JSON metadata row about tablets.",
         "metadata": json.dumps({"task_id": MOD_B, "type": "content"})},
        {"chunk_id": "s-2", "doc_id": "d", "source_file": "Sheet", "doc_chunk_index": 2,
         "text": "Nested metadata row about tablets.",
         "metadata": {"metadata": f"{{'task_id': '{MOD_A}'}}", "type": "content"}},
        {"chunk_id": "s-3", "doc_id": "d", "source_file": "Sheet", "doc_chunk_index": 3,
         "text": "Unparseable metadata row about tablets.", "metadata": "not a dict"},
    ]
    path = si.write_search_db(rows, tmp_path / "s.sqlite")
    with si.SearchIndex(path) as index:
        first = index.chunk("s-0")
        assert first["task_id"] == MOD_A and first["resource_type"] == "pdf"
        assert first["metadata"]["reachable"] is True
        assert index.chunk("s-1")["task_id"] == MOD_B
        assert index.chunk("s-2")["task_id"] == MOD_A
        assert index.chunk("s-3")["metadata"] == {"_raw": "not a dict"}
        assert _ids(index.search(all=["tablets"], modules=["aaaa1111"])) == {"s-0", "s-2"}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_build_search_and_browse(dataset, capsys):
    folder = str(dataset)
    assert main(["index", "search", "--dataset", folder, "--term", "face"]) == 1
    assert "search-db" in capsys.readouterr().err
    assert main(["index", "search-db", "--dataset", folder]) == 0
    assert "15 chunks" in capsys.readouterr().out

    assert main(["index", "search", "--dataset", folder, "--term", "face",
                 "--not", "drooping"]) == 0
    out = capsys.readouterr().out
    assert "1 chunks match" in out and "c-guide-0" in out and "under: Stroke signs" in out

    assert main(["index", "search", "--dataset", folder, "--near", "summarized",
                 "forward", "--within", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1 and payload["hits"][0]["chunk_id"] == "c-gen-1"

    assert main(["index", "search", "--dataset", folder, "--substring", "$18",
                 "--module", "aaaa1111", "--limit", "1"]) == 0
    assert "c-guide-1" in capsys.readouterr().out

    assert main(["index", "search", "--dataset", folder, "--term", "face",
                 "--module", "aaaa"]) == 1
    assert "ambiguous" in capsys.readouterr().err

    assert main(["index", "neighbours", "--dataset", folder, "c-guide-1"]) == 0
    assert "<== asked for" in capsys.readouterr().out
    assert main(["index", "outline", "--dataset", folder, "guide"]) == 0
    assert "Stroke signs" in capsys.readouterr().out
    assert main(["index", "copies", "--dataset", folder, "x-pa-0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overlap"]


def test_cli_out_and_db(dataset, tmp_path, capsys):
    target = tmp_path / "built.sqlite"
    assert main(["index", "search-db", "--dataset", str(dataset), "--out", str(target)]) == 0
    assert target.is_file() and not (dataset / "analysis").exists()
    capsys.readouterr()
    assert main(["index", "search", "--dataset", str(dataset), "--db", str(target),
                 "--regex", r"\d+\.\d"]) == 0
    assert "c-guide-2" in capsys.readouterr().out


def test_cli_prepare_builds_both_unless_told_not_to(dataset, capsys):
    assert main(["index", "prepare", "--dataset", str(dataset), "--no-search-db"]) == 0
    assert not si.search_db_path(dataset).exists()
    assert main(["index", "prepare", "--dataset", str(dataset)]) == 0
    assert si.search_db_path(dataset).is_file()
    capsys.readouterr()
    assert main(["index", "status", "--dataset", str(dataset)]) == 0
    assert "search index: 15 chunks" in capsys.readouterr().out
