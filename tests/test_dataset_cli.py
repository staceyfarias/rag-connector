"""``rag-connector dataset build`` and ``dataset export``, on synthetic data.

The split below is derived by hand from the chunker's documented rules
(``rag_connector.ingest.chunk_loaded_documents``) at chunk_size=30,
chunk_overlap=5; the ids and hashes are then computed from the spec's
definitions written out literally, not recorded from a run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from rag_connector.base import ChunkRecord, RagPipeline, RetrievedChunk
from rag_connector.cli import main
from rag_connector.dataset import read_dataset_folder
from rag_connector.registry import _REGISTRY, ConnectorSpec


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# b.md, character positions:
#   0-3 aaaa, 4 sp, 5-8 bbbb, 9 sp, 10-13 cccc, 14 sp, 15-18 dddd,
#   19-20 "\n\n", 21-24 eeee, 25 sp, 26-29 ffff, 30 sp, 31-34 gggg,
#   35 sp, 36-39 hhhh                                       (length 40)
# Window 1: [0, 30). 30 < 40, so look for a separator in [24, 30): no "\n\n"
#   or "\n" there (they are at 19-20), no ". ", "? ", "! "; a space at 25 > 24,
#   so the window ends at 26. Text "aaaa bbbb cccc dddd\n\neeee" (stripped of
#   the trailing space), span 0..25.
# Window 2 starts at 26 - 5 = 21, runs to the end (61 > 40): "eeee ffff gggg
#   hhhh", span 21..40.
# Window 3 starts at 40 - 5 = 35: " hhhh" strips to "hhhh", span 36..40, which
#   lies inside window 2's span, so it is dropped as pure overlap.
B_MD = "aaaa bbbb cccc dddd\n\neeee ffff gggg hhhh"
A_TXT = "Alpha beta.\n"
EXPECTED = [
    # (chunk_id, text, char_start, char_end, doc_chunk_index, global_index)
    ("a.txt:chunk-0", "Alpha beta.", 0, 11, 0, 0),
    ("b.md:chunk-0", "aaaa bbbb cccc dddd\n\neeee", 0, 25, 0, 1),
    ("b.md:chunk-1", "eeee ffff gggg hhhh", 21, 40, 1, 2),
]
EXPECTED_INVENTORY = _sha("".join(f"{cid}\t{_sha(text)}\n" for cid, text, *_ in EXPECTED))


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_bytes(A_TXT.encode("utf-8"))
    (folder / "b.md").write_bytes(B_MD.encode("utf-8"))
    (folder / "notes.csv").write_text("not a supported document")
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "c.txt").write_text("tool state, not corpus")
    return folder


def test_build_writes_the_hand_derived_split(docs, tmp_path, capsys):
    out = tmp_path / "ds"
    assert main(["dataset", "build", "--folder", str(docs), "--out", str(out),
                 "--chunk-size", "30", "--chunk-overlap", "5"]) == 0

    rows = [json.loads(line) for line in
            (out / "data" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["chunk_id"], r["text"], r["char_start"], r["char_end"],
             r["doc_chunk_index"], r["global_index"]) for r in rows] == EXPECTED
    for row in rows:
        assert "document_filepath" not in row["metadata"]
        assert str(tmp_path) not in json.dumps(row)

    manifest = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["chunk_inventory_sha256"] == EXPECTED_INVENTORY
    assert manifest["dataset_id"] == "ds-" + EXPECTED_INVENTORY[:16]
    assert manifest["name"] == "docs"
    assert manifest["chunking"] == {"method": "rag-connector-chunker",
                                    "chunker_version": "1",
                                    "chunk_size": 30, "chunk_overlap": 5}
    assert manifest["sources"] == {"source_count": 2, "file_types": {"md": 1, "txt": 1}}

    sources = json.loads((out / "data" / "sources.json").read_text(encoding="utf-8"))
    assert sources == [
        {"source_identity": "a.txt", "source_fingerprint": _sha("txt\0" + A_TXT),
         "relative_path": "a.txt", "filename": "a.txt", "file_type": "txt"},
        {"source_identity": "b.md", "source_fingerprint": _sha("md\0" + B_MD),
         "relative_path": "b.md", "filename": "b.md", "file_type": "md"},
    ]

    printed = json.loads(capsys.readouterr().out)
    assert printed["dataset_id"] == manifest["dataset_id"]
    assert printed["chunk_count"] == 3
    assert read_dataset_folder(out).verified


def test_build_refuses_an_output_folder_that_already_holds_something(docs, tmp_path, capsys):
    out = tmp_path / "ds"
    assert main(["dataset", "build", "--folder", str(docs), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["dataset", "build", "--folder", str(docs), "--out", str(out)]) == 1
    assert "not empty" in capsys.readouterr().err


def test_build_refuses_an_overlap_that_cannot_advance(docs, tmp_path, capsys):
    assert main(["dataset", "build", "--folder", str(docs), "--out",
                 str(tmp_path / "ds"), "--chunk-size", "10", "--chunk-overlap", "10"]) == 1
    assert "chunk_overlap" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_build_of_the_same_documents_gets_the_same_id(docs, tmp_path):
    main(["dataset", "build", "--folder", str(docs), "--out", str(tmp_path / "1"),
          "--name", "one"])
    main(["dataset", "build", "--folder", str(docs), "--out", str(tmp_path / "2"),
          "--name", "two"])
    assert (read_dataset_folder(tmp_path / "1").dataset_id
            == read_dataset_folder(tmp_path / "2").dataset_id)


# --- export -----------------------------------------------------------------

class _PagedFakeConnector(RagPipeline):
    """A connector that only pages (list_chunks), carries embeddings and an
    absolute document path on every chunk — all things export must handle."""

    name = "fake"

    def __init__(self, fail=False):
        self.closed = False
        self._fail = fail
        self._chunks = [
            ChunkRecord(chunk_id=f"kb/{i}", doc_id="kb", source_file="kb.pdf",
                        doc_chunk_index=i, global_index=None, text=text,
                        embedding=[0.5, 0.5],
                        metadata={"document_filepath": "/srv/kb.pdf", "page": i + 1})
            for i, text in enumerate(["first passage", "second passage", "third"])
        ]

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=1.0, rank=i)
                for i, c in enumerate(self._chunks[:top_k])]

    def list_chunks(self, *, cursor=None, limit=1000):
        from rag_connector.errors import ConnectorOperationalError
        from rag_connector.models import ChunkPage

        if self._fail:
            raise ConnectorOperationalError("backend unavailable")
        start = int(cursor or 0)
        page = self._chunks[start:start + 2]  # force two pages
        nxt = start + 2
        return ChunkPage(items=page, next_cursor=str(nxt) if nxt < len(self._chunks) else None)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_registered():
    built = {}

    def connect(params):
        pipeline = _PagedFakeConnector(fail=params.get("fail", False))
        built["pipeline"] = pipeline
        return pipeline, {"type": "fake-kb"}

    spec = ConnectorSpec(connector_type="fake-kb", label="Fake KB", description="test",
                         build=lambda c: _PagedFakeConnector(), connect=connect)
    _REGISTRY["fake-kb"] = spec
    try:
        yield built
    finally:
        _REGISTRY.pop("fake-kb", None)


def test_export_freezes_the_connector_corpus_without_embeddings(
        fake_registered, tmp_path, capsys):
    out = tmp_path / "exported"
    assert main(["dataset", "export", "--connector", "fake-kb", "--out", str(out)]) == 0

    rows = [json.loads(line) for line in
            (out / "data" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["chunk_id"] for r in rows] == ["kb/0", "kb/1", "kb/2"]
    assert all("embedding" not in r for r in rows)
    assert [r["metadata"] for r in rows] == [{"page": 1}, {"page": 2}, {"page": 3}]
    assert all(r["global_index"] is None for r in rows)  # not invented

    manifest = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["chunking"] == {"method": "exported-from-connector",
                                    "connector_type": "fake-kb"}
    assert manifest["name"] == "fake-kb"
    expected = _sha("kb/0\t" + _sha("first passage") + "\n"
                    + "kb/1\t" + _sha("second passage") + "\n"
                    + "kb/2\t" + _sha("third") + "\n")
    assert manifest["chunk_inventory_sha256"] == expected
    # The connector reported no source fingerprint: unknown, recorded as "".
    assert json.loads((out / "data" / "sources.json").read_text(encoding="utf-8")) == [
        {"source_identity": "kb", "source_fingerprint": "", "relative_path": "kb.pdf",
         "filename": "kb.pdf", "file_type": "pdf"}]
    assert fake_registered["pipeline"].closed
    assert read_dataset_folder(out).verified


def test_export_of_a_failing_backend_writes_nothing(fake_registered, tmp_path, capsys):
    out = tmp_path / "exported"
    assert main(["dataset", "export", "--connector", "fake-kb",
                 "--params", '{"fail": true}', "--out", str(out)]) == 1
    assert "backend unavailable" in capsys.readouterr().err
    assert not out.exists()
    assert fake_registered["pipeline"].closed


def test_export_of_an_unregistered_type_names_the_registered_ones(tmp_path, capsys):
    assert main(["dataset", "export", "--connector", "nope", "--out",
                 str(tmp_path / "x")]) == 1
    assert "not registered" in capsys.readouterr().err
