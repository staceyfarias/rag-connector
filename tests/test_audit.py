"""Dataset audits (docs/dataset-spec.md, "Audits"), on synthetic corpora.

Each test builds a Dataset, then serves a deliberately changed corpus through a
fake connector and checks that the audit names exactly that change.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rag_connector.audit import (
    AUDITS_DIR,
    audit_dataset,
    connection_sha256,
    list_audits,
    read_audit,
    record_audit,
)
from rag_connector.base import ChunkRecord, RagPipeline
from rag_connector.dataset import chunker_chunking, write_dataset_folder
from rag_connector.errors import ConnectorContractError, UnsupportedCapability


def _chunk(chunk_id, text, *, doc="a.md", idx=0, meta=None):
    return ChunkRecord(
        chunk_id=chunk_id, doc_id=doc, source_file=doc, doc_chunk_index=idx,
        global_index=None, text=text, metadata=dict(meta or {}),
    )


class _Served(RagPipeline):
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def query(self, text, top_k=5, *, answer_metadata=None):
        return []

    def pull_all_chunks(self):
        return list(self._chunks)


ORIGINAL = [
    _chunk("a.md:chunk-0", "Alpha one. Shared tail", idx=0, meta={"section": "intro"}),
    _chunk("a.md:chunk-1", "Shared tail and alpha two.", idx=1, meta={"section": "intro"}),
    _chunk("b.md:chunk-0", "Beta only chunk.", doc="b.md", meta={"section": "body"}),
]


@pytest.fixture
def dataset(tmp_path):
    return write_dataset_folder(
        tmp_path / "ds", ORIGINAL, name="t",
        chunking=chunker_chunking(chunk_size=30, chunk_overlap=10),
    )


def _audit(dataset, served, **kw):
    return audit_dataset(dataset, _Served(served), connector_type="fake", **kw)


def test_identical_backend_is_in_sync(dataset):
    result = _audit(dataset, ORIGINAL)
    assert result.in_sync
    assert result.signals == []
    assert result.summary["chunk_counts"]["unchanged"] == 3
    assert result.summary["backend"]["chunk_inventory_sha256"] == (
        dataset.chunk_inventory_sha256
    )


def test_rebuilt_index_with_new_ids_is_matched_by_text(dataset):
    served = [replace(c, chunk_id=f"uuid-{i}") for i, c in enumerate(ORIGINAL)]
    result = _audit(dataset, served)
    assert not result.in_sync
    assert result.summary["chunk_counts"]["id_changed"] == 3
    assert result.summary["chunk_counts"]["added"] == 0
    assert [s["code"] for s in result.signals] == ["chunk_ids_changed"]
    by_id = {r["chunk_id"]: r for r in result.chunks}
    assert by_id["b.md:chunk-0"]["backend_chunk_id"] == "uuid-2"
    assert by_id["b.md:chunk-0"]["match"] == "text"


def test_removed_document_and_added_document(dataset):
    served = ORIGINAL[:2] + [_chunk("c.md:chunk-0", "Gamma new.", doc="c.md",
                                    meta={"section": "body"})]
    result = _audit(dataset, served)
    docs = {d["doc_id"]: d["status"] for d in result.summary["documents"]}
    assert docs == {"a.md": "unchanged", "b.md": "removed", "c.md": "added"}
    codes = {s["code"] for s in result.signals}
    assert {"chunks_missing", "chunks_added", "documents_removed",
            "documents_added"} <= codes


def test_rechunk_reports_text_changes_and_chunk_counts(dataset):
    served = [
        _chunk("a.md:chunk-0", "Alpha one. Shared tail and alpha two.", idx=0,
               meta={"section": "intro"}),
        ORIGINAL[2],
    ]
    result = _audit(dataset, served)
    counts = result.summary["chunk_counts"]
    assert counts["text_changed"] == 1 and counts["missing"] == 1
    codes = {s["code"] for s in result.signals}
    assert "documents_chunk_count_changed" in codes
    profiles = result.summary["chunking_profiles"]
    assert profiles["dataset"]["neighbour_overlap_chars"]["max"] == len(" Shared tail") - 1
    assert "neighbour_overlap_chars" not in profiles["backend"]


def test_metadata_fields_added_removed_and_changed(dataset):
    served = [replace(c, metadata={"heading": "x"}) for c in ORIGINAL]
    result = _audit(dataset, served)
    codes = {s["code"]: s for s in result.signals}
    assert codes["metadata_keys_added"]["count"] == 1
    assert codes["metadata_keys_removed"]["count"] == 1
    assert codes["chunk_metadata_changed"]["count"] == 3
    assert not result.in_sync  # same ids and texts, but metadata moved


def test_machine_local_path_is_not_a_metadata_change(dataset):
    served = [replace(c, metadata={**c.metadata, "document_filepath": "C:/x"})
              for c in ORIGINAL]
    assert _audit(dataset, served).in_sync


def test_duplicate_text_is_ambiguous_not_guessed(dataset):
    served = [ORIGINAL[0], ORIGINAL[1],
              replace(ORIGINAL[2], chunk_id="n1"), replace(ORIGINAL[2], chunk_id="n2")]
    result = _audit(dataset, served)
    row = next(r for r in result.chunks if r["chunk_id"] == "b.md:chunk-0")
    assert row["status"] == "ambiguous"
    assert row["candidate_backend_chunk_ids"] == ["n1", "n2"]
    assert result.summary["chunk_counts"]["added"] == 0


def test_empty_corpus_is_reported_not_refused(dataset):
    result = _audit(dataset, [])
    assert result.summary["chunk_counts"]["missing"] == 3
    assert result.summary["backend"]["chunk_count"] == 0


def test_a_connector_that_cannot_read_its_corpus_raises(dataset):
    class NoCorpus(RagPipeline):
        def query(self, text, top_k=5, *, answer_metadata=None):
            return []

    with pytest.raises(UnsupportedCapability):
        audit_dataset(dataset, NoCorpus(), connector_type="fake")


def test_duplicate_backend_ids_refuse_the_audit(dataset):
    with pytest.raises(ConnectorContractError):
        _audit(dataset, ORIGINAL + [ORIGINAL[0]])


def test_instances_are_told_apart_by_their_connection(dataset):
    staging = _audit(dataset, ORIGINAL, connection={"url": "https://staging"})
    prod = _audit(dataset, ORIGINAL, connection={"url": "https://prod"})
    s, p = staging.summary["connector"], prod.summary["connector"]
    assert s["connection_sha256"] != p["connection_sha256"]
    assert s["connection_sha256"] == connection_sha256({"url": "https://staging"})
    assert "https://staging" not in json.dumps(staging.summary)


def test_record_keeps_every_audit_and_never_touches_the_core(dataset):
    core_before = {p: (dataset.path / p).read_bytes()
                   for p in ("dataset.json", "data/chunks.jsonl", "data/sources.json")}
    first = record_audit(dataset.path, _audit(dataset, ORIGINAL, host="ragauge",
                                              created_at="2026-10-20T10:00:00Z"))
    second = record_audit(dataset.path, _audit(dataset, ORIGINAL[:2], host="pelorus",
                                               created_at="2026-10-20T11:00:00Z"))
    listed = list_audits(dataset.path)
    assert [a["audit_id"] for a in listed] == [first.audit_id, second.audit_id]
    assert [a["connector"]["host"] for a in listed] == ["ragauge", "pelorus"]
    assert read_audit(second.path).summary["chunk_counts"]["missing"] == 1
    assert first.path.parent == dataset.path / AUDITS_DIR
    envelope = json.loads((dataset.path / ".rag-connector/extension.json").read_text())
    assert envelope["tool"] == "rag-connector"
    assert envelope["core_sha256"] == dataset.core_sha256
    for p, data in core_before.items():
        assert (dataset.path / p).read_bytes() == data


def test_record_refuses_an_audit_of_another_core(dataset, tmp_path):
    other = write_dataset_folder(tmp_path / "other", ORIGINAL[:1], name="o",
                                 chunking=chunker_chunking(chunk_size=30,
                                                           chunk_overlap=10))
    with pytest.raises(ValueError, match="refusing"):
        record_audit(dataset.path, _audit(other, ORIGINAL))
