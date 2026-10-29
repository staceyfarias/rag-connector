"""Lite checks preserve scope, omissions, and failures as distinct evidence."""

import json
from dataclasses import replace

import pytest

from rag_connector import (
    ChunkInventoryEntry,
    ChunkInventoryPage,
    ChunkRecord,
    ConnectorContractError,
    ConnectorOperationalError,
    RagPipeline,
    SnapshotValidity,
    UnsupportedCapability,
    supports,
)
from rag_connector.cli import main
from rag_connector.dataset import chunker_chunking, write_dataset_folder
from rag_connector.registry import _REGISTRY, ConnectorSpec
from rag_connector.sanity import check_dataset


class NoLite(RagPipeline):
    def query(self, text, top_k=5, **kwargs):
        return []

    def pull_all_chunks(self):
        raise AssertionError("Lite checks must never pull the corpus")


class Lite(NoLite):
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.fetches = []

    def count_chunks(self):
        return len(self.chunks)

    def list_chunk_inventory(self, *, cursor=None, limit=1000):
        # Deliberately small pages exercise cursor handling independently of limit.
        offset = int(cursor or 0)
        chunks = list(self.chunks.values())[offset:offset + 2]
        return ChunkInventoryPage([
            ChunkInventoryEntry(c.chunk_id, c.doc_id, c.doc_chunk_index) for c in chunks
        ], str(offset + 2) if offset + 2 < len(self.chunks) else None)

    def get_chunks(self, chunk_ids):
        self.fetches.extend(chunk_ids)
        return {cid: self.chunks[cid] for cid in chunk_ids if cid in self.chunks}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sanity-dataset")
    chunks = [ChunkRecord(str(i), "a.md", "a.md", i, None, f"Text {i}")
              for i in range(10)]
    return write_dataset_folder(tmp_path / "ds", chunks, name="test",
                                chunking=chunker_chunking(chunk_size=30, chunk_overlap=0))


def test_default_reads_only_structure_and_count(dataset):
    pipeline = Lite(dataset.chunks)
    result = check_dataset(dataset, pipeline)
    assert result["count"]["matches"]
    assert result["structure"]["matches"]
    assert result["structure"]["backend_document_counts"] == {"a.md": 10}
    assert result["text_sample"]["status"] == "not_requested"
    assert not pipeline.fetches
    assert "in_sync" not in result
    json.dumps(result)


def test_same_count_different_structure(dataset):
    chunks = list(dataset.chunks)
    chunks[0] = replace(chunks[0], chunk_id="new")
    chunks[1] = replace(chunks[1], doc_chunk_index=99)
    result = check_dataset(dataset, Lite(chunks))
    assert result["count"]["matches"]
    assert not result["structure"]["matches"]
    assert result["structure"]["missing_ids"] == ["0"]
    assert result["structure"]["added_ids"] == ["new"]
    assert result["structure"]["changed_ids"] == ["1"]


def test_sampling_is_replayable_and_compares_exact_text(dataset):
    pipeline = Lite(dataset.chunks)
    first = check_dataset(dataset, pipeline, sample_percent=20, seed=7)["text_sample"]
    assert first["sampled_count"] == 2
    assert first["matches"]
    cid = first["selected_ids"][0]
    pipeline.chunks[cid] = replace(pipeline.chunks[cid], text="Changed")
    second = check_dataset(dataset, pipeline, sample_percent=20, seed=7)["text_sample"]
    assert second["selected_ids"] == first["selected_ids"]
    assert second["changed_ids"] == [cid]
    assert not second["matches"]
    del pipeline.chunks[cid]
    third = check_dataset(dataset, pipeline, sample_percent=20, seed=7)["text_sample"]
    assert third["missing_ids"] == [cid]


def test_small_nonzero_sample_rounds_up_and_full_sample_is_allowed(dataset):
    pipeline = Lite(dataset.chunks)
    assert check_dataset(dataset, pipeline, sample_percent=.1)["text_sample"]["sampled_count"] == 1
    full = check_dataset(dataset, pipeline, sample_percent=100)["text_sample"]
    assert full["sampled_count"] == 10


def test_unsupported_does_not_invoke_full_read_or_base_get_chunks(dataset):
    pipeline = NoLite()
    assert not supports(pipeline, "count_chunks")
    with pytest.raises(UnsupportedCapability):
        pipeline.count_chunks()
    with pytest.raises(UnsupportedCapability):
        pipeline.list_chunk_inventory()
    result = check_dataset(dataset, pipeline, sample_percent=20)
    assert all(result[key]["status"] == "unsupported"
               for key in ("count", "structure", "text_sample"))


def test_count_inventory_and_fetch_are_independently_optional(dataset):
    pipeline = Lite(dataset.chunks)
    pipeline.count_chunks = NoLite().count_chunks
    result = check_dataset(dataset, pipeline)
    assert result["count"]["status"] == "unsupported"
    assert result["structure"]["matches"]
    pipeline = Lite(dataset.chunks)
    pipeline.list_chunk_inventory = NoLite().list_chunk_inventory
    result = check_dataset(dataset, pipeline, sample_percent=20)
    assert result["count"]["matches"]
    assert result["structure"]["status"] == "unsupported"
    assert result["text_sample"]["matches"]


@pytest.mark.parametrize("percent", [-1, 101, float("nan"), float("inf"), True])
def test_invalid_percentage(dataset, percent):
    with pytest.raises(ValueError):
        check_dataset(dataset, NoLite(), sample_percent=percent)


def test_backend_error_is_not_unsupported_or_a_match(dataset):
    class Broken(Lite):
        def list_chunk_inventory(self, **kwargs):
            raise ConnectorOperationalError("offline")

    with pytest.raises(ConnectorOperationalError, match="offline"):
        check_dataset(dataset, Broken(dataset.chunks))


@pytest.mark.parametrize("kind", ["duplicate", "cursor", "invalid"])
def test_invalid_inventory_is_refused(dataset, kind):
    class Broken(Lite):
        def list_chunk_inventory(self, **kwargs):
            if kind == "cursor":
                return ChunkInventoryPage([], "repeat")
            item = ChunkInventoryEntry("x", "a", -1 if kind == "invalid" else 0)
            return ChunkInventoryPage([item, item] if kind == "duplicate" else [item])

    with pytest.raises(ConnectorContractError):
        check_dataset(dataset, Broken(dataset.chunks))


def test_cli_outputs_lite_evidence_and_closes(dataset, monkeypatch, capsys):
    pipeline = Lite(dataset.chunks)
    timestamp = "2026-10-29T12:00:00Z"
    pipeline.check_snapshot = lambda instant: SnapshotValidity(
        "valid" if instant == timestamp else "unknown")
    closed = []
    pipeline.close = lambda: closed.append(True)
    monkeypatch.setitem(_REGISTRY, "lite-test", ConnectorSpec(
        connector_type="lite-test", label="Lite test", description="Test connector",
        build=lambda params: pipeline,
    ))
    assert main(["dataset", "check", "--dataset", str(dataset.path),
                 "--connector", "lite-test", "--sample-percent", "20", "--seed", "7",
                 "--snapshot-at", timestamp]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["text_sample"]["sampled_count"] == 2
    assert result["snapshot"]["status"] == "valid"
    assert closed == [True]


def test_reference_reads_only_requested_payloads():
    from rag_connector.reference import ReferenceRagConnector

    class Collection:
        def count(self):
            return 1

        def get(self, **kwargs):
            calls.append(kwargs)
            result = {"ids": ["a"], "metadatas": [
                {"doc_id": "doc", "source_file": "doc", "doc_chunk_index": 0}
            ]}
            if "documents" in kwargs["include"]:
                result["documents"] = ["Text"]
            return result

    calls = []
    pipeline = ReferenceRagConnector.__new__(ReferenceRagConnector)
    pipeline._collection = Collection()
    assert pipeline.count_chunks() == 1
    page = pipeline.list_chunk_inventory()
    assert page.items == [ChunkInventoryEntry("a", "doc", 0)]
    assert calls[-1]["include"] == ["metadatas"]
    assert pipeline.get_chunks(["a"])["a"].text == "Text"
    assert calls[-1] == {"ids": ["a"], "include": ["documents", "metadatas"]}
    assert pipeline.get_chunks([]) == {}


@pytest.mark.parametrize("status", ["valid", "out_of_sync", "unknown"])
def test_snapshot_validity_is_independent_of_matching_structure(dataset, status):
    seen = []

    class Snapshot(Lite):
        def check_snapshot(self, snapshot_at):
            seen.append(snapshot_at)
            return SnapshotValidity(status, "Backend mutation history")

    timestamp = "2026-10-29T12:00:00Z"
    result = check_dataset(dataset, Snapshot(dataset.chunks), snapshot_at=timestamp)
    assert result["snapshot"] == {"status": status, "snapshot_at": timestamp,
                                   "detail": "Backend mutation history"}
    assert result["structure"]["matches"]
    assert seen == [timestamp]


def test_snapshot_can_be_unsupported_without_losing_structure(dataset):
    result = check_dataset(dataset, Lite(dataset.chunks), snapshot_at="2026-10-29T12:00:00Z")
    assert result["snapshot"]["status"] == "unsupported"
    assert result["structure"]["matches"]
    assert check_dataset(dataset, Lite(dataset.chunks))["snapshot"]["status"] == "not_requested"


@pytest.mark.parametrize("timestamp", ["2026-10-29", "2026-10-29T12:00:00", "bad", 123])
def test_invalid_snapshot_timestamp_is_refused(dataset, timestamp):
    with pytest.raises(ValueError):
        check_dataset(dataset, NoLite(), snapshot_at=timestamp)


def test_snapshot_errors_propagate(dataset):
    class Broken(Lite):
        def check_snapshot(self, snapshot_at):
            raise ConnectorOperationalError("history unavailable")

    with pytest.raises(ConnectorOperationalError, match="history unavailable"):
        check_dataset(dataset, Broken(dataset.chunks), snapshot_at="2026-10-29T12:00:00Z")


def test_invalid_snapshot_return_is_refused(dataset):
    class Broken(Lite):
        def check_snapshot(self, snapshot_at):
            return True

    with pytest.raises(ConnectorContractError):
        check_dataset(dataset, Broken(dataset.chunks), snapshot_at="2026-10-29T12:00:00Z")


@pytest.mark.parametrize("count", [-1, True, "10"])
def test_invalid_count_is_refused(dataset, count):
    pipeline = Lite(dataset.chunks)
    pipeline.count_chunks = lambda: count
    with pytest.raises(ConnectorContractError):
        check_dataset(dataset, pipeline)


def test_count_inventory_disagreement_is_visible(dataset):
    pipeline = Lite(dataset.chunks)
    pipeline.count_chunks = lambda: 11
    result = check_dataset(dataset, pipeline)
    assert not result["count"]["matches"]
    assert result["structure"]["matches"]
    assert not result["structure"]["count_matches_inventory"]


def test_fetch_cannot_return_unrequested_records(dataset):
    pipeline = Lite(dataset.chunks)
    pipeline.get_chunks = lambda ids: {"unexpected": dataset.chunks[0]}
    with pytest.raises(ConnectorContractError):
        check_dataset(dataset, pipeline, sample_percent=20)
