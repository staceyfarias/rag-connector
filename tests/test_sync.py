"""Sync into a copy (docs/dataset-spec.md, "Syncing into a copy").

Each test builds a source Dataset, changes the corpus a fake connector serves,
and checks that the sync writes a new Dataset with its lineage while leaving the
source byte for byte as it was.
"""

from __future__ import annotations

import json

import pytest

from rag_connector.audit import audit_dataset, connection_sha256, record_audit
from rag_connector.base import ChunkRecord, RagPipeline
from rag_connector.cli import main
from rag_connector.dataset import chunker_chunking, read_dataset_folder, write_dataset_folder
from rag_connector.errors import UnsupportedCapability
from rag_connector.registry import _REGISTRY, ConnectorSpec
from rag_connector.sync import SYNCS_DIR, list_syncs, sync_dataset


def _chunk(chunk_id, text, *, doc="a.md", idx=0):
    return ChunkRecord(
        chunk_id=chunk_id, doc_id=doc, source_file=doc, doc_chunk_index=idx,
        global_index=None, text=text, metadata={},
    )


class _Served(RagPipeline):
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def query(self, text, top_k=5, *, answer_metadata=None):
        return []

    def pull_all_chunks(self):
        return list(self._chunks)


ORIGINAL = [
    _chunk("a.md:chunk-0", "Alpha one.", idx=0),
    _chunk("a.md:chunk-1", "Alpha two.", idx=1),
]
# The index was rebuilt: same text, new ids, and one document added.
REBUILT = [
    _chunk("x1", "Alpha one.", idx=0),
    _chunk("x2", "Alpha two.", idx=1),
    _chunk("x3", "Gamma.", doc="c.md"),
]
STAGING = {"url": "https://staging"}


@pytest.fixture
def source(tmp_path):
    return write_dataset_folder(
        tmp_path / "source", ORIGINAL, name="src",
        chunking=chunker_chunking(chunk_size=30, chunk_overlap=10),
    )


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _recorded_audit(source, served, connection=STAGING):
    audit = audit_dataset(source, _Served(served), connector_type="fake",
                          connection=connection, created_at="2026-10-20T10:00:00Z")
    return record_audit(source.path, audit)


def test_sync_writes_a_new_dataset_with_lineage_and_leaves_the_source(source, tmp_path):
    audit = _recorded_audit(source, REBUILT)
    before = _snapshot(source.path)

    result = sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                          connector_type="fake", connection=STAGING,
                          audit_id=audit.audit_id, instance_label="staging")

    assert _snapshot(source.path) == before
    copy = read_dataset_folder(tmp_path / "copy")
    assert copy.verified
    assert [c.chunk_id for c in copy.chunks] == ["x1", "x2", "x3"]
    assert copy.dataset_id != source.dataset_id
    assert copy.name == "src"
    # Lineage between Datasets is in the core, covered by core_sha256 ...
    assert copy.derived_from == {"dataset_id": source.dataset_id,
                                 "core_sha256": source.core_sha256}
    manifest = json.loads((tmp_path / "copy/dataset.json").read_text(encoding="utf-8"))
    assert "audit" not in json.dumps(manifest["derived_from"])
    # ... and what describes the backend is private to the instance.
    record = result.record
    assert result.path.parent == tmp_path / "copy" / SYNCS_DIR
    assert record["source"]["core_sha256"] == source.core_sha256
    assert record["dataset"]["core_sha256"] == copy.core_sha256
    assert record["connector"]["connection_sha256"] == connection_sha256(STAGING)
    assert "https://staging" not in json.dumps(record)
    assert record["audit"]["audit_id"] == audit.audit_id
    assert record["audit"]["backend_matches_audit"] is True
    assert record["same_chunk_inventory_as_source"] is False
    assert list_syncs(tmp_path / "copy")[0]["sync_id"] == record["sync_id"]
    envelope = json.loads((tmp_path / "copy/.rag-connector/extension.json")
                          .read_text(encoding="utf-8"))
    assert envelope["core_sha256"] == copy.core_sha256


def test_lineage_cannot_be_added_to_a_core_afterwards(source, tmp_path):
    sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy", connector_type="fake")
    path = tmp_path / "copy/dataset.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["derived_from"]["dataset_id"] = "ds-0000000000000000"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="core_sha256"):
        read_dataset_folder(tmp_path / "copy")


def test_a_backend_that_moved_on_since_the_audit_is_recorded(source, tmp_path):
    audit = _recorded_audit(source, REBUILT)
    result = sync_dataset(source.path, _Served(REBUILT[:2]), tmp_path / "copy",
                          connector_type="fake", connection=STAGING,
                          audit_id=audit.audit_id)
    assert result.record["audit"]["backend_matches_audit"] is False


def test_an_audit_of_another_instance_is_refused(source, tmp_path):
    audit = _recorded_audit(source, REBUILT, connection={"url": "https://prod"})
    with pytest.raises(ValueError, match="another connector instance"):
        sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                     connector_type="fake", connection=STAGING,
                     audit_id=audit.audit_id)
    assert not (tmp_path / "copy").exists()


def test_an_audit_cannot_be_cited_without_a_connection(source, tmp_path):
    audit = _recorded_audit(source, REBUILT)
    with pytest.raises(ValueError, match="identify the connector instance"):
        sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                     connector_type="fake", audit_id=audit.audit_id)


def test_an_unknown_or_malformed_audit_id_is_refused(source, tmp_path):
    with pytest.raises(FileNotFoundError):
        sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                     connector_type="fake", connection=STAGING,
                     audit_id="audit-20261020T100000Z-deadbeef")
    with pytest.raises(ValueError, match="not an audit id"):
        sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                     connector_type="fake", connection=STAGING,
                     audit_id="../../etc")


def test_an_audit_of_another_connector_type_is_refused(source, tmp_path):
    audit = _recorded_audit(source, REBUILT)
    with pytest.raises(ValueError, match="connector"):
        sync_dataset(source.path, _Served(REBUILT), tmp_path / "copy",
                     connector_type="other", connection=STAGING,
                     audit_id=audit.audit_id)


def test_failures_leave_nothing_behind(source, tmp_path):
    class _NoCorpus(_Served):
        def pull_all_chunks(self):
            raise UnsupportedCapability("no corpus read")

    with pytest.raises(UnsupportedCapability):
        sync_dataset(source.path, _NoCorpus([]), tmp_path / "copy", connector_type="fake")
    with pytest.raises(ValueError, match="at least one chunk"):
        sync_dataset(source.path, _Served([]), tmp_path / "copy", connector_type="fake")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source"]


def test_an_occupied_output_is_refused(source, tmp_path):
    with pytest.raises(FileExistsError):
        sync_dataset(source.path, _Served(REBUILT), source.path, connector_type="fake")


def test_an_unchanged_backend_still_gives_a_copy_and_says_so(source, tmp_path):
    result = sync_dataset(source.path, _Served(ORIGINAL), tmp_path / "copy",
                          connector_type="fake")
    assert result.dataset.dataset_id == source.dataset_id  # same inventory
    assert result.dataset.core_sha256 != source.core_sha256  # a different core
    assert result.record["same_chunk_inventory_as_source"] is True
    assert result.record["audit"] is None


# --- command line ------------------------------------------------------------

@pytest.fixture
def fake_registered():
    def connect(params):
        return _Served(REBUILT), {"type": "fake-kb"}

    _REGISTRY["fake-kb"] = ConnectorSpec(
        connector_type="fake-kb", label="Fake KB", description="test",
        build=lambda c: _Served(REBUILT), connect=connect)
    try:
        yield
    finally:
        _REGISTRY.pop("fake-kb", None)


def test_cli_sync_cites_an_audit_of_the_same_instance(
        fake_registered, source, tmp_path, capsys):
    assert main(["dataset", "audit", "--dataset", str(source.path),
                 "--connector", "fake-kb"]) == 0
    audit_id = json.loads(capsys.readouterr().out)["audit_id"]
    before = _snapshot(source.path)

    out = tmp_path / "copy"
    assert main(["dataset", "sync", "--dataset", str(source.path), "--connector",
                 "fake-kb", "--out", str(out), "--audit", audit_id,
                 "--label", "staging"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["derived_from"]["core_sha256"] == source.core_sha256
    assert printed["sync"]["audit"]["audit_id"] == audit_id
    assert printed["chunk_count"] == 3
    assert _snapshot(source.path) == before


def test_cli_sync_refuses_an_unknown_audit(fake_registered, source, tmp_path, capsys):
    assert main(["dataset", "sync", "--dataset", str(source.path), "--connector",
                 "fake-kb", "--out", str(tmp_path / "copy"),
                 "--audit", "audit-20261020T100000Z-deadbeef"]) == 1
    assert "records no audit" in capsys.readouterr().err
    assert not (tmp_path / "copy").exists()
