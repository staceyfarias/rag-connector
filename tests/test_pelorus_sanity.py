"""Pelorus corpus mutations through a real Chroma collection and connector.

Fixed test vectors avoid a model download; no retrieval quality is measured.
"""

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from rag_connector.audit import audit_dataset
from rag_connector.dataset import exported_chunking, write_dataset_folder
from rag_connector.ingest import load_bundled_chunks
from rag_connector.reference import ReferenceRagConnector
from rag_connector.reference_provision import _chunk_metadata
from rag_connector.sanity import check_dataset
from rag_connector.sync import sync_dataset

pytestmark = pytest.mark.reference_extra

DOCUMENTS = {
    "faq.md", "card_check_in_and_muster_deadlines.md", "itinerary_grand_solar_tour.md",
}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    chunks = [c for c in load_bundled_chunks() if c.doc_id in DOCUMENTS]
    assert {c.doc_id for c in chunks} == DOCUMENTS
    return write_dataset_folder(
        tmp_path_factory.mktemp("pelorus") / "source", chunks,
        name="Pelorus sync scenarios", chunking=exported_chunking("reference"),
    )


class ObservedCollection:
    """Record payload requests while delegating all operations to real Chroma."""

    def __init__(self, collection):
        self.collection = collection
        self.reads = []

    def get(self, **kwargs):
        self.reads.append(kwargs)
        return self.collection.get(**kwargs)

    def count(self):
        return self.collection.count()


@pytest.fixture
def backend(source):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    name = f"sync-test-{uuid4().hex}"
    collection = client.create_collection(name, embedding_function=None)
    collection.add(
        ids=[c.chunk_id for c in source.chunks],
        documents=[c.text for c in source.chunks],
        metadatas=[_chunk_metadata(c) for c in source.chunks],
        embeddings=[[1.0, 0.0, 0.0] for _ in source.chunks],
    )
    # Bind only the corpus operations under test; no embedding client is needed.
    pipeline = ReferenceRagConnector.__new__(ReferenceRagConnector)
    pipeline._collection = ObservedCollection(collection)
    yield pipeline, collection
    client.delete_collection(name)


SCENARIOS = [
    # name, count matches, structure matches, sampled text matches, full audit in sync
    ("unchanged", True, True, True, True),
    ("added_chunk", False, False, True, False),
    ("deleted_chunk", False, False, False, False),
    ("changed_chunk_id", True, False, False, False),
    ("changed_document", True, False, True, False),
    ("changed_position", True, False, True, False),
    ("sampled_text_edit", True, True, False, False),
    ("unsampled_text_edit", True, True, True, False),
    ("metadata_edit", True, True, True, False),
]


@pytest.mark.parametrize("scenario,count_match,structure_match,text_match,full_match", SCENARIOS,
                         ids=[s[0] for s in SCENARIOS])
def test_pelorus_changes(source, backend, tmp_path, scenario, count_match,
                         structure_match, text_match, full_match):
    pipeline, collection = backend
    before = {p.relative_to(source.path): p.read_bytes()
              for p in source.path.rglob("*") if p.is_file()}
    baseline = check_dataset(source, pipeline, sample_percent=20, seed=7)
    selected = baseline["text_sample"]["selected_ids"]
    # Control whether the edited record is in the sample to test both outcomes.
    target = next(c for c in source.chunks if c.chunk_id == selected[0])
    if scenario == "unsampled_text_edit":
        target = next(c for c in source.chunks if c.chunk_id not in selected)
    updated = target
    if scenario == "added_chunk":
        updated = replace(target, chunk_id="scenario-added-chunk",
                          text="Additional sailing notice.",
                          doc_chunk_index=sum(c.doc_id == target.doc_id for c in source.chunks))
    elif scenario == "deleted_chunk":
        collection.delete(ids=[target.chunk_id])
    elif scenario == "changed_chunk_id":
        collection.delete(ids=[target.chunk_id])
        updated = replace(target, chunk_id="scenario-replacement-id")
    elif scenario == "changed_document":
        updated = replace(target, doc_id="different-document.md")
    elif scenario == "changed_position":
        updated = replace(target, doc_chunk_index=999)
    elif scenario in ("sampled_text_edit", "unsampled_text_edit"):
        updated = replace(target, text=target.text + "\nUpdated boarding instruction.")
    elif scenario == "metadata_edit":
        updated = replace(target, metadata={**target.metadata, "review_note": "Revised"})
    if scenario not in ("unchanged", "deleted_chunk"):
        collection.upsert(ids=[updated.chunk_id], documents=[updated.text],
                          metadatas=[_chunk_metadata(updated)], embeddings=[[1.0, 0.0, 0.0]])

    pipeline._collection.reads.clear()
    lite = check_dataset(source, pipeline, sample_percent=20, seed=7,
                         snapshot_at="2026-10-29T12:00:00Z")
    assert lite["count"]["matches"] is count_match
    assert lite["structure"]["matches"] is structure_match
    assert lite["text_sample"]["matches"] is text_match
    assert lite["snapshot"]["status"] == "unsupported"
    # Lite enumeration never hydrates text/vectors; only selected IDs get text.
    for read in pipeline._collection.reads:
        assert "embeddings" not in read["include"]
        if "documents" in read["include"]:
            assert set(read["ids"]) <= set(selected)
        else:
            assert read["include"] == ["metadatas"]

    full = audit_dataset(source, pipeline, connector_type="reference")
    assert full.in_sync is full_match
    synced = sync_dataset(source.path, pipeline, tmp_path / "synced", connector_type="reference")
    assert audit_dataset(synced.dataset, pipeline, connector_type="reference").in_sync
    assert synced.dataset.derived_from["core_sha256"] == source.core_sha256
    after = {p.relative_to(source.path): p.read_bytes()
             for p in source.path.rglob("*") if p.is_file()}
    assert after == before

    (tmp_path / "result.json").write_text(json.dumps({
        "scenario": scenario, "source_chunks": source.chunk_count,
        "source_documents": sorted(DOCUMENTS), "lite": lite,
        "full_audit": dict(full.summary), "synced_chunks": synced.dataset.chunk_count,
        "synced_copy_matches_backend": True, "source_unchanged": True,
    }, indent=2), encoding="utf-8")
