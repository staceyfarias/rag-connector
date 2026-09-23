
import math
import re
from types import SimpleNamespace

import pytest

from rag_connector.base import ChunkRecord, RetrievedChunk
from rag_connector.errors import ConnectorOperationalError
from rag_connector.reference import ReferenceRagConnector
from rag_connector.reference_provision import provision_reference_rag
from rag_connector.registry import get_connector
from rag_connector.validate import validate_pipeline


def _raw_write(connector, records, *, replace=False):
    """Write straight into the demo collection, the way the provisioner does.

    TESTS may manage the demo corpus; the CONNECTOR may not — that split is
    the thing under test elsewhere in this file. Routing these fixtures
    through the provisioner's own internals keeps the connector read-only
    rather than reintroducing a write method for test convenience.
    """
    from rag_connector import reference_provision as provisioning

    client = provisioning._open_client(connector.persist_dir)
    if replace:
        try:
            client.delete_collection(connector.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=connector.collection_name,
        metadata={"hnsw:space": connector.distance},
    )
    provisioning._ingest(collection, connector._embed, records)


class FakeEmbeddingClient:
    model_id = "fake-embedder"
    dimensions = 16

    def _embed(self, text):
        vec = [0.1] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            vec[hash(token) % self.dimensions] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_generate_normalizes_rank_citations_to_chunk_ids():
    pipeline = object.__new__(ReferenceRagConnector)
    pipeline.query = lambda text, top_k: [
        RetrievedChunk("chunk-a", "source a", 0.1, 0),
        RetrievedChunk("chunk-b", "source b", 0.2, 1),
    ]
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
        content='{"answer":"Answer [1]", "citations":["1"]}'))
    result = pipeline.generate("q", top_k=2, llm=llm)
    assert result.citations == ["chunk-a"]


def test_generate_keeps_unmappable_citations_for_zero_support_scoring():
    # A hallucinated citation must survive capture so the citation scorer can
    # assign it zero support -- dropping it here would silently raise precision.
    pipeline = object.__new__(ReferenceRagConnector)
    pipeline.query = lambda text, top_k: [
        RetrievedChunk("chunk-a", "source a", 0.1, 0),
    ]
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
        content='{"answer":"Answer [1][9]", "citations":["1", "9", "ghost-chunk", ""]}'))
    result = pipeline.generate("q", top_k=1, llm=llm)
    # "1" resolves to chunk-a; "9" (out-of-range rank) and "ghost-chunk" are
    # kept verbatim; the empty citation is discarded as noise, not a citation.
    assert result.citations == ["chunk-a", "9", "ghost-chunk"]


def test_generate_captures_missing_llm_as_an_error():
    pipeline = object.__new__(ReferenceRagConnector)
    contexts = [RetrievedChunk("chunk-a", "source a", 0.1, 0)]
    pipeline.query = lambda text, top_k: contexts

    result = pipeline.generate("q", top_k=1)

    assert result.contexts == contexts
    assert result.error == "Reference RAG generation requires an LLM"


@pytest.mark.reference_extra
def test_reference_rag_provisions_and_pulls_contract(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.txt").write_text("alpha apple orchard harvest " * 8, encoding="utf-8")

    records = provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="sandbox_test",
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
    ).pull_all_chunks()

    assert records
    assert records[0].metadata["source_identity"]
    assert records[0].metadata["source_fingerprint"]
    assert records[0].source_file == "alpha.txt"


def _provision(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.txt").write_text(
        "alpha apple orchard harvest " * 8,
        encoding="utf-8",
    )
    (folder / "beta.md").write_text(
        "beta banana grove shipment " * 8,
        encoding="utf-8",
    )
    return provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="reference_test",
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
    )


@pytest.mark.reference_extra
def test_reference_rag_passes_full_connector_validator(tmp_path):
    connector = _provision(tmp_path)
    report = validate_pipeline(
        connector,
        target="reference",
        sample_query="apple orchard",
    )
    assert report.ready, [f"{check.name}: {check.details}" for check in report.failures]


@pytest.mark.reference_extra
def test_reference_query_uses_stable_ids_and_canonical_scores(tmp_path):
    connector = _provision(tmp_path)
    corpus_ids = {chunk.chunk_id for chunk in connector.pull_all_chunks()}
    hits = connector.query("apple orchard", top_k=3)

    assert hits
    assert all(hit.chunk_id in corpus_ids for hit in hits)
    assert [hit.rank for hit in hits] == list(range(len(hits)))
    assert all(hit.raw_score is not None for hit in hits)
    assert all(hit.score == 1.0 - hit.raw_score for hit in hits)


@pytest.mark.reference_extra
def test_reference_fingerprint_and_connection_are_stable(tmp_path):
    connector = _provision(tmp_path)
    assert connector.fingerprint() == connector.fingerprint()
    assert connector.connection() == {
        "type": "reference",
        "persist_dir": str(tmp_path / "chroma"),
        "collection_name": "reference_test",
        "embedding_model": "fake-embedder",
        "distance": "cosine",
        # Undeclared by default: "not stated", never an assumed False.
        "normalized": None,
    }
    assert connector.info()["embedding_fingerprint"]
    assert connector.health().ok


@pytest.mark.reference_extra
def test_reference_run_snapshot_separates_settings_from_observations(tmp_path):
    connector = _provision(tmp_path)

    snapshot = connector.run_snapshot()

    assert snapshot == {
        "schema": "reference-rag.run-snapshot.v1",
        "settings": {
            "backend": "chroma",
            "collection": "reference_test",
            "embedding_provider": "fastembed",
            "embedding_model": "fake-embedder",
            "embedding_dimensions": 16,
            "embedding_fingerprint": connector.info()["embedding_fingerprint"],
            "distance": "cosine",
            "normalized": None,
            "retrieval_mode": "scored",
            "score_convention": "higher_is_better",
        },
        # 4 chunks per document, not 5: the fifth window of each was clamped
        # to the end of the content and produced a span wholly inside its
        # predecessor -- pure overlap, now dropped by chunk_loaded_documents.
        "observations": {"chunk_count": 8},
    }


@pytest.mark.reference_extra
def test_reference_registry_rebuilds_the_same_instance(monkeypatch, tmp_path):
    connector = _provision(tmp_path)
    monkeypatch.setattr(
        "rag_connector.reference.get_embedding_client",
        lambda *args, **kwargs: FakeEmbeddingClient(),
    )
    spec = get_connector("reference")
    assert spec is not None
    rebuilt = spec.build(connector.connection())

    assert rebuilt.fingerprint() == connector.fingerprint()
    assert [hit.chunk_id for hit in rebuilt.query("apple", top_k=2)] == [
        hit.chunk_id for hit in connector.query("apple", top_k=2)
    ]


@pytest.mark.reference_extra
def test_reference_registry_connects_to_a_shared_instance(monkeypatch, tmp_path):
    connector = _provision(tmp_path)
    monkeypatch.setattr(
        "rag_connector.reference.get_embedding_client",
        lambda *args, **kwargs: FakeEmbeddingClient(),
    )
    spec = get_connector("reference")
    assert spec is not None and spec.connect is not None

    connected, connection = spec.connect(connector.connection())

    assert connection == connector.connection()
    assert connected.fingerprint() == connector.fingerprint()
    assert [hit.chunk_id for hit in connected.query("apple", top_k=2)] == [
        hit.chunk_id for hit in connector.query("apple", top_k=2)
    ]


@pytest.mark.reference_extra
def test_reference_pagination_and_fetch_are_consistent(tmp_path):
    connector = _provision(tmp_path)
    first = connector.list_chunks(limit=2)
    assert len(first.items) == 2
    assert first.next_cursor is not None
    second = connector.list_chunks(cursor=first.next_cursor, limit=100)
    all_chunks = connector.pull_all_chunks()
    assert list(first.items) + list(second.items) == all_chunks

    wanted = [all_chunks[0].chunk_id, all_chunks[-1].chunk_id, "missing"]
    fetched = connector.get_chunks(wanted)
    assert set(fetched) == {wanted[0], wanted[1]}


@pytest.mark.reference_extra
def test_reference_does_not_assume_a_normalization_it_was_not_told(tmp_path):
    """Undeclared must stay undeclared. A host reads None as "cannot verify"
    and a bool as an assertion; guessing turns the first into the second."""
    connector = _provision(tmp_path)

    assert connector.describe_space().normalized is None


@pytest.mark.reference_extra
def test_reference_reports_the_declared_normalization(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.txt").write_text("alpha apple orchard " * 8, encoding="utf-8")

    connector = provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="declared",
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
        normalized=True,
    )

    assert connector.describe_space().normalized is True


@pytest.mark.reference_extra
def test_a_declared_normalization_survives_the_connection_round_trip(
    monkeypatch, tmp_path
):
    """The declaration is a property of the space, so a connector rebuilt in a
    fresh process must describe it identically. If it decayed to None on the
    reopen, a host that gated on it at freeze time would quietly stop gating."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.txt").write_text("alpha apple orchard " * 8, encoding="utf-8")
    connector = provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="declared",
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
        normalized=True,
    )
    monkeypatch.setattr(
        "rag_connector.reference.get_embedding_client",
        lambda *args, **kwargs: FakeEmbeddingClient(),
    )
    spec = get_connector("reference")
    assert spec is not None and spec.connect is not None

    rebuilt = spec.build(connector.connection())
    connected, _ = spec.connect(connector.connection())

    assert rebuilt.describe_space().normalized is True
    assert connected.describe_space().normalized is True


@pytest.mark.reference_extra
def test_a_connection_predating_the_declaration_reads_as_unstated(
    monkeypatch, tmp_path
):
    """Old persisted connections have no 'normalized' key. That absence is the
    tri-state's "not stated" already -- no artifact migration required."""
    connector = _provision(tmp_path)
    legacy = connector.connection()
    del legacy["normalized"]
    monkeypatch.setattr(
        "rag_connector.reference.get_embedding_client",
        lambda *args, **kwargs: FakeEmbeddingClient(),
    )

    rebuilt = get_connector("reference").build(legacy)

    assert rebuilt.describe_space().normalized is None


def test_a_non_boolean_normalization_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tri-state"):
        ReferenceRagConnector(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="bad",
            embedding_client=FakeEmbeddingClient(),
            normalized="true",
        )


def _append_second_folder(tmp_path, **kwargs):
    """Provision a SECOND folder into the collection ``_provision`` created."""
    folder = tmp_path / "more_docs"
    folder.mkdir()
    (folder / "gamma.txt").write_text(
        "gamma grape vineyard press " * 8, encoding="utf-8"
    )
    return provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="reference_test",
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
        replace_existing=False,
        **kwargs,
    )


@pytest.mark.reference_extra
def test_appending_continues_the_corpus_ordinals(tmp_path):
    """Chunking restarts global_index at 0 per provision call, so a naive
    append duplicates the ordinals already stored — the one dishonest option
    the validator FAILs. An appended batch must continue where the corpus
    left off."""
    first = _provision(tmp_path).pull_all_chunks()
    appended = _append_second_folder(tmp_path)

    pulled = appended.pull_all_chunks()

    assert len(pulled) > len(first)
    ordinals = [c.global_index for c in pulled]
    assert None not in ordinals
    assert sorted(ordinals) == list(range(len(pulled))), "ordinals collide or gap"


@pytest.mark.reference_extra
def test_an_appended_corpus_still_validates(tmp_path):
    _provision(tmp_path)
    appended = _append_second_folder(tmp_path)

    report = validate_pipeline(appended, target="appended",
                               sample_query="grape vineyard")

    assert report.ready, [f"{c.name}: {c.details}" for c in report.failures]


@pytest.mark.reference_extra
def test_appending_a_document_the_collection_already_holds_is_refused(tmp_path):
    """Re-appending a known document re-produces its chunk_ids, and Chroma's
    duplicate handling varies by version (skip, upsert, error) — none of which
    is an honest append. Refuse by name before anything is written."""
    connector = _provision(tmp_path)
    before = connector.pull_all_chunks()
    folder = tmp_path / "docs"  # the SAME folder _provision ingested

    with pytest.raises(ValueError, match="already holds"):
        provision_reference_rag(
            str(folder),
            persist_dir=str(tmp_path / "chroma"),
            collection_name="reference_test",
            embedding_client=FakeEmbeddingClient(),
            chunk_size=80,
            chunk_overlap=10,
            replace_existing=False,
        )
    assert connector.pull_all_chunks() == before, "a refused append wrote data"


@pytest.mark.reference_extra
def test_appending_to_an_ordinal_less_corpus_stays_ordinal_less(tmp_path):
    """The validator treats mixed ordinals as ambiguous ('supply them for every
    chunk or for none'), so a batch appended to a corpus without ordinals must
    carry none either — not introduce a partial ordering nobody can sort by."""
    connector = _provision(tmp_path)
    # Rewrite the stored corpus without ordinals, as a host-assigns-ordinals
    # black box would have stored it.
    stripped = []
    for chunk in connector.pull_all_chunks():
        chunk.global_index = None
        stripped.append(chunk)
    _raw_write(connector, stripped, replace=True)

    appended = _append_second_folder(tmp_path)

    assert all(c.global_index is None for c in appended.pull_all_chunks())


def test_reference_metadata_omits_an_absent_corpus_ordinal():
    """``global_index=None`` is contract-legal, and the provisioner's metadata
    builder is the shape connector authors copy -- ``int(None)`` here would
    hand them a crash on a record the contract explicitly permits."""
    from rag_connector.reference_provision import _chunk_metadata

    meta = _chunk_metadata(ChunkRecord(
        chunk_id="c:0", doc_id="c", source_file="c.txt",
        doc_chunk_index=0, global_index=None, text="gamma",
    ))

    assert "global_index" not in meta
    assert meta["doc_chunk_index"] == 0


@pytest.mark.reference_extra
def test_reference_round_trips_a_contract_legal_absent_ordinal(tmp_path):
    connector = _provision(tmp_path)
    _raw_write(connector, [ChunkRecord(
        chunk_id="extra:0", doc_id="extra", source_file="extra.txt",
        doc_chunk_index=0, global_index=None, text="gamma grape vineyard",
    )])

    pulled = connector.pull_all_chunks()

    by_id = {chunk.chunk_id: chunk for chunk in pulled}
    assert by_id["extra:0"].global_index is None, "an ordinal was invented"
    # The mixed corpus still sorts deterministically rather than raising on a
    # None/int comparison inside the sort key.
    ordinals = [c.global_index for c in pulled if c.global_index is not None]
    assert ordinals == sorted(ordinals)


@pytest.mark.reference_extra
def test_reference_refuses_a_row_that_lost_its_within_document_ordinal(tmp_path):
    """A -1 sentinel would sail through validate_chunks (which rejects None,
    not -1) and return a chunk claiming an ordinal the store never held."""
    connector = _provision(tmp_path)

    class ForgetfulCollection:
        def get(self, **kwargs):
            return {
                "ids": ["a:0"],
                "documents": ["alpha"],
                "metadatas": [{"chunk_id": "a:0", "doc_id": "a",
                               "source_file": "a.txt"}],
                "embeddings": None,
            }

    connector._collection = ForgetfulCollection()

    with pytest.raises(ValueError, match="doc_chunk_index"):
        connector.pull_all_chunks()


@pytest.mark.reference_extra
def test_reference_returns_the_vectors_the_index_holds(tmp_path):
    """The vectors must come OUT of the index, not from re-embedding the text.

    So the oracle is what the store itself reports (``pull_all_chunks`` reads
    Chroma's stored embeddings), never a fresh call to the embedder -- that
    would be the embedder agreeing with itself, which passes even when the
    index is stale or was written in a different embedding space.
    """
    connector = _provision(tmp_path)
    stored = {c.chunk_id: c.embedding for c in connector.pull_all_chunks()}
    wanted = list(stored)[:3]

    vectors = connector.get_chunk_vectors(wanted)

    assert set(vectors) == set(wanted)
    for chunk_id in wanted:
        assert vectors[chunk_id] == pytest.approx(stored[chunk_id])
        assert all(isinstance(value, float) for value in vectors[chunk_id])


@pytest.mark.reference_extra
def test_reference_chunk_vectors_reproduce_the_reported_score(tmp_path):
    """The reason the capability exists: a host recomputes cosine locally and
    compares it to the score the connector reported. A connector that quietly
    rescaled its scores would fail here while still ranking perfectly."""
    connector = _provision(tmp_path)
    hits = connector.query("apple orchard", top_k=3)
    vectors = connector.get_chunk_vectors([hit.chunk_id for hit in hits])
    query_vector = connector.embed_queries(["apple orchard"]).vectors[0]

    for hit in hits:
        chunk_vector = vectors[hit.chunk_id]
        dot = sum(q * c for q, c in zip(query_vector, chunk_vector))
        norms = math.sqrt(sum(q * q for q in query_vector)) * math.sqrt(
            sum(c * c for c in chunk_vector)
        )
        assert hit.score == pytest.approx(dot / norms, abs=1e-5)


@pytest.mark.reference_extra
def test_reference_omits_unknown_chunk_vector_ids(tmp_path):
    """A missing id is a gap, never a placeholder or a zero vector: a
    fabricated vector is worse than a gap because nothing downstream can
    detect it."""
    connector = _provision(tmp_path)
    known = connector.pull_all_chunks()[0].chunk_id

    vectors = connector.get_chunk_vectors([known, "no-such-chunk"])

    assert set(vectors) == {known}


@pytest.mark.reference_extra
def test_reference_chunk_vectors_short_circuit_on_empty_input(tmp_path):
    connector = _provision(tmp_path)

    class ExplodingCollection:
        def get(self, **kwargs):
            raise AssertionError("the backend must not be consulted for no ids")

    connector._collection = ExplodingCollection()
    assert connector.get_chunk_vectors([]) == {}


@pytest.mark.reference_extra
def test_reference_chunk_vector_backend_failure_is_not_an_empty_result(tmp_path):
    connector = _provision(tmp_path)

    class BrokenCollection:
        def get(self, **kwargs):
            raise TimeoutError("backend unavailable")

    connector._collection = BrokenCollection()
    with pytest.raises(ConnectorOperationalError):
        connector.get_chunk_vectors(["anything"])


@pytest.mark.reference_extra
def test_reference_backend_failures_are_not_empty_results(tmp_path):
    connector = _provision(tmp_path)

    class BrokenCollection:
        def query(self, **kwargs):
            raise TimeoutError("backend unavailable")

    connector._collection = BrokenCollection()
    try:
        connector.query("apple", top_k=3)
    except ConnectorOperationalError as exc:
        assert "backend unavailable" in str(exc)
        assert isinstance(exc.__cause__, TimeoutError)
    else:
        raise AssertionError("backend failure was incorrectly converted to []")


# -- the read/write boundary ----------------------------------------------

def test_connector_exposes_no_way_to_destroy_or_write_a_corpus():
    """A connector is a READ interface onto someone else's RAG system.

    A host reconstructs a connector from a persisted connection document on
    every run. If provisioning lived on the class, that object would carry a
    ``delete_collection`` into every evaluation — one attribute lookup away
    from wiping the corpus under test. The build path belongs to the demo
    system (rag_connector.reference_provision), not to the connector.
    """
    surface = dir(ReferenceRagConnector)
    forbidden = [
        name for name in surface
        if any(word in name.lower() for word in
               ("reset", "ingest", "provision", "delete", "drop", "clear",
                "upsert", "write", "create_from"))
    ]
    assert not forbidden, (
        f"write/destructive methods reachable from the connector: {forbidden}"
    )


def test_the_demo_systems_build_path_still_exists_off_the_connector():
    """The capability is not gone, it moved — users and tests manage the demo
    collection through the provisioner or the CLI."""
    from rag_connector import reference_provision

    assert callable(reference_provision.provision_reference_rag)
    # And it will not be pointed at a collection by default.
    import inspect
    sig = inspect.signature(reference_provision.provision_reference_rag)
    assert sig.parameters["collection_name"].default is inspect.Parameter.empty
