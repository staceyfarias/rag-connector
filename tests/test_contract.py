import pytest

from rag_connector import (
    AnswerGenerator,
    ChunkVectorReader,
    ConnectorContractError,
    ContentPublisher,
    HealthCapability,
    IndexIntrospector,
    PagedCorpusReader,
    QueryEmbedder,
    QueryTelemetryReader,
    RagConnector,
    RagPipeline,
    UnsupportedCapability,
    supports,
    supports_chunk_vectors,
)
from rag_connector.reference import ReferenceRagConnector


class MinimalPipeline(RagPipeline):
    """The smallest legal connector: the two abstract methods and nothing else."""

    def query(self, text, top_k=5):
        return []

    def pull_all_chunks(self):
        return []

# docs/contract.md splits the optional surface in two, and readers act on that
# split: "Optional capabilities" is documented as implemented and safe to build
# against, "Reserved surface" as unwired. An earlier audit got this wrong and
# told connector authors not to build against capabilities the reference
# connector already implements. These tests are what keep the page honest.
#
# The guard comes in two strengths because the protocols do. QueryEmbedder and
# HealthCapability have no base default, so structural presence (issubclass)
# proves an implementation exists. ChunkVectorReader and AnswerGenerator sit
# over concrete RagPipeline defaults, so EVERY pipeline satisfies them
# structurally and issubclass can never fail -- verified: deleting the
# reference's generate override left the old issubclass-only guard green. For
# those rows the real question is supports(): does an override exist?
STRUCTURAL_CAPABILITIES = [
    QueryEmbedder,
    HealthCapability,
]
DEFAULTED_CAPABILITIES = [
    (ChunkVectorReader, "get_chunk_vectors"),
    (AnswerGenerator, "generate"),
]
RESERVED_CAPABILITIES = [
    ContentPublisher,
    QueryTelemetryReader,
    IndexIntrospector,
]


def test_rag_pipeline_satisfies_structural_connector_contract():
    assert issubclass(ReferenceRagConnector, RagPipeline)


@pytest.mark.parametrize("protocol", STRUCTURAL_CAPABILITIES,
                         ids=lambda p: p.__name__)
def test_reference_implements_every_structural_capability(protocol):
    """These protocols have no base default, so issubclass is meaningful: it
    fails the moment the reference stops defining the methods. If one stops
    satisfying the protocol, contract.md's table is lying and the citation
    must be removed."""
    assert issubclass(ReferenceRagConnector, protocol)


@pytest.mark.parametrize(("protocol", "method"), DEFAULTED_CAPABILITIES,
                         ids=[p.__name__ for p, _ in DEFAULTED_CAPABILITIES])
def test_reference_overrides_every_defaulted_capability(protocol, method):
    """For these rows issubclass is a tautology (the base default satisfies
    the protocol for every pipeline), so the guard asks supports(): the
    reference must OVERRIDE the default, or contract.md's citation of it as
    the working example is a lie the old guard could never catch."""
    assert issubclass(ReferenceRagConnector, protocol)   # true for anything
    assert supports(ReferenceRagConnector, method)       # the actual guard


def test_reference_paged_reading_is_the_derived_default():
    """The PagedCorpusReader row is the deliberate exception: the reference
    does NOT override list_chunks (see the note in reference.py) -- it
    satisfies the protocol through the base class deriving pages from
    pull_all_chunks, and contract.md cites exactly that. If an override
    appears, move the table's citation back to reference.py."""
    assert issubclass(ReferenceRagConnector, PagedCorpusReader)
    assert not supports(ReferenceRagConnector, "list_chunks")


@pytest.mark.parametrize("protocol", RESERVED_CAPABILITIES, ids=lambda p: p.__name__)
def test_reserved_capabilities_remain_unimplemented(protocol):
    """The inverse guard: once something here gains an implementation it is no
    longer reserved, and it must move into the Optional capabilities table
    rather than sit under a heading telling people not to use it."""
    assert not issubclass(ReferenceRagConnector, protocol)


def test_contract_is_runtime_checkable():
    assert isinstance(MinimalPipeline(), RagConnector)


def test_a_pipeline_without_chunk_vectors_refuses_out_loud():
    """The whole point of the raising default: a caller that cannot read the
    index must report the connector as UNVERIFIED, not as verified clean. An
    empty dict would be indistinguishable from "checked, all fine"."""
    with pytest.raises(UnsupportedCapability):
        MinimalPipeline().get_chunk_vectors(["anything"])


def test_a_pipeline_without_chunk_vectors_refuses_even_for_an_empty_request():
    # The empty-input short circuit belongs to connectors that *can* answer.
    # A connector that cannot must not look momentarily capable.
    with pytest.raises(UnsupportedCapability):
        MinimalPipeline().get_chunk_vectors([])


def test_supports_reports_overrides_not_structural_presence():
    """The general form of the supports_chunk_vectors question, for every
    method with a concrete base default: 'has the method' is true of all
    pipelines, 'answers it itself' is what a host needs to know."""
    minimal = MinimalPipeline()
    for defaulted in ("generate", "list_chunks", "get_chunk_vectors",
                      "get_chunks", "fingerprint", "info"):
        assert not supports(minimal, defaulted)
    assert supports(minimal, "pull_all_chunks")
    assert supports(minimal, "query")
    assert not supports(minimal, "no_such_method")
    # Classes work the same as instances.
    assert supports(MinimalPipeline, "pull_all_chunks")
    assert not supports(MinimalPipeline, "generate")
    assert supports(ReferenceRagConnector, "generate")


def test_chunk_vector_support_is_advertised_by_implementation_not_isinstance():
    """``isinstance`` cannot answer this one. Because the raising default lives
    on ``RagPipeline``, every pipeline satisfies ``ChunkVectorReader``
    structurally -- so ``supports_chunk_vectors`` is what hosts must ask, and it
    must disagree with ``isinstance`` for a connector that only inherits."""
    minimal = MinimalPipeline()
    assert isinstance(minimal, ChunkVectorReader)
    assert not supports_chunk_vectors(minimal)

    assert supports_chunk_vectors(object.__new__(ReferenceRagConnector))


def test_run_snapshot_is_a_truly_optional_structural_capability():
    from rag_connector import RunSnapshotProvider

    assert not isinstance(MinimalPipeline(), RunSnapshotProvider)
    assert isinstance(object.__new__(ReferenceRagConnector), RunSnapshotProvider)
    assert not supports(MinimalPipeline(), "run_snapshot")
    assert supports(ReferenceRagConnector, "run_snapshot")


# ---------------------------------------------------------------------------
# The chunk metadata contract: what a black box can actually be asked for.
# ---------------------------------------------------------------------------


def _chunk(**overrides):
    from rag_connector import ChunkRecord

    base = dict(
        chunk_id="c1", doc_id="d.txt", source_file="d.txt",
        doc_chunk_index=0, global_index=0, text="hello",
    )
    return ChunkRecord(**{**base, **overrides})


def test_a_connector_may_omit_the_corpus_wide_ordinal():
    """``global_index`` presupposes a stable total order over the corpus.

    A paged or sharded store may not have one and a hosted retrieval API will
    not expose it, so requiring it put one product's need (ordinal-based
    neighbor expansion) into shared vocabulary. A host that wants ordinals
    assigns them at ingest, where it knows the order it read things in.
    """
    RagPipeline.validate_chunks([_chunk(global_index=None)])


def test_the_within_document_ordinal_is_still_required():
    """The relaxation above deliberately does NOT extend to
    ``doc_chunk_index``. A store that returns a chunk can almost always say
    where in its document it sits, and no host backfills this one -- relaxing a
    constraint whose compensating control does not exist only moves the failure
    somewhere quieter. If you are here to 'make them consistent', add the
    backfill first."""
    with pytest.raises(ValueError, match="doc_chunk_index"):
        RagPipeline.validate_chunks([_chunk(doc_chunk_index=None)])


def test_global_index_has_no_default_so_omitting_it_is_explicit():
    """Passing ``None`` is the intended usage; forgetting the field is not.

    A default would have to sit before ``text``, forcing either a default on
    ``text`` -- never absent -- or a field reorder. Reordering a dataclass whose
    consumers may construct positionally misassigns silently, which is exactly
    how EmbeddingSpaceDescriptor drifted once already.
    """
    from rag_connector import ChunkRecord

    with pytest.raises(TypeError):
        ChunkRecord(
            chunk_id="c1", doc_id="d.txt", source_file="d.txt",
            doc_chunk_index=0, text="hello",
        )


def test_the_required_set_still_catches_a_genuinely_broken_ingest():
    """Relaxing one field must not soften the others -- the point of validating
    on pull is that a bad ingest fails loud rather than producing broken qrels
    or uncitable chunks downstream."""
    for missing in ("chunk_id", "doc_id", "source_file", "text"):
        with pytest.raises(ValueError, match=missing):
            RagPipeline.validate_chunks([_chunk(**{missing: None})])


# ---------------------------------------------------------------------------
# Corpus reading: one question at two granularities, either one implementable.
# ---------------------------------------------------------------------------


class _PagingOnlyPipeline(RagPipeline):
    """Implements the primitive a real store actually offers, and nothing else."""

    def __init__(self, total=5):
        self._chunks = [
            _chunk(chunk_id=f"c{i}", global_index=i, text=f"chunk {i}")
            for i in range(total)
        ]
        self.pages_served = 0

    def query(self, text, top_k=5):
        return []

    def list_chunks(self, *, cursor=None, limit=1000):
        from rag_connector.models import ChunkPage

        self.pages_served += 1
        start = int(cursor) if cursor else 0
        page = self._chunks[start : start + limit]
        nxt = start + len(page)
        return ChunkPage(
            items=page,
            next_cursor=str(nxt) if nxt < len(self._chunks) else None,
        )


class _BulkOnlyPipeline(RagPipeline):
    """The pre-existing shape: bulk-pull only. Must keep working untouched."""

    def query(self, text, top_k=5):
        return []

    def pull_all_chunks(self):
        return [_chunk(chunk_id=f"c{i}", global_index=i) for i in range(5)]


def test_a_paging_connector_gets_bulk_pull_for_free():
    """Pagination is what stores natively offer; bulk-pull derives from it at
    no cost. This is the direction that was previously unavailable — the base
    demanded pull_all_chunks, so paged connectors hand-wrote the loop."""
    pipeline = _PagingOnlyPipeline(total=5)

    chunks = pipeline.pull_all_chunks()

    assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2", "c3", "c4"]


def test_a_bulk_only_connector_still_works_and_gains_paging():
    """The existing shape must not break — every shipped connector implements
    pull_all_chunks and nothing else."""
    pipeline = _BulkOnlyPipeline()

    assert len(pipeline.pull_all_chunks()) == 5
    page = pipeline.list_chunks(limit=2)
    assert [c.chunk_id for c in page.items] == ["c0", "c1"]
    assert page.next_cursor is not None


def test_implementing_neither_refuses_instead_of_recursing():
    """Each default derives from the other, so 'implements neither' would be
    infinite recursion. It must be a typed refusal that names both ways out —
    a RecursionError tells an author nothing about what to write."""

    class NoCorpus(RagPipeline):
        def query(self, text, top_k=5):
            return []

    with pytest.raises(UnsupportedCapability, match="list_chunks"):
        NoCorpus().pull_all_chunks()
    with pytest.raises(UnsupportedCapability, match="pull_all_chunks"):
        NoCorpus().list_chunks()


def test_a_repeated_cursor_is_refused_rather_than_looped_forever():
    """This is a black-box contract: a connector CAN be wrong, and hanging is
    the worst way to find that out."""

    class StuckCursor(RagPipeline):
        def query(self, text, top_k=5):
            return []

        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage

            return ChunkPage(items=[_chunk()], next_cursor="always-the-same")

    with pytest.raises(ConnectorContractError, match="repeated cursor"):
        StuckCursor().pull_all_chunks()
