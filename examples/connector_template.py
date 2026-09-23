"""Copy-me starting point for an independently packaged RAG connector.

**Two** methods are required: ``query``, and ONE corpus read. Everything else
below is optional -- ``info`` ships a working base default, and you override it
only to add detail worth freezing into run metadata.

For the corpus read, prefer :meth:`list_chunks` when your store has a paged or
cursor-based listing, which most do. The whole-corpus pull then derives from it
for free. Implementing ``pull_all_chunks`` instead is fine for a store that
only offers a bulk read, but if you find yourself hand-writing a paging loop
inside it, you are writing the adapter the base class already provides -- and a
host that prefers paged reads has nothing to call. This template shows the bulk
form because it is the shorter illustration; see ``list_chunks`` in
``rag_connector/base.py`` for the paged one.

Then publish ``CONNECTOR_SPEC`` through the ``rag_connector.connectors``
entry-point group in your package metadata, and run ``rag-connector validate``
before integrating with a host.
"""

from __future__ import annotations

import os

from rag_connector import ChunkRecord, RagPipeline, RetrievedChunk
from rag_connector.errors import ConnectorOperationalError
from rag_connector.registry import ConnectorSpec


class TemplateRagConnector(RagPipeline):
    """Skeleton adapter around an external RAG system."""

    name = "template-rag"
    retrieval_mode = "scored"

    def __init__(self, *, endpoint: str, collection: str):
        self.endpoint = endpoint
        self.collection = collection
        # Secrets stay out of the connection document. Resolve them here.
        self.api_token = os.environ.get("MY_RAG_API_TOKEN")
        # TODO: construct the backend client.

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return best-first results with IDs shared by ``pull_all_chunks``.

        Convert backend distances to canonical higher-is-better scores and
        preserve the native value in ``raw_score``. Raise
        ``ConnectorOperationalError`` for backend failures; return ``[]`` only
        when retrieval succeeded and genuinely found nothing.
        """

        try:
            # rows = self._client.search(self.collection, text, top_k)
            rows = []
        except Exception as exc:
            raise ConnectorOperationalError(f"retrieval failed: {exc}") from exc

        return [
            RetrievedChunk(
                chunk_id=row.id,
                text=row.text,
                # Already higher-is-better, so nothing was converted -- and
                # raw_score stays None. It records the STORE-NATIVE value when
                # a conversion happened (see cosine_distance_to_similarity /
                # l2_distance_to_score); copying an unconverted score into it
                # just claims evidence that does not exist.
                score=float(row.similarity),
                raw_score=None,
                rank=rank,
                doc_id=row.doc_id,
                metadata=dict(row.metadata),
            )
            for rank, row in enumerate(rows)
        ]

    def pull_all_chunks(self) -> list[ChunkRecord]:
        """Return a complete, stable corpus snapshot.

        A partial pull is an operational failure, not a smaller successful
        corpus. ``doc_chunk_index`` must count from zero without gaps inside
        each document.

        ``global_index`` is OPTIONAL: a corpus-wide ordinal presupposes a
        stable total order that a paged or sharded store may not have, so pass
        ``None`` when your system has none and a host will assign ordinals at
        ingest. Supply it for every chunk or for none, and if you supply it,
        the values must be unique. Note it has no default, so you always pass
        it explicitly -- including ``global_index=None``.

        If you pass ``None``, do not then sort on it: ``validate_chunks``
        accepts an absent ordinal, so a bare ``key=lambda c: c.global_index``
        raises a TypeError from inside the sort rather than reporting anything
        useful.
        """

        rows = []  # TODO: fully paginate the backend corpus.
        chunks = [
            ChunkRecord(
                chunk_id=row.id,
                doc_id=row.doc_id,
                source_file=row.filename,
                doc_chunk_index=row.index_in_document,
                global_index=index,
                text=row.text,
                metadata={"source_fingerprint": row.content_hash},
            )
            for index, row in enumerate(rows)
        ]
        self.validate_chunks(chunks)
        return chunks

    def info(self) -> dict:
        return {
            "name": self.name,
            "type": type(self).__name__,
            "endpoint": self.endpoint,
            "collection": self.collection,
            "retrieval_mode": self.retrieval_mode,
        }

    def run_snapshot(self) -> dict:
        """Optional, secret-free JSON captured beside each evaluation result.

        Keep configured behavior separate from cheap observations of live
        state. Do not put passwords, tokens, full records, or evaluation scores
        here; the host owns evaluation metrics.
        """
        return {
            "schema": "my-rag.run-snapshot.v1",
            "settings": {
                "collection": self.collection,
                # "reranker": self.reranker_name,
                # "retrieval_strategy": self.strategy,
            },
            "observations": {
                # "indexed_chunks": self._client.count(self.collection),
            },
        }


def _build(connection: dict) -> TemplateRagConnector:
    """Reconstruct from a persisted, JSON-serializable, secret-free document."""

    return TemplateRagConnector(
        endpoint=connection["endpoint"],
        collection=connection["collection"],
    )


def _connect(params: dict) -> tuple[TemplateRagConnector, dict]:
    connection = {
        "endpoint": params["endpoint"],
        "collection": params["collection"],
    }
    return _build(connection), connection


CONNECTOR_SPEC = ConnectorSpec(
    connector_type="my-rag",
    label="My RAG",
    description="Attach to the My RAG retrieval service.",
    build=_build,
    connect=_connect,
    params=[
        {
            "name": "endpoint",
            "label": "Endpoint URL",
            "type": "string",
            "required": True,
        },
        {
            "name": "collection",
            "label": "Collection",
            "type": "string",
            "required": True,
        },
    ],
)


def smoke_test(connector: RagPipeline, sample_query: str) -> None:
    """Quick identity preflight; the full validator checks the entire contract."""

    chunks = connector.pull_all_chunks()
    connector.validate_chunks(chunks)
    corpus_ids = {chunk.chunk_id for chunk in chunks}
    hits = connector.query(sample_query, top_k=5)
    unknown_ids = [hit.chunk_id for hit in hits if hit.chunk_id not in corpus_ids]
    if unknown_ids:
        raise AssertionError(
            "query() returned IDs absent from pull_all_chunks(): "
            f"{unknown_ids!r}"
        )
