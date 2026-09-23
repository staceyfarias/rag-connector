"""Provisioning for the Reference RAG — the demo system's OWN implementation.

This module is deliberately not part of the connector.

A connector is a read interface onto a RAG system someone else already built:
:meth:`query` and :meth:`pull_all_chunks`, plus by-id reads. Nothing in that
contract creates, clears, or writes a corpus, because a connector never owns
the system it points at. The Reference RAG is unusual only in that this
project also ships the system it connects to — and building that system is
what lives here, on the far side of the boundary.

The separation is enforced by construction: this module owns the Chroma client
that writes, and hands back a :class:`ReferenceRagConnector` opened on the
finished collection. The connector object a host reconstructs from a persisted
connection document therefore has no method that can destroy a corpus, and no
amount of holding one gets you a ``delete_collection``. That is the same shape
a production connector package uses (its provisioning script is a separate
entry point from its connector), so the Reference RAG stays the honest
template rather than a special case with extra powers.

Users and tests manage the demo collection through here, or through
``rag-connector reference provision``.

Provisioning does not return until :func:`durability_barrier` has read the
corpus back. Chroma 1.5.0 has no flush to call, so "the write finished" and
"a reader can see it" are separate claims, and only the second one is worth
returning on -- see that function.
"""

from __future__ import annotations

from .base import ChunkRecord
from .embedding import EmbeddingClient, get_embedding_client
from .errors import ConnectorOperationalError
from .ingest import (
    DEFAULT_BUNDLED_CORPUS,
    chunk_loaded_documents,
    load_bundled_chunks,
    load_bundled_corpus,
    load_documents,
)
from .reference import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIR,
    ReferenceRagConnector,
    canonical_persist_dir,
)

__all__ = [
    "provision_reference_rag",
    "provision_reference_rag_from_chunks",
    "durability_barrier",
]


def _open_client(persist_dir: str):
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'Provisioning the Reference RAG requires the "reference" extra: '
            'pip install "rag-connector[reference]"'
        ) from exc
    # Same resolved spelling the connector will use, so the write and the read
    # share one Chroma engine instead of racing two. See canonical_persist_dir.
    return chromadb.PersistentClient(path=canonical_persist_dir(persist_dir))


def durability_barrier(collection, expected_count: int) -> None:
    """Prove the corpus is READABLE before provisioning claims to be finished.

    Chroma 1.5.0 exposes no flush, sync, or checkpoint -- there is no call that
    means "make the writes durable now" (see
    :func:`rag_connector.reference.release_store`). So the barrier cannot be a
    flush. What it can be is a read: run, in the writing process, the same
    vector-segment read the CONSUMER will run, and refuse to return until it
    succeeds.

    That is worth doing because of where the failure otherwise lands. The read
    that fails is ``pull_all_chunks`` -- ``get(include=[..., "embeddings"])``,
    which is what makes Chroma construct the HNSW segment reader. Without this,
    a store whose vector segment never materialised is reported minutes later,
    in a different process, as an unreadable corpus with nothing left to point
    at. With it, the same condition is a provisioning failure at the write
    site, naming the collection it just built.

    It is a verification, not a repair: it makes no writes and hides nothing.
    """
    try:
        count = int(collection.count())
    except Exception as exc:
        raise ConnectorOperationalError(
            f"Reference RAG provisioning could not count {collection.name!r} "
            f"after ingest: {exc}"
        ) from exc
    if count < expected_count:
        raise ConnectorOperationalError(
            f"Reference RAG provisioning wrote {expected_count} chunk(s) to "
            f"{collection.name!r} but the store reports {count}."
        )
    try:
        # include=embeddings is the load-bearing part: it is what forces the
        # vector segment reader to be constructed, which is the exact step that
        # fails on a cold read. A metadata-only get would pass over the bug.
        probe = collection.get(limit=1, include=["documents", "embeddings"])
    except Exception as exc:
        raise ConnectorOperationalError(
            f"Reference RAG provisioning wrote {collection.name!r} but its "
            f"vector segment is not readable: {exc}"
        ) from exc
    if not (probe.get("ids") or []):
        raise ConnectorOperationalError(
            f"Reference RAG provisioning wrote {expected_count} chunk(s) to "
            f"{collection.name!r} but a read returned none of them."
        )


def _chunk_metadata(record: ChunkRecord) -> dict:
    """Flatten a ChunkRecord into the metadata Chroma stores.

    ``global_index`` is legally None (a corpus-wide ordinal is a host concern),
    so the key is omitted rather than coerced -- ``int(None)`` would raise on a
    contract-legal record. ``pull_all_chunks`` reads exactly these key names.
    """
    meta = {
        "chunk_id": record.chunk_id,
        "doc_id": record.doc_id,
        "source_file": record.source_file,
        "doc_chunk_index": int(record.doc_chunk_index),
    }
    if record.global_index is not None:
        meta["global_index"] = int(record.global_index)
    if record.char_start is not None:
        meta["char_start"] = int(record.char_start)
    if record.char_end is not None:
        meta["char_end"] = int(record.char_end)
    if record.content_module_id:
        meta["content_module_id"] = record.content_module_id
    for key, value in (record.metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
    return meta


def _align_appended_ordinals(collection, collection_name: str,
                             records: list[ChunkRecord]) -> None:
    """Make an appended batch consistent with the chunks already stored.

    Two silent corruptions live in a naive append. Chunking restarts
    ``global_index`` at 0 on every provision call, so an appended batch would
    duplicate ordinals the collection already holds -- and duplicate corpus
    ordinals are the one dishonest option the validator FAILs. And a folder
    that shares a document with the collection re-produces its chunk_ids;
    Chroma's handling of duplicate ids varies by version (skip, upsert, or
    error), none of which is an honest append -- so the collision is refused
    here, by name, before anything is written.
    """
    try:
        existing = collection.get(include=["metadatas"])
    except Exception as exc:
        raise ConnectorOperationalError(
            f"Reference RAG append read failed: {exc}"
        ) from exc
    existing_ids = set(existing.get("ids") or [])
    if not existing_ids:
        return
    collisions = sorted({r.chunk_id for r in records} & existing_ids)
    if collisions:
        raise ValueError(
            f"Appending would rewrite {len(collisions)} chunk id(s) the "
            f"collection {collection_name!r} already holds (e.g. "
            f"{collisions[0]!r}). Appending is for NEW documents; to "
            "replace existing ones, re-provision without --append."
        )
    ordinals = [
        (meta or {}).get("global_index")
        for meta in (existing.get("metadatas") or [])
    ]
    supplied = [o for o in ordinals if o is not None]
    if supplied and len(supplied) == len(ordinals):
        offset = int(max(supplied)) + 1
        for record in records:
            record.global_index += offset  # freshly chunked: never None
    else:
        # The stored corpus does not carry a complete ordinal set, and the
        # validator treats mixed ordinals as ambiguous ("supply them for every
        # chunk or for none") -- so the appended batch honestly carries none.
        for record in records:
            record.global_index = None


def _ingest(collection, embedder: EmbeddingClient,
            records: list[ChunkRecord], batch_size: int = 100) -> int:
    try:
        embeddings = embedder.embed_documents([r.text for r in records])
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.add(
                ids=[r.chunk_id for r in batch],
                embeddings=[embeddings[i + j] for j in range(len(batch))],
                documents=[r.text for r in batch],
                metadatas=[_chunk_metadata(r) for r in batch],
            )
    except Exception as exc:
        raise ConnectorOperationalError(
            f"Reference RAG ingestion failed: {exc}"
        ) from exc
    return len(records)


def _chunk(documents, source_label: str, chunk_size: int,
           chunk_overlap: int) -> list[ChunkRecord]:
    if not documents:
        raise ValueError(f"No supported documents found in {source_label}")
    records = chunk_loaded_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not records:
        raise ValueError(f"No chunks produced from {source_label}")
    return records


def provision_reference_rag(
    folder_path: str | None = None,
    *,
    chunks: list[ChunkRecord] | None = None,
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    distance: str = "cosine",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    replace_existing: bool = True,
    normalized: bool | None = None,
) -> ReferenceRagConnector:
    """Build the demo corpus, then return a read-only connector onto it.

    This stands in for the ingest a real RAG system performed long before a
    host ever connected to it. The Dataset a host builds is then populated by
    *pulling* through the connector, never from these records directly.

    ``folder_path`` may be omitted, in which case the Pelorus Space corpus
    bundled with this package is used. That is not a convenience: until it
    shipped, ``pip install "rag-connector[reference]"`` installed a Reference
    RAG with no corpus to point at, so the "reference implementation you can
    run" demonstrated nothing without the reader first finding 92 documents of
    their own.

    ``chunks`` provisions from ChunkRecords the caller already has, bypassing
    the loader and the chunker entirely -- see
    :func:`provision_reference_rag_from_chunks`, which is the spelling to
    prefer. ``chunk_size``/``chunk_overlap`` then describe nothing and are
    refused rather than ignored.

    ``replace_existing`` (default True) deletes the collection and rebuilds it
    -- the name says what happens, because the caller is choosing destruction.
    ``collection_name`` is required: this function deletes what it is pointed
    at, so it will not be pointed at anything by default.
    """
    if chunks is not None:
        if folder_path is not None:
            raise ValueError(
                "Pass either folder_path or chunks, not both: chunks are "
                "already split, so a folder to chunk has nothing to do."
            )
        records = list(chunks)
        if not records:
            raise ValueError("No chunks supplied to provision from")
        source_label = f"{len(records)} supplied chunk(s)"
    elif folder_path is None:
        # Loaded inside the resource context: for a zip-imported install the
        # directory only exists for the duration of that block. The documents
        # are read fully here, so nothing downstream depends on the path
        # surviving -- only the ``document_filepath`` metadata records it.
        documents = load_bundled_corpus()
        source_label = f"the bundled {DEFAULT_BUNDLED_CORPUS!r} corpus"
        records = _chunk(documents, source_label, chunk_size, chunk_overlap)
    else:
        documents = load_documents(folder_path)
        source_label = repr(folder_path)
        records = _chunk(documents, source_label, chunk_size, chunk_overlap)

    embedder = embedding_client or get_embedding_client(
        "fastembed", model=embedding_model
    )
    client = _open_client(persist_dir)
    if replace_existing:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # nothing to replace yet
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": distance},
    )
    if not replace_existing:
        _align_appended_ordinals(collection, collection_name, records)
    written = _ingest(collection, embedder, records)
    durability_barrier(collection, written)

    return ReferenceRagConnector(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_client=embedder,
        embedding_model=embedding_model,
        distance=distance,
        normalized=normalized,
    )


def provision_reference_rag_from_chunks(
    chunks: list[ChunkRecord] | None = None,
    **kwargs,
) -> ReferenceRagConnector:
    """Stand up the Reference RAG from an EXISTING chunk set, re-chunking nothing.

    Default: the frozen chunk set bundled with this package. That is the point
    of the frozen file -- Pelorus's pre-canned extracts and clusters and
    RAGauge's pre-canned dataset cite these chunk ids, and a demo that
    re-derives the split before serving them is one chunker change away from
    citing ids that no longer exist. Reading the committed chunks cannot drift;
    re-chunking can.

    Everything else is :func:`provision_reference_rag`'s keyword contract,
    minus ``folder_path``, ``chunk_size`` and ``chunk_overlap``, which describe
    a split that has already happened.
    """
    for unusable in ("folder_path", "chunk_size", "chunk_overlap"):
        if unusable in kwargs:
            raise TypeError(
                f"provision_reference_rag_from_chunks() got {unusable!r}: the "
                "chunks are already split, so nothing here chunks anything."
            )
    if chunks is None:
        chunks = load_bundled_chunks()
    return provision_reference_rag(chunks=chunks, **kwargs)
