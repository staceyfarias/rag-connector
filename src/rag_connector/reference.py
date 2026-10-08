
"""The Reference RAG: a built-in, self-contained RAG for development/learning.

The **Reference RAG** is a *reference implementation* of the :class:`RagPipeline` black-box
interface (see ``docs/contract.md``, "The pipeline") — not a shortcut. It is meant for
development, learning, and baselining; it is **not** a production RAG.

Creating a Dataset from a source folder is one user action but two phases, and
they live in two different modules on purpose:

1. **Provision (Reference-only):** chunk + embed the source documents into an internal
   Chroma collection. This stands in for the ingest a *real* RAG system already did
   before you ever connect to it. It belongs to the demo system, NOT to the
   connector, so it lives in :mod:`rag_connector.reference_provision`.
2. **Pull (identical for every connector):** the Dataset is populated by calling the
   same :meth:`pull_all_chunks` interface a real connector exposes -- defined here.

**This class is read-only by construction.** A connector is a read interface
onto a RAG system someone else already built, so nothing in the contract
creates, clears, or writes a corpus. A host reconstructs a connector from a
persisted connection document on every run; if provisioning lived on this
class, that object would carry a ``delete_collection`` into every evaluation.
It does not, and must not.

Because dataset creation always routes through the connector interface, the Reference RAG
exercises the exact same code path a production connector would -- so it doubles as
the template for writing one. Embedding lives entirely inside this connector; the
harness core never sees vectors (the interface is embedding-agnostic).

**Store lifecycle.** Chroma 1.5.0 publishes no durability primitive -- no
``close``, ``flush``, ``persist`` or ``checkpoint`` on the client, the
collection, or the Rust ``Bindings`` behind them -- and it caches one engine
per client keyed on the raw ``persist_directory`` STRING. Two consequences are
handled here rather than left to callers, because both are invisible until they
bite in another process: :func:`canonical_persist_dir` makes every spelling of
one directory resolve to one engine, and :meth:`ReferenceRagConnector.close`
(built on :func:`release_store`) is the explicit release a writer performs
before handing the store to a separate reader. Reads additionally recover once
from the one known cold-segment failure -- see :meth:`ReferenceRagConnector._read`.
"""

from __future__ import annotations

import importlib.resources
import os
import time

from .base import (
    SCORE_CONVENTION,
    ChunkRecord,
    GeneratedAnswer,
    RagPipeline,
    RetrievedChunk,
    cosine_distance_to_similarity,
)
from .embedding import EmbeddingClient, get_embedding_client
from .errors import ConnectorOperationalError
from .fingerprints import embedding_fingerprint

# The ingest kit is NOT part of the connector contract -- it is what a host runs
# to build a corpus BEFORE a connector reads it. These two names are re-exported
# here only because they lived in this module first and callers import them from
# here; provisioning takes them from ``ingest`` directly.
from .ingest import (  # noqa: F401  (re-exported for backward compatibility)
    LoadedDocument,
    chunk_loaded_documents,
)
from .models import (
    ConnectorHealth,
    DatasetRef,
    EmbeddingBatch,
    EmbeddingSpaceDescriptor,
)
from .prompts import (
    CACHE_BREAKPOINT_AFTER_RAG_BLOCK,
    CACHE_BREAKPOINT_NONE,
    GenerationTrace,
    PromptTemplate,
    RenderedPrompt,
    render_numbered_rag_block,
    render_prompt,
    token_usage_from_response,
    validate_prompt_templates,
)
from .registry import ConnectorSpec, register_connector

DEFAULT_PERSIST_DIR = "./rag_connector_data/reference"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: The instruction text shared by every declared variant below. Factored out so
#: the variants differ in exactly ONE way -- the order of the block and the
#: question -- which is what makes comparing them a measurement of that
#: difference rather than of two unrelated prompts.
_REFERENCE_INSTRUCTIONS = (
    "Answer the question using only the supplied context. If the context "
    "is insufficient, say so. Cite supporting chunks inline as [1], [2], "
    "etc. Return ONLY JSON with keys answer (string) and citations (a list "
    "of the cited chunk_id values)."
)

#: The prompt variants the Reference RAG declares.
#:
#: Two, differing only in whether the question precedes the context, because
#: that one difference is what the cache-breakpoint rule turns on and the pair
#: demonstrates the rule better than either alone.
#:
#: ``question-first-json-v1`` is the prompt this connector has always sent,
#: byte for byte -- the string that used to be hardcoded inside ``generate``.
#: It declares ``cache_breakpoint="none"``, and that is the whole point: the
#: question sits ABOVE the context, so every prefix that reaches the block
#: changes with every question and no breakpoint there could ever be reused.
#: The contract refuses to quietly hoist the block above the query to make one
#: possible, because the reordered prompt is not the prompt this connector
#: ships, and a groundedness number measured on a prompt nobody runs is worth
#: less than no number. ``"none"`` is the honest declaration, and the cost of
#: that honesty -- no cache savings on this variant -- is the correct cost.
#:
#: ``context-first-json-v1`` is the deliberate alternative: an author who wants
#: the saving writes a SECOND variant whose template genuinely puts the context
#: first, declares ``after_rag_block``, and lets a host compare the two on the
#: axis that matters. Its breakpoint is worth having in the fixed-context case
#: -- many questions against one frozen block reuse instructions AND block --
#: and it remains a permission, not a promise: a prefix under the provider's
#: model-dependent minimum is ignored silently, so a caller applies it and then
#: checks ``trace.usage.cache_read_input_tokens``.
REFERENCE_PROMPT_TEMPLATES = validate_prompt_templates((
    PromptTemplate(
        id="question-first-json-v1",
        description=(
            "The Reference RAG's shipped prompt: question first, then the "
            "numbered context block, answering as JSON with chunk_id "
            "citations. Not cacheable in this order, and not reordered to be."
        ),
        template=(
            f"{_REFERENCE_INSTRUCTIONS}\n\n"
            "Question: {query}\n\nContext:\n{rag_block}"
        ),
        cache_breakpoint=CACHE_BREAKPOINT_NONE,
    ),
    PromptTemplate(
        id="context-first-json-v1",
        description=(
            "Same instructions and same JSON answer shape, with the context "
            "block ahead of the question so a cache breakpoint after the "
            "block can cover the instructions and the block together."
        ),
        template=(
            f"{_REFERENCE_INSTRUCTIONS}\n\n"
            "Context:\n{rag_block}\n\nQuestion: {query}"
        ),
        cache_breakpoint=CACHE_BREAKPOINT_AFTER_RAG_BLOCK,
    ),
))

#: The variant ``generate`` uses, so the legacy path and the declared path are
#: provably the same prompt rather than two that happen to look alike.
DEFAULT_REFERENCE_PROMPT_ID = "question-first-json-v1"

#: Wait before the single cold-read retry, so a store that is mid-handoff gets a
#: moment rather than being re-read instantly. Bounded and used at most once per
#: call: this is a recovery, not a retry loop.
COLD_READ_RETRY_DELAY_SECONDS = 0.25

#: Every token that must appear (case-insensitively) in a backend error before
#: :func:`is_cold_segment_read` will call it the known cold-read transient.
#:
#: Chroma 1.5.0's vector-segment reader is Rust-side; these two fragments are
#: literals in ``chromadb_rust_bindings`` ("Error creating hnsw segment reader: "
#: wrapping an inner "Nothing found on disk"). The pair is deliberately narrow.
#: "hnsw segment reader" alone would also match its siblings in the same binary
#: -- "Error reading from hnsw segment reader" and "Error constructing hnsw
#: segment reader" -- which wrap real query and index faults; "nothing found on
#: disk" alone is generic enough to catch unrelated backends. A retry that
#: swallowed either would hide corruption, which is worse than the flake it
#: replaces.
COLD_READ_SIGNATURE = ("hnsw segment reader", "nothing found on disk")


def is_cold_segment_read(exc: BaseException) -> bool:
    """Is ``exc`` the known "the vector segment is not readable yet" failure?

    Scoped to one exact backend signature (:data:`COLD_READ_SIGNATURE`) on
    purpose. The recovery it gates -- throw the engine away and re-open -- is
    only ever correct for a store whose data IS on disk; for a genuinely
    damaged one the re-open fails identically, which is the property that keeps
    this from masking corruption.
    """
    text = str(exc).lower()
    return all(token in text for token in COLD_READ_SIGNATURE)


def canonical_persist_dir(persist_dir: str) -> str:
    """The one spelling of ``persist_dir`` that Chroma must be handed.

    This is not cosmetic. Chroma caches one ``System`` -- one whole engine --
    per client, keyed by the RAW ``persist_directory`` string
    (``SharedSystemClient._get_identifier_from_settings``). So ``"./data"`` and
    ``"C:/proj/data"`` are two cache keys for one directory, and Chroma builds
    two independent engines over the same files, each with its own HNSW index
    cache and its own view of the segments. Two engines on one store is not a
    supported configuration: driving one has been observed to fail the other's
    write with ``InternalError: Error in compaction: Failed to apply logs to
    the metadata segment``.

    This library makes that easy to hit by shipping a RELATIVE
    :data:`DEFAULT_PERSIST_DIR`, which a host naturally absolutises somewhere
    between provisioning and reading. Resolving here means the provisioner and
    every later reader land on the same cache key no matter how the operator
    spelled it.

    Only the path handed to Chroma is canonicalised. ``self.persist_dir``,
    ``connection()`` and ``info()`` keep the operator's own spelling, so no
    persisted connection document changes meaning.
    """
    return os.path.normcase(os.path.realpath(str(persist_dir)))


def release_store(persist_dir: str) -> bool:
    """Stop the Chroma engine holding ``persist_dir`` and forget it. Idempotent.

    Chroma 1.5.0 has no ``close()``, ``flush()`` or ``persist()`` anywhere on
    its Python surface -- not on the client, not on the collection, not on the
    Rust ``Bindings``. The single teardown that exists is
    ``System.stop()``, which reaches ``RustBindingsAPI.stop()`` -> ``del
    self.bindings`` and so drops the Rust engine; it is reachable only through
    the private ``client._system``. This function is that teardown, given a
    name and made safe to call twice.

    ``chromadb.Client.clear_system_cache()`` is NOT an alternative. It empties
    the identifier->System dict without stopping anything, so every engine it
    forgets keeps its worker threads and file handles: looping
    ``PersistentClient`` + ``clear_system_cache`` exhausts the OS outright
    (observed: ``PanicException: OS can't spawn worker thread: Insufficient
    system resources``). Here the System is popped AND stopped.

    Returns True if an engine was found and released.

    Chroma shares one system per path, so this releases the engine for the
    DIRECTORY -- every client in this process that opened it is affected. There
    is no narrower release available; callers who need one need separate
    directories.

    Never raises. This is called on a recovery path and from ``close()``; a
    cleanup helper that throws would replace the failure being recovered from.
    """
    canonical = canonical_persist_dir(persist_dir)
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:  # pragma: no cover - chromadb is an optional extra
        return False
    cache = getattr(SharedSystemClient, "_identifier_to_system", None)
    if not isinstance(cache, dict):
        # A future Chroma reorganised its cache. Releasing is best-effort by
        # construction, so report "nothing released" rather than guessing.
        return False  # pragma: no cover - depends on a chromadb we do not pin
    system = cache.pop(canonical, None)
    if system is None:
        return False
    try:
        system.stop()
    except Exception:  # pragma: no cover - teardown is best-effort
        pass
    return True


def _collection_names(client) -> list[str]:
    """Collection names from a Chroma client, across the 0.5/0.6 API change.

    Chroma 0.6 changed ``list_collections`` to return bare name strings where
    it used to return collection objects. Both shapes are handled because this
    library pins neither, and the failure mode of guessing wrong is a picker
    listing ``<chromadb.Collection object at 0x...>``.
    """
    out = []
    for entry in client.list_collections():
        name = entry if isinstance(entry, str) else getattr(entry, "name", None)
        if name:
            out.append(str(name))
    return sorted(out)


def _reference_datasets(params: dict) -> list[DatasetRef]:
    """List the Chroma collections in a persist directory as selectable datasets.

    Spec-level (see :attr:`ConnectorSpec.list_datasets`): this needs only
    ``persist_dir``, never ``collection_name`` — the field it exists to help
    the operator choose.

    Counts come from Chroma's own ``count()``, which is a metadata read rather
    than a corpus scan, so it stays inside the "cheap and honest" bar for a
    picker. A collection that cannot be opened or counted is still LISTED, with
    ``item_count`` left None and the reason in ``metadata['error']``: a store
    with one sick collection must not lose the other nine from the list.
    """
    persist_dir = str(params.get("persist_dir") or DEFAULT_PERSIST_DIR)
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'Listing Reference RAG datasets requires the "reference" extra: '
            'pip install "rag-connector[reference]"'
        ) from exc
    try:
        # Canonicalised so a picker shares the engine the connector will use
        # rather than standing up a second one over the same files.
        client = chromadb.PersistentClient(path=canonical_persist_dir(persist_dir))
        names = _collection_names(client)
    except Exception as exc:
        raise ConnectorOperationalError(
            f"Reference RAG dataset listing failed for {persist_dir!r}: {exc}"
        ) from exc

    datasets: list[DatasetRef] = []
    for name in names:
        count: int | None = None
        error: str | None = None
        try:
            count = int(client.get_collection(name).count())
        except Exception as exc:
            error = str(exc)
        datasets.append(DatasetRef(
            id=name,
            label=name,
            item_count=count,
            connect_params={"persist_dir": persist_dir, "collection_name": name},
            metadata={"backend": "chroma", "kind": "collection",
                      **({"error": error} if error else {})},
        ))
    return datasets


def _corpus_order(chunk: ChunkRecord) -> tuple:
    """Deterministic corpus order that tolerates a missing ordinal.

    The corpus-wide ordinal is the intended order, but it is legally ``None``,
    and this sort runs BEFORE ``validate_chunks`` -- so it also has to survive a
    row whose metadata is broken rather than raising an unreadable
    ``'<' not supported between 'NoneType' and 'int'`` from inside a sort key.
    Ordinal-bearing chunks lead, in ordinal order; the rest fall back to
    (document, within-document ordinal), which is stable and still meaningful.
    """
    global_index = chunk.global_index
    doc_chunk_index = chunk.doc_chunk_index
    return (
        global_index is None,
        global_index if global_index is not None else 0,
        chunk.doc_id or "",
        doc_chunk_index if doc_chunk_index is not None else -1,
    )


class ReferenceRagConnector(RagPipeline):
    """Built-in reference RAG (FastEmbed + Chroma) behind the black-box interface."""

    name = "reference-fastembed"

    def __init__(
        self,
        *,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str,
        embedding_client: EmbeddingClient | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        distance: str = "cosine",
        normalized: bool | None = None,
        top_k: int | None = None,
    ):
        self.persist_dir = persist_dir
        # The connector's own retrieval depth: how many chunks query() returns
        # when the caller passes no top_k (the evaluation-host case). Part of
        # the system's configuration, so it lives in the connection document;
        # absent means the historical default of 5.
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool)
                                  or top_k < 1):
            raise ValueError(f"top_k must be a positive int or None (got {top_k!r})")
        self.top_k = top_k
        self.collection_name = collection_name
        self.distance = distance
        # Tri-state, and deliberately NOT inferred from the model id: this is
        # the connector author's declaration about the space (see
        # EmbeddingSpaceDescriptor). None means "not stated", which a host reads
        # as unverifiable -- distinct from an asserted False. Defaulting it to a
        # guess would turn "we don't know" into a claim, which is the one thing
        # the tri-state exists to prevent.
        if normalized is not None and not isinstance(normalized, bool):
            raise ValueError(
                f"normalized must be True, False, or None (got "
                f"{normalized!r}). It is a tri-state declaration about the "
                "embedding space, not a free-form value."
            )
        self.normalized = normalized
        self._embed = embedding_client or get_embedding_client(
            "fastembed", model=embedding_model
        )
        # The operator's spelling is kept above for connection()/info(); Chroma
        # is only ever handed the resolved one. See canonical_persist_dir.
        self._chroma_path = canonical_persist_dir(persist_dir)
        self._client = None
        self._collection = None
        self._open()

    def _open(self) -> None:
        """Build (or rebuild) the Chroma client and collection handle.

        Factored out because three callers need exactly this: construction,
        re-opening after :meth:`close`, and the cold-read recovery in
        :meth:`_read`.
        """
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                'The Reference RAG store requires the "reference" extra: '
                'pip install "rag-connector[reference]"'
            ) from exc
        self._client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )

    def close(self) -> bool:
        """Release the Chroma engine backing this connector. Idempotent.

        The handoff barrier a provisioning-side caller wants before another
        PROCESS reads the store it just wrote. Chroma offers no flush, so what
        is on offer is ownership: without this, the engine that opened the
        directory stays alive -- with its worker threads, its open segment
        files and its HNSW cache -- for the rest of the process's life, whether
        or not anyone still holds the client. A second process reading the same
        directory is then sharing it with a live engine, which Chroma's local
        persistent client does not support.

        This is a release, not a poison pill: a later read re-opens
        automatically, so calling it early costs one re-open and never an
        error. See :func:`release_store` for what "release" can mean here.

        Scope, because Chroma's is wider than this object: the engine is
        per-DIRECTORY, not per-connector. Anything else in this process holding
        a client on the same persist_dir is holding the SAME engine, and this
        stops it. That is inherent -- Chroma shares one system per path, so
        there is no narrower release to perform -- but it makes ``close()`` an
        owner's call, not something to sprinkle after every read. Other
        connectors on the directory recover by re-opening; a raw
        ``chromadb`` client held elsewhere does not.
        """
        self._collection = None
        self._client = None
        return release_store(self._chroma_path)

    def _read(self, operation, what: str):
        """Run a backend read, recovering ONCE from the known cold-read failure.

        ``operation`` is a zero-argument callable that must reach through
        ``self._collection`` when it runs, not close over the handle -- the
        recovery replaces that handle before retrying.

        The recovery is: drop the engine (:func:`release_store`), wait
        :data:`COLD_READ_RETRY_DELAY_SECONDS`, re-open, run the read again,
        once. That is the only lever Chroma 1.5.0 gives for this state, and it
        is the same thing that makes a re-run of a failed job succeed against
        the identical files -- done here instead of by hand.

        Everything else raises on the first failure, unchanged. And the retry
        cannot launder a broken store: a segment that is genuinely absent is
        just as absent to a brand-new engine, so the second attempt fails and
        the error is reported -- with its history -- as a
        ``ConnectorOperationalError``. A backend failure must never reach a
        host as an empty result.
        """
        if self._collection is None:
            self._open()
        try:
            return operation()
        except Exception as exc:
            if not is_cold_segment_read(exc):
                raise ConnectorOperationalError(
                    f"Reference RAG {what} failed: {exc}"
                ) from exc
            cold = exc
        release_store(self._chroma_path)
        time.sleep(COLD_READ_RETRY_DELAY_SECONDS)
        try:
            self._open()
            return operation()
        except Exception as exc:
            raise ConnectorOperationalError(
                f"Reference RAG {what} failed: {exc} (still unreadable after "
                f"releasing the Chroma engine for {self._chroma_path!r} and "
                f"re-opening it, so the vector segment is missing rather than "
                f"cold; first failure: {cold})"
            ) from exc

    # NOTE: there is deliberately no provisioning, reset, or ingest method on
    # this class. A connector is a READ interface onto a system someone else
    # built; nothing in the contract creates or clears a corpus. The Reference
    # RAG's own build path lives in :mod:`rag_connector.reference_provision`,
    # so a connector reconstructed from a persisted connection document has no
    # way to destroy the corpus it is meant to read. See that module.

    #: Retrieval depth when neither the caller nor the connection sets one.
    DEFAULT_TOP_K = 5

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Ranked retrieval: embed the query, search Chroma, map to RetrievedChunks.

        The reference pattern for the two rules that matter: ``chunk_id`` comes
        from the stored metadata (the SAME logical id ``pull_all_chunks``
        returns, not a raw store id), and ``score`` is CANONICAL -- Chroma
        returns a cosine *distance* (lower = closer), so it is converted with
        ``cosine_distance_to_similarity`` (1 = exact match) and the store-native
        distance is preserved in ``raw_score`` as evidence. A backend failure is
        raised as ``ConnectorOperationalError`` so a host records a retrieval
        error -- never a fake empty-result zero.

        Depth: the caller's ``top_k`` when given, else this connection's own
        ``top_k``, else :attr:`DEFAULT_TOP_K`.
        """
        depth = top_k or self.top_k or self.DEFAULT_TOP_K
        try:
            emb = self._embed.embed_query(text)
        except Exception as exc:
            raise ConnectorOperationalError(
                f"Reference RAG query failed: {exc}"
            ) from exc
        res = self._read(
            lambda: self._collection.query(
                query_embeddings=[emb],
                n_results=depth,
                include=["documents", "metadatas", "distances"],
            ),
            "query",
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            meta = dict(meta or {})
            out.append(RetrievedChunk(
                chunk_id=meta.get("chunk_id", cid),
                text=doc or "",
                score=cosine_distance_to_similarity(float(dist)),
                rank=rank,
                doc_id=meta.get("doc_id"),
                raw_score=float(dist),
                metadata=meta,
            ))
        return out

    def pull_all_chunks(self) -> list[ChunkRecord]:
        """Rebuild the full contract from store metadata, then sort + validate.

        The reference pattern for corpus enumeration: contract fields are
        reconstructed from what ``_metadata`` persisted (a round-trip through
        the store), everything else goes back into the ``metadata`` passthrough,
        and the result is sorted by ``global_index`` and validated fail-loud
        BEFORE it is returned -- never downstream where a violation would
        surface as a mysteriously broken answer key.
        """
        res = self._read(
            lambda: self._collection.get(
                include=["documents", "metadatas", "embeddings"]
            ),
            "corpus read",
        )
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        embs = res.get("embeddings")
        if embs is None:
            embs = [None] * len(ids)
        known = {
            "chunk_id", "doc_id", "source_file", "doc_chunk_index", "global_index",
            "content_module_id", "char_start", "char_end",
        }
        chunks = []
        for cid, doc, meta, emb in zip(ids, docs, metas, embs):
            meta = dict(meta or {})
            # Absent ordinals stay absent. A ``-1`` sentinel here would be a
            # number the store never held: it passes validate_chunks (which
            # rejects None, not -1), so a row that lost its metadata would be
            # returned as a valid chunk claiming ordinal -1, defeating the
            # fail-loud pull this method promises. None means "no ordinal" --
            # legal for global_index, a contract violation for doc_chunk_index,
            # and validate_chunks below draws exactly that line.
            raw_doc_index = meta.get("doc_chunk_index")
            raw_global_index = meta.get("global_index")
            chunks.append(ChunkRecord(
                chunk_id=meta.get("chunk_id", cid),
                doc_id=meta.get("doc_id", ""),
                source_file=meta.get("source_file", ""),
                doc_chunk_index=(
                    int(raw_doc_index) if raw_doc_index is not None else None
                ),
                global_index=(
                    int(raw_global_index) if raw_global_index is not None else None
                ),
                text=doc or "",
                embedding=list(emb) if emb is not None else None,
                char_start=meta.get("char_start"),
                char_end=meta.get("char_end"),
                content_module_id=meta.get("content_module_id"),
                metadata={k: v for k, v in meta.items() if k not in known},
            ))
        chunks.sort(key=_corpus_order)
        self.validate_chunks(chunks)
        return chunks

    # ``list_chunks`` is inherited. It used to be reimplemented here with
    # exactly the base's logic — slice ``pull_all_chunks`` — which is now the
    # default, so the override said nothing the base did not already say.

    def list_datasets(self) -> list[DatasetRef]:
        """List the sibling collections in this instance's persist directory.

        The reference pattern for the rule that matters here: listing does not
        rebind. ``self`` still reads ``self.collection_name`` afterwards; the
        returned ``connect_params`` are what a HOST feeds back through
        ``connect`` to construct a different instance. A connector that
        reassigned its own collection here would silently change what every
        already-issued fingerprint refers to.
        """
        return _reference_datasets({"persist_dir": self.persist_dir})

    def get_chunk_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        """Read the vectors Chroma actually stores for ``ids`` (see the base docstring).

        The reference pattern for the rule that makes this capability worth
        anything: the vectors come out of the collection, they are never
        recomputed with ``self._embed``. A re-embedding would agree with the
        index by construction and so would pass while the index was stale or
        written in a different embedding space.

        Two Chroma details a copying connector author should notice. Chroma
        returns embeddings as numpy arrays in recent versions, and a numpy
        array's truthiness raises -- hence the explicit ``is None`` checks and
        the conversion to plain floats before the value leaves the boundary.
        And ``get()`` omits unknown ids and may reorder the ones it knows, so
        the dict is built by zipping the RETURNED ids with the returned
        embeddings; assuming request order would mislabel every vector.
        """
        if not ids:
            # Nothing was asked for, so nothing can be missing: an empty dict is
            # the complete and correct answer without a backend round-trip.
            return {}
        res = self._read(
            lambda: self._collection.get(
                ids=list(ids),
                include=["embeddings", "metadatas"],
            ),
            "chunk-vector read",
        )
        got_ids = res.get("ids")
        embeddings = res.get("embeddings")
        if got_ids is None or embeddings is None:
            return {}
        metas = res.get("metadatas")
        if metas is None:
            metas = [None] * len(got_ids)
        vectors: dict[str, list[float]] = {}
        for cid, emb, meta in zip(got_ids, embeddings, metas):
            if emb is None:
                # An indexed chunk with no stored vector is a gap, not a zero
                # vector: leave it out so the caller can see it is unverified.
                continue
            key = (dict(meta or {})).get("chunk_id", cid)
            vectors[key] = [float(value) for value in emb]
        return vectors

    def describe_space(self) -> EmbeddingSpaceDescriptor:
        """Identify the embedding space, asserting only what was declared.

        ``normalized`` is passed through from the constructor rather than
        inferred from the model id. The embedding client is swappable here, so
        a model-name lookup would confidently describe the geometry of a
        *different* embedder than the one actually answering. A host reads
        ``None`` as "cannot verify" and a bool as an assertion, and those must
        not be conflated -- Pelorus, for one, disables its curated tiers on an
        asserted mismatch but only warns on an unstated one.
        """
        return EmbeddingSpaceDescriptor(
            model=self._embed.model_id,
            dimension=self._embed.dimensions,
            normalized=self.normalized,
            metric=self.distance,
            fingerprint=self._embedding_fingerprint(),
        )

    def embed_queries(self, texts: list[str]) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=[self._embed.embed_query(text) for text in texts],
            space=self.describe_space(),
        )

    def health(self) -> ConnectorHealth:
        # Routed through _read so a store that is merely cold reports healthy
        # after the re-open rather than reporting a scary false negative --
        # while a store that is actually broken still reports not-ok.
        try:
            count = self._read(lambda: self._collection.count(), "health check")
        except Exception as exc:
            return ConnectorHealth(ok=False, detail=str(exc))
        return ConnectorHealth(ok=True, detail=f"{count} chunks")

    # -- Generation ---------------------------------------------------------
    #
    # The Reference RAG implements BOTH generation paths, and they share one
    # declared template: ``generate`` retrieves and then renders
    # ``question-first-json-v1``; ``generate_from_prompt`` runs a prompt the
    # host rendered. The prompt string that used to be hardcoded here is now
    # ``REFERENCE_PROMPT_TEMPLATES[0]``, so the legacy path sends the same
    # bytes it always did while the template is finally inspectable, hashable,
    # and choosable before anything runs.

    def prompt_templates(self) -> list[PromptTemplate]:
        """The declared prompt variants (see :data:`REFERENCE_PROMPT_TEMPLATES`).

        A copy of the module-level declaration, which the registry spec serves
        too -- one declared set behind both surfaces, so the variant an
        operator picked from a listing is the variant a bound instance runs.
        """
        return list(REFERENCE_PROMPT_TEMPLATES)

    def render_rag_block(self, chunks) -> str:
        """Render supplied chunks the way this connector renders retrieved ones.

        Identical to what ``generate`` has always built from its own retrieval
        -- ``[n] chunk_id=...`` then the chunk text, blank line between -- so a
        host-supplied block and a self-retrieved one are the same format, and a
        fixed-context measurement is measuring the real prompt.
        """
        return render_numbered_rag_block(chunks)

    def generate(self, text: str, *, top_k: int | None = None, llm=None) -> GeneratedAnswer:
        """Retrieve then synthesize a cited answer (reference connector path)."""
        if llm is None:
            return GeneratedAnswer(
                query=text,
                answer="",
                contexts=self.query(text, top_k=top_k),
                citations=[],
                hydrated_prompt="",
                error="Reference RAG generation requires an LLM",
            )
        contexts = self.query(text, top_k=top_k)
        prompt = render_prompt(
            reference_prompt_template(DEFAULT_REFERENCE_PROMPT_ID),
            query=text,
            rag_block=self.render_rag_block(contexts),
            chunk_ids=[str(c.chunk_id) for c in contexts],
        )
        return self.generate_from_prompt(prompt, contexts=contexts, llm=llm)

    def generate_from_prompt(
        self,
        prompt: RenderedPrompt,
        *,
        contexts=(),
        llm=None,
        decoding=None,
    ) -> GeneratedAnswer:
        """Generate from an already-rendered prompt, returning answer + trace.

        No retrieval: ``prompt`` carries the messages and ``contexts`` carries
        the chunks the caller rendered the block from, which is what makes the
        returned :class:`GeneratedAnswer` interchangeable with ``generate``'s
        for every existing consumer.

        ``hydrated_prompt`` is populated from ``prompt.text`` and ``trace``
        from the same call, so the legacy string and the structured evidence
        cannot disagree.

        ``cache_breakpoint_applied`` stays ``False`` here. The Reference RAG
        talks to a duck-typed ``llm`` handle and never constructs a provider
        request, so it is in no position to place a breakpoint -- and recording
        that honestly is the point: a caller reading
        ``cache_read_input_tokens == 0`` needs to know whether a breakpoint was
        ever applied before concluding anything about caching.
        """
        context_list = list(contexts)
        # Declared defaults first, caller override on top. The Reference RAG
        # cannot APPLY either -- its ``llm`` handle was configured by whoever
        # built it and ``invoke`` takes no decoding kwargs -- so what is
        # recorded here is the declaration. A production connector that owns
        # its generation call should send these and record what it sent.
        effective = {**dict(prompt.decoding), **dict(decoding or {})}

        if llm is None:
            message = "Reference RAG generation requires an LLM"
            return GeneratedAnswer(
                query=prompt.query, answer="", contexts=context_list,
                citations=[], hydrated_prompt=prompt.text, error=message,
                trace=GenerationTrace(prompt=prompt, decoding=effective,
                                      model=_llm_model_name(llm), error=message),
            )

        raw = ""
        response = None
        try:
            response = llm.invoke(_llm_payload(prompt))
            raw = str(response.content)
            answer, citations = _parse_cited_json(raw, context_list)
            error = None
        except Exception as exc:
            # A RUNTIME generation failure is captured, never raised: the run
            # records the failure alongside the prompt that produced it, which
            # is what makes it diagnosable rather than merely fatal.
            answer, citations, error = "", [], str(exc)

        return GeneratedAnswer(
            query=prompt.query, answer=answer, contexts=context_list,
            citations=citations, hydrated_prompt=prompt.text, error=error,
            trace=GenerationTrace(
                prompt=prompt,
                response=raw,
                model=_llm_model_name(llm, response),
                decoding=effective,
                usage=token_usage_from_response(response),
                error=error,
            ),
        )

    def info(self) -> dict:
        emb_fp = self._embedding_fingerprint()
        return {
            "name": self.name,
            "type": type(self).__name__,
            "connector_type": "reference",
            "retrieval_mode": self.retrieval_mode,
            "persist_dir": self.persist_dir,
            "collection": self.collection_name,
            "embedding_provider": "fastembed",
            "embedding_model": self._embed.model_id,
            "dimensions": self._embed.dimensions,
            "distance": self.distance,          # store metric (informational)
            "score_convention": SCORE_CONVENTION,  # scores are similarity, 1=exact
            # Through _read so info() still answers on a connector whose engine
            # has been released by close(), rather than raising on a None handle.
            "chunk_count": self._read(lambda: self._collection.count(), "info"),
            "embedding_fingerprint": emb_fp,
        }

    def run_snapshot(self) -> dict:
        """Return cheap, secret-free context that can explain run deltas.

        These are properties of the live Reference RAG binding, not evaluation
        scores. In particular, ``chunk_count`` is an observation: a host may
        show that the store changed between two captures, but must not place it
        on the same scale as retrieval quality metrics.

        Chunk size and overlap are intentionally absent. The read-only Chroma
        binding cannot prove which ingest settings produced a legacy
        collection (or whether an appended collection used one profile), so
        reporting constructor defaults here would turn a guess into evidence.
        """
        return {
            "schema": "reference-rag.run-snapshot.v1",
            "settings": {
                "backend": "chroma",
                "collection": self.collection_name,
                "embedding_provider": "fastembed",
                "embedding_model": self._embed.model_id,
                "embedding_dimensions": self._embed.dimensions,
                "embedding_fingerprint": self._embedding_fingerprint(),
                "distance": self.distance,
                "normalized": self.normalized,
                "retrieval_mode": self.retrieval_mode,
                "score_convention": SCORE_CONVENTION,
            },
            "observations": {
                "chunk_count": self._read(
                    lambda: self._collection.count(), "run snapshot"
                ),
            },
        }

    def _embedding_fingerprint(self) -> str:
        return embedding_fingerprint(
            provider="fastembed",
            model=self._embed.model_id,
            dimensions=self._embed.dimensions,
        )

    def connection(self) -> dict:
        """Return the secret-free document required to reopen this instance.

        ``normalized`` travels with it. The declaration is a property of the
        space, so a connector rebuilt from this dict in a fresh process has to
        describe that space the same way -- otherwise the assertion silently
        decays to "unknown" on the reopen, and a host that gated on it at
        freeze time quietly stops gating on it at evaluation time.
        """

        doc = {
            "type": "reference",
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "embedding_model": self._embed.model_id,
            "distance": self.distance,
            "normalized": self.normalized,
        }
        # The retrieval depth travels only when one was set, so a default
        # connection's document is the same as before 2026-10-08.
        if self.top_k is not None:
            doc["top_k"] = self.top_k
        return doc


def reference_prompt_template(template_id: str) -> PromptTemplate:
    """Look up one declared Reference RAG variant by id.

    Raises :class:`KeyError` for an unknown id rather than falling back to a
    default. A host that asked for a specific prompt and silently got a
    different one would produce a result labelled with the id it requested and
    measured on the prompt it did not.
    """
    for template in REFERENCE_PROMPT_TEMPLATES:
        if template.id == template_id:
            return template
    known = ", ".join(t.id for t in REFERENCE_PROMPT_TEMPLATES)
    raise KeyError(
        f"Reference RAG declares no prompt template {template_id!r}; "
        f"declared ids are: {known}"
    )


def _reference_prompt_templates() -> list[PromptTemplate]:
    """Zero-arg declaration for the registry, readable BEFORE anything is bound.

    Same list :meth:`ReferenceRagConnector.prompt_templates` serves, from the
    same module-level constant -- a host that renders a prompt picker off a
    registry listing and a host that reads a connected instance must not be
    shown two different sets.
    """
    return list(REFERENCE_PROMPT_TEMPLATES)


def _llm_payload(prompt: RenderedPrompt):
    """What to hand the duck-typed ``llm`` handle for this rendered prompt.

    A prompt with no system message is passed as the single user string --
    byte-identical to the call this connector has always made, so the legacy
    path is unchanged by the template refactor. A prompt WITH a system message
    is passed as role/content pairs, the form the common client libraries
    accept, because flattening a system preamble into the user turn would
    quietly change what the model was told.
    """
    system = prompt.system_message
    if system is None:
        return prompt.user_message.content
    return [("system", system.content), ("human", prompt.user_message.content)]


def _llm_model_name(llm, response=None) -> str | None:
    """Best-effort model identity, preferring what the RESPONSE reported.

    The response is authoritative when it answers: a handle configured with an
    alias, or a service that resolved a version, would otherwise record the
    name that was asked for rather than the model that ran. ``None`` when
    nothing says -- unreported, not guessed.
    """
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        for key in ("model_name", "model"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("model_name", "model", "model_id", "deployment_name"):
        value = getattr(llm, key, None)
        if isinstance(value, str) and value:
            return value
    return None


def _parse_cited_json(raw: str, contexts) -> tuple[str, list[str]]:
    """Parse the JSON answer + citations this connector's prompts ask for.

    Extracted verbatim from ``generate`` so both generation paths normalize
    citations identically; the rules are unchanged. A rank citation (``"1"``)
    maps to that context's ``chunk_id``; an unmappable (hallucinated) citation
    is kept VERBATIM so scoring assigns it zero support instead of the capture
    silently improving precision by dropping it.
    """
    import json

    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("generation response was not JSON")
    parsed = json.loads(raw[start:end + 1])
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        raise ValueError("generation response had no answer")
    valid_ids = {str(c.chunk_id) for c in contexts}
    citations: list[str] = []
    raw_citations = list(parsed.get("citations") or [])
    if not raw_citations:
        import re as _re
        raw_citations = _re.findall(r"\[(\d+)\]", answer)
    for raw_cid in raw_citations:
        cid = str(raw_cid).strip()
        if not cid:
            continue
        if cid in valid_ids:
            normalized = cid
        elif cid.isdigit() and 1 <= int(cid) <= len(contexts):
            normalized = str(contexts[int(cid) - 1].chunk_id)
        else:
            normalized = cid
        if normalized not in citations:
            citations.append(normalized)
    return answer, citations


def _build_reference(connection: dict) -> ReferenceRagConnector:
    """Reconstruct a Reference RAG from a Dataset's persisted connection."""
    return ReferenceRagConnector(
        persist_dir=connection.get("persist_dir", DEFAULT_PERSIST_DIR),
        collection_name=connection["collection_name"],
        embedding_model=connection.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        distance=connection.get("distance", "cosine"),
        # Absent on connections persisted before the declaration existed, which
        # is exactly the tri-state's "not stated" -- no migration needed.
        normalized=connection.get("normalized"),
        # Absent on connections persisted before 2026-10-08 and whenever the
        # operator sets none: the historical depth, 5.
        top_k=connection.get("top_k"),
    )


def _connect_reference(params: dict) -> tuple[ReferenceRagConnector, dict]:
    """Open an already-provisioned Reference RAG as a shareable black box."""

    collection_name = str(params.get("collection_name") or "").strip()
    if not collection_name:
        raise ValueError("collection_name is required")
    connection = {
        "type": "reference",
        "persist_dir": str(params.get("persist_dir") or DEFAULT_PERSIST_DIR),
        "collection_name": collection_name,
        "embedding_model": str(
            params.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        ),
        "distance": str(params.get("distance") or "cosine"),
        "normalized": params.get("normalized"),
    }
    # Stored only when the operator sets a depth, so a default connection's
    # document (and its configuration fingerprint) is unchanged.
    if params.get("top_k") not in (None, ""):
        connection["top_k"] = int(params["top_k"])
    return _build_reference(connection), connection


def _reference_connector_info() -> str:
    """Long-form markdown help for the Reference RAG, read from the package.

    Read through :mod:`importlib.resources` rather than a path relative to this
    file: an editable install resolves to ``src/`` on disk, but a wheel install
    does not, and a relative path would work locally and fail once installed.
    """

    return (
        importlib.resources.files(__package__)
        .joinpath("CONNECTOR-INFO.md")
        .read_text(encoding="utf-8")
    )


register_connector(ConnectorSpec(
    connector_type="reference",
    label="Reference RAG (local)",
    description="Known-good local RAG (FastEmbed + Chroma), provisioned from a source folder.",
    build=_build_reference,
    connect=_connect_reference,
    list_datasets=_reference_datasets,
    connector_info=_reference_connector_info,
    prompt_templates=_reference_prompt_templates,
    params=[
        {
            "name": "persist_dir",
            "label": "Persistence directory",
            "type": "string",
            "required": True,
            "default": DEFAULT_PERSIST_DIR,
        },
        {
            "name": "collection_name",
            "label": "Collection name",
            "type": "string",
            "required": True,
        },
        {
            "name": "embedding_model",
            "label": "FastEmbed model",
            "type": "string",
            "required": True,
            "default": DEFAULT_EMBEDDING_MODEL,
        },
        {
            "name": "top_k",
            "label": "Results per query (retrieval depth)",
            "type": "integer",
            "required": False,
            "help": "How many chunks this RAG returns per query (default 5). Part of "
                    "the system under test; an evaluation host does not override it.",
        },
    ],
))
