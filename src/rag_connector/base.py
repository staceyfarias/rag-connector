"""
The RagPipeline port: the contract every RAG "black box" must satisfy.

A host never needs to know *how* a pipeline chunks, embeds, or retrieves. It
only needs two capabilities:

  * ``query(text, top_k)``      -- run retrieval and get back ranked chunks.
  * ``pull_all_chunks()``       -- enumerate the whole corpus with metadata,
                                  so a host can freeze it as evaluation
                                  evidence.

It also exposes ``get_chunks(chunk_ids)`` for on-demand inspection and drift
diagnostics. The base implementation is correct but expensive (it filters a full
``pull_all_chunks()``); production connectors should override it with direct
by-id retrieval.

Anything implementing those two -- plus a corpus ``fingerprint`` for change
detection -- plugs into a host unchanged. The optional
:class:`~rag_connector.reference.ReferenceRagConnector` is the bundled
known-good implementation; a production connector is just another subclass.

The :class:`ChunkRecord` metadata contract (see ``docs/contract.md``) is what
makes everything downstream work: stable ids for ID-based relevance judgements,
``doc_id`` + ``doc_chunk_index`` for neighbor-window reconstruction, and the
text for content-based judgements.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import ConnectorContractError, UnsupportedCapability

if TYPE_CHECKING:  # ``models`` imports FROM this module, so the runtime import
    from collections.abc import Mapping, Sequence

    from .models import ChunkPage, DatasetRef  # lives inside the method (below).

    # ``prompts`` imports FROM this module too, so the dependency runs one way
    # only: prompts -> base at runtime, base -> prompts for annotations. With
    # ``from __future__ import annotations`` these names never need to exist at
    # runtime, so no import cycle is created.
    from .prompts import GenerationTrace, PromptTemplate, RenderedPrompt

#: Page size the derived ``pull_all_chunks`` asks for. Large enough that paging
#: a big corpus is not chatty, small enough that one page is not a whole-corpus
#: materialization for a connector whose store honours the limit.
_CORPUS_PAGE_SIZE = 1000


@dataclass
class ChunkRecord:
    """One chunk in the corpus, with the metadata every downstream stage needs.

    Required fields are the contract; optional fields are best-effort enrichments
    (a connector that can't supply them should leave them ``None``).

    **``global_index`` is legally ``None``.** A corpus-wide ordinal is not
    something a black-box RAG can generally be asked for — it presupposes a
    stable total order over the corpus, which a paged or sharded store may not
    have and a hosted retrieval API will not expose. Requiring it was one
    product's need (ordinal-based neighbor expansion) sitting in shared
    vocabulary. A host that needs ordinals assigns them at ingest, where it
    knows the order it read things in; a connector that has none passes
    ``None`` and says so honestly.

    It has **no default**, deliberately. ``text`` follows it, so adding one
    would force either a default on ``text`` — which is never absent — or a
    field reorder, and reordering a dataclass whose consumers may construct
    positionally misassigns silently. Passing ``global_index=None`` explicitly
    is the intended usage.

    ``doc_chunk_index`` stays required. It is the *within-document* ordinal, and
    the argument above does not transfer: a store that returns a chunk can
    almost always say where in its document it sits, and no host currently
    backfills it — relaxing a constraint whose compensating control does not
    exist just moves the failure somewhere quieter.
    """

    chunk_id: str                       # stable unique id (store primary key)
    doc_id: str                         # document identity (e.g. source_file)
    source_file: str                    # human-readable document name
    doc_chunk_index: int                # ordinal within its document (for +/-N neighbors)
    global_index: int | None            # corpus-wide ordinal; None = host assigns it
    text: str                           # the chunk text
    embedding: list[float] | None = None
    char_start: int | None = None    # offset into source doc (best-effort)
    char_end: int | None = None
    content_module_id: str | None = None  # product module/channel (stubbed in dev)
    metadata: dict = field(default_factory=dict)  # passthrough for anything extra

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "doc_chunk_index": self.doc_chunk_index,
            "global_index": self.global_index,
            "text": self.text,
            "embedding": self.embedding,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_module_id": self.content_module_id,
            "metadata": self.metadata,
        }


# Required ChunkRecord fields, validated on pull so a bad ingest fails loud.
#
# ``global_index`` is deliberately ABSENT (see ChunkRecord's docstring): a
# corpus-wide ordinal is a host concern, assigned at ingest by whoever knows the
# order they read the corpus in, not something a black box can be asked for. A
# host that needs ordinals enforces and backfills them itself — that is the
# "library standardizes the vocabulary, each product owns its validator" rule
# applied to a field rather than a type.
REQUIRED_CHUNK_FIELDS = ("chunk_id", "doc_id", "source_file",
                         "doc_chunk_index", "text")


# ---------------------------------------------------------------------------
# The score convention (canonical since 2026-07-22): ``RetrievedChunk.score``
# is ALWAYS higher-is-better -- "1.0 = exact match" for cosine similarity.
# Native scale is kept (no invented normalization: L2 has no principled [0,1]
# mapping, and cosine similarity may legitimately be negative, range -1..1).
# A store that returns distances converts with the utilities below and keeps
# the store-native value in ``raw_score`` as frozen evidence.
# ---------------------------------------------------------------------------

def cosine_distance_to_similarity(distance: float) -> float:
    """Chroma-style cosine distance (0=identical .. 2) -> cosine similarity
    (1=identical .. -1). Exact conversion: ``similarity = 1 - distance``."""
    return 1.0 - float(distance)


def l2_distance_to_score(distance: float) -> float:
    """L2/euclidean distance -> canonical direction by NEGATION (0 = exact,
    more negative = farther). This is a labeled direction-flip, not a
    similarity: L2 is unbounded, so any [0,1] normalization would invent a
    convention the raw number doesn't contain."""
    return -float(distance)


def similarity_passthrough(score: float) -> float:
    """Identity for scores that are already higher-is-better (ip/dot-product,
    rerankers, Pinecone similarities). Exists so a connector can state its
    conversion explicitly instead of leaving readers to infer 'no-op'."""
    return float(score)


# Stamped into run metadata by capture so downstream display knows how to read
# stored scores; artifacts without it predate the convention (legacy distance
# semantics, direction inferred from the declared ``distance``).
SCORE_CONVENTION = "higher_is_better"


# ---------------------------------------------------------------------------
# Retrieval mode: a single connector-declared enum naming how a system bounds
# its returned list and whether its per-chunk scores are meaningful. See
# docs/contract.md, "Retrieval modes". Every connector inherits ``scored`` --
# the one behavior that existed before this convention -- so the stamp's legacy
# meaning and the default coincide.
#   scored       -- the connector's configured depth bounds the list (a caller's
#                  top_k overrides it); scores canonical (threshold OK)
#   ordered      -- as scored for cardinality; scores meaningless (evidence
#                  only); threshold REFUSED
#   complete_set -- the SYSTEM decides cardinality; top_k is at most a
#                  pass-through breadth hint, never the metrics cutoff; scores
#                  evidence only; threshold REFUSED; metrics switch to @set
#
# top_k is UNSET by default (2026-10-08): ``query(text)`` retrieves however the
# connector is configured. An evaluation host observes the system as configured
# and does not pass top_k at all; its own metrics cutoff (the k of @k metrics)
# is a host setting that never reaches the connector.
# ---------------------------------------------------------------------------

RETRIEVAL_MODES = ("scored", "ordered", "complete_set")
DEFAULT_RETRIEVAL_MODE = "scored"


class RetrievalModeError(ValueError):
    """An illegal retrieval-mode override or a mode/threshold conflict."""


def resolve_retrieval_mode(live_mode: str | None,
                           dataset_override: str | None) -> tuple[str, str]:
    """Resolve the effective retrieval mode from the live declaration + override.

    ``live_mode`` is what the connector declares now (``None`` -> default
    ``scored``). ``dataset_override`` is the optional per-Dataset opinion stored
    in the connection dict. The ONLY legal override is ``scored`` -> ``ordered``
    ("I distrust this connector's scores"); everything else is refused loudly:
    you cannot mint meaning into scores (``ordered``/``complete_set`` -> anything
    with meaningful scores), and cardinality is a behavioral fact of the system,
    not an operator opinion (any override touching ``complete_set``).

    Returns ``(resolved_mode, source)`` where source is ``"connector"`` or
    ``"dataset_override"``. Raises :class:`RetrievalModeError` on an illegal
    override or an unknown mode value.
    """
    mode = (live_mode or DEFAULT_RETRIEVAL_MODE)
    if mode not in RETRIEVAL_MODES:
        raise RetrievalModeError(
            f"Connector declares unknown retrieval_mode {mode!r}; "
            f"must be one of {RETRIEVAL_MODES}.")
    if dataset_override is None:
        return mode, "connector"
    if dataset_override not in RETRIEVAL_MODES:
        raise RetrievalModeError(
            f"Dataset retrieval_mode override {dataset_override!r} is not a "
            f"valid mode; must be one of {RETRIEVAL_MODES}.")
    if dataset_override == mode:
        return mode, "connector"
    if mode == "scored" and dataset_override == "ordered":
        return "ordered", "dataset_override"
    raise RetrievalModeError(
        f"Illegal Dataset retrieval_mode override {mode!r} -> "
        f"{dataset_override!r}. The only legal override is 'scored' -> "
        "'ordered' (distrust the connector's scores). Scores cannot be minted "
        "into meaning, and cardinality ('complete_set') is decided by the "
        "system and the validator, not by config.")


def refuse_threshold_for_mode(retrieval_mode: str,
                              score_threshold: float | None,
                              *, dataset_name: str = "") -> None:
    """Raise loudly if a score threshold is set for a non-``scored`` mode.

    A threshold filters on per-chunk scores, but under ``ordered`` those scores
    are declared meaningless and under ``complete_set`` the system already chose
    its answer set -- filtering it would measure a hybrid nobody ships. This is
    the apply-or-reject rule (never silently ignore a threshold), and it fires
    for a threshold arriving from the global settings store too, because
    swallowing a stored default is the exact bug class this refusal exists to
    prevent.
    """
    if score_threshold is None or retrieval_mode == "scored":
        return
    where = f" for Dataset {dataset_name!r}" if dataset_name else ""
    raise RetrievalModeError(
        f"A score threshold ({score_threshold}) was requested{where}, but its "
        f"retrieval mode is {retrieval_mode!r} -- scores are not a meaningful "
        "filter in this mode. Either clear the score threshold (including any "
        "stored default on the Settings page), or evaluate a 'scored' "
        "connector. The threshold is refused rather than silently ignored.")


@dataclass
class RetrievedChunk:
    """One result from a retrieval query, in rank order (rank 0 = best).

    ``score`` is canonical: **higher is always better**, in the score's native
    scale (see the conversion utilities above). ``raw_score`` preserves the
    store-native value when a conversion was applied -- evidence stays
    auditable; leave it ``None`` when no conversion happened.

    ``score`` is ``Optional[float]``: required and meaningful under ``scored``
    retrieval, but legally ``None`` under ``ordered`` (scores declared
    meaningless) and ``complete_set`` (e.g. a neighbor-expanded chunk never
    scored against the query). See :func:`resolve_retrieval_mode` and
    ``docs/contract.md``.
    """

    chunk_id: str | None
    text: str
    score: float | None
    rank: int
    doc_id: str | None = None
    raw_score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    """Canonical answer capture returned by a generation-capable connector.

    ``hydrated_prompt`` is the original, opaque prompt capture: the whole
    prompt as one string, after the fact. It is **kept, populated, and not
    repurposed** -- every existing consumer reads it, and a connector that
    knows nothing about prompt templates still fills it in exactly as before.

    ``trace`` is the additive supplement, present only when the connector
    implements the declared-prompt-template capability (see
    :mod:`rag_connector.prompts`). It carries what a single hydrated string
    structurally cannot: the messages with their roles, the model, the decoding
    params, the raw response, token usage, and the template and RAG-block
    hashes as separate values. ``None`` means the connector does not declare
    that capability -- absent, never an empty trace, so "no structured prompt
    evidence" cannot be mistaken for "a prompt with no template".

    When both are present they describe the same call, and
    ``trace.prompt.text`` is the value a connector should use for
    ``hydrated_prompt`` so the two can never disagree.
    """

    query: str
    answer: str
    contexts: list[RetrievedChunk]
    citations: list[str] = field(default_factory=list)
    hydrated_prompt: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    #: Appended LAST on purpose. Consumers construct this dataclass
    #: positionally (``GeneratedAnswer(text, answer, contexts, citations,
    #: prompt)``), so a new field anywhere but the end would silently
    #: misassign arguments in code that still compiles.
    trace: GenerationTrace | None = None


class RagPipeline(ABC):
    """Port for an arbitrary RAG system under evaluation (the black box)."""

    name: str = "rag-pipeline"

    # How this connector bounds its returned list and whether its per-chunk
    # scores are meaningful (see RETRIEVAL_MODES / resolve_retrieval_mode).
    # The default ``scored`` is the only behavior that existed before C2, so
    # every existing connector keeps working with zero changes. Override it on
    # your subclass (``retrieval_mode = "complete_set"``) when the SYSTEM
    # decides cardinality, or ``"ordered"`` when it ranks without meaningful
    # scores. It is surfaced through info() and frozen with the Dataset.
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE

    @abstractmethod
    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run retrieval for a query and return the chunks, best-first.

        ``top_k`` is unset by default: ``None`` means "retrieve however this
        connector is configured" (its own depth, or the system's own set under
        ``complete_set``). A caller MAY pass a positive int to request a depth;
        an evaluation host should not, because the configured system is the
        thing under test. Connectors written before 2026-10-08 with
        ``top_k: int = 5`` stay compatible: a host that omits the argument gets
        their default."""

    # -- Corpus reading -----------------------------------------------------
    #
    # ``list_chunks`` and ``pull_all_chunks`` are the same question at two
    # granularities, so each derives from the other and a connector implements
    # **whichever one its store actually offers**.
    #
    # Prefer ``list_chunks``. Pagination is what real stores natively expose,
    # and deriving bulk-pull from it costs nothing, whereas deriving paging (and
    # by-id reads, and sampling) from bulk-pull makes every one of them
    # O(corpus). The evidence that this was backwards: the one shipped
    # third-party connector implemented ``pull_all_chunks`` by hand-paging its
    # store's list API — writing, as glue, the adapter that belonged here.
    #
    # Neither is abstract, because making one abstract forces the other's
    # implementers to write a stub. The cost is that "implements neither" now
    # fails at call time rather than instantiation time, so the guard below
    # makes that failure loud and names both ways out.

    def pull_all_chunks(self) -> list[ChunkRecord]:
        """Return every chunk in the corpus with the metadata contract satisfied.

        The corpus is the WHOLE collection this connector is configured for.
        Apply only the filters that define the search space (the static
        collection boundary sent with every search, e.g. the partition key of
        an index shared by many customers, disclosed in
        ``info()["internal_filters"]``). Never apply filters that narrow
        results within the space (section/module, delivery mode, access,
        thresholds, top-k): those belong to :meth:`query` alone. A corpus
        filtered by result rules silently drops the content retrieval cannot
        reach, so no test can ever find that defect. Any other exclusion is a
        declared deviation named in :meth:`info`.
        (docs/contract.md, "The corpus read returns the whole collection".)

        Default: page through :meth:`list_chunks` to exhaustion.
        """
        self._require_a_corpus_read_path()
        out: list[ChunkRecord] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = self.list_chunks(cursor=cursor, limit=_CORPUS_PAGE_SIZE)
            out.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                return out
            # A cursor that repeats (or never advances) would loop forever. This
            # is a black-box contract: a connector CAN be wrong, and hanging is
            # the worst way to find out.
            if cursor in seen:
                raise ConnectorContractError(
                    f"{type(self).__name__}.list_chunks returned a repeated "
                    f"cursor {cursor!r}; paging would not terminate."
                )
            seen.add(cursor)

    def list_chunks(
        self,
        *,
        cursor: str | None = None,
        limit: int = _CORPUS_PAGE_SIZE,
    ) -> ChunkPage:
        """Return one page of the corpus, plus the cursor for the next.

        Pages through the WHOLE collection: only the static collection
        boundary, never result-narrowing rules (see :meth:`pull_all_chunks`).

        Default: slice :meth:`pull_all_chunks`, so a connector that can only
        bulk-pull still satisfies :class:`~rag_connector.capabilities.
        PagedCorpusReader`. That default is O(corpus) per page — override it
        with a real paged read if the store has one.
        """
        from .models import ChunkPage  # deferred: models imports from here

        self._require_a_corpus_read_path()
        chunks = self.pull_all_chunks()
        start = int(cursor) if cursor else 0
        page = chunks[start : start + max(0, limit)]
        next_offset = start + len(page)
        return ChunkPage(
            items=page,
            next_cursor=str(next_offset) if next_offset < len(chunks) else None,
        )

    def _require_a_corpus_read_path(self) -> None:
        """Refuse before the two defaults recurse into each other."""
        cls = type(self)
        if (
            cls.pull_all_chunks is RagPipeline.pull_all_chunks
            and cls.list_chunks is RagPipeline.list_chunks
        ):
            raise UnsupportedCapability(
                f"{cls.__name__} implements neither list_chunks nor "
                "pull_all_chunks, so its corpus cannot be read. Implement "
                "list_chunks (preferred — the other is derived from it) or "
                "pull_all_chunks."
            )

    def fingerprint(self) -> str:
        """Stable hash of the corpus, for the baseline cache / change detection.

        Default: hash of sorted ``(chunk_id, text)`` pairs from
        :meth:`pull_all_chunks`. Override with something cheaper if a connector
        can compute identity without pulling the whole corpus.
        """
        items = sorted((c.chunk_id, c.text) for c in self.pull_all_chunks())
        digest = hashlib.sha256(
            json.dumps(items, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        """Fetch a specific set of chunks by id, keyed by ``chunk_id``.

        Default: filter :meth:`pull_all_chunks`. A production connector should
        override with a cheap by-id fetch so integrity checks can *sample* and
        the result inspector can hydrate out-of-snapshot ids without pulling the
        whole corpus. Missing ids must simply be absent from the returned dict.
        """
        wanted = set(chunk_ids)
        return {c.chunk_id: c for c in self.pull_all_chunks() if c.chunk_id in wanted}

    def list_datasets(self) -> list[DatasetRef]:
        """Enumerate the datasets this connector's backend offers.

        Optional capability (:class:`~rag_connector.capabilities.DatasetLister`).
        A bound instance reads ONE dataset; this answers the different question
        of what else is there to bind to, so a host can offer the choice
        instead of making the operator retype a collection name they have to
        remember. Listing is read-only and never changes what ``self`` is bound
        to — rebinding is the host's job, through ``connect``/``build``, and
        producing a new instance is the point: run identity and fingerprints
        are anchored to a binding that does not move under them.

        The default raises :class:`UnsupportedCapability` rather than returning
        an empty list, for the same reason
        :meth:`get_chunk_vectors` does. "This backend cannot enumerate" and
        "this backend has nothing" are opposite facts, and an empty list says
        the second while meaning the first — a host would render "no datasets
        found" over a store holding twenty. Hosts pre-check with
        :func:`~rag_connector.capabilities.supports`
        (``supports(pipeline, "list_datasets")``) instead of catching this.

        Implementations return :class:`~rag_connector.models.DatasetRef`
        entries carrying, at minimum, the ``id`` that identifies the dataset
        and the ``connect_params`` that select it. Report a count only when the
        backend gives one cheaply; leave it ``None`` otherwise rather than
        paying for a corpus scan inside a picker.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} cannot enumerate the datasets available to "
            "it; the dataset it is bound to must be named explicitly."
        )

    def get_chunk_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        """Return the vectors the index ACTUALLY HOLDS for ``ids``, keyed by id.

        Optional capability (:class:`~rag_connector.capabilities.ChunkVectorReader`).
        The default here raises :class:`UnsupportedCapability`, so a connector
        that cannot read its own index says so out loud instead of returning an
        empty dict a caller would mistake for "verified, nothing wrong".

        Three rules make this method worth having:

        * **Indexed vectors, never a re-embedding of the chunk text.** The
          caller recomputes ``dot(unit_query, unit_chunk)`` locally and compares
          it to the score the connector reported, to catch a connector that
          declares cosine but actually returns a rescaled score (say
          ``(1 + cos) / 2``). A rescale preserves ranking, so retrieval keeps
          looking perfect while a calibrated threshold quietly stops meaning
          what it was calibrated to mean. Re-embedding the text would test the
          embedder against itself and pass even when the index is stale or was
          written in a different embedding space -- exactly the two failures
          this check exists to find.
        * **Missing ids are simply absent from the result.** Never a
          placeholder, never a zero vector. A fabricated vector is worse than a
          gap because nothing downstream can detect it; a gap is visible.
        * **Empty input returns an empty dict** without touching the backend.

        A backend failure raises :class:`~rag_connector.errors.ConnectorOperationalError`,
        never an empty or partial dict: silence must not look like success.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} cannot return indexed chunk vectors; "
            "scores can only be trusted as declared."
        )

    def generate(self, text: str, *, top_k: int | None = None, llm=None) -> GeneratedAnswer:
        """Produce an answer through the connector's production-shaped prompt path.

        Retrieval-only connectors may leave this unsupported. ``llm`` exists for
        the built-in Reference RAG; production connectors should
        normally use their own configured generation service.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support generation")

    # -- Declared prompt templates (independently optional capability) ------
    #
    # Three methods, one capability. ``generate`` above builds its prompt
    # internally and reports it afterwards as an opaque string; these three
    # invert that, so a host can read the template BEFORE running anything,
    # supply the context itself, and hold the prompt constant across runs.
    #
    # All three default to raising, so ``supports(pipeline, ...)`` answers
    # truthfully and an existing connector needs zero changes: it implements
    # none of them, declares nothing, and keeps behaving exactly as it does
    # today. See :class:`~rag_connector.capabilities.DeclaredPromptGenerator`
    # and :func:`~rag_connector.capabilities.supports_declared_prompts`.

    def prompt_templates(self) -> Sequence[PromptTemplate]:
        """Enumerate the prompt variants this connector declares.

        Optional capability. Each entry is a
        :class:`~rag_connector.prompts.PromptTemplate` with a stable id, a
        human description, template text carrying ``{rag_block}`` and
        ``{query}``, and the one position where a prompt-cache breakpoint may
        go.

        This is the bound-instance form. The pre-binding form is
        ``ConnectorSpec.prompt_templates`` -- zero-arg, callable straight off a
        registry listing before any params, credentials, or backend exist,
        because a host offering a choice of prompts has to render that choice
        before anything is connected. A connector should serve ONE declared set
        through both (a module-level constant returned by each), so the
        variants an operator picked from are the variants that then run.

        Declaring nothing is a supported configuration: leave this
        unimplemented rather than returning an empty sequence, so a host can
        tell "this connector has no declared prompt" from "this connector
        declared zero prompts", which are different facts.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} declares no prompt templates; its prompt "
            "can only be captured after the fact, through generate()."
        )

    def render_rag_block(self, chunks: Sequence[RetrievedChunk]) -> str:
        """Render ``chunks`` into this connector's own RAG-block format.

        Optional capability. The host supplies the chunks -- which is the whole
        point, since fixing the context is what makes groundedness measurable
        -- and the connector renders them the way it renders them in
        production: its own numbering, its own separators, its own decision
        about whether ids, scores, or source names appear.

        That realism is not decoration. An answer's groundedness depends on
        what the block actually looked like, so a harness that imposed its own
        tidy format would be measuring a prompt nobody ships.
        :func:`~rag_connector.prompts.render_numbered_rag_block` is available
        for connectors whose format happens to be the conventional numbered
        one.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} cannot render a RAG block from supplied "
            "chunks; it only generates from its own retrieval."
        )

    def generate_from_prompt(
        self,
        prompt: RenderedPrompt,
        *,
        contexts: Sequence[RetrievedChunk] = (),
        llm=None,
        decoding: Mapping[str, object] | None = None,
    ) -> GeneratedAnswer:
        """Generate from a fully rendered prompt and return the answer + trace.

        Optional capability. No retrieval happens here: ``prompt`` already
        carries the messages, and ``contexts`` is the chunk list the caller
        rendered the block from, passed through so the returned
        :class:`GeneratedAnswer` is shaped exactly like ``generate``'s and
        every existing consumer of it keeps working.

        The return value must have ``trace`` populated (a
        :class:`~rag_connector.prompts.GenerationTrace`) **and**
        ``hydrated_prompt`` populated -- the new structured evidence
        supplements the legacy string, it does not replace it.

        ``decoding`` overrides the template's declared defaults. Whatever is
        used ends up on the trace, so a run states its decoding params rather
        than leaving them to be inferred from a connector's source.

        Runtime failures follow the same convention as ``generate``: capture
        them as ``GeneratedAnswer(error=...)`` with the trace still attached --
        a failed generation whose prompt is known is diagnosable, one whose
        prompt is lost is not.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} cannot generate from a supplied prompt; "
            "it builds its own prompt inside generate()."
        )

    def info(self) -> dict:
        """Human-readable description of the pipeline (for run metadata).

        ``retrieval_mode`` is included so it freezes with the Dataset's
        ``pipeline_info`` at creation time -- the resolved mode is then read from
        the frozen declaration without re-running the connector.
        """
        return {"name": self.name, "type": type(self).__name__,
                "retrieval_mode": self.retrieval_mode}

    @staticmethod
    def validate_chunks(chunks: list[ChunkRecord]) -> None:
        """Raise if any chunk is missing a required contract field.

        This is where a bad ingest (e.g. metadata that didn't survive the store)
        is caught, rather than silently producing broken qrels downstream.
        """
        for i, c in enumerate(chunks):
            for fname in REQUIRED_CHUNK_FIELDS:
                value = getattr(c, fname, None)
                if value is None or (isinstance(value, str) and value == ""):
                    raise ValueError(
                        f"Chunk #{i} (chunk_id={getattr(c, 'chunk_id', '?')!r}) "
                        f"violates the metadata contract: missing '{fname}'."
                    )
