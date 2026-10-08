"""Structural connector contract and independently optional capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .base import ChunkRecord, GeneratedAnswer, RagPipeline, RetrievedChunk
from .models import (
    ChunkPage,
    ConnectorHealth,
    DatasetRef,
    EmbeddingBatch,
    EmbeddingSpaceDescriptor,
)
from .prompts import PromptTemplate, RenderedPrompt


@runtime_checkable
class RagConnector(Protocol):
    retrieval_mode: str

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]: ...

    def pull_all_chunks(self) -> list[ChunkRecord]: ...

    def fingerprint(self) -> str: ...

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]: ...

    def info(self) -> dict: ...


@runtime_checkable
class HealthCapability(Protocol):
    def health(self) -> ConnectorHealth: ...


@runtime_checkable
class PagedCorpusReader(Protocol):
    def list_chunks(self, *, cursor: str | None = None, limit: int = 1000) -> ChunkPage: ...


@runtime_checkable
class QueryEmbedder(Protocol):
    def describe_space(self) -> EmbeddingSpaceDescriptor: ...

    def embed_queries(self, texts: Sequence[str]) -> EmbeddingBatch: ...


@runtime_checkable
class DatasetLister(Protocol):
    """Enumerate the datasets a backend offers, so a host can offer the choice.

    Distinct from the reserved :class:`IndexIntrospector`, which nothing
    implements: ``list_indexes`` was sketched as *operational* inspection (the
    physical indexes behind one binding, their health and shape), while this is
    a *selection* surface — the units a connector can be bound to, and the
    params that bind to each. A backend may well answer both, differently.
    """

    def list_datasets(self) -> Sequence[DatasetRef]: ...


@runtime_checkable
class RunSnapshotProvider(Protocol):
    """Describe connector-specific configuration and state for one evaluation.

    The returned mapping must be JSON-serializable and secret-free. ``settings``
    contains configured behavior; ``observations`` contains cheap facts about
    the bound system at capture time. Hosts persist the snapshot beside a run
    and may diff it, but connector observations are not evaluation metrics.
    """

    def run_snapshot(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ChunkVectorReader(Protocol):
    def get_chunk_vectors(self, ids: Sequence[str]) -> Mapping[str, Sequence[float]]: ...


def supports(pipeline: object, method_name: str) -> bool:
    """Truthfully report whether ``pipeline`` implements ``method_name`` itself,
    rather than inheriting a :class:`RagPipeline` base default.

    ``isinstance``/``issubclass`` against the capability protocols cannot answer
    this for any method with a concrete base default (``pull_all_chunks``,
    ``list_chunks``, ``get_chunks``, ``get_chunk_vectors``, ``fingerprint``,
    ``generate``, ``info``): the default satisfies the protocol structurally, so
    *every* pipeline "has" the method whether or not anything real answers it.
    Support means "overrides the base default", which is what this checks.

    Accepts an instance or a class. An object that is not a ``RagPipeline``
    subclass counts as supporting any method it defines, since nothing it
    inherits can answer on its behalf.
    """
    attr = getattr(pipeline, method_name, None)
    if attr is None:
        return False
    default = getattr(RagPipeline, method_name, None)
    if default is None:
        return True
    # Bound methods compare via the underlying function; plain functions
    # (class access, staticmethods, instance-attribute callables) compare as-is.
    return getattr(attr, "__func__", attr) is not default


#: The three methods that make up the declared-prompt-template capability.
#: Named once so :func:`supports_declared_prompts`, the validator, and any host
#: reporting partial declarations all agree on the membership.
DECLARED_PROMPT_METHODS = (
    "prompt_templates", "render_rag_block", "generate_from_prompt",
)


def declared_prompt_methods(pipeline: object) -> tuple[str, ...]:
    """Which of :data:`DECLARED_PROMPT_METHODS` ``pipeline`` actually implements.

    The diagnostic behind :func:`supports_declared_prompts`. A host or
    validator reporting a half-declared capability needs to name the missing
    halves, not just say no.
    """
    return tuple(name for name in DECLARED_PROMPT_METHODS
                 if supports(pipeline, name))


def supports_declared_prompts(pipeline: object) -> bool:
    """Truthfully report whether ``pipeline`` declares prompt templates.

    True only when ALL of :data:`DECLARED_PROMPT_METHODS` are overridden. Like
    the other optional capabilities, ``isinstance(pipeline,
    DeclaredPromptGenerator)`` cannot answer this: all three ship
    :class:`~rag_connector.errors.UnsupportedCapability`-raising defaults on
    :class:`~rag_connector.base.RagPipeline` -- so an unsupporting connector
    fails loudly instead of returning an empty template list a host would read
    as "declares nothing" -- which makes every pipeline satisfy the protocol
    structurally.

    All-or-nothing because the capability is only usable as a whole: enumerate
    a variant, render a block, generate from the rendered prompt. Two out of
    three is a bug to report, not a degraded mode to accommodate -- use
    :func:`declared_prompt_methods` to say which parts are there.
    """
    return len(declared_prompt_methods(pipeline)) == len(
        DECLARED_PROMPT_METHODS)


def supports_chunk_vectors(pipeline: object) -> bool:
    """Truthfully report whether ``pipeline`` can read its own indexed vectors.

    ``isinstance(pipeline, ChunkVectorReader)`` cannot answer this. Because
    :meth:`RagPipeline.get_chunk_vectors` ships a default that raises
    :class:`~rag_connector.errors.UnsupportedCapability` -- so that an
    unsupporting connector fails loudly instead of returning an empty dict a
    caller would read as "verified clean" -- *every* pipeline satisfies the
    protocol structurally. (The same is already true of
    :class:`AnswerGenerator` and :meth:`RagPipeline.generate`.)

    The general form of this question is :func:`supports`; this keeps the
    original name for the one capability where the dishonesty was first found.
    """
    return supports(pipeline, "get_chunk_vectors")


@runtime_checkable
class AnswerGenerator(Protocol):
    def generate(self, text: str, *, top_k: int | None = None, llm=None) -> GeneratedAnswer: ...


@runtime_checkable
class DeclaredPromptGenerator(Protocol):
    """Generate from a prompt the connector DECLARED and the host rendered.

    Strictly additive over :class:`AnswerGenerator`, and independently
    optional: a connector may implement one, both, or neither, and one that
    implements neither is unchanged by this capability's existence.

    ``AnswerGenerator.generate`` retrieves and prompts in one opaque step, then
    reports the prompt afterwards as a string. This capability separates the
    three things that step fuses together -- what the prompt SAYS
    (``prompt_templates``), what context it is GIVEN
    (``render_rag_block``, from chunks the host supplies), and what the model
    DID with them (``generate_from_prompt``, returning a structured trace).
    Separating them is what lets a host hold the prompt constant, vary it
    deliberately, hash the template independently of the query, and inspect
    what was sent before rather than after the fact.

    **Declared as a unit.** All three methods, or none. Enumerating variants
    with nothing able to run them, or accepting a rendered prompt nobody can
    enumerate a template for, is a half-declaration a host cannot use --
    :func:`supports_declared_prompts` is the check that treats it as one
    capability, and the conformance validator FAILS a partial declaration
    rather than working around it.
    """

    def prompt_templates(self) -> Sequence[PromptTemplate]: ...

    def render_rag_block(self, chunks: Sequence[RetrievedChunk]) -> str: ...

    def generate_from_prompt(
        self,
        prompt: RenderedPrompt,
        *,
        contexts: Sequence[RetrievedChunk] = (),
        llm=None,
        decoding: Mapping[str, Any] | None = None,
    ) -> GeneratedAnswer: ...


@runtime_checkable
class ContentPublisher(Protocol):
    def upsert(self, items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

    def delete(self, external_ids: Sequence[str]) -> Mapping[str, Any]: ...


@runtime_checkable
class QueryTelemetryReader(Protocol):
    def read_queries(
        self,
        *,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> tuple[Sequence[Mapping[str, Any]], str | None]: ...


@runtime_checkable
class IndexIntrospector(Protocol):
    def list_indexes(self) -> Sequence[Mapping[str, Any]]: ...

