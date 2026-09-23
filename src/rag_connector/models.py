"""Portable value types shared by connector implementations and hosts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import ChunkRecord, GeneratedAnswer, RetrievedChunk

# Product-neutral spelling for hosts that retrieve items other than literal
# chunks. It remains the same runtime type for RAGauge artifact compatibility.
RetrievedItem = RetrievedChunk


@dataclass(frozen=True, slots=True)
class ScoreSemantics:
    comparable: bool
    higher_is_better: bool | None
    metric: str | None = None
    transform: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Optional response envelope for transports that carry response metadata."""

    items: Sequence[RetrievedChunk] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkPage:
    items: Sequence[ChunkRecord]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """One selectable dataset a connector can be bound to and read.

    A "dataset" here is whatever the backend calls the unit a connector binds
    to — a Chroma collection, a Pinecone index or namespace, a hosted
    knowledge base. The word is the host's, because the host is the thing that
    has to offer the choice; the backend's own noun travels in ``metadata`` so
    nothing is renamed behind the operator's back.

    ``connect_params`` is what makes the listing *actionable* rather than
    decorative: it is the fragment of a connector's ``params`` that selects
    this dataset, so a host rebinds by merging it into the params it already
    collected and calling ``connect`` again. Without it a host would have to
    know which param name each connector selects on — connector-specific
    knowledge the registry exists to keep out of hosts.

    ``item_count`` is **best-effort and legally None**. It is here because a
    cheap count is the single most useful thing for telling two collections
    apart in a picker, and absent when the backend cannot answer without a
    scan. None means "not reported", never zero: a listing that reported an
    unknown size as an empty dataset would make a populated collection look
    like a mistake.
    """

    id: str
    label: str | None = None
    item_count: int | None = None
    connect_params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "item_count": self.item_count,
            "connect_params": dict(self.connect_params),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceDescriptor:
    """Identity of the embedding space a connector queries in.

    ``metric`` and ``normalized`` are **tri-state**: a value asserts the
    property, ``None`` means the connector cannot (or does not) report it.
    The distinction is load-bearing. A host's similarity thresholds are
    calibrated in the units of some specific geometry — typically cosine over
    L2-normalized vectors — so a connector reporting a *different* metric is
    declaring those numbers invalid for its space, while a connector reporting
    ``None`` is declaring only that it does not know. Unverifiable is not the
    same as incompatible, and a connector that guesses a plausible default
    turns the first into the second silently.

    ``fingerprint`` is the stable identity of the space itself — what binds
    every derived artifact (cached vectors, centroids, classifiers, saved
    clusterings) to the space it was computed in, so that a model or
    normalization change invalidates them instead of quietly mis-serving
    against them. A connector that can reuse an existing identity for its space
    should do so rather than deriving a second one: two fingerprints for one
    space is how drift detection stops working. Connectors that cannot identify
    their space leave it ``None`` — a host then knows it cannot detect drift,
    rather than wrongly believing it can.
    """

    model: str | None
    dimension: int | None
    normalized: bool | None = None
    metric: str | None = None
    fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: Sequence[Sequence[float]]
    space: EmbeddingSpaceDescriptor


@dataclass(frozen=True, slots=True)
class ConnectorDescriptor:
    connector_type: str
    name: str
    contract_version: str = "0.1"
    retrieval_mode: str = "scored"
    score_semantics: ScoreSemantics = field(
        default_factory=lambda: ScoreSemantics(
            comparable=True,
            higher_is_better=True,
        )
    )
    capabilities: frozenset[str] = field(default_factory=frozenset)
    connector_config_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    embedding_space_fingerprint: str | None = None
    retrieval_policy_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConnectorHealth:
    """Whether the connector can serve right now, and why not if it cannot.

    ``detail`` is for a human reading an operations screen, so it should name
    the actual obstacle ("collection 'docs' not found") rather than restate the
    verdict ("unhealthy").
    """

    ok: bool
    detail: str = ""


__all__ = [
    "ChunkPage",
    "ChunkRecord",
    "ConnectorDescriptor",
    "ConnectorHealth",
    "DatasetRef",
    "EmbeddingBatch",
    "EmbeddingSpaceDescriptor",
    "GeneratedAnswer",
    "RetrievedChunk",
    "RetrievedItem",
    "RetrievalResult",
    "ScoreSemantics",
]

