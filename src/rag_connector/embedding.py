"""Minimal local embedding adapter used by the optional Reference RAG."""

from __future__ import annotations

from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Vector widths that are KNOWN for these model ids, so the common case costs no
# probe. Anything absent is measured (see ``FastEmbedEmbeddingClient.
# dimensions``) rather than assumed: a guessed width is asserted downstream as
# fact -- through ``describe_space()`` and, worse, baked into the embedding
# fingerprint that binds every derived artifact to its space.
_KNOWN_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

#: Text embedded once, lazily, to measure an uncatalogued model's vector width.
_DIMENSION_PROBE = "dimension probe"


class EmbeddingClient(Protocol):
    model_id: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbeddingClient:
    """Local embeddings via FastEmbed with no cloud credentials."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        dimensions: int | None = None,
        model=None,
    ):
        self.model_id = model_id
        self._dimensions = dimensions or _KNOWN_DIMENSIONS.get(model_id)
        if model is not None:
            self._model = model
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                'Reference RAG embeddings require the "reference" extra: '
                'pip install "rag-connector[reference]"'
            ) from exc
        self._model = TextEmbedding(model_name=model_id)

    @property
    def dimensions(self) -> int:
        """The vector width this client actually produces.

        Measured from the model on first use for any id the catalogue above
        does not cover, because this number does not stay local: it is asserted
        through ``describe_space()`` and hashed into the embedding fingerprint.
        A plausible-looking default there turns "we cannot verify this space"
        into "this space is 384-dimensional" -- the same claim a correct
        connector makes, so nothing downstream can tell them apart, and two
        genuinely different spaces can end up sharing a fingerprint.

        The probe costs one short embedding, once, against a model ``__init__``
        has already loaded. If the model cannot embed, that failure surfaces
        here instead of being papered over with a number nobody measured.
        """
        if self._dimensions is None:
            self._dimensions = len(self.embed_query(_DIMENSION_PROBE))
        return self._dimensions

    @staticmethod
    def _to_list(vector) -> list[float]:
        tolist = getattr(vector, "tolist", None)
        values = tolist() if tolist else vector
        return [float(value) for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_list(vector) for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        query_embed = getattr(self._model, "query_embed", None)
        if query_embed is not None:
            return self._to_list(next(iter(query_embed(text))))
        return self._to_list(next(iter(self._model.embed([text]))))


def get_embedding_client(
    provider: str = "fastembed",
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int | None = None,
    _model=None,
) -> EmbeddingClient:
    if provider != "fastembed":
        raise ValueError("Reference RAG supports only the local 'fastembed' provider")
    return FastEmbedEmbeddingClient(
        model_id=model,
        dimensions=dimensions,
        model=_model,
    )

