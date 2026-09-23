"""The embedding client must report the width it actually produces.

``dimensions`` does not stay local: it is asserted through
``describe_space()`` and hashed into the embedding fingerprint that binds
derived artifacts to their space. A guess there is indistinguishable from a
measurement, which is the whole problem.
"""

from rag_connector.embedding import FastEmbedEmbeddingClient


class FakeModel:
    """Stands in for ``fastembed.TextEmbedding``, counting its own use."""

    def __init__(self, width: int):
        self.width = width
        self.embed_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        return [[0.5] * self.width for _ in texts]


def test_an_uncatalogued_model_is_measured_not_guessed():
    # 1024 wide, and not in the small known-dimensions catalogue: the old
    # `.get(model_id, 384)` fallback reported 384 for exactly this case.
    client = FastEmbedEmbeddingClient("acme/unlisted-v2", model=FakeModel(1024))

    assert client.dimensions == 1024


def test_the_reported_width_matches_the_vectors_actually_produced():
    """The invariant a guessed default breaks: a connector reporting 384 while
    emitting 1024-wide vectors is describing a space that does not exist."""
    client = FastEmbedEmbeddingClient("acme/unlisted-v2", model=FakeModel(1024))

    assert len(client.embed_query("anything")) == client.dimensions


def test_a_catalogued_model_costs_no_probe():
    model = FakeModel(384)
    client = FastEmbedEmbeddingClient("BAAI/bge-small-en-v1.5", model=model)

    assert client.dimensions == 384
    assert model.embed_calls == 0


def test_an_explicit_dimension_is_honored_without_probing():
    model = FakeModel(1024)
    client = FastEmbedEmbeddingClient(
        "acme/unlisted-v2", dimensions=256, model=model
    )

    assert client.dimensions == 256
    assert model.embed_calls == 0


def test_the_measurement_happens_once():
    model = FakeModel(1024)
    client = FastEmbedEmbeddingClient("acme/unlisted-v2", model=model)

    assert client.dimensions == 1024
    assert client.dimensions == 1024

    assert model.embed_calls == 1
