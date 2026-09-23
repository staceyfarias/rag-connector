"""Dataset listing: enumerate what a backend offers, without rebinding anything."""

import pytest

from rag_connector import ChunkRecord, DatasetRef, RagPipeline, RetrievedChunk, supports
from rag_connector.errors import UnsupportedCapability
from rag_connector.reference import _collection_names, _reference_datasets
from rag_connector.reference_provision import provision_reference_rag
from rag_connector.registry import ConnectorSpec, get_connector


class FakeEmbeddingClient:
    model_id = "fake-embedder"
    dimensions = 8

    def _embed(self, text):
        vec = [0.1] * self.dimensions
        for i, ch in enumerate(text[:32]):
            vec[i % self.dimensions] += ord(ch) % 7
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class SilentConnector(RagPipeline):
    """A connector that cannot enumerate -- it never overrides list_datasets."""

    name = "silent"

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id="c0", text="alpha", score=1.0, rank=0)]

    def pull_all_chunks(self):
        return [ChunkRecord("c0", "d0", "d0.txt", 0, 0, "alpha")]


# -- The default: unsupported is said out loud, never faked as "none found" ----


def test_unsupported_listing_raises_instead_of_returning_an_empty_list():
    # An empty list would render as "this store has no datasets" over a store
    # holding twenty. The two facts are opposite; only one of them is true.
    with pytest.raises(UnsupportedCapability):
        SilentConnector().list_datasets()


def test_supports_distinguishes_the_base_default_from_a_real_implementation():
    # This is how a host decides whether to show a picker, without calling and
    # catching -- the base default satisfies the protocol structurally.
    assert supports(SilentConnector(), "list_datasets") is False
    from rag_connector.reference import ReferenceRagConnector

    assert supports(ReferenceRagConnector, "list_datasets") is True


# -- Spec level: enumerate before anything is bound ---------------------------


def test_spec_without_a_lister_reports_the_capability_as_absent():
    spec = ConnectorSpec(
        connector_type="unlisted",
        label="Unlisted",
        description="",
        build=lambda connection: SilentConnector(),
    )
    assert spec.list_datasets is None
    assert spec.supports_dataset_listing is False
    assert spec.to_public_dict()["supports_dataset_listing"] is False


def test_reference_spec_advertises_listing_to_hosts():
    spec = get_connector("reference")
    assert spec is not None
    assert spec.supports_dataset_listing is True
    assert spec.to_public_dict()["supports_dataset_listing"] is True


def _provision(tmp_path, collection_name, filename="alpha.txt"):
    folder = tmp_path / f"docs_{collection_name}"
    folder.mkdir()
    (folder / filename).write_text("alpha apple orchard " * 6, encoding="utf-8")
    return provision_reference_rag(
        str(folder),
        persist_dir=str(tmp_path / "chroma"),
        collection_name=collection_name,
        embedding_client=FakeEmbeddingClient(),
        chunk_size=60,
        chunk_overlap=10,
    )


@pytest.mark.reference_extra
def test_reference_lists_collections_from_persist_dir_alone(tmp_path):
    # The point of the spec-level surface: no collection_name is supplied,
    # because that is the field the operator is being helped to choose.
    _provision(tmp_path, "first_collection")
    _provision(tmp_path, "second_collection")

    listed = _reference_datasets({"persist_dir": str(tmp_path / "chroma")})

    assert [d.id for d in listed] == ["first_collection", "second_collection"]
    assert all(isinstance(d, DatasetRef) for d in listed)
    assert all(d.item_count and d.item_count > 0 for d in listed)
    assert all(d.metadata["kind"] == "collection" for d in listed)


@pytest.mark.reference_extra
def test_listed_connect_params_rebind_to_the_chosen_dataset(monkeypatch, tmp_path):
    # The listing is actionable: a host merges connect_params into the params
    # it already collected and calls connect -- it never has to know that this
    # particular connector selects on a field called "collection_name".
    _provision(tmp_path, "wanted")
    _provision(tmp_path, "unwanted")
    monkeypatch.setattr(
        "rag_connector.reference.get_embedding_client",
        lambda *args, **kwargs: FakeEmbeddingClient(),
    )
    spec = get_connector("reference")
    assert spec is not None and spec.connect is not None

    chosen = next(
        d for d in spec.list_datasets({"persist_dir": str(tmp_path / "chroma")})
        if d.id == "wanted"
    )
    connected, connection = spec.connect(dict(chosen.connect_params))

    assert connected.collection_name == "wanted"
    assert connection["collection_name"] == "wanted"


@pytest.mark.reference_extra
def test_listing_does_not_rebind_the_instance_that_listed(tmp_path):
    # Rebinding is a host-side construction of a NEW instance. If listing moved
    # this instance's binding, every fingerprint already issued from it would
    # quietly start referring to a different corpus.
    connector = _provision(tmp_path, "home")
    _provision(tmp_path, "elsewhere")
    before = connector.fingerprint()

    ids = {d.id for d in connector.list_datasets()}

    assert ids == {"home", "elsewhere"}
    assert connector.collection_name == "home"
    assert connector.fingerprint() == before


@pytest.mark.reference_extra
def test_an_unreadable_collection_stays_listed_without_inventing_a_count(
    monkeypatch, tmp_path
):
    # One sick collection must not delete the healthy ones from the picker, and
    # its size must read as unknown rather than as an empty dataset.
    _provision(tmp_path, "healthy")

    import chromadb

    real_client = chromadb.PersistentClient

    class BrokenCount:
        def __init__(self, path):
            self._inner = real_client(path=path)

        def list_collections(self):
            return ["healthy", "sick"]

        def get_collection(self, name):
            if name == "sick":
                raise RuntimeError("collection is corrupt")
            return self._inner.get_collection(name)

    monkeypatch.setattr(chromadb, "PersistentClient",
                        lambda path: BrokenCount(path))

    listed = _reference_datasets({"persist_dir": str(tmp_path / "chroma")})

    by_id = {d.id: d for d in listed}
    assert set(by_id) == {"healthy", "sick"}
    assert by_id["sick"].item_count is None
    assert "corrupt" in by_id["sick"].metadata["error"]
    assert by_id["healthy"].item_count > 0


def test_collection_names_survives_both_chroma_return_shapes():
    # Chroma 0.6 returns names; earlier versions returned collection objects.
    class NameObject:
        def __init__(self, name):
            self.name = name

    class OldClient:
        def list_collections(self):
            return [NameObject("b"), NameObject("a")]

    class NewClient:
        def list_collections(self):
            return ["b", "a"]

    assert _collection_names(OldClient()) == ["a", "b"]
    assert _collection_names(NewClient()) == ["a", "b"]
