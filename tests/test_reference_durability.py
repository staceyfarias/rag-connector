"""The cold-read barrier: path canonicalisation, close(), and the one retry.

Background, because the shape of these tests is dictated by it. Chroma 1.5.0
exposes NO durability primitive: no ``close``, ``flush``, ``persist`` or
``checkpoint`` on the client, the collection, or the Rust ``Bindings``. Its one
teardown is ``System.stop()`` (``RustBindingsAPI.stop()`` -> ``del
self.bindings``), reachable only through the private ``client._system``; and
``clear_system_cache()`` is not a substitute, because it forgets engines
without stopping them.

So the barrier here is built from three things that ARE available, and each is
pinned below:

1. one engine per store, by canonicalising the path Chroma keys its system
   cache on;
2. an explicit release (``close()``), because otherwise the writing engine owns
   the directory for the rest of the process;
3. a single, signature-scoped re-open-and-retry on the read path.

What these tests do NOT claim: that the underlying cold-read race is dead. It
is not reproducible on demand, so nothing here could honestly assert that. They
pin the mechanism -- that the retry fires for exactly one signature, fires
once, and cannot turn a broken store into a passing read.
"""

import re

import pytest

from rag_connector import reference as reference_module
from rag_connector.errors import ConnectorOperationalError
from rag_connector.reference import (
    COLD_READ_SIGNATURE,
    ReferenceRagConnector,
    canonical_persist_dir,
    is_cold_segment_read,
    release_store,
)
from rag_connector.reference_provision import durability_barrier, provision_reference_rag

COLD_MESSAGE = "Error creating hnsw segment reader: Nothing found on disk"


class FakeEmbeddingClient:
    model_id = "fake-embedder"
    dimensions = 16

    def _embed(self, text):
        vec = [0.1] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            vec[hash(token) % self.dimensions] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _provision(tmp_path, persist_dir=None, collection="durability_test"):
    folder = tmp_path / "docs"
    if not folder.exists():
        folder.mkdir()
        (folder / "alpha.txt").write_text(
            "alpha apple orchard harvest " * 8, encoding="utf-8"
        )
    return provision_reference_rag(
        str(folder),
        persist_dir=persist_dir or str(tmp_path / "chroma"),
        collection_name=collection,
        embedding_client=FakeEmbeddingClient(),
        chunk_size=80,
        chunk_overlap=10,
    )


def _systems():
    from chromadb.api.shared_system_client import SharedSystemClient

    return SharedSystemClient._identifier_to_system


# -- the signature is narrow -------------------------------------------------

def test_only_the_exact_cold_read_signature_is_treated_as_recoverable():
    """The retry is scoped to one backend failure, deliberately.

    A retry that also swallowed neighbouring hnsw errors would be hiding
    corruption behind a delay, which is strictly worse than the flake it
    replaces -- so each near miss below must NOT match.
    """
    assert is_cold_segment_read(RuntimeError(COLD_MESSAGE))
    # Case-insensitive: the fragments come from Rust, not from a Python format
    # string, and we do not control their capitalisation across versions.
    assert is_cold_segment_read(RuntimeError(COLD_MESSAGE.upper()))

    near_misses = [
        # The WRITER, not the reader: a failure to build the index is a real
        # write fault and must surface immediately.
        "Error creating hnsw segment writer: Nothing found on disk",
        # A read that got as far as the index and failed inside it.
        "Error reading from hnsw segment reader: Error querying knn",
        # Generic absence from some other backend.
        "Nothing found on disk",
        # A different segment entirely.
        "Error reading from metadata segment reader: Error reading from sqlite",
        "Collection is missing HNSW configuration",
    ]
    for message in near_misses:
        assert not is_cold_segment_read(RuntimeError(message)), message


def test_the_signature_tokens_are_lowercase_so_matching_stays_case_insensitive():
    # is_cold_segment_read lowercases the message and not the tokens; an
    # upper-case token here would silently never match.
    assert all(token == token.lower() for token in COLD_READ_SIGNATURE)


# -- one engine per store ----------------------------------------------------

@pytest.mark.reference_extra
def test_two_spellings_of_one_directory_resolve_to_one_chroma_engine(
    tmp_path, monkeypatch
):
    """Chroma keys its system cache on the RAW persist_directory string.

    ``./chroma`` and the absolute path are one directory but two cache keys, so
    unresolved they produce two independent Rust engines over the same files --
    an unsupported configuration that has been observed to fail a write with
    "Error in compaction: Failed to apply logs to the metadata segment". This
    library invites exactly that by defaulting persist_dir to a RELATIVE path.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma").mkdir()
    _provision(tmp_path, persist_dir=str(tmp_path / "chroma"))

    before = set(_systems())
    relative = ReferenceRagConnector(
        persist_dir="./chroma",
        collection_name="durability_test",
        embedding_client=FakeEmbeddingClient(),
    )
    added = set(_systems()) - before

    assert added == set(), (
        "opening the same store by a different spelling built a SECOND Chroma "
        f"engine: {added}"
    )
    assert relative._chroma_path == canonical_persist_dir(str(tmp_path / "chroma"))
    assert relative.pull_all_chunks()


@pytest.mark.reference_extra
def test_canonicalisation_does_not_rewrite_the_persisted_connection(tmp_path):
    """Only the path handed to Chroma is resolved.

    ``persist_dir`` travels in the connection document and is shown in
    ``info()``. Rewriting it would change persisted artifacts for a fix that is
    purely about which engine gets opened.
    """
    given = str(tmp_path / "chroma")
    connector = _provision(tmp_path, persist_dir=given)

    assert connector.connection()["persist_dir"] == given
    assert connector.info()["persist_dir"] == given
    assert connector.persist_dir == given


# -- release -----------------------------------------------------------------

@pytest.mark.reference_extra
def test_close_stops_the_engine_and_forgets_it(tmp_path):
    connector = _provision(tmp_path)
    path = connector._chroma_path
    assert path in _systems(), "provisioning should have left an engine open"

    released = connector.close()

    assert released is True
    assert path not in _systems(), "close() must evict the system, not just drop refs"


@pytest.mark.reference_extra
def test_close_is_idempotent_and_a_release_rather_than_a_poison_pill(tmp_path):
    """Closing early must cost a re-open, never an error.

    ``close()`` is a lifecycle call a host makes when handing a store to
    another process. If it left the connector unusable, every caller would need
    to know whether anyone else still wanted to read -- so a later read
    re-opens instead.
    """
    connector = _provision(tmp_path)
    expected = len(connector.pull_all_chunks())

    assert connector.close() is True
    assert connector.close() is False, "second close released a second engine"

    assert len(connector.pull_all_chunks()) == expected
    assert connector.health().ok
    # info() reads chunk_count off the collection too, so it is on the same
    # hook: a released engine must re-open, not raise on a dropped handle.
    assert connector.info()["chunk_count"] == expected


def test_release_store_on_an_unopened_directory_is_a_no_op(tmp_path):
    assert release_store(str(tmp_path / "never_opened")) is False


# -- the retry ---------------------------------------------------------------

class _ColdOnce:
    """A collection that fails the first read with the cold signature."""

    def __init__(self, payload, failures=1, exc=None):
        self.payload = payload
        self.failures = failures
        self.calls = 0
        self.exc = exc or RuntimeError(COLD_MESSAGE)

    def _maybe_fail(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc

    def get(self, **kwargs):
        self._maybe_fail()
        return self.payload

    def query(self, **kwargs):
        self._maybe_fail()
        return self.payload

    def count(self):
        self._maybe_fail()
        return len(self.payload.get("ids", []))


@pytest.fixture()
def instant_retry(monkeypatch):
    monkeypatch.setattr(reference_module, "COLD_READ_RETRY_DELAY_SECONDS", 0)


def _pin_collection(connector, monkeypatch, stub):
    """Install ``stub`` and keep the recovery's re-open from replacing it."""
    connector._collection = stub
    monkeypatch.setattr(connector, "_open", lambda: None)


@pytest.mark.reference_extra
def test_a_cold_read_is_retried_once_and_succeeds(tmp_path, monkeypatch, instant_retry):
    connector = _provision(tmp_path)
    real = connector._collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    stub = _ColdOnce(real, failures=1)
    _pin_collection(connector, monkeypatch, stub)

    chunks = connector.pull_all_chunks()

    assert chunks, "the retry should have produced the corpus"
    assert stub.calls == 2, "expected exactly one retry, not a loop"


@pytest.mark.reference_extra
def test_the_engine_is_released_before_the_retry(tmp_path, monkeypatch, instant_retry):
    """Re-reading through the SAME engine would be a blind retry.

    The recovery only means anything because it throws the engine away first --
    that is the whole content of "re-open", and it is what makes a fresh
    process succeed where the failing one did not.
    """
    connector = _provision(tmp_path)
    real = connector._collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    _pin_collection(connector, monkeypatch, _ColdOnce(real, failures=1))
    released = []
    monkeypatch.setattr(
        reference_module, "release_store", lambda path: released.append(path) or True
    )

    connector.pull_all_chunks()

    assert released == [connector._chroma_path]


@pytest.mark.reference_extra
def test_a_persistent_cold_read_is_reported_not_masked(
    tmp_path, monkeypatch, instant_retry
):
    """The property that keeps this from hiding corruption.

    A segment that is genuinely absent is just as absent to a brand-new engine,
    so the second attempt fails and the caller gets an operational error --
    never an empty corpus.
    """
    connector = _provision(tmp_path)
    stub = _ColdOnce({"ids": []}, failures=99)
    _pin_collection(connector, monkeypatch, stub)

    with pytest.raises(ConnectorOperationalError) as raised:
        connector.pull_all_chunks()

    assert stub.calls == 2, "a failing store must not be retried indefinitely"
    message = str(raised.value)
    assert "corpus read failed" in message
    assert "re-opening" in message, "the error should say the recovery was tried"
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.reference_extra
def test_an_unrelated_backend_failure_is_not_retried(
    tmp_path, monkeypatch, instant_retry
):
    connector = _provision(tmp_path)
    stub = _ColdOnce({"ids": []}, failures=99, exc=TimeoutError("backend unavailable"))
    _pin_collection(connector, monkeypatch, stub)

    with pytest.raises(ConnectorOperationalError) as raised:
        connector.pull_all_chunks()

    assert stub.calls == 1, "only the cold-read signature earns a retry"
    assert "backend unavailable" in str(raised.value)
    assert isinstance(raised.value.__cause__, TimeoutError)


@pytest.mark.reference_extra
def test_query_and_chunk_vectors_get_the_same_recovery(
    tmp_path, monkeypatch, instant_retry
):
    """The cold segment breaks every vector read, not only pull_all_chunks."""
    connector = _provision(tmp_path)
    payload = {
        "ids": [["a"]],
        "documents": [["text"]],
        "metadatas": [[{"chunk_id": "a"}]],
        "distances": [[0.25]],
    }
    _pin_collection(connector, monkeypatch, _ColdOnce(payload, failures=1))
    assert connector.query("apple", top_k=1)[0].chunk_id == "a"

    flat = {"ids": ["a"], "embeddings": [[0.5, 0.5]], "metadatas": [{"chunk_id": "a"}]}
    _pin_collection(connector, monkeypatch, _ColdOnce(flat, failures=1))
    assert connector.get_chunk_vectors(["a"]) == {"a": [0.5, 0.5]}


@pytest.mark.reference_extra
def test_health_recovers_from_a_cold_segment_but_still_reports_a_broken_one(
    tmp_path, monkeypatch, instant_retry
):
    connector = _provision(tmp_path)
    _pin_collection(connector, monkeypatch, _ColdOnce({"ids": ["a"]}, failures=1))
    assert connector.health().ok is True

    _pin_collection(connector, monkeypatch, _ColdOnce({"ids": []}, failures=99))
    assert connector.health().ok is False


# -- the provisioning barrier ------------------------------------------------

class _StubCollection:
    name = "stub_collection"

    def __init__(self, count, get_result=None, get_exc=None):
        self._count = count
        self._get_result = get_result or {"ids": ["a"]}
        self._get_exc = get_exc

    def count(self):
        return self._count

    def get(self, **kwargs):
        if self._get_exc:
            raise self._get_exc
        return self._get_result


def test_the_barrier_fails_at_the_write_site_when_the_segment_is_unreadable():
    """Without this the same condition surfaces in another process, later.

    The read that fails downstream is ``get(include=[..., "embeddings"])`` --
    the call that makes Chroma construct the HNSW segment reader. Provisioning
    runs it here so an unmaterialised segment is a provisioning failure naming
    the collection, not a mysterious empty corpus at evaluation time.
    """
    collection = _StubCollection(3, get_exc=RuntimeError(COLD_MESSAGE))

    with pytest.raises(ConnectorOperationalError, match="vector segment is not readable"):
        durability_barrier(collection, 3)


def test_the_barrier_catches_a_short_write():
    with pytest.raises(ConnectorOperationalError, match="reports 1"):
        durability_barrier(_StubCollection(1), 3)


def test_the_barrier_catches_a_store_that_reports_rows_but_returns_none():
    with pytest.raises(ConnectorOperationalError, match="returned none"):
        durability_barrier(_StubCollection(3, get_result={"ids": []}), 3)


def test_the_barrier_passes_a_healthy_store():
    assert durability_barrier(_StubCollection(3), 3) is None


@pytest.mark.reference_extra
def test_provisioning_actually_runs_the_barrier(tmp_path, monkeypatch):
    """Pinned so the barrier cannot be dropped from the write path silently."""
    from rag_connector import reference_provision

    seen = []
    monkeypatch.setattr(
        reference_provision,
        "durability_barrier",
        lambda collection, expected: seen.append(expected),
    )

    _provision(tmp_path)

    assert seen and seen[0] > 0, "provisioning returned without verifying the store"


def test_the_barrier_probe_asks_for_embeddings(monkeypatch):
    """A metadata-only read would pass straight over the bug.

    The vector segment reader is only constructed when embeddings are
    requested, so this include list is the load-bearing detail of the barrier.
    """
    asked = {}

    class Recording(_StubCollection):
        def get(self, **kwargs):
            asked.update(kwargs)
            return {"ids": ["a"]}

    durability_barrier(Recording(3), 3)

    assert "embeddings" in asked.get("include", [])
