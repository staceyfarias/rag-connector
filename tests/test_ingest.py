"""The ingest kit is declared API, and is NOT the connector contract.

These tests pin the two things that make it usable from outside this repo: it
imports on a bare core install, and the names it was promoted from still work.
"""

import hashlib
import importlib.resources
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest


def test_ingest_imports_with_no_optional_extras_installed():
    """``import rag_connector.ingest`` must not require the ``reference`` extra.

    Every parser (``pypdf``, ``python-docx``, ``striprtf``) and ``chromadb`` is
    imported inside the function that needs it. That is invisible from this
    suite's own environment, where the extras ARE installed -- so the check runs
    in a subprocess with those modules blocked at import time, which is what a
    ``pip install rag-connector`` user actually has.
    """
    program = textwrap.dedent(
        """
        import sys

        BLOCKED = {"chromadb", "pypdf", "docx", "striprtf", "fastembed"}


        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked for this test: {name}")
                return None


        for name in list(sys.modules):
            if name.split(".")[0] in BLOCKED:
                del sys.modules[name]
        sys.meta_path.insert(0, Blocker())

        from rag_connector.ingest import (
            LoadedDocument,
            chunk_loaded_documents,
            document_source_identity,
            load_documents,
        )

        doc = LoadedDocument(
            filename="a.txt",
            filepath="a.txt",
            relative_path="a.txt",
            file_type="txt",
            content="alpha beta gamma delta " * 20,
            source_identity="id",
            source_fingerprint="fp",
        )
        records = chunk_loaded_documents([doc], chunk_size=80, chunk_overlap=10)
        assert records, "the kit imported but produced nothing"
        print("OK", len(records))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


def test_the_pre_move_import_paths_still_resolve():
    """``document_loader`` and ``reference`` re-export what moved to ``ingest``."""
    from rag_connector import document_loader, ingest, reference

    assert document_loader.load_documents is ingest.load_documents
    assert document_loader.LoadedDocument is ingest.LoadedDocument
    assert reference.chunk_loaded_documents is ingest.chunk_loaded_documents
    assert reference.LoadedDocument is ingest.LoadedDocument


# --- chunking invariants -----------------------------------------------------
#
# Two defects lived in chunk_loaded_documents for as long as it existed, and
# both are invisible from a chunk-count assertion. They are pinned here as
# properties that must hold over WHOLE corpora, at several parameter settings,
# rather than as a fixed expected output nobody can read.


def assert_chunking_invariants(documents, *, chunk_size, chunk_overlap):
    """Every invariant chunk_loaded_documents promises, over a whole corpus."""
    from rag_connector.ingest import chunk_loaded_documents

    records = chunk_loaded_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    assert records, "a non-empty corpus produced no chunks"
    content_by_doc = {doc.source_identity: doc.content for doc in documents}

    previous_by_doc = {}
    for record in records:
        source = content_by_doc[record.doc_id]

        # (a) The recorded span IS the text. The chunker strips surrounding
        # whitespace off each window; recording the unstripped window made
        # source[char_start:char_end] != text for essentially every chunk,
        # which silently breaks any caller resolving a citation by offset.
        assert source[record.char_start:record.char_end] == record.text, (
            f"{record.chunk_id}: span {record.char_start}:{record.char_end} "
            f"does not hold the chunk text"
        )

        previous = previous_by_doc.get(record.doc_id)
        if previous is not None:
            # (b) No chunk is wholly contained in its predecessor. The final
            # window of a document is clamped to the end of the content, which
            # produced a trailing chunk of pure overlap -- an extra embedding,
            # an extra row, and a duplicate retrieval hit for no new text.
            assert not (
                previous.char_start <= record.char_start
                and record.char_end <= previous.char_end
            ), f"{record.chunk_id} is wholly inside {previous.chunk_id}"
            # Chunks still advance and still overlap in document order.
            assert record.char_start > previous.char_start
        previous_by_doc[record.doc_id] = record

        # Chunk ids stay dense per document even when a chunk is dropped,
        # and carry the self-describing ``:chunk-N`` suffix.
        assert record.chunk_id == f"{record.doc_id}:chunk-{record.doc_chunk_index}"

    for doc_id in {r.doc_id for r in records}:
        indices = [r.doc_chunk_index for r in records if r.doc_id == doc_id]
        assert indices == list(range(len(indices)))
    assert [r.global_index for r in records] == list(range(len(records)))
    return records


def _realistic_corpus(tmp_path):
    """Documents shaped like real source files, not like a chunker's happy path.

    The span defect only shows on chunks whose window boundary lands on the
    separator the chunker searched for, so the text below deliberately mixes
    paragraph breaks, single newlines, sentence ends, long unbroken runs, and
    trailing whitespace -- and includes documents whose length falls just past
    a chunk boundary, which is what produced the redundant trailing chunk.
    """
    from rag_connector.ingest import load_documents

    paragraph = (
        "Departure windows for the Ceres transfer open twice a synodic period. "
        "Passengers holding a scrubbed booking keep their fare class.\n"
        "Rebooking is automatic; no fee applies.\n\n"
    )
    (tmp_path / "windows.txt").write_text(paragraph * 9, encoding="utf-8")
    (tmp_path / "trailing.md").write_text(
        paragraph * 6 + "  \n\n   ", encoding="utf-8"
    )
    (tmp_path / "unbroken.txt").write_text(
        "A" * 40 + " " + ("longtokenwithoutanybreaks" * 60), encoding="utf-8"
    )
    (tmp_path / "short.txt").write_text("  one short line.  \n", encoding="utf-8")
    nested = tmp_path / "archive"
    nested.mkdir()
    (nested / "unicode.md").write_text(
        ("Ríos de Venus — the glass floor tour. 展望台からの眺め。\n\n" * 25),
        encoding="utf-8",
    )
    return load_documents(str(tmp_path))


def test_every_chunk_span_holds_exactly_its_own_text(tmp_path):
    documents = _realistic_corpus(tmp_path)
    for chunk_size, chunk_overlap in [
        (1000, 200), (400, 80), (120, 30), (80, 10), (77, 0),
    ]:
        assert_chunking_invariants(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


def test_a_trailing_chunk_of_pure_overlap_is_not_emitted():
    """The reported case: a 2282-character document used to end in a 200-char
    chunk whose span was already wholly inside the one before it."""
    from rag_connector.ingest import LoadedDocument, chunk_loaded_documents

    content = ("word " * 456) + "end."
    assert len(content) == 2284
    doc = LoadedDocument(
        filename="d.txt", filepath="d.txt", relative_path="d.txt",
        file_type="txt", content=content,
        source_identity="doc", source_fingerprint="fp",
    )
    spans = [
        (r.char_start, r.char_end)
        for r in chunk_loaded_documents([doc], chunk_size=1000, chunk_overlap=200)
    ]
    assert spans[-1][1] == len(content)
    assert not (spans[-2][0] <= spans[-1][0] and spans[-1][1] <= spans[-2][1])


def test_an_overlap_that_cannot_advance_is_refused_not_repaired():
    """An overlap at least as wide as the window used to be silently replaced
    with ``chunk_size // 4``. The caller then held a corpus split at parameters
    they never chose and were never told about -- and the parameters are what a
    stored chunk set claims to be reproducible from, so the record of how it was
    made was wrong for anyone who tried."""
    from rag_connector.ingest import LoadedDocument, chunk_loaded_documents

    doc = LoadedDocument(
        filename="d.txt", filepath="d.txt", relative_path="d.txt",
        file_type="txt", content="word " * 400,
        source_identity="doc", source_fingerprint="fp",
    )

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_loaded_documents([doc], chunk_size=1000, chunk_overlap=1000)

    # And one character below the window is still a legal, if useless, split.
    assert chunk_loaded_documents([doc], chunk_size=1000, chunk_overlap=999)


# --- the bundled corpus ------------------------------------------------------


def test_the_bundled_corpus_ships_and_is_the_full_28_documents():
    """Package data that fails to ship is invisible from an editable install.

    This package carried no corpus at all until now, so an installed
    ``rag-connector[reference]`` had a Reference RAG and nothing to point it
    at. Read through the same ``importlib.resources`` path production uses, so
    a missing package-data declaration fails here rather than at a user's
    first demo.
    """
    from rag_connector.ingest import (
        BUNDLED_CORPORA,
        DEFAULT_BUNDLED_CORPUS,
        bundled_corpus_path,
        load_bundled_corpus,
    )

    assert DEFAULT_BUNDLED_CORPUS in BUNDLED_CORPORA
    with bundled_corpus_path() as path:
        files = sorted(p.name for p in pathlib.Path(path).iterdir() if p.is_file())
        assert len(files) == 28
        assert {p.rsplit(".", 1)[1] for p in files} == {"txt", "md"}
        # Pelorus's own state directory must not travel with the documents.
        assert not (pathlib.Path(path) / ".pelorus").exists()

    documents = load_bundled_corpus()
    assert len(documents) == 28
    assert all(doc.content.strip() for doc in documents)


def test_the_bundled_corpus_is_marked_mit():
    notice = (
        importlib.resources.files("rag_connector")
        .joinpath("corpus")
        .joinpath("README.md")
        .read_text(encoding="utf-8")
    )
    assert "MIT" in notice


def test_chunking_invariants_hold_over_the_bundled_corpus():
    """The property tests above use constructed text; this one uses the real
    92 documents, which is where both defects were actually measured."""
    from rag_connector.ingest import load_bundled_corpus

    documents = load_bundled_corpus()
    for chunk_size, chunk_overlap in [(1000, 200), (512, 64), (256, 0)]:
        assert_chunking_invariants(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


def test_unknown_bundled_corpus_is_refused_by_name():
    from rag_connector.ingest import bundled_corpus_path

    with pytest.raises(ValueError, match="pelorus_space"):
        with bundled_corpus_path("not_a_corpus"):
            pass


class _FakeEmbeddingClient:
    """Deterministic stand-in: this test is about the corpus, not the model."""

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


@pytest.mark.reference_extra
def test_provisioning_needs_no_folder_from_the_caller(tmp_path):
    """``provision_reference_rag()`` with no folder builds the bundled corpus."""
    from rag_connector.reference_provision import provision_reference_rag

    connector = provision_reference_rag(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="bundled_demo",
        embedding_client=_FakeEmbeddingClient(),
    )
    try:
        snapshot = connector.run_snapshot()
        # Exactly the bundled split: a partial or fixture corpus cannot pass.
        assert snapshot["observations"]["chunk_count"] == 426
    finally:
        connector.close()


# --- the frozen pre-chunked corpus -------------------------------------------


def test_the_frozen_chunk_set_ships_and_is_the_426_chunks():
    """The pre-chunked corpus is package data with the same invisibility trap
    as the documents: an editable install reads it off ``src/`` either way."""
    from rag_connector.ingest import bundled_chunks_path, load_bundled_chunks

    with bundled_chunks_path() as path:
        assert pathlib.Path(path).is_file()

    chunks = load_bundled_chunks()
    assert len(chunks) == 426
    assert len({c.chunk_id for c in chunks}) == 426
    assert [c.global_index for c in chunks] == list(range(426))


def test_frozen_chunk_ids_are_readable_and_self_describing():
    from rag_connector.ingest import load_bundled_chunks

    by_id = {c.chunk_id: c for c in load_bundled_chunks()}
    first = by_id["booking_cancellation_policy.md:chunk-0"]
    assert first.doc_id == "booking_cancellation_policy.md"
    assert first.source_file == "booking_cancellation_policy.md"
    assert first.doc_chunk_index == 0
    assert first.text.strip() == first.text and first.text


def test_the_committed_chunk_set_has_not_drifted_from_the_kit():
    """The whole value of a frozen corpus is that it cannot silently stop being
    what the kit produces. Regenerate and compare, byte for byte."""
    from rag_connector.ingest import (
        build_bundled_chunks,
        bundled_chunks_path,
        chunks_to_jsonl,
    )

    regenerated = chunks_to_jsonl(build_bundled_chunks())
    with bundled_chunks_path() as path:
        raw = pathlib.Path(path).read_bytes()
    committed = raw.decode("utf-8")
    assert regenerated == committed, (
        "the committed chunk set no longer matches what the ingest kit "
        "produces from the bundled documents. If the change is intended, "
        "re-ground every citation that depends on these ids and run "
        "tools/freeze_bundled_chunks.py; otherwise revert the change."
    )

    # And the file is the one the README publishes. README.md prints these
    # three numbers as the way a stranger checks their own reproduction, so
    # they are a promise this package makes to people who cannot run this
    # suite. Without the assertion the documented digest and the shipped file
    # drift apart the first time the corpus is edited on purpose: the
    # comparison above stays green, because both sides moved together.
    assert hashlib.sha256(raw).hexdigest() == (
        "5638e7143fe4b664e874d00aa8722cafaf97972debd1fce8b2d1c1914a0548b4"
    ), "the frozen chunk set changed; update the digest published in README.md"
    assert len(raw) == 613818
    assert raw.count(b"\n") == 426
    assert b"\r\n" not in raw, ".gitattributes pins this file to LF"


def test_the_frozen_form_round_trips_through_its_own_reader():
    from rag_connector.ingest import (
        build_bundled_chunks,
        chunks_from_jsonl,
        chunks_to_jsonl,
        load_bundled_chunks,
    )

    loaded = load_bundled_chunks()
    assert chunks_to_jsonl(chunks_from_jsonl(chunks_to_jsonl(loaded))) == \
        chunks_to_jsonl(loaded)
    # And the loaded records are the built records: nothing is lost in the file.
    assert [c.to_dict() for c in loaded] == [
        c.to_dict() for c in build_bundled_chunks()
    ]


def test_the_frozen_shape_is_the_one_rag_eval_already_writes():
    """One corpus, one chunk set, one set of ids, read by both products
    without an adapter -- which only holds if the fields are the fields."""
    import json

    from rag_connector.ingest import bundled_chunks_path

    with bundled_chunks_path() as path:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 426
    for line in (lines[0], lines[-1]):
        row = json.loads(line)
        assert list(row) == [
            "chunk_id", "doc_id", "source_file", "doc_chunk_index",
            "global_index", "char_start", "char_end", "content_module_id",
            "text", "metadata",
        ]
        assert set(row["metadata"]) == {
            "source_identity", "source_fingerprint", "relative_path",
            "filename", "file_type",
        }


def test_the_frozen_form_carries_no_machine_specific_path():
    """``document_filepath`` is where the chunker happened to read from. Freezing
    one machine's filesystem into a file two products read everywhere else is a
    lie that only shows up on someone else's box."""
    from rag_connector.ingest import bundled_chunks_path

    with bundled_chunks_path() as path:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "document_filepath" not in text
    assert "C:\\\\" not in text


def test_an_unmodelled_field_is_refused_rather_than_dropped():
    from rag_connector.ingest import chunks_from_jsonl

    row = (
        '{"chunk_id": "a:chunk-0", "doc_id": "a", "source_file": "a.md", '
        '"doc_chunk_index": 0, "global_index": 0, "text": "t", '
        '"section_heading": "Refunds"}'
    )
    with pytest.raises(ValueError, match="section_heading"):
        chunks_from_jsonl(row)


@pytest.mark.reference_extra
def test_provisioning_from_the_frozen_chunks_re_chunks_nothing(tmp_path, monkeypatch):
    """A caller must be able to stand up the Reference RAG on the frozen ids
    without the chunker running at all."""
    from rag_connector import reference_provision
    from rag_connector.reference_provision import provision_reference_rag_from_chunks

    calls = []

    def _tripwire(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the chunker ran")

    # Patch the name reference_provision actually calls. It imported the
    # function directly, so patching rag_connector.ingest would leave this test
    # passing whether or not the chunker ran.
    monkeypatch.setattr(reference_provision, "chunk_loaded_documents", _tripwire)

    # The tripwire is armed: the folder path still trips it.
    with pytest.raises(AssertionError, match="the chunker ran"):
        reference_provision.provision_reference_rag(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="unused",
            embedding_client=_FakeEmbeddingClient(),
        )
    calls.clear()

    connector = provision_reference_rag_from_chunks(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="frozen_demo",
        embedding_client=_FakeEmbeddingClient(),
    )
    try:
        assert not calls, "provisioning from frozen chunks re-chunked the corpus"
        pulled = connector.pull_all_chunks()
        assert len(pulled) == 426
        assert "booking_cancellation_policy.md:chunk-0" in {
            c.chunk_id for c in pulled
        }
    finally:
        connector.close()


def test_chunking_parameters_are_refused_when_provisioning_from_chunks(tmp_path):
    from rag_connector.reference_provision import provision_reference_rag_from_chunks

    with pytest.raises(TypeError, match="chunk_size"):
        provision_reference_rag_from_chunks(
            chunk_size=512,
            persist_dir=str(tmp_path / "chroma"),
            collection_name="nope",
            embedding_client=_FakeEmbeddingClient(),
        )
