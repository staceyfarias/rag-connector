import pytest

from rag_connector.document_loader import load_documents


def test_a_non_utf8_file_fails_naming_the_file(tmp_path):
    """A folder load must still fail loud on undecodable text, but the bare
    UnicodeDecodeError said WHICH byte, not which file — leaving the operator
    to bisect a folder by hand."""
    (tmp_path / "good.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "legacy.txt").write_bytes(b"caf\xe9 latin-1 caf\xe9")

    with pytest.raises(ValueError, match="legacy.txt"):
        load_documents(str(tmp_path))


def test_load_documents_recurses_and_fingerprints_stably(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (nested / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"ignored")

    first = load_documents(str(tmp_path))
    second = load_documents(str(tmp_path))

    assert [doc.relative_path for doc in first] == ["a.txt", "nested/b.md"]
    assert [doc.source_identity for doc in first] == [
        doc.source_identity for doc in second
    ]
    assert [doc.source_fingerprint for doc in first] == [
        doc.source_fingerprint for doc in second
    ]


def test_load_documents_rejects_missing_folder(tmp_path):
    missing = tmp_path / "missing"
    try:
        load_documents(str(missing))
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing folder was accepted")



def test_dot_directories_are_not_ingested(tmp_path):
    """A corpus folder is a working directory: it holds ``.git``, and a Pelorus
    dataset folder holds ``.pelorus``. Neither is a document. The walk was safe
    only by accident -- ``.pelorus`` happens to hold no supported extension --
    so the exclusion is pinned with files that would otherwise load."""
    (tmp_path / "real.md").write_text("real", encoding="utf-8")
    for hidden in (".pelorus", ".git"):
        directory = tmp_path / hidden
        directory.mkdir()
        (directory / "state.md").write_text("internal state", encoding="utf-8")
    nested = tmp_path / "archive" / ".cache"
    nested.mkdir(parents=True)
    (nested / "stale.txt").write_text("stale", encoding="utf-8")
    (tmp_path / "archive" / "kept.txt").write_text("kept", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert [doc.relative_path for doc in documents] == ["archive/kept.txt", "real.md"]


def test_source_identity_is_the_readable_relative_path(tmp_path):
    """One identity function across this library and Pelorus, and it is the
    path. The hash it replaced was a pure function of the same path, so it
    bought no uniqueness -- only opacity on the MCP and connector surfaces
    these ids reach."""
    nested = tmp_path / "Archive"
    nested.mkdir()
    (nested / "Booking_Policy.MD").write_text("policy", encoding="utf-8")

    [document] = load_documents(str(tmp_path))

    assert document.source_identity == "archive/booking_policy.md"


def test_load_order_is_the_same_on_every_platform(tmp_path):
    """Sorting ``Path`` objects sorts on a platform-dependent key: Windows
    casefolds and separates with a backslash, POSIX compares the raw string
    with a forward slash. ``global_index`` and the corpus fingerprint are both
    derived from load order, so the two platforms would derive different chunk
    sets from one corpus. The expected order below is the POSIX spelling of
    each relative path, sorted -- and it is the answer on both."""
    (tmp_path / "Alpha.md").write_text("A", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("b", encoding="utf-8")
    (tmp_path / "delta.md").write_text("d", encoding="utf-8")
    nested = tmp_path / "Zone"
    nested.mkdir()
    (nested / "inner.md").write_text("z", encoding="utf-8")
    deep = tmp_path / "Zone-extra"
    deep.mkdir()
    (deep / "inner.md").write_text("x", encoding="utf-8")

    loaded = load_documents(str(tmp_path))

    assert [doc.relative_path for doc in loaded] == [
        "Alpha.md",
        "Zone-extra/inner.md",
        "Zone/inner.md",
        "beta.txt",
        "delta.md",
    ]


def test_a_byte_order_mark_is_not_part_of_the_document(tmp_path):
    """A BOM is an encoding artifact, not text. Read as plain UTF-8 it survives
    as U+FEFF at offset 0, so it becomes the first character of chunk 0 -- inside
    the text a citation resolves to, and inside the fingerprint over it. One
    editor that writes a BOM would then change the corpus."""
    (tmp_path / "bom.md").write_bytes("Booking policy".encode("utf-8-sig"))
    (tmp_path / "plain.md").write_text("Booking policy", encoding="utf-8")

    by_name = {doc.filename: doc for doc in load_documents(str(tmp_path))}

    assert by_name["bom.md"].content == "Booking policy"
    assert not by_name["bom.md"].content.startswith("﻿")
    # And the same text really is the same document, whatever wrote it.
    assert (
        by_name["bom.md"].source_fingerprint
        == by_name["plain.md"].source_fingerprint
    )


def test_a_missing_parser_names_the_extra_that_provides_it(tmp_path, monkeypatch):
    """A core install has no parsers, and pointing a folder load at one that
    happens to hold a PDF is an ordinary thing to do. Unguarded it raised
    ``No module named 'pypdf'`` -- an import the caller never wrote, and no
    route out of it. (``None`` in ``sys.modules`` is how Python spells "this
    import fails now"; monkeypatch puts the real module back.)"""
    import sys

    (tmp_path / "brochure.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(ImportError) as caught:
        load_documents(str(tmp_path))

    message = str(caught.value)
    assert ".pdf" in message and "pypdf" in message
    assert 'pip install "rag-connector[reference]"' in message
