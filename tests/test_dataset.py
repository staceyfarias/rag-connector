"""The Dataset folder spec (docs/dataset-spec.md), tested against hand-derived values.

Every expected hash below is computed in the test from the spec's own
definition, written out literally — never by calling the function under test
and recording what it returned. Two digests are additionally pinned as literal
hex because an independent implementation produced them: testset-kit's
``testset_kit/core.py::chunk_inventory_sha256`` (kit 0.5.0), run on the same
input on 2026-09-24. The spec adopts that definition exactly, so agreeing with
it is the point.

Test data is synthetic, or the bundled pelorus_space corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_connector.base import ChunkRecord
from rag_connector.dataset import (
    DATASET_SPEC_ID,
    DATASET_SPEC_VERSION,
    LEGACY_LAYOUT_RAGAUGE_V0,
    DatasetIntegrityError,
    chunk_inventory_sha256,
    chunker_chunking,
    dataset_id_for_inventory,
    exported_chunking,
    list_extensions,
    read_dataset_folder,
    read_legacy_dataset_folder,
    source_manifest_from_chunks,
    write_dataset_folder,
    write_extension_envelope,
)
from rag_connector.fingerprints import corpus_fingerprint


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(chunk_id, text, *, doc="d.txt", idx=0, gidx=None, meta=None):
    return ChunkRecord(
        chunk_id=chunk_id, doc_id=doc, source_file=doc, doc_chunk_index=idx,
        global_index=gidx, text=text,
        metadata=dict(meta or {}),
    )


# The three synthetic chunks `dataset build` produces from the docs folder in
# tests/test_dataset_cli.py, hand-split there. Reused here so the pinned kit
# digest covers the same input.
SYNTHETIC = [
    ("a.txt:chunk-0", "Alpha beta."),
    ("b.md:chunk-0", "aaaa bbbb cccc dddd\n\neeee"),
    ("b.md:chunk-1", "eeee ffff gggg hhhh"),
]
#: testset-kit's chunk_inventory_sha256 on SYNTHETIC (independent implementation).
SYNTHETIC_INVENTORY_BY_TESTSET_KIT = (
    "bfc47256c82fc7b2a2b0a487d7f2e4603fb41dfc7a2c360c72b24321fc6a8272"
)
#: testset-kit's chunk_inventory_sha256 on the bundled frozen 605-chunk split.
PELORUS_SPACE_INVENTORY_BY_TESTSET_KIT = (
    "42e45a7af6eda157862c0092b3200e9dd096a19fb54c2f0b3eabcb8c9fe371f7"
)


# --- the chunk inventory ----------------------------------------------------

def test_chunk_inventory_is_the_testset_kit_definition_written_out():
    # Lines sorted by id: <id>\t<sha256(text)>\n, then sha256 of the whole.
    expected = _sha(
        "a.txt:chunk-0\t" + _sha("Alpha beta.") + "\n"
        + "b.md:chunk-0\t" + _sha("aaaa bbbb cccc dddd\n\neeee") + "\n"
        + "b.md:chunk-1\t" + _sha("eeee ffff gggg hhhh") + "\n"
    )
    rows = [{"chunk_id": cid, "text": text} for cid, text in reversed(SYNTHETIC)]
    assert chunk_inventory_sha256(rows) == expected
    assert expected == SYNTHETIC_INVENTORY_BY_TESTSET_KIT


def test_chunk_inventory_accepts_chunk_records_and_ignores_input_order():
    records = [_chunk(cid, text) for cid, text in SYNTHETIC]
    assert chunk_inventory_sha256(records) == SYNTHETIC_INVENTORY_BY_TESTSET_KIT
    assert chunk_inventory_sha256(records[::-1]) == SYNTHETIC_INVENTORY_BY_TESTSET_KIT


def test_chunk_inventory_sorts_by_code_point_not_by_natural_order():
    # "chunk-10" < "chunk-2" by code point; a natural sort would swap them.
    expected = _sha("x:chunk-10\t" + _sha("ten") + "\n" + "x:chunk-2\t" + _sha("two") + "\n")
    rows = [{"chunk_id": "x:chunk-2", "text": "two"}, {"chunk_id": "x:chunk-10", "text": "ten"}]
    assert chunk_inventory_sha256(rows) == expected


def test_a_rechunk_that_refills_the_same_ids_changes_the_inventory():
    before = [{"chunk_id": "d:chunk-0", "text": "old words"}]
    after = [{"chunk_id": "d:chunk-0", "text": "new words"}]
    assert chunk_inventory_sha256(before) != chunk_inventory_sha256(after)


def test_the_bundled_frozen_split_has_the_inventory_testset_kit_computes():
    from rag_connector.ingest import load_bundled_chunks

    assert chunk_inventory_sha256(load_bundled_chunks()) == (
        PELORUS_SPACE_INVENTORY_BY_TESTSET_KIT
    )


def test_dataset_id_is_ds_plus_the_first_16_hex_of_the_inventory():
    assert dataset_id_for_inventory(SYNTHETIC_INVENTORY_BY_TESTSET_KIT) == "ds-bfc47256c82fc7b2"
    with pytest.raises(ValueError):
        dataset_id_for_inventory("not-a-hash")


# --- the source list (ported from RAGauge) ----------------------------------

def test_source_manifest_uses_metadata_identity_first_and_the_first_chunk_seen():
    chunks = [
        _chunk("x:0", "one", doc="docs/x.md", meta={
            "source_identity": "docs/x.md", "source_fingerprint": "fp-x",
            "relative_path": "docs/x.md", "file_type": "md"}),
        _chunk("x:1", "two", doc="docs/x.md", idx=1, meta={
            "source_identity": "docs/x.md", "source_fingerprint": "IGNORED"}),
        # No metadata at all: identity falls back to doc_id, file_type to suffix,
        # fingerprint is "" (unknown), relative_path to source_file.
        _chunk("a:0", "three", doc="a.txt"),
    ]
    assert source_manifest_from_chunks(chunks) == [
        {"source_identity": "a.txt", "source_fingerprint": "", "relative_path": "a.txt",
         "filename": "a.txt", "file_type": "txt"},
        {"source_identity": "docs/x.md", "source_fingerprint": "fp-x",
         "relative_path": "docs/x.md", "filename": "docs/x.md", "file_type": "md"},
    ]


# --- writing and reading ----------------------------------------------------

def _write(tmp_path, chunks=None, **kwargs):
    chunks = chunks if chunks is not None else [
        _chunk(cid, text, doc=cid.split(":")[0], idx=int(cid[-1]), gidx=i,
               meta={"source_identity": cid.split(":")[0],
                     "document_filepath": "C:/Users/someone/docs/" + cid.split(":")[0]})
        for i, (cid, text) in enumerate(SYNTHETIC)
    ]
    kwargs.setdefault("name", "synthetic")
    kwargs.setdefault("chunking", chunker_chunking(chunk_size=30, chunk_overlap=5))
    kwargs.setdefault("created_at", "2026-09-24T00:00:00Z")
    return write_dataset_folder(tmp_path / "ds", chunks, **kwargs)


def test_written_manifest_carries_every_core_field(tmp_path):
    dataset = _write(tmp_path)
    manifest = json.loads((tmp_path / "ds" / "dataset.json").read_text(encoding="utf-8"))

    assert manifest["spec"] == DATASET_SPEC_ID == "rag-connector-dataset"
    assert manifest["spec_version"] == DATASET_SPEC_VERSION == "1.0"
    assert manifest["dataset_id"] == "ds-bfc47256c82fc7b2"
    assert manifest["name"] == "synthetic"
    assert manifest["created_at"] == "2026-09-24T00:00:00Z"
    assert manifest["chunking"] == {
        "method": "rag-connector-chunker", "chunker_version": "1",
        "chunk_size": 30, "chunk_overlap": 5,
    }
    assert manifest["sources"] == {"source_count": 2, "file_types": {"md": 1, "txt": 1}}
    assert manifest["chunk_count"] == 3
    assert manifest["chunk_inventory_sha256"] == SYNTHETIC_INVENTORY_BY_TESTSET_KIT
    assert dataset.verified and not dataset.legacy
    assert dataset.dataset_id == "ds-bfc47256c82fc7b2"


def test_core_sha256_is_the_documented_digest_of_the_three_files(tmp_path):
    _write(tmp_path)
    folder = tmp_path / "ds"
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "core_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    expected = hashlib.sha256((
        "rag-connector-dataset-core/1\n"
        "dataset.json\t" + hashlib.sha256(canonical).hexdigest() + "\n"
        "data/chunks.jsonl\t"
        + hashlib.sha256((folder / "data" / "chunks.jsonl").read_bytes()).hexdigest() + "\n"
        "data/sources.json\t"
        + hashlib.sha256((folder / "data" / "sources.json").read_bytes()).hexdigest() + "\n"
    ).encode("utf-8")).hexdigest()
    assert manifest["core_sha256"] == expected


def test_corpus_fingerprint_is_the_existing_source_level_one(tmp_path):
    dataset = _write(tmp_path)
    sources = json.loads((tmp_path / "ds" / "data" / "sources.json").read_text("utf-8"))
    assert dataset.corpus_fingerprint == corpus_fingerprint(sources)


def test_the_absolute_document_path_and_embeddings_are_never_written(tmp_path):
    chunks = [_chunk("d:0", "text", meta={"document_filepath": "/home/me/d.txt",
                                          "keep": "yes"})]
    chunks[0].embedding = [0.1, 0.2]
    dataset = _write(tmp_path, chunks=chunks)
    raw = (tmp_path / "ds" / "data" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "document_filepath" not in raw and "/home/me" not in raw
    assert "embedding" not in json.loads(raw.splitlines()[0])
    assert dataset.chunks[0].metadata == {"keep": "yes"}
    # The caller's own record is untouched.
    assert chunks[0].metadata["document_filepath"] == "/home/me/d.txt"


def test_files_are_written_with_lf_line_endings(tmp_path):
    _write(tmp_path)
    for name in ("dataset.json", "data/chunks.jsonl", "data/sources.json"):
        assert b"\r\n" not in (tmp_path / "ds" / name).read_bytes()


def test_identical_content_gets_the_same_id_whatever_the_name_or_time(tmp_path):
    one = _write(tmp_path / "1", name="first", created_at="2026-01-01T00:00:00Z")
    two = _write(tmp_path / "2", name="second", created_at="2027-01-01T00:00:00Z")
    assert one.dataset_id == two.dataset_id
    assert one.core_sha256 != two.core_sha256  # different manifests, same content


def test_writing_refuses_to_overwrite(tmp_path):
    _write(tmp_path)
    before = (tmp_path / "ds" / "dataset.json").read_bytes()
    with pytest.raises(FileExistsError):
        _write(tmp_path, name="again")
    assert (tmp_path / "ds" / "dataset.json").read_bytes() == before


def test_writing_into_an_existing_empty_directory_is_allowed(tmp_path):
    (tmp_path / "ds").mkdir()
    assert _write(tmp_path).verified


def test_writing_refuses_a_non_empty_directory_even_without_a_manifest(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        _write(tmp_path)


def test_writing_leaves_no_staging_directory_behind(tmp_path):
    _write(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


@pytest.mark.parametrize("chunks, message", [
    ([], "at least one chunk"),
    ([_chunk("d:0", "x"), _chunk("d:0", "y", idx=1)], "duplicate chunk_id"),
    ([_chunk("d:0", "   ")], "blank text"),
    ([_chunk("", "x")], "chunk_id"),
])
def test_writing_refuses_chunks_that_break_the_contract(tmp_path, chunks, message):
    with pytest.raises(ValueError, match=message):
        _write(tmp_path, chunks=chunks)
    assert not (tmp_path / "ds").exists()


def test_writing_requires_a_chunking_block_with_a_method(tmp_path):
    with pytest.raises(ValueError, match="chunking.method"):
        _write(tmp_path, chunking={"chunk_size": 10})
    with pytest.raises(ValueError, match="chunk_overlap"):
        _write(tmp_path, chunking={"method": "rag-connector-chunker",
                                   "chunker_version": "1", "chunk_size": 10})


def test_an_exported_chunking_block_names_the_connector(tmp_path):
    dataset = _write(tmp_path, chunking=exported_chunking("my-rag"))
    assert dataset.chunking == {"method": "exported-from-connector",
                                "connector_type": "my-rag"}
    with pytest.raises(ValueError):
        exported_chunking("")


def test_another_tools_chunking_method_is_allowed(tmp_path):
    dataset = _write(tmp_path, chunking={"method": "acme-splitter", "tokens": 256})
    assert dataset.chunking["method"] == "acme-splitter"


# --- verification on read ---------------------------------------------------

def _edit_json(path: Path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_a_changed_chunk_text_is_refused(tmp_path):
    _write(tmp_path)
    path = tmp_path / "ds" / "data" / "chunks.jsonl"
    path.write_bytes(path.read_bytes().replace(b"Alpha beta.", b"Alpha gamma"))
    with pytest.raises(DatasetIntegrityError, match="chunk_inventory_sha256"):
        read_dataset_folder(tmp_path / "ds")


def test_a_changed_source_list_is_refused(tmp_path):
    _write(tmp_path)
    _edit_json(tmp_path / "ds" / "data" / "sources.json",
               lambda rows: rows[0].update(source_fingerprint="tampered"))
    with pytest.raises(DatasetIntegrityError, match="core_sha256"):
        read_dataset_folder(tmp_path / "ds")


@pytest.mark.parametrize("change", [
    lambda m: m.update(name="renamed"),
    lambda m: m.update(description="a human edit"),  # even an unknown key
    lambda m: m.pop("created_at"),
])
def test_any_manifest_edit_after_writing_is_refused(tmp_path, change):
    _write(tmp_path)
    _edit_json(tmp_path / "ds" / "dataset.json", change)
    with pytest.raises(DatasetIntegrityError, match="core_sha256"):
        read_dataset_folder(tmp_path / "ds")


def test_reformatting_the_manifest_whitespace_is_not_a_change(tmp_path):
    _write(tmp_path)
    _edit_json(tmp_path / "ds" / "dataset.json", lambda m: None)  # re-indented
    assert read_dataset_folder(tmp_path / "ds").verified


def test_unknown_manifest_keys_written_with_the_core_are_ignored(tmp_path):
    """A writer on a later minor version may add keys; this reader ignores them.

    Built by hand to the spec's definition, so the reader is checked against the
    spec rather than against its own writer.
    """
    dataset = _write(tmp_path)
    folder = tmp_path / "ds"
    manifest = dict(dataset.manifest)
    manifest["spec_version"] = "1.7"
    manifest["future_field"] = {"anything": [1, 2]}
    body = {k: v for k, v in manifest.items() if k != "core_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    manifest["core_sha256"] = hashlib.sha256((
        "rag-connector-dataset-core/1\n"
        "dataset.json\t" + hashlib.sha256(canonical).hexdigest() + "\n"
        "data/chunks.jsonl\t"
        + hashlib.sha256((folder / "data" / "chunks.jsonl").read_bytes()).hexdigest() + "\n"
        "data/sources.json\t"
        + hashlib.sha256((folder / "data" / "sources.json").read_bytes()).hexdigest() + "\n"
    ).encode("utf-8")).hexdigest()
    (folder / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

    again = read_dataset_folder(folder)
    assert again.verified and again.spec_version == "1.7"


def test_a_different_major_version_is_refused(tmp_path):
    _write(tmp_path)
    _edit_json(tmp_path / "ds" / "dataset.json", lambda m: m.update(spec_version="2.0"))
    with pytest.raises(DatasetIntegrityError, match="major version 1"):
        read_dataset_folder(tmp_path / "ds")


def test_a_missing_core_file_is_refused(tmp_path):
    _write(tmp_path)
    (tmp_path / "ds" / "data" / "sources.json").unlink()
    with pytest.raises(DatasetIntegrityError, match="missing a core file"):
        read_dataset_folder(tmp_path / "ds")


def test_a_folder_without_a_manifest_is_not_a_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_dataset_folder(tmp_path)


def test_reading_ignores_testsets_and_extensions(tmp_path):
    _write(tmp_path)
    (tmp_path / "ds" / "testsets" / "mine").mkdir(parents=True)
    (tmp_path / "ds" / "testsets" / "mine" / "testset.csv").write_text("q\n")
    (tmp_path / "ds" / "extensions" / "other-tool").mkdir(parents=True)
    (tmp_path / "ds" / "extensions" / "other-tool" / "data.bin").write_bytes(b"\0")
    assert read_dataset_folder(tmp_path / "ds").verified


def test_the_bundled_corpus_round_trips_as_a_verified_dataset(tmp_path):
    from rag_connector.ingest import (
        BUNDLED_CHUNK_OVERLAP,
        BUNDLED_CHUNK_SIZE,
        load_bundled_chunks,
    )

    dataset = write_dataset_folder(
        tmp_path / "pelorus", load_bundled_chunks(), name="Pelorus Space",
        chunking=chunker_chunking(chunk_size=BUNDLED_CHUNK_SIZE,
                                  chunk_overlap=BUNDLED_CHUNK_OVERLAP),
    )
    assert dataset.chunk_count == 605
    assert dataset.source_summary["source_count"] == 92
    assert dataset.chunk_inventory_sha256 == PELORUS_SPACE_INVENTORY_BY_TESTSET_KIT
    assert dataset.dataset_id == "ds-" + PELORUS_SPACE_INVENTORY_BY_TESTSET_KIT[:16]
    # The frozen chunk file is already in the Dataset's row format, and carries
    # no document_filepath, so the Dataset's chunks.jsonl is that file exactly.
    from rag_connector.ingest import bundled_chunks_path
    with bundled_chunks_path() as frozen:
        assert (tmp_path / "pelorus" / "data" / "chunks.jsonl").read_bytes() == (
            Path(frozen).read_bytes()
        )


# --- the legacy RAGauge v0 layout -------------------------------------------

def _write_v0(folder: Path, *, chunk_count=2):
    """A RAGauge v0 folder, shaped as RAGauge's store writes one (synthetic)."""
    (folder / "data").mkdir(parents=True)
    rows = [
        {"chunk_id": "doc.md:chunk-0", "doc_id": "doc.md", "source_file": "doc.md",
         "doc_chunk_index": 0, "global_index": 0, "char_start": 0, "char_end": 5,
         "content_module_id": "doc.md", "text": "Hello",
         "metadata": {"source_identity": "doc.md", "source_fingerprint": "fp",
                      "file_type": "md", "relative_path": "doc.md"}},
        {"chunk_id": "doc.md:chunk-1", "doc_id": "doc.md", "source_file": "doc.md",
         "doc_chunk_index": 1, "global_index": 1, "char_start": 6, "char_end": 11,
         "content_module_id": "doc.md", "text": "World",
         "metadata": {"source_identity": "doc.md", "source_fingerprint": "fp",
                      "file_type": "md", "relative_path": "doc.md"}},
    ]
    (folder / "data" / "chunks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    sources = [{"source_identity": "doc.md", "source_fingerprint": "fp",
                "relative_path": "doc.md", "filename": "doc.md", "file_type": "md"}]
    (folder / "data" / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    manifest = {
        "id": "my-dataset-0123456789ab", "name": "My Dataset", "connector_type": "sandbox",
        "corpus_fingerprint": corpus_fingerprint(sources), "pipeline_fingerprint": "abcd",
        "chunk_count": chunk_count, "source_count": 1, "context": "",
        "context_version": "2026-07-09T13:36:44", "created_at": "2026-07-08T11:12:56",
    }
    (folder / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_legacy_folder_is_refused_by_the_verifying_reader_by_default(tmp_path):
    folder = _write_v0(tmp_path / "v0")
    with pytest.raises(DatasetIntegrityError, match="legacy RAGauge"):
        read_dataset_folder(folder)


def test_a_legacy_folder_reads_unverified_and_is_never_written(tmp_path):
    folder = _write_v0(tmp_path / "v0")
    before = _snapshot(folder)

    dataset = read_legacy_dataset_folder(folder)
    also = read_dataset_folder(folder, allow_legacy=True)

    assert _snapshot(folder) == before
    for d in (dataset, also):
        assert d.layout == LEGACY_LAYOUT_RAGAUGE_V0 and d.legacy
        assert not d.verified
        assert d.core_sha256 is None and d.chunking is None
        assert d.dataset_id == "my-dataset-0123456789ab"
        assert d.chunk_count == 2
        assert d.chunk_inventory_sha256 == _sha(
            "doc.md:chunk-0\t" + _sha("Hello") + "\n"
            + "doc.md:chunk-1\t" + _sha("World") + "\n")
        assert d.warnings[0].startswith("legacy RAGauge v0 layout")


def test_a_legacy_count_mismatch_is_reported_not_refused(tmp_path):
    folder = _write_v0(tmp_path / "v0", chunk_count=7)
    dataset = read_legacy_dataset_folder(folder)
    assert any("chunk_count recorded 7" in w for w in dataset.warnings)


def test_the_legacy_reader_refuses_a_spec_folder(tmp_path):
    _write(tmp_path)
    with pytest.raises(DatasetIntegrityError):
        read_legacy_dataset_folder(tmp_path / "ds")


# --- extensions -------------------------------------------------------------

def test_an_extension_envelope_records_the_core_it_was_built_on(tmp_path):
    dataset = _write(tmp_path)
    area = write_extension_envelope(tmp_path / "ds", "ragauge", tool_version="0.9",
                                    format_version="1",
                                    created_at="2026-09-24T01:00:00Z")
    assert area == tmp_path / "ds" / "extensions" / "ragauge"
    assert json.loads((area / "extension.json").read_text(encoding="utf-8")) == {
        "tool": "ragauge", "tool_version": "0.9", "format_version": "1",
        "core_sha256": dataset.core_sha256, "created_at": "2026-09-24T01:00:00Z",
    }
    # Writing an extension does not touch the core.
    assert read_dataset_folder(tmp_path / "ds").core_sha256 == dataset.core_sha256


def test_extensions_are_discovered_by_listing_envelopes_not_by_the_manifest(tmp_path):
    dataset = _write(tmp_path)
    folder = tmp_path / "ds"
    write_extension_envelope(folder, "zeta", tool_version="1", format_version="1")
    write_extension_envelope(folder, "alpha", tool_version="2", format_version="3")
    (folder / "extensions" / "no-envelope").mkdir()
    (folder / "extensions" / "broken").mkdir()
    (folder / "extensions" / "broken" / "extension.json").write_text("{nope")

    found = list_extensions(folder)
    assert [e.tool for e in found] == ["alpha", "broken", "zeta"]
    assert found[0].problems == () and found[0].built_on(dataset)
    assert found[0].tool_version == "2" and found[0].format_version == "3"
    assert found[1].problems and "not valid JSON" in found[1].problems[0]
    assert "extensions" not in json.dumps(dataset.manifest)


def test_an_envelope_whose_tool_disagrees_with_its_area_is_flagged(tmp_path):
    dataset = _write(tmp_path)
    area = tmp_path / "ds" / "extensions" / "mine"
    area.mkdir(parents=True)
    (area / "extension.json").write_text(json.dumps({
        "tool": "someone-else", "tool_version": "1", "format_version": "1",
        "core_sha256": dataset.core_sha256, "created_at": "x"}))
    [envelope] = list_extensions(tmp_path / "ds")
    assert any("someone-else" in p for p in envelope.problems)


def test_no_extensions_directory_lists_nothing(tmp_path):
    _write(tmp_path)
    assert list_extensions(tmp_path / "ds") == []


def test_an_existing_envelope_is_replaced_only_on_request(tmp_path):
    _write(tmp_path)
    folder = tmp_path / "ds"
    write_extension_envelope(folder, "tool", tool_version="1", format_version="1")
    with pytest.raises(FileExistsError):
        write_extension_envelope(folder, "tool", tool_version="2", format_version="1")
    write_extension_envelope(folder, "tool", tool_version="2", format_version="1",
                             overwrite=True)
    assert list_extensions(folder)[0].tool_version == "2"


@pytest.mark.parametrize("bad", ["", "Upper", "../escape", "a/b", ".hidden"])
def test_extension_tool_names_are_plain_lowercase_directory_names(tmp_path, bad):
    _write(tmp_path)
    with pytest.raises(ValueError):
        write_extension_envelope(tmp_path / "ds", bad, tool_version="1", format_version="1")


def test_an_envelope_cannot_be_written_on_a_tampered_or_legacy_folder(tmp_path):
    _write(tmp_path)
    _edit_json(tmp_path / "ds" / "dataset.json", lambda m: m.update(name="x"))
    with pytest.raises(DatasetIntegrityError):
        write_extension_envelope(tmp_path / "ds", "t", tool_version="1", format_version="1")
    folder = _write_v0(tmp_path / "v0")
    with pytest.raises(DatasetIntegrityError):
        write_extension_envelope(folder, "t", tool_version="1", format_version="1")


def test_the_dataset_module_imports_with_no_optional_extras():
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import sys
        BLOCKED = {"chromadb", "pypdf", "docx", "striprtf", "fastembed"}
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(name)
                return None
        sys.meta_path.insert(0, Blocker())
        import rag_connector.dataset
        print("OK")
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
