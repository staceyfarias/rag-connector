"""The Dataset folder: a portable, write-once corpus snapshot any tool can read.

This module implements ``docs/dataset-spec.md`` — spec id
:data:`DATASET_SPEC_ID`, version :data:`DATASET_SPEC_VERSION`, versioned
independently of the package and of the connector contract. Read that page for
the rules; the docstrings here say how each function honours them.

A Dataset is a folder::

    <dataset>/
      dataset.json          CORE  manifest (this spec)
      data/chunks.jsonl     CORE  one chunk row per line
      data/sources.json     CORE  the source list
      testsets/<name>/      reserved, shared: defined by testset-kit, not here
      extensions/<tool>/    one area per tool, with an extension.json envelope

The three CORE files are written once, together, by :func:`write_dataset_folder`,
and never again. Everything a tool learns about a Dataset afterwards — a human
description, a scan, curated extracts — goes into that tool's own
``extensions/<tool>/`` area, whose contents this library never defines.

Like the ingest kit, this is **not** part of the connector contract: a connector
is read-only, and a Dataset is something a host writes. Nothing here imports an
optional extra.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import ChunkRecord, RagPipeline
from .fingerprints import corpus_fingerprint
from .ingest import CHUNKER_METHOD, CHUNKER_VERSION, chunks_from_jsonl, chunks_to_jsonl

__all__ = [
    "CHUNKING_EXPORTED",
    "DATASET_SPEC_ID",
    "DATASET_SPEC_VERSION",
    "Dataset",
    "DatasetIntegrityError",
    "EXTENSION_ENVELOPE_FILE",
    "ExtensionEnvelope",
    "LEGACY_LAYOUT_RAGAUGE_V0",
    "LAYOUT_CORE",
    "chunk_inventory_sha256",
    "chunker_chunking",
    "compute_core_sha256",
    "dataset_id_for_inventory",
    "exported_chunking",
    "list_extensions",
    "read_dataset_folder",
    "read_legacy_dataset_folder",
    "source_manifest_from_chunks",
    "write_dataset_folder",
    "write_extension_envelope",
]

#: The spec this module writes and reads. Versioned on its own: a change to the
#: package or the connector contract does not move it, and a change to it is
#: announced in ``docs/dataset-spec.md``.
DATASET_SPEC_ID = "rag-connector-dataset"
DATASET_SPEC_VERSION = "1.0"

#: The major version this reader understands. A higher MINOR is read (its new
#: keys are ignored); a different MAJOR is refused.
_SUPPORTED_MAJOR = "1"

#: ``Dataset.layout`` for a folder written to this spec.
LAYOUT_CORE = DATASET_SPEC_ID
#: ``Dataset.layout`` for a RAGauge folder written before this spec existed.
LEGACY_LAYOUT_RAGAUGE_V0 = "ragauge-v0"

#: ``chunking.method`` for a Dataset exported from a connector's corpus read.
CHUNKING_EXPORTED = "exported-from-connector"

MANIFEST_FILE = "dataset.json"
CHUNKS_FILE = "data/chunks.jsonl"
SOURCES_FILE = "data/sources.json"
CORE_FILES = (MANIFEST_FILE, CHUNKS_FILE, SOURCES_FILE)
TESTSETS_DIR = "testsets"
EXTENSIONS_DIR = "extensions"
EXTENSION_ENVELOPE_FILE = "extension.json"

#: Domain tag hashed in front of the core digest, so a core hash can never be
#: mistaken for a hash of anything else, and a future definition can change it.
_CORE_HASH_TAG = "rag-connector-dataset-core/1\n"

#: Chunk-metadata keys stripped on write. ``document_filepath`` is the absolute
#: path the chunker read from — one machine's filesystem, meaningless on any
#: other and a leak of that machine's layout. ``source_file`` and
#: ``relative_path`` already carry the portable identity. Same rule the frozen
#: bundled chunk set follows (``ingest._UNFROZEN_METADATA_KEYS``).
_STRIPPED_METADATA_KEYS = ("document_filepath",)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class DatasetIntegrityError(ValueError):
    """A Dataset folder does not match what its manifest says it holds.

    Raised by the readers instead of returning a Dataset, because a corpus
    whose recorded identity no longer matches its contents is not a smaller or
    older corpus — it is an unknown one, and everything that cites its chunk ids
    would silently cite different text.
    """


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def chunk_inventory_sha256(rows: Iterable[Any]) -> str:
    """The chunk-inventory fingerprint (method ``chunk-inventory-v1``).

    Adopted **exactly** from testset-kit (``testset_kit/core.py``,
    ``chunk_inventory_sha256``), so a Test Set's recorded inventory and a
    Dataset's are the same number: for every chunk, sorted by ``chunk_id``
    (Python string order, i.e. by Unicode code point), the line
    ``<chunk_id>\\t<sha256 of the chunk text, UTF-8, lowercase hex>\\n``; the
    result is the sha256 of those lines concatenated, UTF-8, full lowercase hex.

    ``rows`` may be :class:`~rag_connector.base.ChunkRecord` objects or
    mappings with ``chunk_id`` and ``text`` keys. A missing text hashes as the
    empty string, as testset-kit's does; a Dataset never holds one, because
    :func:`write_dataset_folder` refuses blank text.

    Chunk ids and texts are all it covers. That is deliberate: they are what an
    answer key cites, and a re-chunk that refills the same positional ids with
    different text changes it.
    """
    pairs = []
    for row in rows:
        chunk_id = _row_value(row, "chunk_id")
        text = _row_value(row, "text")
        pairs.append((str(chunk_id), str(text or "")))
    lines = [
        f"{chunk_id}\t{hashlib.sha256(text.encode('utf-8')).hexdigest()}\n"
        for chunk_id, text in sorted(pairs, key=lambda pair: pair[0])
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def dataset_id_for_inventory(inventory_sha256: str) -> str:
    """The Dataset id: ``ds-`` plus the first 16 hex digits of the inventory.

    Content-derived, so the same chunk ids and texts always get the same id,
    wherever and whenever they are written, and a re-chunk gets a new one. It
    names the chunked content only — not the name, the time or the tool.
    """
    if not _SHA256_RE.match(inventory_sha256 or ""):
        raise ValueError(f"not a sha256 hex digest: {inventory_sha256!r}")
    return f"ds-{inventory_sha256[:16]}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_core_sha256(manifest: Mapping[str, Any], chunks_bytes: bytes,
                        sources_bytes: bytes) -> str:
    """The core digest over the three core files.

    ``dataset.json`` carries this value, so it cannot hash its own bytes. It is
    therefore hashed in canonical form — the manifest **without** its
    ``core_sha256`` key, serialised with sorted keys, no whitespace, UTF-8 —
    while the two data files are hashed byte for byte::

        sha256(
          "rag-connector-dataset-core/1\\n"
          "dataset.json\\t"      + sha256(canonical manifest) + "\\n"
          "data/chunks.jsonl\\t" + sha256(file bytes)         + "\\n"
          "data/sources.json\\t" + sha256(file bytes)         + "\\n"
        )

    Every manifest key is inside it, including keys a reader does not know.
    That is what makes the core write-once: adding, changing or removing any
    key after the fact is a mismatch a reader refuses.
    """
    body = {key: value for key, value in manifest.items() if key != "core_sha256"}
    parts = [
        (MANIFEST_FILE, hashlib.sha256(_canonical_json(body)).hexdigest()),
        (CHUNKS_FILE, hashlib.sha256(chunks_bytes).hexdigest()),
        (SOURCES_FILE, hashlib.sha256(sources_bytes).hexdigest()),
    ]
    text = _CORE_HASH_TAG + "".join(f"{name}\t{digest}\n" for name, digest in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The source list
# ---------------------------------------------------------------------------

def source_manifest_from_chunks(chunks: Iterable[ChunkRecord]) -> list[dict[str, str]]:
    """One row per source document, derived from the chunks that cite it.

    Ported from RAGauge (``src/datasets/connect.py``, lines 61-75, 2026-09-24),
    which builds its Dataset source list this way; moving it here gives every
    tool one definition instead of a copy each. The behaviour is unchanged:

    * a source is identified by ``metadata["source_identity"]``, falling back
      to ``doc_id`` and then ``source_file``;
    * the first chunk seen for a source supplies its row;
    * ``source_fingerprint`` comes from chunk metadata and is ``""`` when the
      connector did not report one — **unknown, not a fingerprint of
      nothing**; ``relative_path`` falls back to ``source_file``; ``filename``
      is ``source_file``; ``file_type`` falls back to the suffix of
      ``source_file``;
    * rows are sorted by ``source_identity``.

    The result is what :func:`~rag_connector.fingerprints.corpus_fingerprint`
    digests.
    """
    by_identity: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        identity = meta.get("source_identity") or chunk.doc_id or chunk.source_file
        by_identity.setdefault(identity, {
            "source_identity": identity,
            "source_fingerprint": meta.get("source_fingerprint", ""),
            "relative_path": meta.get("relative_path") or chunk.source_file,
            "filename": chunk.source_file,
            "file_type": meta.get(
                "file_type", Path(chunk.source_file).suffix.lstrip(".")
            ),
        })
    return sorted(by_identity.values(), key=lambda item: item["source_identity"])


def _source_summary(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    file_types: dict[str, int] = {}
    for row in sources:
        kind = str(row.get("file_type") or "")
        file_types[kind] = file_types.get(kind, 0) + 1
    return {
        "source_count": len(sources),
        "file_types": dict(sorted(file_types.items())),
    }


# ---------------------------------------------------------------------------
# Chunking blocks
# ---------------------------------------------------------------------------

def chunker_chunking(*, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    """The ``chunking`` block for a split made by this package's chunker."""
    return {
        "method": CHUNKER_METHOD,
        "chunker_version": CHUNKER_VERSION,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
    }


def exported_chunking(connector_type: str) -> dict[str, Any]:
    """The ``chunking`` block for chunks read out of a connector as they are.

    The connector's system chunked them, by rules this library cannot see, so
    the block says where they came from and claims nothing about how.
    """
    if not connector_type:
        raise ValueError("an exported Dataset must name the connector_type it came from")
    return {"method": CHUNKING_EXPORTED, "connector_type": str(connector_type)}


def _chunking_problems(chunking: Any) -> list[str]:
    if not isinstance(chunking, Mapping):
        return ["chunking must be an object"]
    method = chunking.get("method")
    if not isinstance(method, str) or not method:
        return ["chunking.method must be a non-empty string"]
    problems = []
    if method == CHUNKER_METHOD:
        for key in ("chunker_version",):
            if not isinstance(chunking.get(key), str) or not chunking.get(key):
                problems.append(f"chunking.{key} must be a non-empty string")
        for key in ("chunk_size", "chunk_overlap"):
            value = chunking.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"chunking.{key} must be a non-negative integer")
    elif method == CHUNKING_EXPORTED:
        if not isinstance(chunking.get("connector_type"), str) or not chunking.get(
            "connector_type"
        ):
            problems.append("chunking.connector_type must name the connector")
    return problems


# ---------------------------------------------------------------------------
# The typed result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """A Dataset folder as read — its manifest facts, chunks and sources.

    ``verified`` is True only for a folder written to this spec whose recorded
    ``chunk_inventory_sha256``, ``core_sha256``, ``corpus_fingerprint``,
    ``chunk_count`` and ``dataset_id`` were all recomputed and matched. A
    legacy folder is never verified: it recorded none of those hashes, so
    ``chunk_inventory_sha256`` on it is *computed at read time*, not a recorded
    fact, and ``core_sha256`` is ``None``.
    """

    path: Path
    layout: str
    spec_version: str | None
    dataset_id: str
    name: str
    created_at: str | None
    chunking: Mapping[str, Any] | None
    source_summary: Mapping[str, Any]
    chunk_count: int
    chunk_inventory_sha256: str
    core_sha256: str | None
    corpus_fingerprint: str | None
    chunks: list[ChunkRecord] = field(repr=False)
    sources: list[dict[str, Any]] = field(repr=False)
    manifest: Mapping[str, Any] = field(repr=False)
    verified: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def legacy(self) -> bool:
        return self.layout != LAYOUT_CORE


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _prepared_chunks(chunks: Iterable[ChunkRecord]) -> list[ChunkRecord]:
    """Validate, then copy each record with machine-local metadata removed.

    The caller's records are not mutated: stripping a path out of someone
    else's objects is a side effect nobody asked for.
    """
    records = list(chunks)
    if not records:
        raise ValueError("a Dataset needs at least one chunk")
    bad = [r for r in records if not isinstance(r, ChunkRecord)]
    if bad:
        raise TypeError(
            f"chunks must be ChunkRecord objects, got {type(bad[0]).__name__}"
        )
    RagPipeline.validate_chunks(records)
    seen: set[str] = set()
    out = []
    for record in records:
        if record.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {record.chunk_id!r}")
        seen.add(record.chunk_id)
        if not record.text.strip():
            raise ValueError(f"chunk {record.chunk_id!r} has blank text")
        metadata = {
            key: value for key, value in (record.metadata or {}).items()
            if key not in _STRIPPED_METADATA_KEYS
        }
        out.append(replace(record, metadata=metadata, embedding=None))
    return out


def _sources_bytes(sources: Sequence[Mapping[str, Any]]) -> bytes:
    return (json.dumps(list(sources), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_dataset_folder(
    out_dir: str | os.PathLike[str],
    chunks: Iterable[ChunkRecord],
    *,
    name: str,
    chunking: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]] | None = None,
    created_at: str | None = None,
) -> Dataset:
    """Write a new Dataset's core — once — and return it as read back.

    ``out_dir`` must not exist, or must be an empty directory. Anything else is
    refused rather than overwritten: the core is write-once, and a folder that
    already holds something may hold another Dataset whose ids other artifacts
    cite. The files are written into a temporary sibling directory and moved
    into place only when all three are complete, so a failed write leaves no
    half-Dataset behind.

    ``chunking`` is required — a Dataset that cannot say how it was split
    cannot be reproduced or compared. Use :func:`chunker_chunking` or
    :func:`exported_chunking`, or any mapping with a non-empty ``method``.

    ``sources`` defaults to :func:`source_manifest_from_chunks`. Chunk records
    are validated against the chunk contract, must have unique ids and
    non-blank text, and are written without embeddings and without
    ``metadata["document_filepath"]``.

    The returned :class:`Dataset` is the result of :func:`read_dataset_folder`
    on what was written, so it has been verified end to end.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a Dataset needs a non-empty name")
    problems = _chunking_problems(chunking)
    if problems:
        raise ValueError("invalid chunking block: " + "; ".join(problems))

    target = Path(out_dir)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"{target} exists and is not a directory")
        if any(target.iterdir()):
            raise FileExistsError(
                f"{target} is not empty. A Dataset's core is written once; "
                "choose a new output directory rather than overwriting one."
            )

    records = _prepared_chunks(chunks)
    source_rows = (
        [dict(row) for row in sources] if sources is not None
        else source_manifest_from_chunks(records)
    )
    chunks_bytes = chunks_to_jsonl(records).encode("utf-8")
    sources_bytes = _sources_bytes(source_rows)
    inventory = chunk_inventory_sha256(records)

    manifest: dict[str, Any] = {
        "spec": DATASET_SPEC_ID,
        "spec_version": DATASET_SPEC_VERSION,
        "dataset_id": dataset_id_for_inventory(inventory),
        "name": name,
        "created_at": created_at or _utc_now(),
        "chunking": dict(chunking),
        "sources": _source_summary(source_rows),
        "chunk_count": len(records),
        "chunk_inventory_sha256": inventory,
        "corpus_fingerprint": corpus_fingerprint(source_rows),
    }
    manifest["core_sha256"] = compute_core_sha256(manifest, chunks_bytes, sources_bytes)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-staging-", dir=target.parent))
    try:
        (staging / "data").mkdir()
        (staging / CHUNKS_FILE).write_bytes(chunks_bytes)
        (staging / SOURCES_FILE).write_bytes(sources_bytes)
        (staging / MANIFEST_FILE).write_bytes(manifest_bytes)
        if target.exists():
            target.rmdir()  # empty, checked above
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_dataset_folder(target)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _load_manifest(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{folder} holds no {MANIFEST_FILE}; it is not a Dataset")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DatasetIntegrityError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetIntegrityError(f"{path} must hold a JSON object")
    return manifest


def _is_legacy_v0(manifest: Mapping[str, Any]) -> bool:
    return "spec" not in manifest and isinstance(manifest.get("id"), str)


def read_dataset_folder(path: str | os.PathLike[str], *,
                        allow_legacy: bool = False) -> Dataset:
    """Read a Dataset folder, verify it, and return it typed.

    Every recorded identity is recomputed from the files and must match —
    ``chunk_inventory_sha256`` from the chunk rows, ``core_sha256`` from the
    three core files, ``corpus_fingerprint`` from the source list,
    ``chunk_count`` and ``dataset_id``. Any mismatch raises
    :class:`DatasetIntegrityError`; a Dataset is never returned half-trusted.

    Unknown manifest keys are ignored (they are still inside ``core_sha256``).
    A higher minor ``spec_version`` is read; a different major is refused.
    ``testsets/`` and ``extensions/`` are not read here — see
    :func:`list_extensions`.

    A RAGauge folder written before this spec is refused unless
    ``allow_legacy=True``, which delegates to :func:`read_legacy_dataset_folder`
    and returns an *unverified* Dataset — so a caller that asked for a verified
    read cannot be handed one that was not.
    """
    folder = Path(path)
    manifest = _load_manifest(folder)
    if manifest.get("spec") != DATASET_SPEC_ID:
        if _is_legacy_v0(manifest):
            if allow_legacy:
                return read_legacy_dataset_folder(folder)
            raise DatasetIntegrityError(
                f"{folder} is a legacy RAGauge Dataset (no {DATASET_SPEC_ID!r} "
                "manifest); it records no chunk-inventory or core hash to "
                "verify. Read it with read_legacy_dataset_folder(), or pass "
                "allow_legacy=True, and treat it as unverified."
            )
        raise DatasetIntegrityError(
            f"{folder / MANIFEST_FILE} is not a {DATASET_SPEC_ID!r} manifest "
            f"(spec={manifest.get('spec')!r})"
        )
    version = str(manifest.get("spec_version") or "")
    if version.split(".")[0] != _SUPPORTED_MAJOR:
        raise DatasetIntegrityError(
            f"{folder} is {DATASET_SPEC_ID} version {version!r}; this reader "
            f"understands major version {_SUPPORTED_MAJOR} only"
        )

    problems: list[str] = []
    for key in ("dataset_id", "name", "chunk_inventory_sha256", "core_sha256",
                "corpus_fingerprint"):
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            problems.append(f"missing or empty {key}")
    count = manifest.get("chunk_count")
    if not isinstance(count, int) or isinstance(count, bool):
        problems.append("chunk_count must be an integer")
    problems.extend(_chunking_problems(manifest.get("chunking")))
    if problems:
        raise DatasetIntegrityError(f"{folder / MANIFEST_FILE}: " + "; ".join(problems))

    try:
        chunks_bytes = (folder / CHUNKS_FILE).read_bytes()
        sources_bytes = (folder / SOURCES_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise DatasetIntegrityError(f"{folder} is missing a core file: {exc}") from exc
    try:
        chunks = chunks_from_jsonl(chunks_bytes.decode("utf-8"))
        sources = json.loads(sources_bytes.decode("utf-8"))
    except ValueError as exc:
        raise DatasetIntegrityError(f"{folder}: unreadable core data: {exc}") from exc
    if not isinstance(sources, list) or not all(isinstance(r, dict) for r in sources):
        raise DatasetIntegrityError(
            f"{folder / SOURCES_FILE} must hold a JSON array of objects"
        )

    inventory = chunk_inventory_sha256(chunks)
    mismatches = []
    if inventory != manifest["chunk_inventory_sha256"]:
        mismatches.append(
            f"chunk_inventory_sha256 recorded {manifest['chunk_inventory_sha256']} "
            f"but the chunk rows hash to {inventory}"
        )
    core = compute_core_sha256(manifest, chunks_bytes, sources_bytes)
    if core != manifest["core_sha256"]:
        mismatches.append(
            f"core_sha256 recorded {manifest['core_sha256']} but the core files "
            f"hash to {core}"
        )
    if len(chunks) != count:
        mismatches.append(f"chunk_count recorded {count} but there are {len(chunks)} rows")
    if dataset_id_for_inventory(inventory) != manifest["dataset_id"]:
        mismatches.append(
            f"dataset_id {manifest['dataset_id']!r} is not the id of this chunk "
            f"inventory ({dataset_id_for_inventory(inventory)!r})"
        )
    fingerprint = corpus_fingerprint(sources)
    if fingerprint != manifest["corpus_fingerprint"]:
        mismatches.append(
            f"corpus_fingerprint recorded {manifest['corpus_fingerprint']} but "
            f"the source list hashes to {fingerprint}"
        )
    if mismatches:
        raise DatasetIntegrityError(
            f"{folder} does not match its manifest — its core was changed after "
            "it was written, or was never written by a conforming writer. "
            + "; ".join(mismatches)
        )

    return Dataset(
        path=folder,
        layout=LAYOUT_CORE,
        spec_version=version,
        dataset_id=manifest["dataset_id"],
        name=manifest["name"],
        created_at=manifest.get("created_at"),
        chunking=dict(manifest["chunking"]),
        source_summary=dict(manifest.get("sources") or {}),
        chunk_count=count,
        chunk_inventory_sha256=inventory,
        core_sha256=core,
        corpus_fingerprint=fingerprint,
        chunks=chunks,
        sources=sources,
        manifest=manifest,
        verified=True,
    )


def read_legacy_dataset_folder(path: str | os.PathLike[str]) -> Dataset:
    """Read a RAGauge Dataset folder written before this spec ("v0 layout").

    **Legacy, read-only, never migrated in place.** Those folders have the same
    three files (``dataset.json``, ``data/chunks.jsonl``, ``data/sources.json``)
    and the same chunk row, but their manifest is RAGauge's own — ``id`` rather
    than ``dataset_id``, no spec id, and no recorded inventory or core hash. So
    this reader:

    * writes nothing, ever — not the manifest, not a hash, not a marker;
    * returns ``verified=False`` and ``core_sha256=None``;
    * *computes* ``chunk_inventory_sha256`` from the rows (useful for binding a
      Test Set to it, but a read-time calculation, not a recorded fact);
    * keeps the legacy ``id`` as ``dataset_id`` — it is RAGauge's name for the
      folder, not a content-derived id;
    * reports, rather than refuses, a chunk count or corpus fingerprint that
      disagrees with the rows, in ``warnings``. RAGauge validates those folders
      its own way; this reader's job is to keep them readable.

    To bring one under the spec, write a NEW Dataset from its chunks with
    :func:`write_dataset_folder`; it gets a new, content-derived id.
    """
    folder = Path(path)
    manifest = _load_manifest(folder)
    if not _is_legacy_v0(manifest):
        raise DatasetIntegrityError(
            f"{folder / MANIFEST_FILE} is not a legacy RAGauge manifest"
        )
    try:
        chunks = chunks_from_jsonl((folder / CHUNKS_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetIntegrityError(f"{folder} is missing {CHUNKS_FILE}") from exc
    except ValueError as exc:
        raise DatasetIntegrityError(f"{folder / CHUNKS_FILE}: {exc}") from exc
    sources_path = folder / SOURCES_FILE
    sources = (
        json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.is_file()
        else source_manifest_from_chunks(chunks)
    )

    warnings = [
        "legacy RAGauge v0 layout: no recorded chunk-inventory or core hash; "
        "read unverified"
    ]
    if not sources_path.is_file():
        warnings.append(f"{SOURCES_FILE} is missing; the source list was derived from chunks")
    recorded_count = manifest.get("chunk_count")
    if recorded_count is not None and recorded_count != len(chunks):
        warnings.append(
            f"chunk_count recorded {recorded_count} but there are {len(chunks)} rows"
        )
    recorded_fp = manifest.get("corpus_fingerprint")
    computed_fp = corpus_fingerprint(sources)
    if recorded_fp and recorded_fp != computed_fp:
        warnings.append(
            f"corpus_fingerprint recorded {recorded_fp} but the source list hashes "
            f"to {computed_fp}"
        )

    return Dataset(
        path=folder,
        layout=LEGACY_LAYOUT_RAGAUGE_V0,
        spec_version=None,
        dataset_id=manifest["id"],
        name=str(manifest.get("name") or manifest["id"]),
        created_at=manifest.get("created_at"),
        chunking=None,
        source_summary=_source_summary(sources),
        chunk_count=len(chunks),
        chunk_inventory_sha256=chunk_inventory_sha256(chunks),
        core_sha256=None,
        corpus_fingerprint=recorded_fp or None,
        chunks=chunks,
        sources=sources,
        manifest=manifest,
        verified=False,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtensionEnvelope:
    """One tool's ``extensions/<tool>/extension.json``, as read.

    The envelope is the only part of an extension area this spec defines. What
    else the tool keeps there is the tool's own format. ``problems`` is empty
    for a well-formed envelope; a malformed one is still listed, with its
    problems, rather than skipped — a tool's area that silently disappears from
    a listing is harder to diagnose than one that says what is wrong with it.
    """

    tool: str
    path: Path
    tool_version: str | None
    format_version: str | None
    core_sha256: str | None
    created_at: str | None
    raw: Mapping[str, Any] = field(repr=False)
    problems: tuple[str, ...] = ()

    def built_on(self, dataset: Dataset) -> bool:
        """True when this extension records the core of ``dataset``."""
        return bool(self.core_sha256) and self.core_sha256 == dataset.core_sha256


def _envelope_problems(tool_dir: str, raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["extension.json must hold a JSON object"]
    problems = []
    if raw.get("tool") != tool_dir:
        problems.append(
            f"tool is {raw.get('tool')!r} but the area is extensions/{tool_dir}/"
        )
    for key in ("tool_version", "format_version", "created_at"):
        if not isinstance(raw.get(key), str) or not raw.get(key):
            problems.append(f"{key} must be a non-empty string")
    if not _SHA256_RE.match(str(raw.get("core_sha256") or "")):
        problems.append("core_sha256 must be a sha256 hex digest")
    return problems


def list_extensions(folder: str | os.PathLike[str]) -> list[ExtensionEnvelope]:
    """Every ``extensions/*/extension.json`` in a Dataset folder, sorted by tool.

    This is how extensions are discovered: the manifest never lists them,
    because recording one there would mean rewriting the write-once core every
    time a tool arrived. A directory under ``extensions/`` without an envelope
    is not an extension and is not listed. Nothing inside an area other than
    its envelope is read.
    """
    root = Path(folder) / EXTENSIONS_DIR
    if not root.is_dir():
        return []
    out = []
    for envelope_path in sorted(root.glob(f"*/{EXTENSION_ENVELOPE_FILE}")):
        tool_dir = envelope_path.parent.name
        try:
            raw = json.loads(envelope_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            out.append(ExtensionEnvelope(
                tool=tool_dir, path=envelope_path.parent, tool_version=None,
                format_version=None, core_sha256=None, created_at=None, raw={},
                problems=(f"extension.json is not valid JSON: {exc}",),
            ))
            continue
        data = raw if isinstance(raw, dict) else {}
        out.append(ExtensionEnvelope(
            tool=tool_dir,
            path=envelope_path.parent,
            tool_version=data.get("tool_version"),
            format_version=data.get("format_version"),
            core_sha256=data.get("core_sha256"),
            created_at=data.get("created_at"),
            raw=data,
            problems=tuple(_envelope_problems(tool_dir, raw)),
        ))
    return out


def write_extension_envelope(
    folder: str | os.PathLike[str],
    tool: str,
    *,
    tool_version: str,
    format_version: str,
    created_at: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create ``extensions/<tool>/`` with its ``extension.json`` and return the area.

    The Dataset is read and verified first, and its ``core_sha256`` recorded in
    the envelope, so an extension always says which core it was built on. A
    legacy folder has no core hash and is refused.

    ``tool`` is a lowercase name (``[a-z0-9][a-z0-9._-]*``) and is the
    directory name. An existing envelope is refused unless ``overwrite=True``:
    only the tool that owns the area should replace it, and doing so should be
    a decision rather than a side effect. Nothing else in the area is touched.
    """
    if not _TOOL_NAME_RE.match(tool or ""):
        raise ValueError(
            f"tool name {tool!r} must be lowercase letters, digits, '.', '_' or "
            "'-', starting with a letter or digit"
        )
    for key, value in (("tool_version", tool_version), ("format_version", format_version)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
    dataset = read_dataset_folder(folder)
    area = Path(folder) / EXTENSIONS_DIR / tool
    envelope_path = area / EXTENSION_ENVELOPE_FILE
    if envelope_path.exists() and not overwrite:
        raise FileExistsError(
            f"{envelope_path} already exists; pass overwrite=True to replace "
            "your own tool's envelope"
        )
    envelope = {
        "tool": tool,
        "tool_version": tool_version,
        "format_version": format_version,
        "core_sha256": dataset.core_sha256,
        "created_at": created_at or _utc_now(),
    }
    area.mkdir(parents=True, exist_ok=True)
    envelope_path.write_bytes(
        (json.dumps(envelope, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return area
