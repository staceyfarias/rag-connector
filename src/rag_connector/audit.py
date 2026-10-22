"""Dataset audits: compare a frozen Dataset with a connector's corpus as it is now.

A Dataset is write-once. The system it was taken from, or attached to, is not:
an index can be rebuilt (new chunk ids for the same text), a corpus re-ingested
with other chunking settings or metadata fields, and documents added or
removed. An **audit** reads the connector's whole corpus once and records,
chunk by chunk and document by document, how it differs from the Dataset.

An audit changes nothing. It does not touch the core, does not edit an
attachment, and does not repair anything; it is evidence a person or an
application uses to decide what to do — reattach, take a new Dataset as a copy
(a re-chunk is always a new Dataset, spec rule 3), or nothing. A Dataset may
hold any number of audits, from any application and any connector instance,
each in its own directory under ``.rag-connector/audits/``.

Matching follows the order the attachment design fixes, and is **never
positional**:

1. the same chunk id, then
2. the exact text (sha256), only when exactly one unmatched backend chunk and
   exactly one unmatched Dataset chunk carry it.

Everything else is reported, not guessed: a text shared by several candidates
is ``ambiguous``, never assigned to one of them.

See ``docs/dataset-spec.md`` ("Audits").
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import ChunkRecord, RagPipeline
from .dataset import (
    _STRIPPED_METADATA_KEYS,
    Dataset,
    chunk_inventory_sha256,
    read_dataset_folder,
)
from .errors import ConnectorContractError

__all__ = [
    "AUDIT_FORMAT_VERSION",
    "AUDITS_DIR",
    "CHUNK_STATUSES",
    "DOCUMENT_STATUSES",
    "DatasetAudit",
    "audit_dataset",
    "connection_sha256",
    "list_audits",
    "read_audit",
    "record_audit",
]

#: The format of what :func:`record_audit` writes. Bumped on any change a
#: reader of an older audit must know about.
AUDIT_FORMAT_VERSION = "1"

#: This package's private area in a Dataset folder (spec: "Tool-private areas").
PRIVATE_AREA = ".rag-connector"
AUDITS_DIR = f"{PRIVATE_AREA}/audits"
_TOOL = "rag-connector"

#: One status per Dataset chunk, plus ``added`` for backend chunks no Dataset
#: chunk matched.
CHUNK_STATUSES = (
    "unchanged",       # same id, same text (metadata may still differ)
    "text_changed",    # same id, different text
    "id_changed",      # same text, under a different id
    "ambiguous",       # same text as several unmatched chunks; not assigned
    "missing",         # neither the id nor the text is in the backend
    "added",           # a backend chunk that matched no Dataset chunk
)

DOCUMENT_STATUSES = (
    "unchanged",  # every chunk unchanged, nothing added under its doc_id
    "changed",    # anything else, while the document is still present
    "removed",    # no chunk found, and no backend chunk has its doc_id
    "added",      # a backend doc_id with no Dataset chunk behind it
)

#: Chunk fields compared besides ``text`` and ``metadata``. ``global_index`` is
#: left out: it is a host-assigned ordinal (legally ``None`` from a connector),
#: so a difference says nothing about the backend.
_COMPARED_FIELDS = (
    "doc_id", "source_file", "doc_chunk_index", "char_start", "char_end",
    "content_module_id",
)

#: The longest overlap looked for between neighbouring chunks, in characters.
_MAX_OVERLAP_SCAN = 4000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def connection_sha256(connection: Mapping[str, Any]) -> str:
    """Identify one connector instance's configuration, without storing it.

    sha256 over the connection document as canonical JSON (keys sorted, no
    whitespace). Two applications that configured their own instances of the
    same connector type get different values whenever their configurations
    differ, which is what lets an audit say *which* instance it describes.
    The document itself is not recorded: it is the host's, and naming an
    endpoint is not the Dataset's business.
    """
    payload = json.dumps(
        dict(connection), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _sha(payload)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetAudit:
    """One audit: what the connector held, compared with the Dataset.

    ``summary`` is the JSON written to ``audit.json``; ``chunks`` is one row
    per Dataset chunk plus one per ``added`` backend chunk, written to
    ``chunks.jsonl``. ``path`` is set once the audit is recorded.
    """

    audit_id: str
    summary: Mapping[str, Any]
    chunks: list[dict[str, Any]] = field(repr=False)
    path: Path | None = None

    @property
    def in_sync(self) -> bool:
        return bool(self.summary.get("in_sync"))

    @property
    def signals(self) -> list[Mapping[str, Any]]:
        return list(self.summary.get("signals", []))


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

def _metadata(record: ChunkRecord) -> dict[str, Any]:
    # The Dataset writer strips machine-local keys; strip the same keys from the
    # backend side, or every chunk would report a metadata change.
    return {
        key: value for key, value in (record.metadata or {}).items()
        if key not in _STRIPPED_METADATA_KEYS
    }


def _differences(old: ChunkRecord, new: ChunkRecord) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = [name for name in _COMPARED_FIELDS
              if getattr(old, name) != getattr(new, name)]
    if fields:
        out["fields_changed"] = fields
    old_meta, new_meta = _metadata(old), _metadata(new)
    added = sorted(set(new_meta) - set(old_meta))
    removed = sorted(set(old_meta) - set(new_meta))
    changed = sorted(k for k in set(old_meta) & set(new_meta)
                     if old_meta[k] != new_meta[k])
    if added:
        out["metadata_keys_added"] = added
    if removed:
        out["metadata_keys_removed"] = removed
    if changed:
        out["metadata_keys_changed"] = changed
    return out


def _overlap(previous: str, following: str) -> int:
    """Longest suffix of ``previous`` that is a prefix of ``following``.

    Prefix function over ``following + sep + tail(previous)``: linear, not the
    quadratic try-every-length loop.
    """
    tail = previous[-_MAX_OVERLAP_SCAN:]
    head = following[:_MAX_OVERLAP_SCAN]
    s = head + "\x00" + tail
    pi = [0] * len(s)
    for i in range(1, len(s)):
        k = pi[i - 1]
        while k and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi[-1]


def _chunking_profile(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """What the chunks show about how they were split — observed, not declared.

    Lengths are in characters. ``overlap`` is the exact repeated text between
    neighbouring chunks of one document (ordered by ``doc_chunk_index``); a
    chunker that trims or re-joins at boundaries shows less overlap than it was
    configured with, so read it as a comparison between two sides, never as a
    recovered setting.
    """
    if not chunks:
        return {"chunk_count": 0, "document_count": 0}
    lengths = sorted(len(c.text) for c in chunks)
    by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    overlaps = []
    for doc_chunks in by_doc.values():
        ordered = sorted(doc_chunks, key=lambda c: c.doc_chunk_index)
        for prev, nxt in zip(ordered, ordered[1:]):
            overlaps.append(_overlap(prev.text, nxt.text))
    meta_keys = Counter(k for c in chunks for k in _metadata(c))
    profile: dict[str, Any] = {
        "chunk_count": len(chunks),
        "document_count": len(by_doc),
        "chars": {
            "min": lengths[0],
            "median": statistics.median(lengths),
            "mean": round(statistics.fmean(lengths), 1),
            "p90": lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))],
            "max": lengths[-1],
        },
        "chunks_per_document_median": statistics.median(
            len(v) for v in by_doc.values()
        ),
        "metadata_keys": dict(sorted(meta_keys.items())),
    }
    if overlaps:
        profile["neighbour_overlap_chars"] = {
            "pairs": len(overlaps),
            "pairs_with_overlap": sum(1 for o in overlaps if o > 0),
            "median": statistics.median(overlaps),
            "max": max(overlaps),
        }
    return profile


def _match(
    dataset_chunks: Sequence[ChunkRecord], backend_chunks: Sequence[ChunkRecord]
) -> tuple[list[dict[str, Any]], dict[str, ChunkRecord]]:
    """One row per Dataset chunk, plus ``added`` rows; and the claimed backend ids."""
    backend_by_id = {c.chunk_id: c for c in backend_chunks}
    rows: dict[str, dict[str, Any]] = {}
    claimed: dict[str, ChunkRecord] = {}  # backend id -> backend chunk

    # 1. By id.
    unmatched = []
    for d in dataset_chunks:
        b = backend_by_id.get(d.chunk_id)
        if b is None:
            unmatched.append(d)
            continue
        claimed[b.chunk_id] = b
        row: dict[str, Any] = {"chunk_id": d.chunk_id, "doc_id": d.doc_id,
                               "backend_chunk_id": b.chunk_id, "match": "id"}
        if _sha(d.text) == _sha(b.text):
            row["status"] = "unchanged"
        else:
            row["status"] = "text_changed"
            row["text_sha256"] = _sha(d.text)
            row["backend_text_sha256"] = _sha(b.text)
        row.update(_differences(d, b))
        rows[d.chunk_id] = row

    # 2. By exact text, one-to-one only.
    free_by_text: dict[str, list[ChunkRecord]] = defaultdict(list)
    for b in backend_chunks:
        if b.chunk_id not in claimed:
            free_by_text[_sha(b.text)].append(b)
    unmatched_by_text: dict[str, list[ChunkRecord]] = defaultdict(list)
    for d in unmatched:
        unmatched_by_text[_sha(d.text)].append(d)
    for text_sha, ds in unmatched_by_text.items():
        candidates = free_by_text.get(text_sha, [])
        for d in ds:
            row = {"chunk_id": d.chunk_id, "doc_id": d.doc_id}
            if len(ds) == 1 and len(candidates) == 1:
                b = candidates[0]
                claimed[b.chunk_id] = b
                row.update({"status": "id_changed", "backend_chunk_id": b.chunk_id,
                            "match": "text"})
                row.update(_differences(d, b))
            elif candidates:
                row.update({
                    "status": "ambiguous",
                    "candidate_backend_chunk_ids": sorted(c.chunk_id for c in candidates),
                })
            else:
                row["status"] = "missing"
            rows[d.chunk_id] = row
        if candidates and not (len(ds) == 1 and len(candidates) == 1):
            # Ambiguous candidates are not "added": they hold a Dataset text.
            for b in candidates:
                claimed.setdefault(b.chunk_id, b)

    ordered = [rows[d.chunk_id] for d in dataset_chunks]
    for b in backend_chunks:
        if b.chunk_id not in claimed:
            ordered.append({"chunk_id": None, "doc_id": b.doc_id,
                            "backend_chunk_id": b.chunk_id, "status": "added",
                            "backend_text_sha256": _sha(b.text)})
    return ordered, claimed


def _documents(
    dataset_chunks: Sequence[ChunkRecord],
    backend_chunks: Sequence[ChunkRecord],
    rows: Sequence[Mapping[str, Any]],
    claimed: Mapping[str, ChunkRecord],
) -> list[dict[str, Any]]:
    ds_count = Counter(c.doc_id for c in dataset_chunks)
    be_count = Counter(c.doc_id for c in backend_chunks)
    statuses: dict[str, Counter] = defaultdict(Counter)
    added_by_doc: Counter = Counter()
    matched_backend_docs: set[str] = {c.doc_id for c in claimed.values()}
    for row in rows:
        if row["status"] == "added":
            added_by_doc[row["doc_id"]] += 1
        else:
            statuses[row["doc_id"]][row["status"]] += 1
    out = []
    for doc_id in sorted(ds_count):
        counts = statuses[doc_id]
        found = ds_count[doc_id] - counts["missing"]
        if found == 0 and be_count[doc_id] == 0:
            status = "removed"
        elif (counts["unchanged"] == ds_count[doc_id]
              and added_by_doc[doc_id] == 0):
            status = "unchanged"
        else:
            status = "changed"
        out.append({
            "doc_id": doc_id, "status": status,
            "dataset_chunks": ds_count[doc_id],
            "backend_chunks": be_count[doc_id],
            "chunk_statuses": dict(sorted(counts.items())),
            "added_chunks": added_by_doc[doc_id],
        })
    for doc_id in sorted(set(be_count) - set(ds_count)):
        if doc_id in matched_backend_docs:
            continue  # holds Dataset text under a new doc_id: not a new document
        out.append({
            "doc_id": doc_id, "status": "added", "dataset_chunks": 0,
            "backend_chunks": be_count[doc_id], "chunk_statuses": {},
            "added_chunks": added_by_doc[doc_id],
        })
    # A backend doc_id that also carries matched Dataset text, and is new.
    for doc_id in sorted((set(be_count) - set(ds_count)) & matched_backend_docs):
        out.append({
            "doc_id": doc_id, "status": "changed", "dataset_chunks": 0,
            "backend_chunks": be_count[doc_id], "chunk_statuses": {},
            "added_chunks": added_by_doc[doc_id],
            "note": "a doc_id the Dataset does not have, holding Dataset text",
        })
    return out


def _signals(
    chunk_counts: Mapping[str, int],
    doc_counts: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
    dataset_profile: Mapping[str, Any],
    backend_profile: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Named observations, each with its count. Facts, never a verdict."""
    out: list[dict[str, Any]] = []

    def add(code: str, count: int, summary: str) -> None:
        if count:
            out.append({"code": code, "count": count, "summary": summary})

    add("chunk_ids_changed", chunk_counts.get("id_changed", 0),
        "The backend holds Dataset text under different chunk ids, as an index "
        "rebuild does. Any id mapping made before is stale for these chunks: "
        "reattach the Dataset to this connector instance.")
    add("chunk_text_changed", chunk_counts.get("text_changed", 0),
        "Chunk ids the Dataset has now hold different text in the backend.")
    add("chunks_missing", chunk_counts.get("missing", 0),
        "Dataset chunks whose id and text are both absent from the backend.")
    add("chunks_added", chunk_counts.get("added", 0),
        "Backend chunks that match no Dataset chunk.")
    add("chunks_ambiguous", chunk_counts.get("ambiguous", 0),
        "Dataset chunks whose text matches several backend chunks; not assigned.")
    add("documents_removed", doc_counts.get("removed", 0),
        "Documents with no chunk left in the backend, by id, text or doc_id.")
    add("documents_added", doc_counts.get("added", 0),
        "Backend documents with no Dataset chunk behind them.")
    rechunked = sum(
        1 for d in documents
        if d["status"] == "changed" and d["dataset_chunks"]
        and d["backend_chunks"] and d["dataset_chunks"] != d["backend_chunks"]
    )
    add("documents_chunk_count_changed", rechunked,
        "Documents present on both sides with a different number of chunks, as a "
        "re-chunk with other size or overlap settings does. Compare the two "
        "chunking profiles.")
    ds_keys = set(dataset_profile.get("metadata_keys", {}))
    be_keys = set(backend_profile.get("metadata_keys", {}))
    add("metadata_keys_added", len(be_keys - ds_keys),
        "Metadata keys the backend's chunks carry that the Dataset's do not: "
        + ", ".join(sorted(be_keys - ds_keys)))
    add("metadata_keys_removed", len(ds_keys - be_keys),
        "Metadata keys the Dataset's chunks carry that the backend's do not: "
        + ", ".join(sorted(ds_keys - be_keys)))
    add("chunk_metadata_changed",
        sum(1 for r in rows if r.get("metadata_keys_added")
            or r.get("metadata_keys_removed") or r.get("metadata_keys_changed")),
        "Matched chunks whose metadata differs.")
    add("chunk_fields_changed",
        sum(1 for r in rows if r.get("fields_changed")),
        "Matched chunks whose doc_id, source_file, position or module differs.")
    return out


def audit_dataset(
    dataset: Dataset,
    pipeline: RagPipeline,
    *,
    connector_type: str,
    connection: Mapping[str, Any] | None = None,
    instance_label: str | None = None,
    host: str | None = None,
    created_at: str | None = None,
) -> DatasetAudit:
    """Compare ``dataset`` with ``pipeline``'s whole corpus, as it is now.

    Reads the corpus once through ``pull_all_chunks`` (which pages through
    ``list_chunks`` where that is what the connector implements). A connector
    that cannot read its corpus raises
    :class:`~rag_connector.errors.UnsupportedCapability` and a backend failure
    raises: neither is ever reported as "every chunk missing". An empty corpus
    read *is* reported — the contract makes it a fact, not an error.

    Identify the connector instance the audit describes: ``connector_type``
    always; ``connection`` (the instance's connection document, recorded only
    as :func:`connection_sha256`); ``instance_label`` and ``host`` as the
    application names them ("staging", "pelorus"). Two instances of one
    connector type are different instances, even over the same backend.

    Writes nothing. :func:`record_audit` stores the result in the Dataset.
    """
    if not isinstance(connector_type, str) or not connector_type:
        raise ValueError("connector_type must name the connector")
    backend = list(pipeline.pull_all_chunks())
    RagPipeline.validate_chunks(backend)
    seen: set[str] = set()
    duplicates = sorted({c.chunk_id for c in backend
                         if c.chunk_id in seen or seen.add(c.chunk_id)})
    if duplicates:
        # Two chunks under one id is a broken corpus read, not a backend state:
        # no comparison against it means anything.
        raise ConnectorContractError(
            f"{type(pipeline).__name__}'s corpus read returned duplicate chunk "
            f"ids ({len(duplicates)}, e.g. {duplicates[:3]}); cannot audit."
        )

    rows, claimed = _match(dataset.chunks, backend)
    documents = _documents(dataset.chunks, backend, rows, claimed)
    chunk_counts = Counter(r["status"] for r in rows)
    doc_counts = Counter(d["status"] for d in documents)
    dataset_profile = _chunking_profile(dataset.chunks)
    backend_profile = _chunking_profile(backend)
    signals = _signals(chunk_counts, doc_counts, rows, dataset_profile,
                       backend_profile, documents)
    backend_inventory = chunk_inventory_sha256(backend) if backend else None
    when = created_at or _utc_now()
    audit_id = "audit-" + when.replace("-", "").replace(":", "") + "-" + secrets.token_hex(4)

    summary: dict[str, Any] = {
        "audit_id": audit_id,
        "format_version": AUDIT_FORMAT_VERSION,
        "created_at": when,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "core_sha256": dataset.core_sha256,
            "chunk_inventory_sha256": dataset.chunk_inventory_sha256,
            "chunk_count": dataset.chunk_count,
            "chunking": dict(dataset.chunking or {}),
            "verified": dataset.verified,
        },
        "connector": {
            "connector_type": connector_type,
            "connection_sha256": (connection_sha256(connection)
                                  if connection is not None else None),
            "instance_label": instance_label,
            "host": host,
            "retrieval_mode": getattr(pipeline, "retrieval_mode", None),
        },
        "backend": {
            "chunk_count": len(backend),
            "chunk_inventory_sha256": backend_inventory,
        },
        # Same ids and same texts: the inventories are the same definition.
        "in_sync": (backend_inventory == dataset.chunk_inventory_sha256
                    and not any(s["code"] in ("chunk_metadata_changed",
                                              "chunk_fields_changed")
                                for s in signals)),
        "chunk_counts": {s: chunk_counts.get(s, 0) for s in CHUNK_STATUSES},
        "document_counts": {s: doc_counts.get(s, 0) for s in DOCUMENT_STATUSES},
        "signals": signals,
        "chunking_profiles": {"dataset": dataset_profile, "backend": backend_profile},
        "documents": documents,
    }
    return DatasetAudit(audit_id=audit_id, summary=summary, chunks=rows)


# ---------------------------------------------------------------------------
# Recording and reading
# ---------------------------------------------------------------------------

def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def _ensure_private_area(folder: Path, dataset: Dataset) -> None:
    from importlib.metadata import version

    envelope_path = folder / PRIVATE_AREA / "extension.json"
    if envelope_path.exists():
        return
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(envelope_path, {
        "tool": _TOOL,
        "tool_version": version("rag-connector"),
        "format_version": AUDIT_FORMAT_VERSION,
        "core_sha256": dataset.core_sha256,
        "created_at": _utc_now(),
    })


def record_audit(folder: str | os.PathLike[str], audit: DatasetAudit) -> DatasetAudit:
    """Store ``audit`` in the Dataset at ``folder`` and return it with its path.

    Writes ``.rag-connector/audits/<audit_id>/audit.json`` and ``chunks.jsonl``,
    and nothing else (the area's ``extension.json`` is created on first use).
    The Dataset is re-read and verified first, and must be the one audited:
    an audit recorded beside another core would describe a corpus the folder
    does not hold. A legacy folder has no core hash and is refused. An existing
    audit directory is never overwritten.
    """
    target = Path(folder)
    dataset = read_dataset_folder(target)
    recorded = audit.summary["dataset"]
    if dataset.core_sha256 != recorded.get("core_sha256"):
        raise ValueError(
            f"{target} holds core {dataset.core_sha256}, but the audit was made "
            f"against {recorded.get('core_sha256')}; refusing to record it here."
        )
    _ensure_private_area(target, dataset)
    area = target / AUDITS_DIR / audit.audit_id
    area.mkdir(parents=True, exist_ok=False)
    _write_json(area / "audit.json", audit.summary)
    (area / "chunks.jsonl").write_bytes("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in audit.chunks
    ).encode("utf-8"))
    return DatasetAudit(audit_id=audit.audit_id, summary=audit.summary,
                        chunks=audit.chunks, path=area)


def read_audit(path: str | os.PathLike[str]) -> DatasetAudit:
    """Read one recorded audit directory."""
    area = Path(path)
    summary = json.loads((area / "audit.json").read_text(encoding="utf-8"))
    chunks_path = area / "chunks.jsonl"
    rows = [json.loads(line) for line in
            chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return DatasetAudit(audit_id=summary["audit_id"], summary=summary,
                        chunks=rows, path=area)


def list_audits(folder: str | os.PathLike[str]) -> list[Mapping[str, Any]]:
    """Every recorded audit's ``audit.json``, oldest first. Reads nothing else.

    Each entry gains ``path``. An unreadable ``audit.json`` is listed with a
    ``problem`` rather than skipped.
    """
    root = Path(folder) / AUDITS_DIR
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/audit.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("audit.json must hold a JSON object")
        except ValueError as exc:
            out.append({"audit_id": summary_path.parent.name,
                        "path": str(summary_path.parent), "problem": str(exc)})
            continue
        summary["path"] = str(summary_path.parent)
        out.append(summary)
    out.sort(key=lambda s: (str(s.get("created_at") or ""), s["audit_id"]))
    return out

