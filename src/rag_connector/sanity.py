"""Lightweight Dataset checks with explicit, independently optional evidence.

No full-corpus fallback, persisted audit, or claim of full synchronization.
Backend failures propagate; unsupported operations are reported separately.
"""

from __future__ import annotations

import math
import random
import secrets
from collections import Counter
from datetime import datetime

from .base import ChunkRecord, RagPipeline
from .capabilities import supports
from .dataset import Dataset
from .errors import ConnectorContractError, UnsupportedCapability
from .models import ChunkInventoryEntry, ChunkInventoryPage, SnapshotValidity

__all__ = ["check_dataset"]


def _inventory(pipeline: RagPipeline) -> dict[str, ChunkInventoryEntry]:
    entries: dict[str, ChunkInventoryEntry] = {}
    cursor = None
    cursors: set[str] = set()
    while True:
        page = pipeline.list_chunk_inventory(cursor=cursor, limit=1000)
        if not isinstance(page, ChunkInventoryPage):
            raise ConnectorContractError("Expected ChunkInventoryPage")
        for item in page.items:
            if (not isinstance(item, ChunkInventoryEntry)
                    or not isinstance(item.chunk_id, str) or not item.chunk_id.strip()
                    or not isinstance(item.doc_id, str) or not item.doc_id.strip()
                    or type(item.doc_chunk_index) is not int or item.doc_chunk_index < 0):
                raise ConnectorContractError("Invalid chunk inventory entry")
            if item.chunk_id in entries:
                raise ConnectorContractError(f"Duplicate inventory id: {item.chunk_id}")
            entries[item.chunk_id] = item
        cursor = page.next_cursor
        if cursor is None:
            return entries
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise ConnectorContractError("Invalid or repeated inventory cursor")
        cursors.add(cursor)


def check_dataset(
    dataset: Dataset, pipeline: RagPipeline, *, sample_percent: float = 0,
    seed: int | None = None,
    snapshot_at: str | None = None,
) -> dict:
    """Compare count, full structure, and optionally sampled text.

    Percent is 0..100, rounded up to whole chunks. Sample uniformly without
    replacement from the frozen Dataset, not similarity-query results. Record
    the seed and IDs for replay. Sampling requires an overridden ``get_chunks``
    that fetches by ID without scanning the corpus; the base fallback is never
    called. A successful sample says nothing definitive about untested text,
    other metadata, vectors, or retrieval policy. The returned dict is JSON-safe.
    """
    if (isinstance(sample_percent, bool) or not isinstance(sample_percent, (int, float))
            or not math.isfinite(sample_percent) or not 0 <= sample_percent <= 100):
        raise ValueError("sample_percent must be a finite percentage from 0 to 100")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    if snapshot_at is not None:
        try:
            instant = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("snapshot_at must be an ISO 8601 timestamp with timezone") from exc
        if instant.utcoffset() is None:
            raise ValueError("snapshot_at must include a timezone")
    result = {
        "format_version": "1", "mode": "lite",
        "dataset": {"dataset_id": dataset.dataset_id,
                    "core_sha256": dataset.core_sha256, "chunk_count": dataset.chunk_count},
    }
    result["snapshot"] = {"status": "not_requested"}
    if snapshot_at is not None:
        snapshot = {"snapshot_at": snapshot_at}
        result["snapshot"] = snapshot
        try:
            if not supports(pipeline, "check_snapshot"):
                raise UnsupportedCapability("check_snapshot is not implemented")
            validity = pipeline.check_snapshot(snapshot_at)
        except UnsupportedCapability as exc:
            snapshot.update(status="unsupported", reason=str(exc))
        else:
            if not isinstance(validity, SnapshotValidity):
                raise ConnectorContractError("check_snapshot must return SnapshotValidity")
            snapshot.update(status=validity.status, detail=validity.detail)
    backend_count = None
    try:
        if not supports(pipeline, "count_chunks"):
            raise UnsupportedCapability("count_chunks is not implemented")
        backend_count = pipeline.count_chunks()
    except UnsupportedCapability as exc:
        result["count"] = {"status": "unsupported", "reason": str(exc)}
    else:
        if type(backend_count) is not int or backend_count < 0:
            raise ConnectorContractError("count_chunks must return a nonnegative integer")
        result["count"] = {
            "status": "checked", "backend_count": backend_count,
            "matches": backend_count == dataset.chunk_count,
        }
    source = {c.chunk_id: c for c in dataset.chunks}
    try:
        if not supports(pipeline, "list_chunk_inventory"):
            raise UnsupportedCapability("list_chunk_inventory is not implemented")
        backend = _inventory(pipeline)
    except UnsupportedCapability as exc:
        result["structure"] = {"status": "unsupported", "reason": str(exc)}
    else:
        missing = sorted(source.keys() - backend.keys())
        added = sorted(backend.keys() - source.keys())
        changed = sorted(cid for cid in source.keys() & backend.keys()
                         if (source[cid].doc_id, source[cid].doc_chunk_index)
                         != (backend[cid].doc_id, backend[cid].doc_chunk_index))
        result["structure"] = {
            "status": "checked", "matches": not (missing or added or changed),
            "backend_count": len(backend), "missing_ids": missing, "added_ids": added,
            "changed_ids": changed,
            "dataset_document_counts": dict(sorted(
                Counter(c.doc_id for c in source.values()).items())),
            "backend_document_counts": dict(sorted(
                Counter(c.doc_id for c in backend.values()).items())),
            "count_matches_inventory": (None if backend_count is None
                                        else backend_count == len(backend)),
        }
    sample = {"requested_percent": sample_percent, "population_count": len(source)}
    result["text_sample"] = sample
    if sample_percent == 0:
        sample.update(status="not_requested", sampled_count=0)
        return result
    seed = secrets.randbits(64) if seed is None else seed
    n = min(len(source), math.ceil(len(source) * sample_percent / 100))
    ids = random.Random(seed).sample(sorted(source), n)
    sample.update(seed=seed, selected_ids=ids, selected_count=n)
    if not supports(pipeline, "get_chunks"):
        sample.update(status="unsupported", reason="Requires a direct get_chunks implementation")
        return result
    fetched = {}
    try:
        for offset in range(0, len(ids), 100):
            batch_ids = ids[offset:offset + 100]
            batch = pipeline.get_chunks(batch_ids)
            if not isinstance(batch, dict) or batch.keys() - set(batch_ids):
                raise ConnectorContractError("get_chunks returned unexpected IDs")
            for cid, chunk in batch.items():
                if (not isinstance(chunk, ChunkRecord) or chunk.chunk_id != cid
                        or not isinstance(chunk.text, str)):
                    raise ConnectorContractError("Invalid get_chunks record")
            fetched.update(batch)
    except UnsupportedCapability as exc:
        sample.update(status="unsupported", reason=str(exc))
        return result
    missing = sorted(set(ids) - fetched.keys())
    changed = sorted(cid for cid in fetched if fetched[cid].text != source[cid].text)
    sample.update(status="checked", sampled_count=len(fetched), missing_ids=missing,
                  changed_ids=changed, matches=not (missing or changed))
    return result
