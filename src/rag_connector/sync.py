"""Sync into a copy: a new Dataset from a backend's current corpus, with lineage.

A Dataset's core is write-once, and Test Sets cite its chunk ids, so a Dataset
is never updated to follow a backend that has moved on (spec rule 9). A
**sync** takes a *new* Dataset from one connector instance's corpus as it is
now, and records where it came from. The source Dataset is read and verified,
never written: not its core, not its private area.

Lineage is recorded in two places, split by what kind of fact it is:

- ``derived_from`` in the new core's manifest — the source ``dataset_id`` and
  ``core_sha256``. A fact about two Datasets, true for as long as both exist,
  so it is written once with the core, covered by ``core_sha256``, and travels
  with every shared copy.
- ``.rag-connector/syncs/<sync-id>/sync.json`` in the new Dataset — the
  connector instance (``connection_sha256``) and the audit that motivated the
  copy. Those describe one backend (rule 9), and the audit lives in the
  source's private area, so this record is private and does not travel.

Migrating the source's Test Sets to the new chunk ids is a separate decision,
and nothing here makes it.

See ``docs/dataset-spec.md`` ("Syncing into a copy").
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import (
    AUDITS_DIR,
    PRIVATE_AREA,
    _ensure_private_area,
    _utc_now,
    _write_json,
    connection_sha256,
)
from .base import RagPipeline
from .dataset import Dataset, export_dataset_from_connector, read_dataset_folder

__all__ = [
    "SYNCS_DIR",
    "SYNC_FORMAT_VERSION",
    "DatasetSync",
    "list_syncs",
    "sync_dataset",
]

#: The format of ``sync.json``. Bumped on any change a reader must know about.
SYNC_FORMAT_VERSION = "1"

SYNCS_DIR = f"{PRIVATE_AREA}/syncs"

#: ``audit-<UTC time>-<8 hex>``, as :func:`rag_connector.audit.audit_dataset`
#: makes them. Checked before the id is used as a path.
_AUDIT_ID_RE = re.compile(r"^audit-[0-9A-Za-z.+]+-[0-9a-f]{8}$")


@dataclass(frozen=True)
class DatasetSync:
    """The new Dataset a sync wrote, and its ``sync.json`` record."""

    dataset: Dataset
    record: Mapping[str, Any]
    path: Path


def _source_audit(source: Dataset, audit_id: str, *, connector_type: str,
                  connection: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read ``audit_id`` from the source and check it describes this sync.

    Refused unless the audit was made against the source's core, by the same
    connector type, for the same connector instance: an audit of another
    instance says nothing about this one, even over identical chunks (rule 9).
    """
    if not _AUDIT_ID_RE.match(audit_id or ""):
        raise ValueError(f"{audit_id!r} is not an audit id")
    path = source.path / AUDITS_DIR / audit_id / "audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"{source.path} records no audit {audit_id!r}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    recorded_core = (summary.get("dataset") or {}).get("core_sha256")
    if recorded_core != source.core_sha256:
        raise ValueError(
            f"audit {audit_id} was made against core {recorded_core}, not the "
            f"source's {source.core_sha256}; refusing to cite it"
        )
    instance = summary.get("connector") or {}
    if instance.get("connector_type") != connector_type:
        raise ValueError(
            f"audit {audit_id} describes a {instance.get('connector_type')!r} "
            f"connector, not {connector_type!r}; refusing to cite it"
        )
    if connection is None or not instance.get("connection_sha256"):
        raise ValueError(
            f"audit {audit_id} can be cited only when both it and this sync "
            "identify the connector instance by its connection; one does not"
        )
    if instance["connection_sha256"] != connection_sha256(connection):
        raise ValueError(
            f"audit {audit_id} was made for another connector instance "
            "(connection_sha256 differs); an audit is never shared across "
            "instances"
        )
    return summary


def sync_dataset(
    source: str | os.PathLike[str],
    pipeline: RagPipeline,
    out_dir: str | os.PathLike[str],
    *,
    connector_type: str,
    connection: Mapping[str, Any] | None = None,
    audit_id: str | None = None,
    instance_label: str | None = None,
    host: str | None = None,
    name: str | None = None,
    created_at: str | None = None,
) -> DatasetSync:
    """Write a new Dataset at ``out_dir`` from ``pipeline``'s corpus, as it is now.

    ``source`` is the Dataset folder being followed. It is read and verified
    (a legacy folder records no core hash to cite, and is refused) and is
    never written. The new Dataset is :func:`export_dataset_from_connector`'s,
    with ``derived_from`` naming the source; ``name`` defaults to the source's.

    ``audit_id``, when given, names an audit recorded in the source that
    motivated the copy. It must have been made against the source's core for
    the same connector instance — ``connection`` is then required — or the sync
    is refused. Whether the backend's chunk inventory is still the one the
    audit saw is recorded as ``backend_matches_audit``: the audit is cited as
    the reason, and that flag says whether it still describes the copy taken.

    ``out_dir`` must not exist or be empty. The core and the sync record are
    written into a temporary sibling and moved into place together, so a
    failure (a backend error, an empty corpus) leaves nothing behind.
    """
    if not isinstance(connector_type, str) or not connector_type:
        raise ValueError("connector_type must name the connector")
    source_dataset = read_dataset_folder(source)
    audit = (_source_audit(source_dataset, audit_id, connector_type=connector_type,
                           connection=connection)
             if audit_id is not None else None)

    target = Path(out_dir)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"{target} exists and is not a directory")
        if any(target.iterdir()):
            raise FileExistsError(
                f"{target} is not empty. A sync writes a new Dataset; choose a "
                "new output directory."
            )
    when = created_at or _utc_now()
    sync_id = "sync-" + when.replace("-", "").replace(":", "") + "-" + secrets.token_hex(4)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-sync-", dir=target.parent))
    try:
        written = export_dataset_from_connector(
            pipeline, staging, connector_type=connector_type,
            name=name or source_dataset.name, created_at=when,
            derived_from=source_dataset,
        )
        record: dict[str, Any] = {
            "sync_id": sync_id,
            "format_version": SYNC_FORMAT_VERSION,
            "created_at": when,
            "source": {
                "dataset_id": source_dataset.dataset_id,
                "core_sha256": source_dataset.core_sha256,
                "chunk_inventory_sha256": source_dataset.chunk_inventory_sha256,
                "chunk_count": source_dataset.chunk_count,
            },
            "dataset": {
                "dataset_id": written.dataset_id,
                "core_sha256": written.core_sha256,
                "chunk_inventory_sha256": written.chunk_inventory_sha256,
                "chunk_count": written.chunk_count,
            },
            # Same ids and texts as the source; metadata may still differ.
            "same_chunk_inventory_as_source": (
                written.chunk_inventory_sha256 == source_dataset.chunk_inventory_sha256
            ),
            "connector": {
                "connector_type": connector_type,
                "connection_sha256": (connection_sha256(connection)
                                      if connection is not None else None),
                "instance_label": instance_label,
                "host": host,
                "retrieval_mode": getattr(pipeline, "retrieval_mode", None),
            },
            "audit": None if audit is None else {
                "audit_id": audit["audit_id"],
                "created_at": audit.get("created_at"),
                "in_sync": audit.get("in_sync"),
                "backend_chunk_inventory_sha256":
                    (audit.get("backend") or {}).get("chunk_inventory_sha256"),
                "backend_matches_audit": (
                    (audit.get("backend") or {}).get("chunk_inventory_sha256")
                    == written.chunk_inventory_sha256
                ),
            },
        }
        _ensure_private_area(staging, written)
        area = staging / SYNCS_DIR / sync_id
        area.mkdir(parents=True)
        _write_json(area / "sync.json", record)
        if target.exists():
            target.rmdir()  # empty, checked above
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetSync(dataset=read_dataset_folder(target), record=record,
                       path=target / SYNCS_DIR / sync_id)


def list_syncs(folder: str | os.PathLike[str]) -> list[Mapping[str, Any]]:
    """Every ``sync.json`` in a Dataset folder, oldest first, each with ``path``.

    A Dataset written by :func:`sync_dataset` holds one. An unreadable record
    is listed with a ``problem`` rather than skipped.
    """
    root = Path(folder) / SYNCS_DIR
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for record_path in sorted(root.glob("*/sync.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("sync.json must hold a JSON object")
        except ValueError as exc:
            out.append({"sync_id": record_path.parent.name,
                        "path": str(record_path.parent), "problem": str(exc)})
            continue
        record["path"] = str(record_path.parent)
        out.append(record)
    out.sort(key=lambda r: (str(r.get("created_at") or ""), r["sync_id"]))
    return out
