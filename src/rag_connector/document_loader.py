"""Compatibility shim: the document loader now lives in :mod:`rag_connector.ingest`.

``load_documents`` and ``LoadedDocument`` moved when loading and chunking were
promoted into one declared **ingest kit** — a host building its own corpus needs
both, and they were split across two modules for no reason other than history.
The names are re-exported here so existing imports keep working; new code should
import from :mod:`rag_connector.ingest`.
"""

from __future__ import annotations

from .ingest import LoadedDocument, load_documents

__all__ = ["LoadedDocument", "load_documents"]
