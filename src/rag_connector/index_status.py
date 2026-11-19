"""Which of a Dataset's indexes exist, whether they are current, and what builds them.

A Dataset is created by pulling chunks; its indexes are built by a separate step.
:func:`index_status` is what a host asks after a pull (or before a search) so it can
offer to build what is missing and say exactly what that needs:

- **search index** (``analysis/rag-connector/search.sqlite``) and **book index**
  (``index.json`` beside it, plus the same entries as tables in the search index):
  built from the chunk text by rules, no model. ``rag-connector index prepare``
  builds both.
- **signal index**: index lines a model writes per document. Optional; it calls a
  model (the ``llm`` extra and ``ANTHROPIC_API_KEY``) and costs credits.
  ``rag-connector index signal`` builds it.

Each index is ``missing``, ``stale`` (built from other chunks, or by an older
rag-connector) or ``current``. Nothing here builds anything, calls a model, or
reads a credential's value: the key is only tested for being set.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

__all__ = ["BUILD_COMMAND", "SIGNAL_COMMAND", "index_status", "signal_buildable"]

BUILD_COMMAND = "rag-connector index prepare --dataset {dataset}"
SIGNAL_COMMAND = "rag-connector index signal --dataset {dataset}"


def signal_buildable() -> dict[str, Any]:
    """Whether the signal index can be built in this environment: ``{"ok", "reason"}``.
    Checks that the ``anthropic`` package (the ``llm`` extra) is installed and that
    ``ANTHROPIC_API_KEY`` is set; never reads or returns the key."""
    missing = []
    if importlib.util.find_spec("anthropic") is None:
        missing.append('the model client is not installed (pip install "rag-connector[llm]")')
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing.append("ANTHROPIC_API_KEY is not set")
    if missing:
        return {"ok": False, "reason": "; ".join(missing)}
    return {"ok": True, "reason": "model client installed and ANTHROPIC_API_KEY set"}


def _search(folder: Path) -> dict[str, Any]:
    from . import search_index

    path = search_index.search_db_path(folder)
    if not path.is_file():
        return {"state": "missing", "path": str(path)}
    try:
        index = search_index.SearchIndex(path)
    except search_index.SearchIndexError as exc:
        return {"state": "stale", "path": str(path), "reasons": [str(exc)]}
    try:
        reasons = index.stale_reasons(folder)
        meta = dict(index.meta)
    finally:
        index.close()
    out: dict[str, Any] = {"state": "stale" if reasons else "current", "path": str(path),
                           "built_at": meta.get("built_at"), "chunks": meta.get("chunk_count")}
    if reasons:
        out["reasons"] = reasons
    return out


def _book(folder: Path) -> dict[str, Any]:
    from . import corpus_index

    path = corpus_index.index_path(folder)
    if not path.is_file():
        return {"state": "missing", "path": str(path)}
    found = corpus_index.load(folder)
    if found is None:
        return {"state": "stale", "path": str(path),
                "reasons": ["built from other chunks than this Dataset holds"]}
    counts = found.get("counts") or {}
    return {"state": "current", "path": str(path), "built_at": found.get("built_at"),
            "documents": counts.get("documents"), "chunks": counts.get("chunks"),
            "entries": counts.get("entries"), "threads": counts.get("threads")}


def _signal(folder: Path) -> dict[str, Any]:
    from . import signal_index

    path = signal_index.signal_path(folder)
    if not path.is_file():
        return {"state": "missing", "path": str(path)}
    found = signal_index.load(folder)
    if found is None:
        return {"state": "stale", "path": str(path),
                "reasons": ["built from other chunks than this Dataset holds"]}
    return {"state": "current", "path": str(path), "built_at": found.get("built_at"),
            "documents": len(found.get("documents") or []),
            "models": list(found.get("models") or [])}


def index_status(dataset_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """The state of every index of the Dataset at ``dataset_dir``.

    ``{"search_index", "book_index", "signal_index"}``: each ``{"state":
    missing|stale|current, "path", ...}`` (``reasons`` when stale);
    ``"ready_to_search"``: the search index is current, so search, lookup and
    SQL work; ``"build"``: the command that builds or rebuilds the search and
    book indexes when either is not current, else ``None``; ``"signal"``:
    ``{"buildable", "reason", "command"}`` for the optional model-written index.
    """
    folder = Path(dataset_dir)
    search, book, signal = _search(folder), _book(folder), _signal(folder)
    can = signal_buildable()
    needs_build = search["state"] != "current" or book["state"] != "current"
    return {
        "dataset": str(folder),
        "search_index": search,
        "book_index": book,
        "signal_index": signal,
        "ready_to_search": search["state"] == "current",
        "build": BUILD_COMMAND.format(dataset=folder) if needs_build else None,
        "build_uses_model": False,
        "signal": {"buildable": can["ok"], "reason": can["reason"],
                   "command": SIGNAL_COMMAND.format(dataset=folder),
                   "uses_model": True},
    }
