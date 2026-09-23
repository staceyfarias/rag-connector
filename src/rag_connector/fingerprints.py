
"""Stable source, corpus, and embedding fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def document_source_identity(dataset_folder: str, filepath: str) -> str:
    """Return a deterministic source identity for a file within a dataset.

    The identity IS the dataset-relative path, normalized (forward slashes,
    casefolded for stable dedup on case-insensitive filesystems). It was
    previously the sha256 of that path; the hash added nothing but opacity --
    it is a pure function of the path, so it never bought uniqueness the path
    lacks, and both a file rename and any downstream consumer see the same
    thing either way. The readable form flows through the whole document layer
    (chunk ids are ``{source_identity}:chunk-N``) and out to the MCP /
    connector surfaces, where a legible id is worth more than a fixed-width
    one. The path is not used as a filesystem name anywhere, so path characters
    are safe.

    This is the single spelling shared with Pelorus, which defined the same
    name and signature over the readable path while this module hashed it. Two
    meanings for one identity function across two repos is a corpus difference
    nobody can see; the readable one wins.

    Note for :func:`corpus_fingerprint`: it iterates a manifest sorted by
    source identity, so changing this function changes that sort order and
    therefore the corpus fingerprint of any multi-document corpus. That is a
    corpus-domain fingerprint, deliberately sensitive to what the corpus is;
    it is not the embedding fingerprint, which is frozen for artifact
    compatibility below.
    """
    try:
        relative = Path(filepath).resolve().relative_to(Path(dataset_folder).resolve())
    except ValueError:
        relative = Path(filepath).name
    return str(relative).replace("\\", "/").casefold()


def document_source_fingerprint(file_type: str, content: str) -> str:
    """Stable parsed-source fingerprint: file type + parsed text content."""
    digest = hashlib.sha256()
    digest.update((file_type or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update((content or "").encode("utf-8"))
    return digest.hexdigest()


def corpus_fingerprint(source_manifest: Iterable[dict[str, str]]) -> str:
    """Stable corpus fingerprint from a sorted source manifest."""
    digest = hashlib.sha256()
    for item in sorted(source_manifest, key=lambda x: x.get("source_identity", "")):
        digest.update((item.get("relative_path") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update((item.get("file_type") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update((item.get("source_fingerprint") or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def embedding_fingerprint_payload(
    *,
    provider: str,
    model: str,
    dimensions: int,
    normalize: bool = True,
    model_version: str | None = None,
) -> dict[str, Any]:
    # Every value below — including the historical "rag-eval." adapter string,
    # which predates this library being product-neutral — is hashed into the
    # embedding fingerprint that binds persisted artifacts to their space.
    # Renaming the adapter string (or any key) would change every fingerprint
    # and read as "the embedding model changed" on corpora that did not change.
    # The payload is FROZEN for artifact compatibility; extend it only behind a
    # new "version" value.
    return {
        "version": 1,
        "provider": provider,
        "model": model,
        "model_version": model_version,
        "configured_dimension": dimensions,
        "vector_dimension": dimensions,
        "normalize": normalize,
        "adapter": "rag-eval.embedding_client.v1",
        "query_purpose": "query",
        "document_purpose": "document",
    }


def embedding_fingerprint(**kwargs) -> str:
    payload = embedding_fingerprint_payload(**kwargs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


