"""Validation and stable identity for optional connector run snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .errors import ConnectorContractError

RUN_SNAPSHOT_MAX_BYTES = 65_536
_SECRET_KEY_HINTS = (
    "password", "secret", "token", "api_key", "apikey", "credential",
)


def _secret_key_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(hint in str(key).lower() for hint in _SECRET_KEY_HINTS):
                found.append(path)
            found.extend(_secret_key_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_secret_key_paths(child, f"{prefix}[{index}]"))
    return found


def validate_run_snapshot(snapshot: Any) -> dict:
    """Return a normalized JSON snapshot or raise a contract error.

    Normalization is a strict JSON round trip. Besides rejecting NaN, infinity,
    bytes and arbitrary objects, it converts JSON-compatible tuples to arrays so
    every host hashes and persists exactly the same value.
    """
    problems: list[str] = []
    if not isinstance(snapshot, dict):
        problems.append(f"returned {type(snapshot).__name__}, not dict")
    else:
        if not isinstance(snapshot.get("schema"), str) or not snapshot["schema"].strip():
            problems.append("'schema' must be a non-empty string")
        for section in ("settings", "observations"):
            if not isinstance(snapshot.get(section), dict):
                problems.append(f"'{section}' must be a JSON object")
        secrets = _secret_key_paths(snapshot)
        if secrets:
            problems.append(f"secret-looking keys persisted: {secrets!r}")

    encoded: str | None = None
    if not problems:
        try:
            encoded = json.dumps(
                snapshot, allow_nan=False, ensure_ascii=False,
                separators=(",", ":"), sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            problems.append(f"not strict JSON: {exc}")
        if encoded is not None and len(encoded.encode("utf-8")) > RUN_SNAPSHOT_MAX_BYTES:
            problems.append(
                f"encoded snapshot exceeds the {RUN_SNAPSHOT_MAX_BYTES}-byte limit"
            )

    if problems:
        raise ConnectorContractError("Invalid run_snapshot(): " + "; ".join(problems))
    return json.loads(encoded)


def run_snapshot_fingerprint(snapshot: Any) -> str:
    """SHA-256 identity of the normalized snapshot returned by the connector."""
    normalized = validate_run_snapshot(snapshot)
    encoded = json.dumps(
        normalized, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
