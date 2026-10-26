"""The neutral metadata filter ``query(..., filter=...)`` takes.

A small, documented subset of MongoDB's query syntax (also Pinecone's metadata
filter language, and close to Chroma's ``where``), so a connector to any of
those backends can pass it through or translate it mechanically:

* A **field condition** is ``{"field": value}`` (equality, ``$eq``) or
  ``{"field": {"$op": operand, ...}}``. Several operators on one field are
  ANDed.
* **Comparison operators**: ``$eq``, ``$ne`` (operand: a scalar),
  ``$in``, ``$nin`` (a non-empty list of scalars), ``$gt``, ``$gte``, ``$lt``,
  ``$lte`` (a number), ``$exists`` (a bool).
* **Logical operators**: ``$and`` and ``$or``, each a non-empty list of
  filters.
* Top-level keys are ANDed. ``{}`` (and ``None``) means "no constraint".

A scalar is a ``str``, ``int``, ``float`` or ``bool``. ``null`` values, bare
lists (array equality) and nested documents are not part of the subset: say
``{"field": {"$in": [...]}}`` instead of ``{"field": [...]}``. Range operators
take numbers only, which every backend above can honour.

:func:`validate_filter` is the single gate: it returns a normalized deep copy
or raises :class:`~rag_connector.errors.InvalidFilterError` naming the path of
the first problem. :func:`filter_operators` lists the operators a filter uses,
so a connector (or host) can compare them with what it supports and raise
:class:`~rag_connector.errors.UnsupportedFilterError` instead of ignoring any
part of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .errors import InvalidFilterError, UnsupportedFilterError

#: Comparison operators a field condition may use.
COMPARISON_OPERATORS = frozenset({
    "$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$exists",
})
#: Logical operators that combine filters.
LOGICAL_OPERATORS = frozenset({"$and", "$or"})
#: Every operator of the neutral syntax. A connector that passes filters
#: straight to a backend that speaks the whole subset declares this.
FILTER_OPERATORS = COMPARISON_OPERATORS | LOGICAL_OPERATORS

_SCALAR_OPERATORS = frozenset({"$eq", "$ne"})
_LIST_OPERATORS = frozenset({"$in", "$nin"})
_RANGE_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})


def _is_scalar(value: Any) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return isinstance(value, (str, int, float, bool))


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _fail(path: str, message: str) -> InvalidFilterError:
    return InvalidFilterError(f"filter{path}: {message}")


def _validate_condition(field: str, condition: Any, path: str) -> Any:
    if isinstance(condition, Mapping):
        if not condition:
            raise _fail(path, f"the condition on {field!r} is empty")
        out: dict[str, Any] = {}
        for op, operand in condition.items():
            op_path = f"{path}[{op!r}]"
            if not isinstance(op, str) or not op.startswith("$"):
                raise _fail(
                    op_path,
                    f"{op!r} is not an operator; nested documents are not "
                    f"part of the filter syntax (field {field!r})")
            if op in LOGICAL_OPERATORS:
                raise _fail(op_path, f"{op} combines filters and belongs at "
                                     f"filter level, not inside field {field!r}")
            if op not in COMPARISON_OPERATORS:
                raise _fail(op_path, f"unknown operator {op!r}; the syntax has "
                                     + ", ".join(sorted(FILTER_OPERATORS)))
            if op in _SCALAR_OPERATORS:
                if not _is_scalar(operand):
                    raise _fail(op_path, f"{op} takes a string, number or "
                                         f"bool, not {operand!r}")
                out[op] = operand
            elif op in _LIST_OPERATORS:
                if (not isinstance(operand, (list, tuple)) or not operand
                        or not all(_is_scalar(v) for v in operand)):
                    raise _fail(op_path, f"{op} takes a non-empty list of "
                                         f"strings, numbers or bools, not "
                                         f"{operand!r}")
                out[op] = list(operand)
            elif op in _RANGE_OPERATORS:
                if not _is_number(operand):
                    raise _fail(op_path, f"{op} takes a finite number, not "
                                         f"{operand!r}")
                out[op] = operand
            else:  # $exists
                if not isinstance(operand, bool):
                    raise _fail(op_path, f"$exists takes true or false, not "
                                         f"{operand!r}")
                out[op] = operand
        return out
    if _is_scalar(condition):
        return condition
    if isinstance(condition, (list, tuple)):
        raise _fail(path, f"a list is not a value for {field!r}; use "
                          "{\"$in\": [...]} to match any of several values")
    raise _fail(path, f"{condition!r} is not a value for {field!r}; values "
                      "are strings, numbers or bools (null is not supported)")


def _validate(filt: Any, path: str) -> dict:
    if not isinstance(filt, Mapping):
        raise _fail(path, f"a filter is an object, not {type(filt).__name__}")
    out: dict[str, Any] = {}
    for key, value in filt.items():
        key_path = f"{path}[{key!r}]"
        if not isinstance(key, str) or not key:
            raise _fail(key_path, "field names are non-empty strings")
        if key in LOGICAL_OPERATORS:
            if not isinstance(value, (list, tuple)) or not value:
                raise _fail(key_path, f"{key} takes a non-empty list of "
                                      "filters")
            parts = [_validate(part, f"{key_path}[{i}]")
                     for i, part in enumerate(value)]
            if any(not part for part in parts):
                raise _fail(key_path, f"{key} contains an empty filter")
            out[key] = parts
        elif key.startswith("$"):
            raise _fail(key_path,
                        f"{key!r} is not allowed at filter level; only $and "
                        "and $or combine filters (comparison operators go "
                        "inside a field condition)")
        else:
            out[key] = _validate_condition(key, value, key_path)
    return out


def validate_filter(filt: Mapping[str, Any] | None) -> dict:
    """Return a normalized deep copy of ``filt``, or raise
    :class:`InvalidFilterError`.

    ``None`` and ``{}`` both return ``{}`` -- no constraint. Tuples become
    lists, so the result is JSON-serializable and safe to persist or pass to a
    backend client.
    """
    if filt is None:
        return {}
    return _validate(filt, "")


def filter_operators(filt: Mapping[str, Any] | None) -> frozenset[str]:
    """Every operator ``filt`` uses (validated first). A bare value counts as
    ``$eq``; an empty filter uses none."""
    used: set[str] = set()

    def walk(node: Mapping[str, Any]) -> None:
        for key, value in node.items():
            if key in LOGICAL_OPERATORS:
                used.add(key)
                for part in value:
                    walk(part)
            elif isinstance(value, Mapping):
                used.update(value)
            else:
                used.add("$eq")

    walk(validate_filter(filt))
    return frozenset(used)


def filter_fields(filt: Mapping[str, Any] | None) -> frozenset[str]:
    """Every metadata field ``filt`` constrains (validated first)."""
    fields: set[str] = set()

    def walk(node: Mapping[str, Any]) -> None:
        for key, value in node.items():
            if key in LOGICAL_OPERATORS:
                for part in value:
                    walk(part)
            else:
                fields.add(key)

    walk(validate_filter(filt))
    return frozenset(fields)


def require_filter_operators(
    filt: Mapping[str, Any] | None,
    supported: Iterable[str],
    *,
    connector: str = "this connector",
) -> dict:
    """Validate ``filt`` and check it uses only ``supported`` operators.

    Returns the normalized filter. Raises :class:`InvalidFilterError` for a
    malformed filter and :class:`UnsupportedFilterError` (naming the
    operators) when any part of it cannot be honoured -- a connector calls
    this at the top of ``query`` so no part of a filter is ever dropped
    silently. An empty filter passes whatever is supported.
    """
    normalized = validate_filter(filt)
    if not normalized:
        return normalized
    allowed = frozenset(supported)
    missing = filter_operators(normalized) - allowed
    if not allowed:
        raise UnsupportedFilterError(
            f"{connector} does not support query filters", operators=missing)
    if missing:
        raise UnsupportedFilterError(
            f"{connector} cannot honour filter operator(s) "
            f"{', '.join(sorted(missing))}; it supports "
            f"{', '.join(sorted(allowed))}",
            operators=missing)
    return normalized


def and_filters(*filters: Mapping[str, Any] | None) -> dict:
    """AND several filters into one (each validated). Empty and ``None``
    parts are dropped; one remaining part is returned as is; none gives
    ``{}``. Never merges keys, so two conditions on one field both hold."""
    parts = [validate_filter(f) for f in filters]
    parts = [p for p in parts if p]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
