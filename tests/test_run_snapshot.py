from __future__ import annotations

import math

import pytest

from rag_connector import run_snapshot_fingerprint, validate_run_snapshot
from rag_connector.errors import ConnectorContractError


def _snapshot(**overrides):
    return {
        "schema": "demo.run-snapshot.v1",
        "settings": {"strategy": "hybrid"},
        "observations": {"indexed_chunks": 3},
        **overrides,
    }


def test_snapshot_fingerprint_is_order_independent_and_normalized():
    left = _snapshot(settings={"b": 2, "a": (1, "x")})
    right = _snapshot(settings={"a": [1, "x"], "b": 2})

    assert validate_run_snapshot(left)["settings"]["a"] == [1, "x"]
    assert run_snapshot_fingerprint(left) == run_snapshot_fingerprint(right)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_snapshot_rejects_non_json_numbers(value):
    with pytest.raises(ConnectorContractError, match="strict JSON"):
        validate_run_snapshot(_snapshot(observations={"value": value}))


def test_snapshot_rejects_nested_credentials():
    with pytest.raises(ConnectorContractError, match="api_token"):
        validate_run_snapshot(_snapshot(settings={"auth": {"api_token": "x"}}))
