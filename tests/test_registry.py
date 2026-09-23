import importlib.metadata
import json

import pytest

from rag_connector import ChunkRecord, RagPipeline, RetrievedChunk
from rag_connector import registry as reg
from rag_connector.errors import ConnectorRegistrationError
from rag_connector.registry import (
    ConnectorSpec,
    all_connectors,
    connectable_connectors,
    get_connector,
    register_connector,
    registered_types,
    validate_connection,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reg.discover_connectors()
    saved = dict(reg._REGISTRY)
    yield
    reg._REGISTRY.clear()
    reg._REGISTRY.update(saved)


class FakeConnector(RagPipeline):
    name = "fake-rag"

    def __init__(self, *, endpoint: str = "x"):
        self.endpoint = endpoint

    def query(self, text: str, top_k: int = 5):
        return [RetrievedChunk(chunk_id="c0", text="alpha", score=0.1, rank=0)]

    def pull_all_chunks(self):
        return [
            ChunkRecord(
                chunk_id="c0",
                doc_id="d0",
                source_file="doc.txt",
                doc_chunk_index=0,
                global_index=0,
                text="alpha beta",
            )
        ]


def _fake_spec(connector_type="fake-rag", connect=True):
    return ConnectorSpec(
        connector_type=connector_type,
        label="Fake RAG",
        description="test connector",
        build=lambda connection: FakeConnector(
            endpoint=connection.get("endpoint", "x")
        ),
        connect=(
            lambda params: (
                FakeConnector(endpoint=params["endpoint"]),
                {"endpoint": params["endpoint"]},
            )
        )
        if connect
        else None,
        params=[
            {
                "name": "endpoint",
                "label": "Endpoint",
                "type": "string",
                "required": True,
            }
        ],
    )


def test_register_and_lookup():
    spec = register_connector(_fake_spec())
    assert get_connector("fake-rag") is spec
    assert "fake-rag" in registered_types()


def test_reregister_same_spec_is_idempotent():
    spec = register_connector(_fake_spec())
    assert register_connector(spec) is spec


def test_duplicate_type_different_spec_raises():
    register_connector(_fake_spec())
    # Still a ValueError: hosts caught it that way before the boundary grew a
    # typed error, and ConnectorRegistrationError inherits both so they keep
    # working. Do not "tidy" this into the typed class only.
    with pytest.raises(ValueError, match="already registered"):
        register_connector(_fake_spec())


def test_duplicate_type_raises_the_typed_registration_error():
    register_connector(_fake_spec())
    with pytest.raises(ConnectorRegistrationError, match="already registered"):
        register_connector(_fake_spec())


def test_connectable_excludes_build_only_specs():
    register_connector(_fake_spec("dropdown-rag", connect=True))
    register_connector(_fake_spec("hidden-rag", connect=False))
    types = [spec.connector_type for spec in connectable_connectors()]
    assert "dropdown-rag" in types
    assert "hidden-rag" not in types


def test_discovery_registers_reference_rag():
    assert "reference" in registered_types()
    reference = get_connector("reference")
    assert reference is not None
    assert reference.user_connectable
    assert reference in all_connectors()


def test_public_dict_shape():
    spec = _fake_spec()
    assert spec.to_public_dict() == {
        "connector_type": "fake-rag",
        "label": "Fake RAG",
        "description": "test connector",
        "params": spec.params,
        # A host renders a dataset picker off this flag, so it is always
        # present and always a bool -- never absent for a connector that
        # cannot enumerate, which a host would have to read as "unknown".
        "supports_dataset_listing": False,
    }


def test_connector_info_is_optional_and_absent_by_default():
    spec = _fake_spec()
    assert spec.connector_info is None


def test_connector_info_stays_out_of_the_public_dict():
    # ``to_public_dict`` is serialized to JSON for hosts. A callable cannot
    # travel that way, and the consumers of that dict (dataset pickers) do not
    # want the long-form text either -- a host reads the attribute directly off
    # the spec object where it needs it.
    spec = _fake_spec()
    spec.connector_info = lambda: "# Fake\n"
    assert "connector_info" not in spec.to_public_dict()
    assert "guide" not in json.dumps(spec.to_public_dict())


def test_reference_rag_supplies_its_own_connector_info():
    reference = get_connector("reference")
    assert reference is not None
    assert reference.connector_info is not None
    text = reference.connector_info()
    # Markdown, authored by the connector, naming the three parameters an
    # operator has to fill in. Read from inside the installed package, so this
    # also proves the bundled file is reachable without a relative path.
    assert text.startswith("# Reference RAG (local)")
    for param in ("persist_dir", "collection_name", "embedding_model"):
        assert param in text


def test_connection_must_be_json_serializable():
    validate_connection({"endpoint": "https://example.test", "timeout": 10})
    with pytest.raises(ConnectorRegistrationError):
        validate_connection({"bad": object()})


def test_connection_round_trips_as_plain_json():
    connection = {"endpoint": "https://example.test", "collection": "docs"}
    assert json.loads(json.dumps(connection)) == connection


# ---------------------------------------------------------------------------
# Entry-point discovery: once-only ON SUCCESS.
#
# The bug these pin: discovery used to mark itself done before loading entry
# points, so a broken third-party plugin raised on the first call and then
# every later call returned silently with that plugin missing -- the registry
# looked fine and the connector had simply vanished.
# ---------------------------------------------------------------------------

class _FakeEntryPoint:
    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader
        self.loads = 0

    def load(self):
        self.loads += 1
        return self._loader()


def _explode():
    raise RuntimeError("connector package is broken")


def _publish(monkeypatch, *entry_points):
    """Present ``entry_points`` as the installed connector plugins.

    Layers over the suite-wide fixture that hides ambient plugins, so these
    tests see exactly what they declare and nothing from site-packages.
    """

    def fake_entry_points(**kwargs):
        if kwargs.get("group") == reg.ENTRY_POINT_GROUP:
            return tuple(entry_points)
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


@pytest.fixture
def fresh_discovery(monkeypatch):
    monkeypatch.setattr(reg, "_DISCOVERY_COMPLETE", False)
    monkeypatch.setattr(reg, "_LOADED_ENTRY_POINTS", set())


def test_a_broken_entry_point_fails_every_call_not_only_the_first(
    monkeypatch, fresh_discovery
):
    broken = _FakeEntryPoint("broken", "pkg:boom", _explode)
    _publish(monkeypatch, broken)

    with pytest.raises(RuntimeError, match="broken"):
        reg.discover_connectors()

    # The silent-partial-registry bug: this second call used to return cleanly.
    with pytest.raises(RuntimeError, match="broken"):
        reg.discover_connectors()


def test_a_failed_discovery_does_not_re_register_what_already_loaded(
    monkeypatch, fresh_discovery
):
    """A factory entry point mints a NEW spec per load, so a naive retry would
    collide with the spec it registered on the previous attempt."""
    good = _FakeEntryPoint("good", "pkg:good", lambda: _fake_spec("plugin-rag"))
    broken = _FakeEntryPoint("broken", "pkg:boom", _explode)
    _publish(monkeypatch, good, broken)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="broken"):
            reg.discover_connectors()

    assert "plugin-rag" in reg._REGISTRY
    assert good.loads == 1, "the working plugin was loaded again on retry"


def test_discovery_completes_once_the_broken_plugin_is_fixed(
    monkeypatch, fresh_discovery
):
    good = _FakeEntryPoint("good", "pkg:good", lambda: _fake_spec("plugin-rag"))
    broken = _FakeEntryPoint("broken", "pkg:boom", _explode)
    _publish(monkeypatch, good, broken)
    with pytest.raises(RuntimeError, match="broken"):
        reg.discover_connectors()

    repaired = _FakeEntryPoint("broken", "pkg:boom", lambda: _fake_spec("late-rag"))
    _publish(monkeypatch, good, repaired)
    reg.discover_connectors()

    assert {"plugin-rag", "late-rag"} <= set(reg._REGISTRY)
    assert good.loads == 1


def test_build_reconstructs_connector_from_connection():
    spec = _fake_spec()
    rebuilt = spec.build({"endpoint": "https://rag.example"})
    assert isinstance(rebuilt, FakeConnector)
    assert rebuilt.endpoint == "https://rag.example"
