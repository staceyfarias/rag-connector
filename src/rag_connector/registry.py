"""Connector registration, reconstruction, and package discovery."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import RagPipeline
from .errors import ConnectorRegistrationError
from .models import DatasetRef
from .prompts import PromptTemplate

ENTRY_POINT_GROUP = "rag_connector.connectors"
ParamField = dict[str, Any]


@dataclass
class ConnectorSpec:
    """Everything a host needs to offer and reconstruct a connector."""

    connector_type: str
    label: str
    description: str
    build: Callable[[dict], RagPipeline]
    connect: Callable[[dict], tuple[RagPipeline, dict]] | None = None
    params: list[ParamField] = field(default_factory=list)
    #: Optional. Enumerate the backend's datasets from PARTIAL params — enough
    #: to reach the backend (a persist dir, an endpoint, credentials) but NOT
    #: the field that selects a dataset, since that is what the host is asking
    #: about. Symmetric with ``connect``: a callable the connector author
    #: supplies, guarded by :attr:`supports_dataset_listing`.
    #:
    #: This is the surface hosts should use. ``RagPipeline.list_datasets`` on a
    #: bound instance answers the same question, but reaching it means binding
    #: first, and binding takes the parameter the host is trying to choose —
    #: so the instance method serves a connector already bound (showing an
    #: operator what else is there), while a picker shown BEFORE anything
    #: exists needs this one. A connector may implement either or both.
    list_datasets: Callable[[dict], list[DatasetRef]] | None = None
    #: Optional. Long-form markdown describing what this connector IS and what
    #: its parameters mean — authored by the connector's author, for an operator
    #: choosing or configuring it. Zero-arg because it is read from a registry
    #: listing, BEFORE anything is bound: no params, no credentials, no backend.
    #:
    #: Distinct from :attr:`description`, which is one line for a table row, and
    #: from ``RagPipeline.info``, which is machine-readable run metadata off a
    #: connected instance. The contract says nothing about where the text comes
    #: from — a bundled file, a docstring, something computed — only that a call
    #: returns markdown.
    connector_info: Callable[[], str] | None = None
    #: Optional. The prompt variants this connector declares it can be
    #: measured on -- see :class:`~rag_connector.prompts.PromptTemplate`.
    #: Zero-arg for the same reason :attr:`connector_info` is: a host renders
    #: a prompt-variant picker from a registry listing, BEFORE anything is
    #: bound, so this must answer with no params, no credentials and no
    #: backend. ``RagPipeline.prompt_templates`` answers the same question off
    #: a connected instance; a connector should serve one declared set through
    #: both, so the variant an operator chose is the variant that runs.
    #:
    #: Guarded by :attr:`supports_prompt_templates`, and deliberately absent
    #: from :meth:`to_public_dict` -- a callable cannot be JSON, and a picker
    #: reads the attribute off the spec object where it needs the templates.
    prompt_templates: Callable[[], Sequence[PromptTemplate]] | None = None

    @property
    def user_connectable(self) -> bool:
        return self.connect is not None

    @property
    def supports_dataset_listing(self) -> bool:
        return self.list_datasets is not None

    @property
    def supports_prompt_templates(self) -> bool:
        return self.prompt_templates is not None

    @property
    def connection_schema(self) -> list[ParamField]:
        """Compatibility name for hosts that describe fields as a schema."""

        return self.params

    def to_public_dict(self) -> dict:
        return {
            "connector_type": self.connector_type,
            "label": self.label,
            "description": self.description,
            "params": self.params,
            "supports_dataset_listing": self.supports_dataset_listing,
        }


_REGISTRY: dict[str, ConnectorSpec] = {}
_DISCOVERY_COMPLETE = False
#: Entry points already loaded and registered, so a retry after a failing
#: sibling does not re-load them (a factory entry point returns a NEW spec each
#: call, which would then collide with the one it already registered).
_LOADED_ENTRY_POINTS: set[str] = set()


def validate_connection(connection: Mapping[str, Any]) -> None:
    try:
        json.dumps(dict(connection))
    except (TypeError, ValueError) as exc:
        raise ConnectorRegistrationError(
            "connection configuration must be JSON-serializable"
        ) from exc


def register_connector(spec: ConnectorSpec) -> ConnectorSpec:
    existing = _REGISTRY.get(spec.connector_type)
    if existing is not None and existing is not spec:
        raise ConnectorRegistrationError(
            f"Connector type {spec.connector_type!r} is already registered "
            f"(by {existing.label!r}). Connector types must be unique."
        )
    _REGISTRY[spec.connector_type] = spec
    return spec


register = register_connector


def _load_spec(value: Any, *, source: str) -> ConnectorSpec:
    spec = value() if callable(value) and not isinstance(value, ConnectorSpec) else value
    if not isinstance(spec, ConnectorSpec):
        raise ConnectorRegistrationError(f"{source} did not provide ConnectorSpec")
    return spec


def discover_connectors() -> None:
    """Register bundled and independently installed connector packages.

    Discovery is once-only **on success**. A broken third-party entry point
    raises, and raises again on the next call, rather than marking discovery
    done on the way in — which would report the failure once and then serve a
    silently partial registry forever after, the shape where a connector is
    missing from the UI and nobody is told why.
    """

    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE:
        return

    # Importing the optional Reference module is safe without its heavy extras:
    # Chroma and FastEmbed are loaded only when an instance is constructed.
    importlib.import_module("rag_connector.reference")

    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        key = f"{entry_point.name}={entry_point.value}"
        if key in _LOADED_ENTRY_POINTS:
            continue
        register_connector(
            _load_spec(entry_point.load(), source=f"entry point {entry_point.name!r}")
        )
        _LOADED_ENTRY_POINTS.add(key)

    _DISCOVERY_COMPLETE = True


discover = discover_connectors


def get_connector(connector_type: str) -> ConnectorSpec | None:
    discover_connectors()
    return _REGISTRY.get(connector_type)


get = get_connector


def all_connectors() -> list[ConnectorSpec]:
    discover_connectors()
    return sorted(_REGISTRY.values(), key=lambda spec: spec.label.casefold())


def all_specs() -> tuple[ConnectorSpec, ...]:
    return tuple(all_connectors())


def connectable_connectors() -> list[ConnectorSpec]:
    return [spec for spec in all_connectors() if spec.user_connectable]


def registered_types() -> list[str]:
    discover_connectors()
    return sorted(_REGISTRY)

