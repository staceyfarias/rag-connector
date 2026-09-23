"""Typed failures at the RAG Connector boundary."""


class ConnectorError(RuntimeError):
    """Base class for connector-boundary failures."""


class ConnectorOperationalError(ConnectorError):
    """The connector supports an operation, but its backend failed.

    Hosts must not turn this into an empty retrieval result: an operational
    failure is missing evidence, not evidence that nothing matched.
    """


class UnsupportedCapability(ConnectorError):
    """The connector does not implement a requested optional capability."""


class ConnectorContractError(ConnectorError):
    """A connector returned data that violates the portable contract."""


class ConnectorRegistrationError(ConnectorError, ValueError):
    """A connector could not be registered or reconstructed safely.

    Also a ``ValueError``, following ``io.UnsupportedOperation``: a duplicate
    connector type used to raise a bare ``ValueError``, and hosts already catch
    it that way. Inheriting both lets every registration failure carry the typed
    boundary class without breaking an ``except ValueError`` that predates it.
    """

