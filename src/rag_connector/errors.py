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



class InvalidFilterError(ConnectorError, ValueError):
    """A ``query(..., filter=...)`` filter is not in the neutral syntax.

    Raised by :func:`rag_connector.filters.validate_filter` before any backend
    call. Also a ``ValueError``: the caller passed a malformed argument.
    """


class UnsupportedFilterError(UnsupportedCapability):
    """A connector cannot honour a query filter, or an operator or field in it.

    A connector raises this rather than ignore any part of a filter: a filter
    that is silently dropped turns a narrowed query into an unnarrowed one,
    and every result measured through it into the wrong evidence. A subclass
    of :class:`UnsupportedCapability`, so hosts that already catch that keep
    working. ``operators`` names the operators that could not be honoured
    (empty when the reason is a field, or filtering as a whole).
    """

    def __init__(self, message: str, *, operators=(), fields=()):
        super().__init__(message)
        self.operators = frozenset(operators)
        self.fields = frozenset(fields)
