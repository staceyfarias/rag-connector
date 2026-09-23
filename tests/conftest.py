"""Suite-wide isolation from the ambient environment."""

import importlib.metadata
import importlib.util

import pytest

_REAL_ENTRY_POINTS = importlib.metadata.entry_points


@pytest.fixture(autouse=True)
def _no_third_party_entry_points(monkeypatch):
    """Hide installed 'rag_connector.connectors' entry points from discovery.

    This suite tests the library, not whichever connector packages happen to
    be installed on the developer's machine: without this, a locally installed
    third-party connector package registers itself -- or crashes -- inside
    these tests, so the suite's result depends on ambient site-packages. The
    bundled reference connector is unaffected; it registers by module import,
    not through an entry point.
    """
    from rag_connector.registry import ENTRY_POINT_GROUP

    def filtered(**kwargs):
        if kwargs.get("group") == ENTRY_POINT_GROUP:
            return ()
        return _REAL_ENTRY_POINTS(**kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", filtered)


#: Modules the ``reference`` extra installs. The Reference RAG is an optional
#: implementation, and the suite that covers it is optional with it.
_REFERENCE_EXTRA_MODULES = ("chromadb", "fastembed")


def pytest_collection_modifyitems(config, items):
    """Skip the Reference RAG's tests when its extra is not installed.

    ``pip install "rag-connector[dev]"`` installs the test runner but not
    chromadb or fastembed, and the tests that stand a Reference RAG up then
    failed -- 41 red results describing an absent optional dependency rather
    than a broken library. A missing optional extra is a skip, and the skip
    says how to turn those tests back on.

    Marked tests, not whole modules: both Reference RAG files also hold tests
    that need no backend at all, and skipping those would quietly shrink the
    core's own coverage on a core install.
    """
    missing = [
        name
        for name in _REFERENCE_EXTRA_MODULES
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=(
            f"needs the \"reference\" extra ({', '.join(missing)} not "
            'installed): pip install "rag-connector[dev,reference]"'
        )
    )
    for item in items:
        if "reference_extra" in item.keywords:
            item.add_marker(skip)
