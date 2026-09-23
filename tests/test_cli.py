import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from rag_connector.cli import main


def test_module_execution_is_not_a_silent_no_op(monkeypatch, capsys):
    """``python -m rag_connector.cli list`` must actually run the CLI.

    Every other test here calls ``main()`` directly, which is precisely how a
    missing ``__main__`` guard stays invisible: the module form imported, ran
    nothing, printed nothing, and exited 0. The author guide diagnoses a
    registration problem by a connector being ABSENT from this listing, so an
    empty success is the one answer it must never give by accident.
    """
    monkeypatch.setattr(sys, "argv", ["rag_connector.cli", "list"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rag_connector.cli", run_name="__main__")

    assert exit_info.value.code == 0
    assert "reference" in capsys.readouterr().out


def test_list_includes_reference_connector(capsys):
    assert main(["list"]) == 0
    assert "reference\tReference RAG (local)" in capsys.readouterr().out


def test_reference_provision_prints_reusable_connection(monkeypatch, capsys):
    closed = []
    connector = SimpleNamespace(
        connection=lambda: {
            "type": "reference",
            "collection_name": "example",
            "persist_dir": "data",
        },
        info=lambda: {"name": "reference-fastembed", "chunk_count": 3},
        close=lambda: closed.append(True),
    )
    observed = {}

    def provision(folder, **kwargs):
        observed["folder"] = folder
        observed.update(kwargs)
        return connector

    monkeypatch.setattr(
        "rag_connector.reference_provision.provision_reference_rag",
        provision,
    )

    assert main(
        [
            "reference",
            "provision",
            "docs",
            "--collection",
            "example",
            "--persist-dir",
            "data",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["connection"]["collection_name"] == "example"
    assert observed["folder"] == "docs"
    # The flag names the destruction it performs: --append opts out, and the
    # default replaces the collection.
    assert observed["replace_existing"] is True
    # Not passed on the command line, so it stays unstated rather than False.
    assert observed["normalized"] is None
    # This command writes a store and exits; the next reader is another
    # process. Chroma has no flush, so releasing the engine is the handoff --
    # printing the connection while still owning the directory is the bug.
    assert closed == [True], "provision must release the store before exiting"


def test_reference_provision_forwards_the_normalization_declaration(monkeypatch):
    """The declaration has to be reachable from the documented provisioning
    path, not only by writing Python against the constructor."""
    observed = {}

    def provision(folder, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            connection=lambda: {}, info=lambda: {}, close=lambda: True
        )

    monkeypatch.setattr(
        "rag_connector.reference_provision.provision_reference_rag",
        provision,
    )

    assert main(["reference", "provision", "docs", "--collection", "example",
                 "--normalized", "true"]) == 0
    assert observed["normalized"] is True

    assert main(["reference", "provision", "docs", "--collection", "example",
                 "--normalized", "false"]) == 0
    assert observed["normalized"] is False


def test_validate_subcommand_forwards_arguments(monkeypatch):
    observed = {}

    def validate(args, *, prog=None):
        observed["args"] = args
        observed["prog"] = prog
        return 7

    monkeypatch.setattr("rag_connector.validate.main", validate)
    assert main(["validate", "--import", "example:Connector"]) == 7
    assert observed["args"] == ["--import", "example:Connector"]
    # The console script reports its own invocation, not the module form.
    assert observed["prog"] == "rag-connector validate"
