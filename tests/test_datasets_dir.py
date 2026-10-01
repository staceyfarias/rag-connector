"""The shared Datasets directory: how it is resolved, listed and used by the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_connector import datasets_dir as dd
from rag_connector.cli import main
from rag_connector.dataset import read_dataset_folder


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake profile folder, with no environment override in play."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    monkeypatch.delenv(dd.DATASETS_DIR_ENV, raising=False)
    return fake


def test_default_is_a_visible_folder_in_the_profile(home):
    path, source = dd.resolve_datasets_dir()
    assert path == (home / "rag-connector" / "datasets").resolve()
    assert source == dd.SOURCE_DEFAULT
    assert not any(part.startswith(".") for part in path.relative_to(home.resolve()).parts)


def test_resolution_order_argument_then_env_then_config_then_default(home, monkeypatch, tmp_path):
    dd.set_datasets_dir(tmp_path / "from-config")
    assert dd.resolve_datasets_dir() == ((tmp_path / "from-config").resolve(), dd.SOURCE_CONFIG)

    monkeypatch.setenv(dd.DATASETS_DIR_ENV, str(tmp_path / "from-env"))
    assert dd.resolve_datasets_dir() == ((tmp_path / "from-env").resolve(), dd.SOURCE_ENV)

    assert dd.resolve_datasets_dir(tmp_path / "arg") == (
        (tmp_path / "arg").resolve(), dd.SOURCE_ARGUMENT)

    monkeypatch.delenv(dd.DATASETS_DIR_ENV)
    dd.set_datasets_dir(None)
    assert dd.resolve_datasets_dir()[1] == dd.SOURCE_DEFAULT


def test_resolving_never_creates_the_directory_but_create_true_does(home):
    path = dd.datasets_dir()
    assert not path.exists()
    assert dd.datasets_dir(create=True).is_dir()


def test_setting_keeps_other_config_keys(home):
    config = dd.config_path()
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"other": 1}), encoding="utf-8")
    dd.set_datasets_dir(home / "x")
    stored = json.loads(config.read_text(encoding="utf-8"))
    assert stored["other"] == 1 and "datasets_dir" in stored
    dd.set_datasets_dir(None)
    assert json.loads(config.read_text(encoding="utf-8")) == {"other": 1}


def test_an_unreadable_config_is_an_error_not_the_default(home):
    config = dd.config_path()
    config.parent.mkdir(parents=True)
    config.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        dd.resolve_datasets_dir()


def test_tilde_in_a_configured_path_is_expanded(home):
    dd.set_datasets_dir("~/elsewhere")
    path, _ = dd.resolve_datasets_dir()
    assert path == (home / "elsewhere").resolve()


def test_folder_name_for():
    assert dd.folder_name_for("Pelorus Space v2.1") == "pelorus-space-v2.1"
    assert dd.folder_name_for("  ../..  ") == "dataset"
    assert dd.folder_name_for("") == "dataset"


def _write_manifest(folder: Path, manifest) -> None:
    folder.mkdir(parents=True)
    (folder / "dataset.json").write_text(
        manifest if isinstance(manifest, str) else json.dumps(manifest), encoding="utf-8")


def test_listing_reads_manifests_only_and_skips_non_datasets(home, tmp_path):
    root = tmp_path / "ds"
    _write_manifest(root / "b-spec", {"spec": "rag-connector-dataset", "dataset_id": "ds-1",
                                      "name": "B", "chunk_count": 3, "created_at": "t"})
    _write_manifest(root / "a-legacy", {"id": "old-1", "name": "A"})
    _write_manifest(root / "c-broken", "{nope")
    _write_manifest(root / "_archive", {"dataset_id": "ds-skip"})
    _write_manifest(root / ".hidden", {"dataset_id": "ds-skip"})
    (root / "no-manifest").mkdir()
    (root / "stray.txt").write_text("x", encoding="utf-8")

    entries = dd.list_dataset_folders(root)

    assert [e.path.name for e in entries] == ["a-legacy", "b-spec", "c-broken"]
    legacy, spec, broken = entries
    assert legacy.legacy and legacy.dataset_id == "old-1" and not legacy.verified
    assert not spec.legacy and spec.dataset_id == "ds-1" and spec.chunk_count == 3
    assert broken.problem and broken.dataset_id is None


def test_listing_a_missing_directory_is_empty(home, tmp_path):
    assert dd.list_dataset_folders(tmp_path / "nowhere") == []


# --- CLI ---------------------------------------------------------------------

def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "My Docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Title\n\nSome words about a thing.\n", encoding="utf-8")
    return docs


def test_dataset_build_without_out_writes_into_the_datasets_directory(home, tmp_path, capsys):
    assert main(["dataset", "build", "--folder", str(_docs(tmp_path))]) == 0
    summary = json.loads(capsys.readouterr().out)

    expected = (home / "rag-connector" / "datasets" / "my-docs").resolve()
    assert Path(summary["path"]).resolve() == expected
    assert read_dataset_folder(expected).dataset_id == summary["dataset_id"]


def test_dataset_build_still_refuses_a_non_empty_target(home, tmp_path, capsys):
    docs = _docs(tmp_path)
    assert main(["dataset", "build", "--folder", str(docs)]) == 0
    capsys.readouterr()
    assert main(["dataset", "build", "--folder", str(docs)]) == 1
    assert "rag-connector dataset" in capsys.readouterr().err


def test_dataset_build_with_out_ignores_the_datasets_directory(home, tmp_path, capsys):
    out = tmp_path / "elsewhere"
    assert main(["dataset", "build", "--folder", str(_docs(tmp_path)), "--out", str(out)]) == 0
    capsys.readouterr()
    assert (out / "dataset.json").is_file()
    assert not (home / "rag-connector").exists()


def test_dataset_list_prints_one_json_line_per_dataset(home, tmp_path, capsys):
    assert main(["dataset", "build", "--folder", str(_docs(tmp_path))]) == 0
    built = json.loads(capsys.readouterr().out)
    assert main(["dataset", "list"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["dataset_id"], r["name"]) for r in lines] == [(built["dataset_id"], "My Docs")]


def test_datasets_dir_command_shows_sets_and_clears(home, tmp_path, capsys):
    assert main(["datasets-dir"]) == 0
    assert capsys.readouterr().out.strip() == str((home / "rag-connector" / "datasets").resolve())

    target = tmp_path / "mine"
    assert main(["datasets-dir", "--set", str(target), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["path"] == str(target.resolve())
    assert shown["source"] == "config" and shown["exists"] is False

    assert main(["datasets-dir", "--unset", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "default"


def test_datasets_dir_command_reports_a_bad_config(home, capsys):
    config = dd.config_path()
    config.parent.mkdir(parents=True)
    config.write_text("[]", encoding="utf-8")
    assert main(["datasets-dir"]) == 1
    assert "datasets-dir" in capsys.readouterr().err
