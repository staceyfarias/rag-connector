"""The shared Datasets directory: one place every product looks for Datasets.

A Dataset is a folder (``docs/dataset-spec.md``). So that RAGauge, Pelorus Query
and testset-kit can share them without copying, this package names one directory
where Datasets live by default, and every product that reads this module
resolves the same one. A Dataset dropped into it is found; ``dataset build``
writes into it when no ``--out`` is given.

Resolution, first match wins:

1. an explicit path the caller passes;
2. the ``RAG_CONNECTOR_DATASETS_DIR`` environment variable;
3. the ``datasets_dir`` setting in ``~/.rag-connector/config.json``;
4. the default: ``<checkout>/datasets`` when this package is running from a
   source checkout (a clone, or an editable install of one; the folder is
   gitignored there), else ``~/rag-connector/datasets`` (``%USERPROFILE%`` on
   Windows).

A clone gets its own folder so that everything built from it, RAGauge and
Pelorus included when they install it editable, shares one place that sits next
to the pristine copies in ``examples/``. The default is an ordinary visible
folder, not a hidden one, because people drop Dataset folders into it. A
product may let its user configure a different directory of its own; that is
that product's setting, and it falls back to this one when unset.

Listing reads only each folder's ``dataset.json``. It does **not** verify the
Dataset: verification hashes every chunk, which is the cost of *opening* one
(:func:`rag_connector.dataset.read_dataset_folder`), not of showing a list.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Environment variable that overrides the configured and default directory.
DATASETS_DIR_ENV = "RAG_CONNECTOR_DATASETS_DIR"
#: The setting key, in :func:`config_path`'s file.
CONFIG_KEY = "datasets_dir"

SOURCE_ARGUMENT = "argument"
SOURCE_ENV = "environment"
SOURCE_CONFIG = "config"
SOURCE_DEFAULT = "default"


def _source_checkout_root() -> Path | None:
    """The repository root when running from a source checkout, else ``None``.

    A checkout has ``pyproject.toml`` and ``src/rag_connector`` two levels above
    this file; an installed wheel (site-packages) has neither.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "rag_connector").is_dir():
        return root
    return None


def default_datasets_dir() -> Path:
    """``<checkout>/datasets`` from a source checkout, else ``~/rag-connector/datasets``
    (the profile folder, whatever the OS)."""
    checkout = _source_checkout_root()
    if checkout is not None:
        return checkout / "datasets"
    return Path.home() / "rag-connector" / "datasets"


def config_path() -> Path:
    """``~/.rag-connector/config.json``; the same relative place on every OS."""
    return Path.home() / ".rag-connector" / "config.json"


def _read_config() -> dict[str, Any]:
    path = config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # A config that cannot be read is not "no setting": silently falling
        # back to the default would send Datasets somewhere the user did not
        # choose. Say so.
        raise ValueError(f"{path} could not be read as JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return raw


def _clean(value: str | os.PathLike[str]) -> Path:
    text = os.path.expandvars(os.fspath(value))
    # A leading "~" means the same profile folder the default uses.
    if text == "~" or text.startswith(("~/", "~\\")):
        return (Path.home() / text[2:]).resolve()
    return Path(text).resolve()


def resolve_datasets_dir(
    explicit: str | os.PathLike[str] | None = None,
) -> tuple[Path, str]:
    """Return ``(directory, source)``; ``source`` says which rule chose it.

    Does not create the directory. ``source`` is one of ``"argument"``,
    ``"environment"``, ``"config"`` or ``"default"``.
    """
    if explicit not in (None, ""):
        return _clean(explicit), SOURCE_ARGUMENT
    env = os.environ.get(DATASETS_DIR_ENV, "").strip()
    if env:
        return _clean(env), SOURCE_ENV
    configured = _read_config().get(CONFIG_KEY)
    if configured not in (None, ""):
        if not isinstance(configured, str):
            raise ValueError(f"{config_path()}: {CONFIG_KEY!r} must be a string path")
        return _clean(configured), SOURCE_CONFIG
    return default_datasets_dir().resolve(), SOURCE_DEFAULT


def datasets_dir(explicit: str | os.PathLike[str] | None = None, *,
                 create: bool = False) -> Path:
    """The directory Datasets live in. ``create=True`` makes it if missing."""
    path, _source = resolve_datasets_dir(explicit)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def set_datasets_dir(path: str | os.PathLike[str] | None) -> Path:
    """Persist the directory in the user config; ``None`` clears the setting.

    Returns the config file written. Other keys in it are kept.
    """
    config = _read_config()
    if path is None:
        config.pop(CONFIG_KEY, None)
    else:
        config[CONFIG_KEY] = str(_clean(path))
    target = config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def folder_name_for(name: str) -> str:
    """A safe folder name for a Dataset called ``name`` (``"Pelorus Space"`` ->
    ``"pelorus-space"``). Never empty."""
    slug = _UNSAFE.sub("-", name.strip()).strip("-.").lower()
    return slug[:64] or "dataset"


@dataclass(frozen=True)
class DatasetEntry:
    """One folder in the Datasets directory, from its manifest alone.

    ``verified`` is always ``False`` here: nothing was hashed. Open the folder
    with :func:`rag_connector.dataset.read_dataset_folder` for that. ``legacy``
    marks a RAGauge folder written before the spec. ``problem`` is set, and the
    other fields are ``None``, when the manifest could not be read.
    """

    path: Path
    dataset_id: str | None
    name: str | None
    chunk_count: int | None
    created_at: str | None
    legacy: bool = False
    verified: bool = False
    problem: str | None = None


def list_dataset_folders(
    directory: str | os.PathLike[str] | None = None,
) -> list[DatasetEntry]:
    """Dataset folders directly inside the Datasets directory, sorted by path.

    A child directory counts if it holds a ``dataset.json``. Hidden folders
    (``.x``) and ``_x`` folders are skipped: they are some tool's scratch
    space, not Datasets. A missing directory is an empty list, not an error.
    """
    root = datasets_dir(directory)
    if not root.is_dir():
        return []
    entries: list[DatasetEntry] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        manifest_path = child / "dataset.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("dataset.json must hold a JSON object")
        except (OSError, ValueError) as exc:
            entries.append(DatasetEntry(child, None, None, None, None,
                                        problem=f"unreadable dataset.json: {exc}"))
            continue
        legacy = "spec" not in manifest and isinstance(manifest.get("id"), str)
        dataset_id = manifest.get("dataset_id") or manifest.get("id")
        count = manifest.get("chunk_count")
        entries.append(DatasetEntry(
            path=child,
            dataset_id=dataset_id if isinstance(dataset_id, str) else None,
            name=manifest.get("name") if isinstance(manifest.get("name"), str) else None,
            chunk_count=count if isinstance(count, int) else None,
            created_at=manifest.get("created_at")
            if isinstance(manifest.get("created_at"), str) else None,
            legacy=legacy,
        ))
    return entries
