"""Check that a built wheel and sdist carry the files a consumer reads.

A green suite says nothing about what ships. Every file this script looks for
is read at runtime through ``importlib.resources`` or by a consumer's type
checker, and an editable install resolves all of them off ``src/`` on disk --
so a package-data declaration that drops one is invisible from every local run
and visible only here. It has been wrong before: the package shipped with no
corpus at all, and an installed ``rag-connector[reference]`` had a Reference
RAG with nothing to point it at.

    python -m build
    python tools/verify_artifact.py dist/*.whl dist/*.tar.gz

Exits non-zero, naming what is missing, if any artifact is incomplete.
"""

from __future__ import annotations

import pathlib
import sys
import tarfile
import zipfile

#: The bundled corpus, as shipped: 28 documents (corpus v2.1) plus the frozen split whose
#: chunk ids Pelorus's extracts and RAGauge's dataset both cite.
EXPECTED_CORPUS_DOCUMENTS = 28
CORPUS_DIR = "rag_connector/corpus/pelorus_space/"
FROZEN_CHUNKS = "rag_connector/corpus/pelorus_space.chunks.jsonl"
#: Package data with the same invisibility, and a worse failure: a consumer's
#: type checker silently ignores the annotations while the classifiers still
#: advertise ``Typing :: Typed``.
REQUIRED_FILES = ("rag_connector/py.typed", "rag_connector/CONNECTOR-INFO.md")


def _members(path: pathlib.Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        # An sdist nests everything under <name>-<version>/; drop that prefix so
        # one set of expectations reads both artifacts.
        return [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]


def check(path: pathlib.Path) -> list[str]:
    names = _members(path)
    if path.suffix != ".whl":
        names = [name[len("src/"):] if name.startswith("src/") else name for name in names]
    problems = []
    documents = [name for name in names if name.startswith(CORPUS_DIR)]
    if len(documents) != EXPECTED_CORPUS_DOCUMENTS:
        problems.append(
            f"{len(documents)} corpus documents under {CORPUS_DIR}, "
            f"expected {EXPECTED_CORPUS_DOCUMENTS}"
        )
    for required in (FROZEN_CHUNKS, *REQUIRED_FILES):
        if required not in names:
            problems.append(f"missing {required}")
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python tools/verify_artifact.py <artifact> [artifact ...]")
        return 2
    failed = False
    for argument in argv:
        path = pathlib.Path(argument)
        problems = check(path)
        if problems:
            failed = True
            for problem in problems:
                print(f"{path.name}: {problem}")
        else:
            print(f"{path.name}: OK ({EXPECTED_CORPUS_DOCUMENTS} corpus documents, "
                  "frozen chunks, py.typed, connector info)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
