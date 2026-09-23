"""Regenerate the frozen pre-chunked form of a bundled corpus.

Run this ONLY when the chunk set is meant to change -- a document edited, a
corpus added, the chunker's behaviour deliberately altered. Everything
downstream cites these ids: Pelorus's pre-canned extracts and clusters and
RAGauge's pre-canned dataset and TestSet are all built against this exact
split, so regenerating without re-grounding them leaves both products citing
chunks that no longer exist.

``tests/test_ingest.py`` runs the same generation and fails if the committed
file differs, so the file cannot drift from the kit that made it by accident --
only by someone running this and committing the result.

    python tools/freeze_bundled_chunks.py [corpus_name ...]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rag_connector.ingest import (  # noqa: E402
    BUNDLED_CORPORA,
    build_bundled_chunks,
    chunks_to_jsonl,
)

CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "rag_connector" / "corpus"


def main(argv: list[str]) -> int:
    names = argv or list(BUNDLED_CORPORA)
    for name in names:
        if name not in BUNDLED_CORPORA:
            print(f"unknown corpus {name!r}; available: {', '.join(BUNDLED_CORPORA)}")
            return 2
        records = build_bundled_chunks(name)
        target = CORPUS_DIR / f"{name}.chunks.jsonl"
        # newline="" keeps the "\n" chunks_to_jsonl wrote from becoming "\r\n"
        # on Windows: the frozen file is pinned to LF in .gitattributes, and a
        # rewrite that flips every line ending is a 605-line diff saying
        # nothing.
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(chunks_to_jsonl(records))
        print(f"{target}: {len(records)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
