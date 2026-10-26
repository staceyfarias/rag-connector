"""A full-text search index over a Dataset: one SQLite file any tool can read.

:func:`build` reads a verified Dataset and writes
``analysis/rag-connector/search.sqlite`` (or a path given with ``out=``): every
chunk row with its text and metadata, two FTS5 indexes over the text, the
copy groups, and the heading each chunk sits under. :func:`open_index` opens
it read-only and refuses one built on another core.

- Word search (``chunks_fts``, tokenizer ``unicode61 remove_diacritics 2``):
  AND / OR / NOT, exact phrase, prefix, NEAR, ranked by bm25.
- Substring search (``chunks_trigram``, tokenizer ``trigram``): case-insensitive
  substrings, for the tokens word search splits (``$18``, ``41.5``,
  ``BE-FAST``) and for partial words.
- Regular expressions, through a ``regexp`` function registered on the
  connection (Python ``re``).

What is searched is the chunk text with Markdown backslash escapes removed
(``\\$18`` is searched as ``$18``): column ``plain``, which is NULL where
that changes nothing, so readers use ``COALESCE(plain, text)`` or the
``chunks_plain`` view. ``text`` is the chunk text exactly as the Dataset
holds it.

The file is evidence about the corpus, not corpus (spec rule 4): every row is
keyed by the Dataset's own ``chunk_id``, and the ``meta`` table records the
``core_sha256`` it was built from. Standard library only; the layout is
documented in ``docs/dataset-spec.md`` so other tools can read the file
directly.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import ChunkRecord
from .dataset import CHUNKS_FILE, MANIFEST_FILE, Dataset, read_dataset_folder

__all__ = [
    "AmbiguousIdError",
    "FORMAT",
    "FORMAT_VERSION",
    "SEARCH_DB",
    "SearchIndex",
    "SearchIndexError",
    "SearchIndexMissing",
    "SearchResult",
    "StaleSearchIndex",
    "build",
    "open_index",
    "search_db_path",
    "write_search_db",
]

FORMAT = "rag-connector-search-index"
FORMAT_VERSION = "1"
SEARCH_DB = "search.sqlite"

#: Overlap copies: word shingles of this many words.
OVERLAP_SHINGLE = 8
#: ... sharing at least this share of the smaller chunk's shingles.
OVERLAP_CONTAINMENT = 0.8
#: Two overlapping chunks are one passage (grouped) only when the shared
#: shingles are also at least this share of the LARGER chunk. A short chunk
#: inside many long ones (boilerplate) is still recorded as a pair, but does
#: not chain those long chunks into one group.
OVERLAP_GROUP_LARGER = 0.5
#: Chunks with fewer words are too short to call an overlap (identical and
#: near copies of them are still found).
OVERLAP_MIN_WORDS = 30
#: Candidates are found from 1 in this many shingles (chosen by hash, so the
#: same shingle is sampled in every chunk); each candidate pair is then
#: checked on all its shingles.
OVERLAP_SAMPLE = 4
#: A sampled shingle in more chunks than this is boilerplate and is not used
#: to find candidates.
OVERLAP_MAX_POSTING = 1000

SNIPPET_BEFORE = 100
SNIPPET_AFTER = 200


class SearchIndexError(ValueError):
    """A search index cannot answer: stale, malformed, or a bad query."""


class SearchIndexMissing(FileNotFoundError):
    """No search index where one was expected."""


class StaleSearchIndex(SearchIndexError):
    """The index was built from a different core than the Dataset holds now."""


class AmbiguousIdError(SearchIndexError):
    """An id prefix matches more than one id (or none)."""


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~]|\n)")


def plain_text(text: str) -> str:
    """Chunk text with Markdown backslash escapes removed: ``\\$18`` -> ``$18``,
    ``\\[00:09\\]`` -> ``[00:09]``, and a backslash line break -> a line break."""
    return _MD_ESCAPE.sub(lambda m: m.group(1), text)


# Runs of characters that UTF-8 read as cp1252 turns one character into:
# "donâ€™t" for "don't".
_MOJIBAKE = re.compile(
    "[\u00c2-\u00f4][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e"
    "\u0192\u02c6\u02dc\u2013\u2014\u2018-\u201e\u2020-\u2022\u2026\u2030"
    "\u2039\u203a\u20ac\u2122]{1,3}")


def _demojibake(text: str) -> str:
    def fix(match: re.Match) -> str:
        try:
            return match.group(0).encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return match.group(0)
    return _MOJIBAKE.sub(fix, text)


_WORDS = re.compile(r"\w+")


def _near_words(text: str) -> list[str]:
    """Words once case, punctuation, accents and encoding damage are ignored."""
    text = unicodedata.normalize("NFKD", _demojibake(plain_text(text)).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WORDS.findall(text)


def _identical_key(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(?:\*\*|__)([^*_\n]{2,120}?)(?:\*\*|__)\s*:?\s*$")
_QUESTION = re.compile(
    r"^\s*(?:[-*]\s+|\d{1,3}[.)]\s+|Q\s*[:.]\s*)?([A-Z0-9\"'(][^\n]{6,118}\?)\s*$")
_SENTENCE_BREAK = re.compile(r"[.!?]\s")
_TIMESTAMP = re.compile(r"\d{1,2}:\d{2}|\bspk_\d")
_ENDS_BLOCK = re.compile(r"[.!?:)\]*_]\s*$")


def _clean_heading(raw: str) -> str:
    title = re.sub(r"[*_`]+", "", raw).strip(" #:")
    return " ".join(title.split())


def headings_in(text: str) -> list[tuple[int, str]]:
    """(offset, heading) for each heading line in ``text``: a Markdown heading,
    a bold-only line, or a FAQ question line: one short sentence ending in
    '?', starting a block (after a blank line or a finished sentence), and
    not a transcript line."""
    found = []
    offset = 0
    previous = ""
    for line in text.split("\n"):
        title = ""
        hit = _MD_HEADING.match(line) or _BOLD_LINE.match(line)
        if hit:
            title = _clean_heading(hit.group(1))
        else:
            hit = _QUESTION.match(line)
            if (hit and not _TIMESTAMP.search(line)
                    and not _SENTENCE_BREAK.search(hit.group(1))
                    and (not previous.strip() or _ENDS_BLOCK.search(previous))):
                title = _clean_heading(hit.group(1))
        if title and len(title) <= 160:
            found.append((offset, title))
        offset += len(line) + 1
        previous = line
    return found


# ---------------------------------------------------------------------------
# Chunk rows
# ---------------------------------------------------------------------------

def _metadata(raw: Any) -> dict[str, Any]:
    """Metadata as a dict. A string holding a JSON object or a Python-literal
    dict is parsed (never evaluated); anything else is kept under ``_raw``."""
    if isinstance(raw, Mapping):
        meta = dict(raw)
    elif raw is None or raw == "":
        meta = {}
    elif isinstance(raw, str):
        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except ValueError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                parsed = None
        meta = dict(parsed) if isinstance(parsed, Mapping) else {"_raw": raw}
    else:
        meta = {"_raw": raw}
    inner = meta.get("metadata")
    if isinstance(inner, str) and inner.strip().startswith("{"):
        parsed_inner = _metadata(inner)
        if "_raw" not in parsed_inner:
            meta["metadata"] = parsed_inner
    return meta


def _field(meta: Mapping[str, Any], key: str) -> Any:
    value = meta.get(key)
    if value in (None, "") and isinstance(meta.get("metadata"), Mapping):
        value = meta["metadata"].get(key)
    return None if value in (None, "") else value


def _row(chunk: ChunkRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, ChunkRecord):
        raw = {
            "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
            "source_file": chunk.source_file, "doc_chunk_index": chunk.doc_chunk_index,
            "text": chunk.text, "content_module_id": chunk.content_module_id,
            "metadata": chunk.metadata,
        }
    else:
        raw = dict(chunk)
    meta = _metadata(raw.get("metadata"))
    text = str(raw.get("text") or "")
    source_file = str(raw.get("source_file") or "")
    doc_id = str(raw.get("doc_id") or source_file)
    module = raw.get("content_module_id")
    task_id = _field(meta, "task_id") or (str(module) if module else None)
    index = raw.get("doc_chunk_index")
    return {
        "chunk_id": str(raw["chunk_id"]),
        "doc_id": doc_id,
        "doc_chunk_index": int(index) if index is not None else None,
        "source": str(_field(meta, "source_document_name") or source_file or doc_id),
        "source_file": source_file,
        "chunk_label": _str_or_none(_field(meta, "chunk_label")),
        "task_id": _str_or_none(task_id),
        "step_id": _str_or_none(_field(meta, "step_id")),
        "source_scope": _str_or_none(_field(meta, "source_scope")),
        "resource_type": _str_or_none(_field(meta, "resource_type")),
        "type": _str_or_none(_field(meta, "type")),
        "content_module_id": _str_or_none(module),
        "text": text,
        "plain": plain_text(text),
        "metadata": meta,
    }


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Copy groups
# ---------------------------------------------------------------------------

def _groups_by_key(rows: list[dict], key) -> list[list[str]]:
    by: dict[str, list[str]] = {}
    for row in rows:
        by.setdefault(key(row), []).append(row["chunk_id"])
    return [sorted(ids) for ids in by.values() if len(ids) > 1]


def _shingles(key: str) -> set[str]:
    words = key.split(" ")
    n = OVERLAP_SHINGLE
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _overlap_pairs(reps: list[tuple[str, str]]
                   ) -> list[tuple[str, str, int, float, float]]:
    """(a, b, shared shingles, containment of the smaller, share of the larger)
    for chunk pairs whose
    smaller chunk has >= OVERLAP_CONTAINMENT of its shingles in the other.
    ``reps`` is (chunk_id, near key: the words joined by single spaces)."""
    sampled: list[set[int]] = []
    sizes: list[int] = []
    postings: dict[int, list[int]] = {}
    for i, (_cid, key) in enumerate(reps):
        shingles = _shingles(key)
        sizes.append(len(shingles))
        mine = set()
        for shingle in shingles:
            h = zlib.crc32(shingle.encode("utf-8"))
            if h % OVERLAP_SAMPLE == 0:
                mine.add(h)
        sampled.append(mine)
        for h in mine:
            postings.setdefault(h, []).append(i)

    cache: dict[int, set[str]] = {}

    def full(i: int) -> set[str]:
        if i not in cache:
            if len(cache) > 512:
                cache.clear()
            cache[i] = _shingles(reps[i][1])
        return cache[i]

    pairs = []
    for i, mine in enumerate(sampled):
        if not mine:
            continue
        counts: Counter[int] = Counter()
        for h in mine:
            posting = postings[h]
            if len(posting) > OVERLAP_MAX_POSTING:
                continue
            for j in posting:
                if j > i:
                    counts[j] += 1
        for j, shared_sampled in counts.items():
            smaller = min(len(mine), len(sampled[j]))
            if shared_sampled < 0.5 * smaller:
                continue
            shared = len(full(i) & full(j))
            containment = shared / min(sizes[i], sizes[j])
            if containment >= OVERLAP_CONTAINMENT:
                a, b = sorted((reps[i][0], reps[j][0]))
                larger = shared / max(sizes[i], sizes[j])
                pairs.append((a, b, shared, round(containment, 4), round(larger, 4)))
    return sorted(pairs)


def _components(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return [sorted(ids) for ids in groups.values()]


def copy_groups(rows: list[dict]) -> tuple[list[tuple[str, str, str]], dict[str, str],
                                           list[tuple[str, str, int, float, float]]]:
    """(group_id, kind, chunk_id) memberships, each chunk's representative, and
    the overlap pairs between representatives."""
    memberships: list[tuple[str, str, str]] = []
    ident_key = {row["chunk_id"]: _identical_key(row["text"]) for row in rows}
    identical = sorted(_groups_by_key(rows, lambda r: ident_key[r["chunk_id"]]))
    for n, ids in enumerate(identical, 1):
        memberships.extend((f"I{n}", "identical", cid) for cid in ids)

    # One string per chunk, not a list of words: a word list per chunk costs
    # far more memory on a large corpus.
    near_key = {row["chunk_id"]: " ".join(_near_words(row["text"])) for row in rows}
    near_all = _groups_by_key(rows, lambda r: near_key[r["chunk_id"]])
    # A near group adds something only when its members are not all one text.
    near = sorted(ids for ids in near_all if len({ident_key[c] for c in ids}) > 1)
    for n, ids in enumerate(near, 1):
        memberships.extend((f"N{n}", "near", cid) for cid in ids)

    rep_of = {row["chunk_id"]: row["chunk_id"] for row in rows}
    members_of: dict[str, list[str]] = {}
    for ids in near_all:
        for cid in ids:
            rep_of[cid] = ids[0]
    for cid, rep in rep_of.items():
        members_of.setdefault(rep, []).append(cid)

    reps = sorted((cid, near_key[cid]) for cid in members_of
                  if near_key[cid] and near_key[cid].count(" ") + 1 >= OVERLAP_MIN_WORDS)
    pairs = _overlap_pairs(reps)
    components = sorted(_components(
        (a, b) for a, b, _shared, _smaller, larger in pairs
        if larger >= OVERLAP_GROUP_LARGER))
    for n, rep_ids in enumerate(components, 1):
        ids = sorted(cid for rep in rep_ids for cid in members_of[rep])
        memberships.extend((f"O{n}", "overlap", cid) for cid in ids)
    return memberships, rep_of, pairs


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
  n INTEGER PRIMARY KEY,
  chunk_id TEXT NOT NULL UNIQUE,
  doc_id TEXT NOT NULL,
  doc_chunk_index INTEGER,
  source TEXT NOT NULL,
  source_file TEXT NOT NULL,
  chunk_label TEXT,
  task_id TEXT,
  step_id TEXT,
  source_scope TEXT,
  resource_type TEXT,
  type TEXT,
  content_module_id TEXT,
  text TEXT NOT NULL,
  plain TEXT,
  heading_before TEXT,
  headings TEXT NOT NULL,
  copy_rep TEXT NOT NULL,
  metadata TEXT NOT NULL
);
CREATE INDEX chunks_doc ON chunks (doc_id, doc_chunk_index);
CREATE INDEX chunks_task ON chunks (task_id);
CREATE INDEX chunks_resource_type ON chunks (resource_type);
CREATE INDEX chunks_type ON chunks (type);
CREATE INDEX chunks_source ON chunks (source COLLATE NOCASE);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  plain, source, content='', tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE chunks_trigram USING fts5(
  plain, content='', tokenize='trigram', detail=none);
CREATE VIEW chunks_plain AS SELECT n, chunk_id, COALESCE(plain, text) AS plain FROM chunks;
CREATE TABLE copy_groups (
  group_id TEXT NOT NULL, kind TEXT NOT NULL, chunk_id TEXT NOT NULL,
  PRIMARY KEY (kind, chunk_id));
CREATE INDEX copy_groups_group ON copy_groups (group_id);
CREATE TABLE overlap_pairs (
  chunk_a TEXT NOT NULL, chunk_b TEXT NOT NULL, shared_shingles INTEGER NOT NULL,
  containment REAL NOT NULL, containment_larger REAL NOT NULL,
  PRIMARY KEY (chunk_a, chunk_b));
"""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("rag-connector")
    except PackageNotFoundError:
        return "unknown"


def write_search_db(chunks: Iterable[ChunkRecord | Mapping[str, Any]],
                    path: str | os.PathLike[str], *,
                    meta: Mapping[str, Any] | None = None) -> Path:
    """Write a search index over ``chunks`` to ``path`` and return the path.

    The low-level writer: no Dataset is read, and ``meta`` is recorded as
    given (:func:`build` passes the Dataset's identity). The file is written
    beside ``path`` and moved into place when complete, so a reader never sees
    half an index, and a rebuild replaces it whole.
    """
    rows = [_row(chunk) for chunk in chunks]
    seen: set[str] = set()
    for row in rows:
        if row["chunk_id"] in seen:
            raise ValueError(f"duplicate chunk_id {row['chunk_id']!r}")
        seen.add(row["chunk_id"])
    rows.sort(key=lambda r: (r["doc_id"], r["doc_chunk_index"] is None,
                             r["doc_chunk_index"] or 0, r["chunk_id"]))

    # Headings: the nearest one before each chunk, carried along its document.
    carry: dict[str, str | None] = {}
    for row in rows:
        found = headings_in(row["plain"])
        row["heading_before"] = carry.get(row["doc_id"])
        row["headings"] = found
        if found:
            carry[row["doc_id"]] = found[-1][1]

    memberships, rep_of, pairs = copy_groups(rows)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.building-{os.getpid()}")
    if staging.exists():
        staging.unlink()
    conn = sqlite3.connect(staging)
    try:
        conn.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
        conn.executescript(SCHEMA)
        for n, row in enumerate(rows, 1):
            conn.execute(
                "INSERT INTO chunks (n, chunk_id, doc_id, doc_chunk_index, source, "
                "source_file, chunk_label, task_id, step_id, source_scope, "
                "resource_type, type, content_module_id, text, plain, heading_before, "
                "headings, copy_rep, metadata) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (n, row["chunk_id"], row["doc_id"], row["doc_chunk_index"], row["source"],
                 row["source_file"], row["chunk_label"], row["task_id"], row["step_id"],
                 row["source_scope"], row["resource_type"], row["type"],
                 row["content_module_id"], row["text"],
                 None if row["plain"] == row["text"] else row["plain"],
                 row["heading_before"],
                 json.dumps(row["headings"], ensure_ascii=False),
                 rep_of[row["chunk_id"]],
                 json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True,
                            default=str)))
            conn.execute("INSERT INTO chunks_fts (rowid, plain, source) VALUES (?,?,?)",
                         (n, row["plain"], row["source"]))
            conn.execute("INSERT INTO chunks_trigram (rowid, plain) VALUES (?,?)",
                         (n, row["plain"]))
        conn.executemany("INSERT INTO copy_groups VALUES (?,?,?)", memberships)
        conn.executemany("INSERT INTO overlap_pairs VALUES (?,?,?,?,?)", pairs)
        info = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "built_at": _now(),
            "tool": "rag-connector",
            "tool_version": _tool_version(),
            "sqlite_version": sqlite3.sqlite_version,
            "chunk_count": str(len(rows)),
            "document_count": str(len({r["doc_id"] for r in rows})),
            "word_tokenizer": "unicode61 remove_diacritics 2",
            "substring_tokenizer": "trigram",
            "searched_text": "COALESCE(plain, text): the text with Markdown backslash "
                             "escapes removed (plain is NULL where that changes nothing)",
            "overlap": json.dumps({
                "shingle_words": OVERLAP_SHINGLE, "containment": OVERLAP_CONTAINMENT,
                "group_larger": OVERLAP_GROUP_LARGER,
                "min_words": OVERLAP_MIN_WORDS, "sample": OVERLAP_SAMPLE,
                "max_posting": OVERLAP_MAX_POSTING}),
        }
        info.update({key: str(value) for key, value in (meta or {}).items()
                     if value is not None})
        conn.executemany("INSERT INTO meta VALUES (?,?)", sorted(info.items()))
        conn.commit()
        conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('optimize')")
        conn.execute("INSERT INTO chunks_trigram (chunks_trigram) VALUES ('optimize')")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    except BaseException:
        conn.close()
        staging.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(staging, target)
    return target


def search_db_path(dataset_dir: str | os.PathLike[str]) -> Path:
    """``<dataset>/analysis/rag-connector/search.sqlite``."""
    from .corpus_index import analysis_dir
    return analysis_dir(dataset_dir) / SEARCH_DB


def build(dataset_dir: str | os.PathLike[str], out: str | os.PathLike[str] | None = None,
          *, dataset: Dataset | None = None) -> Path:
    """Build the search index for a Dataset and return the file's path.

    Without ``out`` the file goes in the Dataset's analysis area,
    ``analysis/rag-connector/search.sqlite``, and the area's envelope is
    rewritten to list it (an area built on another core is replaced first,
    as for the book index). With ``out`` (a file path, or a directory to hold
    ``search.sqlite``) nothing in the Dataset folder is written. The core is
    never written. ``dataset`` skips re-reading a Dataset already read.
    """
    from . import corpus_index

    folder = Path(dataset_dir)
    if out is None:
        ds = corpus_index.open_area(folder)
        target = search_db_path(folder)
    else:
        ds = dataset if dataset is not None else read_dataset_folder(folder)
        target = Path(out)
        if target.is_dir():
            target = target / SEARCH_DB
    if ds.core_sha256 is None:
        raise SearchIndexError(f"{folder} records no core_sha256 to build against")
    meta = {
        "dataset_id": ds.dataset_id,
        "dataset_name": ds.name,
        "core_sha256": ds.core_sha256,
        "chunk_inventory_sha256": ds.chunk_inventory_sha256,
        "chunks_file_bytes": (folder / CHUNKS_FILE).stat().st_size,
    }
    path = write_search_db(ds.chunks, target, meta=meta)
    if out is None:
        corpus_index.write_envelope(folder)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def open_index(dataset_dir: str | os.PathLike[str], *,
               path: str | os.PathLike[str] | None = None,
               allow_stale: bool = False) -> SearchIndex:
    """Open a Dataset's search index read-only.

    ``path`` reads an index built elsewhere (``build(..., out=)``); it is still
    checked against ``dataset_dir``. Raises :class:`SearchIndexMissing` when
    there is no file and :class:`StaleSearchIndex` when the Dataset's recorded
    ``core_sha256`` (or the size of ``data/chunks.jsonl``) differs from what
    the index was built from, unless ``allow_stale``. The check reads the
    manifest, not the whole core; :meth:`SearchIndex.verify` re-reads it.
    """
    folder = Path(dataset_dir)
    target = Path(path) if path is not None else search_db_path(folder)
    if target.is_dir():
        target = target / SEARCH_DB
    if not target.is_file():
        raise SearchIndexMissing(
            f"no search index at {target}; `rag-connector index search-db "
            f"--dataset {folder}` builds one")
    index = SearchIndex(target)
    index.dataset_dir = folder
    problems = index.stale_reasons(folder)
    if problems and not allow_stale:
        index.close()
        raise StaleSearchIndex(
            f"the search index at {target} was not built from this Dataset: "
            + "; ".join(problems) + ". Rebuild it with `rag-connector index "
            f"search-db --dataset {folder}`.")
    return index


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _fts_phrase(term: str) -> str:
    """One user term as a safe FTS5 phrase. Several words are a phrase; a
    trailing ``*`` is a prefix. Everything else is inside double quotes, so
    no user text is read as FTS5 syntax."""
    term = " ".join(str(term).split())
    prefix = term.endswith("*")
    core = term.rstrip("*").strip()
    if not _WORDS.search(core):
        raise SearchIndexError(
            f"{term!r} has no letters or digits for word search; use a substring")
    return _quote(core) + (" *" if prefix else "")


def _term_pattern(term: str) -> re.Pattern | None:
    term = " ".join(str(term).split())
    prefix = term.endswith("*")
    words = _WORDS.findall(term.rstrip("*"))
    if not words:
        return None
    body = r"\W+".join(re.escape(w) for w in words)
    return re.compile(r"\b" + body + (r"\w*" if prefix else r"\b"), re.IGNORECASE)


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


_REGEX_CACHE: dict[str, re.Pattern] = {}


def _regexp(pattern: str, value: str | None) -> bool:
    if value is None:
        return False
    compiled = _REGEX_CACHE.get(pattern)
    if compiled is None:
        compiled = _REGEX_CACHE[pattern] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    return compiled.search(value) is not None


def _contains(value: str | None, needle: str) -> bool:
    return value is not None and needle.casefold() in value.casefold()


@dataclass
class SearchResult:
    """What a search found. ``total`` counts every hit; ``hits`` holds them all
    unless the caller passed ``limit``. The counts cover every hit."""

    total: int
    hits: list[dict[str, Any]]
    by_doc: list[dict[str, Any]]
    by_module: list[dict[str, Any]]
    query: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return len(self.hits) < self.total

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "shown": len(self.hits), "query": self.query,
                "by_module": self.by_module, "by_doc": self.by_doc, "hits": self.hits}


_HIT_COLUMNS = ("n", "chunk_id", "doc_id", "doc_chunk_index", "chunk_label", "source",
                "task_id", "step_id", "source_scope", "resource_type", "type",
                "COALESCE(plain, text) AS plain", "heading_before", "headings")


class SearchIndex:
    """A search index file, open read-only."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.dataset_dir: Path | None = None
        self.conn = _connect_ro(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function("regexp", 2, _regexp, deterministic=True)
        self.conn.create_function("contains", 2, _contains, deterministic=True)
        try:
            self.meta = {row["key"]: row["value"]
                         for row in self.conn.execute("SELECT key, value FROM meta")}
        except sqlite3.DatabaseError as exc:
            self.conn.close()
            raise SearchIndexError(f"{self.path} is not a search index: {exc}") from exc
        if self.meta.get("format") != FORMAT:
            self.conn.close()
            raise SearchIndexError(f"{self.path} is not a {FORMAT} file")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SearchIndex:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- freshness ---------------------------------------------------------

    def stale_reasons(self, dataset_dir: str | os.PathLike[str]) -> list[str]:
        """Why this index does not describe the Dataset (empty when it does)."""
        folder = Path(dataset_dir)
        problems = []
        if self.meta.get("format_version") != FORMAT_VERSION:
            problems.append(f"format version {self.meta.get('format_version')!r}, "
                            f"this reader reads {FORMAT_VERSION!r}")
        try:
            manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return problems + [f"cannot read {folder / MANIFEST_FILE}: {exc}"]
        core = manifest.get("core_sha256")
        if core != self.meta.get("core_sha256"):
            problems.append(f"built from core {self.meta.get('core_sha256')}, "
                            f"the Dataset records {core}")
        try:
            size = (folder / CHUNKS_FILE).stat().st_size
        except OSError as exc:
            return problems + [f"cannot read {folder / CHUNKS_FILE}: {exc}"]
        if str(size) != self.meta.get("chunks_file_bytes"):
            problems.append(f"{CHUNKS_FILE} is {size} bytes, the index was built "
                            f"from {self.meta.get('chunks_file_bytes')} bytes")
        return problems

    def verify(self, dataset_dir: str | os.PathLike[str] | None = None) -> None:
        """Re-read and verify the whole Dataset and compare its chunk inventory."""
        folder = Path(dataset_dir or self.dataset_dir or "")
        dataset = read_dataset_folder(folder)
        if dataset.chunk_inventory_sha256 != self.meta.get("chunk_inventory_sha256"):
            raise StaleSearchIndex(
                f"chunk inventory {dataset.chunk_inventory_sha256} differs from the "
                f"index's {self.meta.get('chunk_inventory_sha256')}")

    # -- ids -----------------------------------------------------------------

    def _by_prefix(self, column: str, table: str, value: str, what: str) -> list[str]:
        value = str(value).strip()
        if not value:
            raise AmbiguousIdError(f"an empty {what} id")
        exact = self.conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (value,)).fetchone()
        if exact:
            return [value]
        rows = self.conn.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE {column} >= ? AND {column} < ? "
            f"ORDER BY {column} LIMIT 6", (value, value + "\U0010ffff")).fetchall()
        return [row[0] for row in rows]

    def resolve_chunk(self, chunk: str) -> str:
        """A chunk id from an id or an unambiguous prefix of one."""
        found = self._by_prefix("chunk_id", "chunks", chunk, "chunk")
        if len(found) != 1:
            raise AmbiguousIdError(_ambiguity("chunk", chunk, found))
        return found[0]

    def resolve_module(self, module: str) -> str:
        """A module (``task_id``) from an id or an unambiguous prefix of one."""
        found = self._by_prefix("task_id", "chunks", module, "module")
        if len(found) != 1:
            raise AmbiguousIdError(_ambiguity("module", module, found))
        return found[0]

    def resolve_doc(self, doc: str) -> list[str]:
        """Document ids for a doc id, an unambiguous doc id prefix, or an exact
        title (case-insensitive; a title may name several documents)."""
        found = self._by_prefix("doc_id", "chunks", doc, "document")
        if len(found) == 1:
            return found
        titled = [row[0] for row in self.conn.execute(
            "SELECT DISTINCT doc_id FROM chunks WHERE source = ? COLLATE NOCASE "
            "ORDER BY doc_id", (str(doc).strip(),))]
        if titled:
            return titled
        raise AmbiguousIdError(_ambiguity("document", doc, found))

    # -- search ------------------------------------------------------------

    def search(self, all: Sequence[str] | str = (), any: Sequence[str] | str = (),  # noqa: A002
               not_: Sequence[str] | str = (), phrase: str | Sequence[str] | None = None,
               near: Sequence[Any] | None = None, regex: str | Sequence[str] | None = None,
               substring: str | Sequence[str] | None = None, word: bool = True,
               modules: Sequence[str] | str = (), general_only: bool = False,
               docs: Sequence[str] | str = (), exclude_docs: Sequence[str] | str = (),
               types: Sequence[str] | str = (), titles: bool = False,
               limit: int | None = None) -> SearchResult:
        """Chunks matching every condition given.

        ``all``: every term; ``any``: at least one; ``not_``: none. A term of
        several words is a phrase and a trailing ``*`` makes a prefix. With
        ``word=True`` (default) terms are whole words (accents ignored);
        with ``word=False`` they are case-insensitive substrings. ``phrase``:
        exact word sequence(s). ``near``: ``(term, term, ..., N)``, all terms
        within N words of each other. ``regex``: Python regular
        expression(s), case-insensitive unless the pattern says ``(?-i)``.
        ``substring``: case-insensitive substring(s), punctuation included.
        All of these search ``plain``; ``titles=True`` also lets word terms
        match the document title.

        Filters: ``modules`` (task ids or unambiguous prefixes),
        ``general_only`` (chunks with no module; with ``modules`` too, those
        modules plus general), ``docs`` / ``exclude_docs`` (doc ids, prefixes,
        or exact titles), ``types`` (``resource_type`` or ``type``).

        Every hit is returned unless ``limit`` is given; ``total`` always
        counts them all. Ranked by bm25 when there is a word condition, else
        in document order.
        """
        all_terms, any_terms, not_terms = _as_list(all), _as_list(any), _as_list(not_)
        phrases, regexes, substrings = (_as_list(phrase), _as_list(regex),
                                        _as_list(substring))
        where: list[str] = []
        params: list[Any] = []
        fts_parts: list[str] = []
        fts_not: list[str] = []
        patterns: list[re.Pattern] = []
        substr_needles: list[str] = list(substrings)

        def substring_condition(needle: str) -> tuple[str, list[Any]]:
            # The trigram index narrows to chunks holding every trigram of the
            # needle; contains() then checks the needle itself.
            trigrams = list(dict.fromkeys(needle[i:i + 3] for i in range(len(needle) - 2)))
            if trigrams:
                return ("(c.n IN (SELECT rowid FROM chunks_trigram WHERE chunks_trigram "
                        "MATCH ?) AND contains(COALESCE(c.plain, c.text), ?))",
                        [" AND ".join(_quote(t) for t in trigrams), needle])
            return "contains(COALESCE(c.plain, c.text), ?)", [needle]

        if word:
            fts_parts += [_fts_phrase(t) for t in all_terms]
            if any_terms:
                fts_parts.append("(" + " OR ".join(_fts_phrase(t) for t in any_terms) + ")")
            fts_not += [_fts_phrase(t) for t in not_terms]
            patterns += [p for p in map(_term_pattern, all_terms + any_terms) if p]
        else:
            for needle in all_terms:
                sql, args = substring_condition(needle)
                where.append(sql)
                params += args
            if any_terms:
                ors = [substring_condition(needle) for needle in any_terms]
                where.append("(" + " OR ".join(sql for sql, _ in ors) + ")")
                params += [a for _, args in ors for a in args]
            for needle in not_terms:
                sql, args = substring_condition(needle)
                where.append(f"NOT ({sql})")
                params += args
            substr_needles += all_terms + any_terms
        for text in phrases:
            words = _WORDS.findall(text)
            if not words:
                raise SearchIndexError(f"phrase {text!r} has no words; use a substring")
            fts_parts.append(_quote(" ".join(words)))
            patterns.append(_term_pattern(text))
        if near:
            *near_terms, distance = list(near)
            if len(near_terms) < 2 or not isinstance(distance, int) or distance < 0:
                raise SearchIndexError("near is (term, term, ..., N) with N >= 0 words")
            fts_parts.append("NEAR(" + " ".join(_fts_phrase(t) for t in near_terms)
                             + f", {distance})")
            patterns += [p for p in map(_term_pattern, near_terms) if p]
        for pattern in regexes:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise SearchIndexError(f"bad regular expression {pattern!r}: {exc}") from exc
            where.append("regexp(?, COALESCE(c.plain, c.text))")
            params.append(pattern)
        for needle in substrings:
            sql, args = substring_condition(needle)
            where.append(sql)
            params += args

        resolved_modules = [self.resolve_module(m) for m in _as_list(modules)]
        if resolved_modules or general_only:
            ors = []
            if resolved_modules:
                ors.append(f"c.task_id IN ({_marks(resolved_modules)})")
                params += resolved_modules
            if general_only:
                ors.append("c.task_id IS NULL")
            where.append("(" + " OR ".join(ors) + ")")
        doc_ids = [d for doc in _as_list(docs) for d in self.resolve_doc(doc)]
        if doc_ids:
            where.append(f"c.doc_id IN ({_marks(doc_ids)})")
            params += doc_ids
        excluded = [d for doc in _as_list(exclude_docs) for d in self.resolve_doc(doc)]
        if excluded:
            where.append(f"c.doc_id NOT IN ({_marks(excluded)})")
            params += excluded
        type_list = _as_list(types)
        if type_list:
            where.append(f"(c.resource_type IN ({_marks(type_list)}) "
                         f"OR c.type IN ({_marks(type_list)}))")
            params += type_list + type_list

        column = "{plain source}" if titles else "plain"
        if fts_parts:
            expr = " AND ".join(fts_parts)
            if fts_not:
                expr = f"({expr}) NOT ({' OR '.join(fts_not)})"
            match = f"{column} : ({expr})"
            weights = "1.0, 1.0" if titles else "1.0, 0.0"
            sql = (f"SELECT c.n, bm25(chunks_fts, {weights}) AS score FROM chunks_fts "
                   f"JOIN chunks c ON c.n = chunks_fts.rowid WHERE chunks_fts MATCH ?")
            all_params = [match] + params
            if where:
                sql += " AND " + " AND ".join(where)
            sql += " ORDER BY score, c.n"
        else:
            if fts_not:
                where.append("c.n NOT IN (SELECT rowid FROM chunks_fts WHERE chunks_fts "
                             "MATCH ?)")
                params.append(f"{column} : ({' OR '.join(fts_not)})")
            if not where:
                raise SearchIndexError("give at least one search term or filter")
            sql = ("SELECT c.n, NULL AS score FROM chunks c WHERE " + " AND ".join(where)
                   + " ORDER BY c.source, c.doc_id, c.doc_chunk_index, c.chunk_id")
            all_params = params
        try:
            ranked = self.conn.execute(sql, all_params).fetchall()
        except sqlite3.OperationalError as exc:
            raise SearchIndexError(f"search failed: {exc}") from exc

        order = [row["n"] for row in ranked]
        scores = {row["n"]: row["score"] for row in ranked}
        rows = self._rows_by_n(order)
        by_doc: dict[str, dict[str, Any]] = {}
        by_module: Counter[str | None] = Counter()
        for row in rows:
            entry = by_doc.setdefault(row["doc_id"], {"doc_id": row["doc_id"],
                                                      "source": row["source"],
                                                      "task_id": row["task_id"], "hits": 0})
            entry["hits"] += 1
            by_module[row["task_id"]] += 1
        shown = rows if limit is None else rows[:max(0, int(limit))]
        groups = self._groups_for([row["chunk_id"] for row in shown])
        regex_compiled = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in regexes]
        hits = []
        for row in shown:
            hits.append(_hit(row, scores.get(row["n"]), groups.get(row["chunk_id"], {}),
                             patterns, regex_compiled, substr_needles))
        query = {key: value for key, value in {
            "all": all_terms, "any": any_terms, "not": not_terms, "phrase": phrases,
            "near": list(near) if near else None, "regex": regexes,
            "substring": substrings, "word": word, "modules": resolved_modules,
            "general_only": general_only, "docs": doc_ids, "exclude_docs": excluded,
            "types": type_list, "titles": titles, "limit": limit,
        }.items() if value not in (None, [], False) or key == "word"}
        return SearchResult(
            total=len(rows), hits=hits,
            by_doc=sorted(by_doc.values(), key=lambda d: (-d["hits"], d["source"],
                                                          d["doc_id"])),
            by_module=[{"task_id": task, "hits": count} for task, count in
                       sorted(by_module.items(), key=lambda kv: (-kv[1], kv[0] or ""))],
            query=query)

    def _rows_by_n(self, order: list[int]) -> list[sqlite3.Row]:
        found: dict[int, sqlite3.Row] = {}
        cols = ", ".join(_HIT_COLUMNS)
        for start in range(0, len(order), 500):
            batch = order[start:start + 500]
            for row in self.conn.execute(
                    f"SELECT {cols} FROM chunks WHERE n IN ({_marks(batch)})", batch):
                found[row["n"]] = row
        return [found[n] for n in order]

    def _groups_for(self, chunk_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        group_ids: set[str] = set()
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start:start + 500]
            for row in self.conn.execute(
                    "SELECT group_id, kind, chunk_id FROM copy_groups "
                    f"WHERE chunk_id IN ({_marks(batch)})", batch):
                out.setdefault(row["chunk_id"], {})[row["kind"]] = {"group": row["group_id"]}
                group_ids.add(row["group_id"])
        sizes: dict[str, int] = {}
        ids = sorted(group_ids)
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            for row in self.conn.execute(
                    "SELECT group_id, COUNT(*) AS size FROM copy_groups "
                    f"WHERE group_id IN ({_marks(batch)}) GROUP BY group_id", batch):
                sizes[row["group_id"]] = row["size"]
        for kinds in out.values():
            for info in kinds.values():
                info["size"] = sizes.get(info["group"], 0)
        return out

    # -- reading chunks ------------------------------------------------------

    def chunk(self, chunk: str) -> dict[str, Any]:
        """One chunk, whole: every column, with metadata and headings parsed."""
        cid = self.resolve_chunk(chunk)
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (cid,)).fetchone()
        return _full(row)

    def neighbors(self, chunk: str, radius: int = 1) -> list[dict[str, Any]]:
        """The chunk and up to ``radius`` chunks either side in its document,
        in document order, each whole. ``target`` marks the one asked for."""
        cid = self.resolve_chunk(chunk)
        here = self.conn.execute("SELECT doc_id, doc_chunk_index FROM chunks "
                                 "WHERE chunk_id = ?", (cid,)).fetchone()
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY doc_chunk_index, chunk_id",
            (here["doc_id"],)).fetchall()
        position = next(i for i, row in enumerate(rows) if row["chunk_id"] == cid)
        radius = max(0, int(radius))
        out = []
        for row in rows[max(0, position - radius):position + radius + 1]:
            item = _full(row)
            item["target"] = row["chunk_id"] == cid
            out.append(item)
        return out

    neighbours = neighbors

    def outline(self, doc: str) -> list[dict[str, Any]]:
        """A document's chunks in order: index, label, first heading, the
        heading carried in from earlier chunks, and size. ``doc`` is a doc
        id, an unambiguous prefix, an exact title, or a chunk id in it."""
        try:
            doc_ids = self.resolve_doc(doc)
        except AmbiguousIdError:
            cid = self.resolve_chunk(doc)
            doc_ids = [self.conn.execute("SELECT doc_id FROM chunks WHERE chunk_id = ?",
                                         (cid,)).fetchone()[0]]
        if len(doc_ids) != 1:
            raise AmbiguousIdError(_ambiguity("document", doc, doc_ids))
        out = []
        for row in self.conn.execute(
                "SELECT chunk_id, doc_id, source, doc_chunk_index, chunk_label, task_id, "
                "heading_before, headings, length(COALESCE(plain, text)) AS chars FROM chunks "
                "WHERE doc_id = ? ORDER BY doc_chunk_index, chunk_id", (doc_ids[0],)):
            headings = json.loads(row["headings"])
            out.append({
                "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
                "source": row["source"], "doc_chunk_index": row["doc_chunk_index"],
                "chunk_label": row["chunk_label"], "task_id": row["task_id"],
                "heading": headings[0][1] if headings else None,
                "headings": [title for _offset, title in headings],
                "heading_before": row["heading_before"], "chars": row["chars"],
            })
        return out

    def copies(self, chunk: str) -> dict[str, Any]:
        """The chunk's copies by kind: ``identical`` (same text, whitespace
        aside), ``near`` (same once case, punctuation, accents and encoding
        damage are ignored) and ``overlap``.

        ``overlap`` lists the chunk's overlap group (the same passage chunked
        differently or lightly reworded) and every chunk it was paired with
        directly, including a longer chunk it merely sits inside. Each carries
        ``grouped``, and, when the two were compared directly, ``containment``
        (the share of the smaller chunk's shingles in the other) and
        ``containment_larger`` (the same count over the larger chunk); both
        are ``None`` for a group member linked only through another chunk.
        """
        cid = self.resolve_chunk(chunk)
        rep = self.conn.execute("SELECT copy_rep FROM chunks WHERE chunk_id = ?",
                                (cid,)).fetchone()[0]
        out: dict[str, Any] = {"chunk_id": cid, "identical": [], "near": [], "overlap": []}
        member_sql = ("SELECT c.chunk_id, c.doc_id, c.source, c.doc_chunk_index, c.task_id, "
                      "c.copy_rep FROM chunks c ")
        grouped: set[str] = set()
        overlap_rows: dict[str, sqlite3.Row] = {}
        for kind, group in self.conn.execute(
                "SELECT kind, group_id FROM copy_groups WHERE chunk_id = ?", (cid,)):
            members = self.conn.execute(
                member_sql + "JOIN copy_groups g ON g.chunk_id = c.chunk_id "
                "WHERE g.group_id = ? AND c.chunk_id != ? ORDER BY c.source, c.chunk_id",
                (group, cid)).fetchall()
            out[f"{kind}_group"] = group
            if kind == "overlap":
                grouped = {m["chunk_id"] for m in members}
                overlap_rows.update({m["chunk_id"]: m for m in members})
            else:
                out[kind] = [_copy_item(m) for m in members]
        direct: dict[str, tuple[float, float]] = {}
        for a, b, smaller, larger in self.conn.execute(
                "SELECT chunk_a, chunk_b, containment, containment_larger FROM "
                "overlap_pairs WHERE chunk_a = ? OR chunk_b = ?", (rep, rep)):
            direct[b if a == rep else a] = (smaller, larger)
        if direct:
            reps = sorted(direct)
            for m in self.conn.execute(
                    member_sql + f"WHERE c.copy_rep IN ({_marks(reps)}) AND c.chunk_id != ?",
                    [*reps, cid]):
                overlap_rows.setdefault(m["chunk_id"], m)
        items = []
        for m in sorted(overlap_rows.values(), key=lambda m: (m["source"], m["chunk_id"])):
            item = _copy_item(m)
            pair = (1.0, 1.0) if m["copy_rep"] == rep else direct.get(m["copy_rep"])
            item["grouped"] = m["chunk_id"] in grouped
            item["containment"], item["containment_larger"] = pair or (None, None)
            items.append(item)
        out["overlap"] = items
        return out

    def modules(self) -> list[dict[str, Any]]:
        """Every module (task_id; ``None`` for general) with its chunk count."""
        return [{"task_id": row[0], "chunks": row[1], "documents": row[2]}
                for row in self.conn.execute(
                    "SELECT task_id, COUNT(*), COUNT(DISTINCT doc_id) FROM chunks "
                    "GROUP BY task_id ORDER BY task_id")]


def _marks(values: Sequence[Any]) -> str:
    return ",".join("?" * len(values))


def _ambiguity(what: str, given: str, found: list[str]) -> str:
    if not found:
        return f"no {what} id starts with {given!r}"
    shown = ", ".join(found[:5])
    more = " ..." if len(found) > 5 else ""
    return f"{what} {given!r} is ambiguous: {shown}{more}; give more characters"


def _copy_item(row: sqlite3.Row) -> dict[str, Any]:
    return {"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "source": row["source"],
            "doc_chunk_index": row["doc_chunk_index"], "task_id": row["task_id"]}


def _full(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item.pop("n", None)
    if item.get("plain") is None:
        item["plain"] = item["text"]
    item["metadata"] = json.loads(item["metadata"])
    item["headings"] = [title for _offset, title in json.loads(item["headings"])]
    return item


def _first_position(plain: str, patterns: list[re.Pattern], regexes: list[re.Pattern],
                    needles: list[str]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for pattern in [*regexes, *patterns]:
        match = pattern.search(plain)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.end())
    folded = plain.casefold()
    for needle in needles:
        at = folded.find(needle.casefold())
        if at >= 0 and (best is None or at < best[0]):
            best = (at, at + len(needle))
    return best


def _snippet(plain: str, span: tuple[int, int] | None) -> str:
    start, end = span or (0, 0)
    lo = max(0, start - SNIPPET_BEFORE)
    hi = min(len(plain), end + SNIPPET_AFTER)
    text = " ".join(plain[lo:hi].split())
    return ("..." if lo > 0 else "") + text + ("..." if hi < len(plain) else "")


def _hit(row: sqlite3.Row, score: float | None, groups: dict[str, Any],
         patterns: list[re.Pattern], regexes: list[re.Pattern],
         needles: list[str]) -> dict[str, Any]:
    plain = row["plain"]
    span = _first_position(plain, patterns, regexes, needles)
    heading = row["heading_before"]
    at = span[0] if span else 0
    for offset, title in json.loads(row["headings"]):
        if offset <= at:
            heading = title
    return {
        "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
        "doc_chunk_index": row["doc_chunk_index"], "chunk_label": row["chunk_label"],
        "source": row["source"], "task_id": row["task_id"], "step_id": row["step_id"],
        "source_scope": row["source_scope"], "resource_type": row["resource_type"],
        "type": row["type"], "heading": heading,
        "score": None if score is None else round(-score, 4),
        "snippet": _snippet(plain, span), "copies": groups,
    }

