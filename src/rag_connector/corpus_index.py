"""A book index over a Dataset: names, titles and headings, and the chunks that hold them.

Freezing the chunks and indexing them are separate. :func:`prepare` reads a
verified Dataset and writes, into the shared analysis area
``analysis/rag-connector/``, a term-to-chunk locator a person or an agent can
search the way it searches a book's index: a name, a document title, or a
section heading, and the chunk ids that hold it.

No model is called. Names and headings are taken from the chunk text.
Ordinary wording is not copied into the index; a reader still opens the
chunks, and uses the previous and next chunk id when the passage continues.
Identical copies of a chunk are one location, with the other ids beside it.

The index is evidence about the corpus, not corpus (spec rule 4): every
location is a ``chunk_id`` from the Dataset's own rows, and the area's
envelope records the ``core_sha256`` it was built from. The model-labelled
companion is :mod:`rag_connector.signal_index`.

See ``docs/dataset-spec.md`` ("Known tool areas").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import ChunkRecord
from .dataset import (
    ANALYSIS_DIR,
    ANALYSIS_ENVELOPE_FILE,
    Dataset,
    read_dataset_folder,
    write_analysis_envelope,
)

__all__ = [
    "ANALYSIS_TOOL",
    "analysis_dir",
    "build",
    "find",
    "format_find",
    "index_dir",
    "index_path",
    "load",
    "markdown_path",
    "prepare",
]

#: The analysis area this module writes: ``analysis/rag-connector/``.
ANALYSIS_TOOL = "rag-connector"
#: The format of the area's contents (index.json and its companions).
FORMAT_VERSION = "1"
INDEX_NAME = "index.json"
INDEX_MD = "index.md"
KIND = "rag-connector-corpus-index"
VERSION = 2
MAX_LOCATIONS = 12
MAX_COPIES = 8

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z", re.I)
_WORD = r"[A-Z][a-z]+(?:'[a-z]+)?"
_INITIAL = r"[A-Z]\."
_START = rf"(?:St\.|Ft\.|{_WORD})"
_CONT = rf"(?:(?:of|de|du|la|van|von)\s+)?(?:St\.|Ft\.|{_INITIAL}|{_WORD})"
_PHRASE = re.compile(rf"\b{_START}(?:\s+{_CONT}){{1,5}}\b")
_SINGLE = re.compile(rf"\b{_WORD}\b")
# A capitalized word and a number: "Route 66", "Building 4", "Sec. 12".
_NUMBERED = re.compile(r"\b([A-Z][a-z]+\.?)\s+(\d+[A-Za-z]?(?:\.\d+)?)\b")
_POSTAL = re.compile(r"\d{5}")
_TOWN = re.compile(r"(?:\s*[A-Z][A-Za-z.'-]*)*\s*")
# A year after one of these is a date, not part of a name ("Since 1947").
_BEFORE_YEAR = frozenset("""
since until through fall spring summer winter fiscal class copyright
established est founded year
""".split())
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_CONNECTORS = frozenset("of de du la van von".split())
_LEADING = frozenset("the a an any this these that those".split())
# Capitalized words that name nothing on their own: calendar words, document
# structure, and common English. General on purpose; a corpus that needs more
# passes them to build(stopwords=...), and the index records what it used.
_SINGLE_STOP = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec since until through
and but nor yet
chapter section sections article part page note table figure appendix introduction
paragraph slide item step line box suite
overview summary contents index manual guide
the this that these those each every following additional other another such
have has had do does did make made take taken give given get got
there here both name street contact address membership events announcements
bring build built break brief call come came going
shall upon pursuant said same under into over about after before
when where what which who how not from with your their its
""".split())


def analysis_dir(folder: str | os.PathLike[str]) -> Path:
    """``<dataset>/analysis/rag-connector/``: the shared area the indexes go in."""
    return Path(folder) / ANALYSIS_DIR / ANALYSIS_TOOL


def index_path(folder: str | os.PathLike[str]) -> Path:
    return analysis_dir(folder) / INDEX_NAME


def markdown_path(folder: str | os.PathLike[str]) -> Path:
    return analysis_dir(folder) / INDEX_MD


def index_dir(folder: str | os.PathLike[str]) -> Path:
    return analysis_dir(folder) / "index"


def _doc_of(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("doc_id") or chunk.get("source_file") or "")


def _rows(chunks: Iterable[ChunkRecord | Mapping[str, Any]]) -> list[dict]:
    """Chunk rows as dicts, without embeddings."""
    rows = []
    for chunk in chunks:
        row = chunk.to_dict() if isinstance(chunk, ChunkRecord) else dict(chunk)
        row.pop("embedding", None)
        rows.append(row)
    return rows


def human_source(source: str) -> bool:
    """A title a person can read. Step ids are two uuids joined by '::'."""
    if "::" in source:
        left, right = source.split("::", 1)
        if _UUID.fullmatch(left) and _UUID.fullmatch(right):
            return False
    return any(c.isalpha() for c in source)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _digest(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def _letter(term: str) -> str:
    ch = term.casefold()[:1]
    return ch if ch.isalpha() else "0"


def _good_heading(raw: str) -> str:
    title = " ".join(raw.strip().strip("#").split())
    words = title.split()
    if not title or len(title) > 80 or len(words) > 12:
        return ""
    if ". " in title:
        return ""
    if _YEAR.fullmatch(title):
        return ""
    return title


def _headings(text: str) -> list[str]:
    found = []
    for match in _HEADING.finditer(text):
        title = _good_heading(match.group(2))
        if title and title not in found:
            found.append(title)
    return found


def _line_initial(text: str, start: int) -> bool:
    """A capital that only starts a line or a sentence is a weak name."""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i >= 0 and text[i] == "|":
        return False
    if i < 0 or text[i] in "\n\r.!?":
        return True
    return False


def _after_address_comma(text: str, start: int) -> bool:
    """Only a town name between a comma and here: "12 Main St, Delray Beach 33483"."""
    comma = text.rfind(",", 0, start)
    return comma >= 0 and bool(_TOWN.fullmatch(text[comma + 1:start]))


def _numbered(text: str, stop: frozenset[str]):
    """Numbered names in ``text`` as (match, lead word, line-initial)."""
    for match in _NUMBERED.finditer(text):
        word, number = match.group(1), match.group(2)
        bare = word.rstrip(".").casefold()
        if len(bare) < 3 or bare in stop:
            continue
        if word.endswith(".") and len(bare) > 4:
            continue  # a sentence ending, then a list number; not "Sec." or "Bldg."
        if _YEAR.fullmatch(number) and bare in _BEFORE_YEAR:
            continue
        if _POSTAL.fullmatch(number) and _after_address_comma(text, match.start()):
            continue  # a town and its postal code
        yield match, bare, _line_initial(text, match.start())


def numbered_leads(texts: Iterable[str],
                   stop: frozenset[str] = _SINGLE_STOP) -> dict[str, tuple[bool, int]]:
    """How the corpus uses each lead word of a numbered name.

    For each word: whether it is ever used mid-sentence, and how many distinct
    numbers that are not year-like follow it. A capital that only starts a line
    ("Arrive 30 minutes early") is a weak name; one the corpus also uses
    mid-sentence ("... at Post 1966") is not. And a year-like number is part of
    a name only after a word the corpus numbers things with: "Post 1966" when
    there is also a "Post 12", but not "Chartered 1946".
    """
    mid: set[str] = set()
    numbers: dict[str, set[str]] = {}
    for text in texts:
        for match, bare, initial in _numbered(text, stop):
            if not initial:
                mid.add(bare)
            if not _YEAR.fullmatch(match.group(2)):
                numbers.setdefault(bare, set()).add(match.group(2))
    return {bare: (bare in mid, len(numbers.get(bare, ())))
            for bare in mid | set(numbers)}


def _phrase_surface(surface: str, stop: frozenset[str]) -> str:
    """Drop a leading article and a trailing initial. Keep a phrase that
    names something: at least two words, one of them not a common word."""
    parts = surface.split()
    while parts and parts[0].casefold().rstrip(".") in _LEADING:
        parts = parts[1:]
    while parts and (parts[-1].casefold().rstrip(".") in _LEADING | _CONNECTORS
                     or re.fullmatch(r"[A-Z]\.", parts[-1])):
        parts = parts[:-1]
    if len(parts) < 2:
        return ""
    content = [
        part for part in parts
        if part.casefold().rstrip(".") not in _CONNECTORS
        and not re.fullmatch(r"[A-Z]\.", part)
        and part.casefold() not in {"st.", "ft."}
    ]
    if not any(part.casefold() not in stop for part in content):
        return ""
    return " ".join(parts)


def _mentions(text: str, stop: frozenset[str] = _SINGLE_STOP,
              leads: Mapping[str, tuple[bool, int]] | None = None
              ) -> list[tuple[str, str]]:
    """(kind, surface) pairs. kind is 'entity'. Headings are handled apart."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    covered: list[tuple[int, int]] = []

    def add(kind: str, surface: str) -> None:
        surface = " ".join(surface.split()).strip(" ,;:")
        key = surface.casefold()
        if not surface or key in seen or _YEAR.fullmatch(surface):
            return
        seen.add(key)
        found.append((kind, surface))

    for match, bare, initial in _numbered(text, stop):
        if leads is not None:
            used_mid, counted = leads.get(bare, (False, 0))
            if initial and not used_mid:
                continue
            if _YEAR.fullmatch(match.group(2)) and counted < 2:
                continue
        add("entity", match.group(0))
        covered.append(match.span())
    for match in _PHRASE.finditer(text):
        surface = _phrase_surface(match.group(0), stop)
        if surface:
            add("entity", surface)
        covered.append(match.span())
    for match in _SINGLE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in covered):
            continue
        word = match.group(0)
        if len(word) < 4 or word.casefold() in stop or _line_initial(text, match.start()):
            continue
        add("entity", word)
    return found


def _location(chunk: dict, section: str, prev_id: str, next_id: str,
              doc_chunks: int) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "doc_id": _doc_of(chunk),
        "source": str(chunk.get("source_file") or _doc_of(chunk)),
        "section": section,
        "prev": prev_id,
        "next": next_id,
        "doc_chunks": doc_chunks,
        "copies": [],
    }


def _prefer_primary(chunks: list[dict]) -> dict:
    def rank(chunk: dict) -> tuple:
        source = str(chunk.get("source_file") or "")
        return (0 if human_source(source) else 1, source,
                int(chunk.get("doc_chunk_index") or 0), str(chunk["chunk_id"]))
    return min(chunks, key=rank)


def build(chunks: Iterable[ChunkRecord | Mapping[str, Any]], *,
          stopwords: Iterable[str] = ()) -> dict[str, Any]:
    """The index body: entries and threads. No dataset files are read.

    ``stopwords`` adds capitalized words that name nothing in this corpus (a
    title used on every page, say) to the general list.
    """
    stop = _SINGLE_STOP | {word.casefold() for word in stopwords}
    chunks = _rows(chunks)
    leads = numbered_leads((str(c.get("text") or "") for c in chunks), stop)
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(_doc_of(chunk), []).append(chunk)
    groups = []
    for doc_id, group in by_doc.items():
        group.sort(key=lambda c: (int(c.get("doc_chunk_index") or 0), str(c["chunk_id"])))
        groups.append((doc_id, group))
    groups.sort(key=lambda item: (str(item[1][0].get("source_file") or item[0]), item[0]))

    text_groups: dict[str, list[dict]] = {}
    for _, group in groups:
        for chunk in group:
            text_groups.setdefault(_digest(str(chunk.get("text") or "")), []).append(chunk)
    primary_of = {digest: _prefer_primary(same) for digest, same in text_groups.items()}

    raw_entries: dict[str, dict[str, Any]] = {}
    raw_threads: list[dict[str, Any]] = []

    def add_entry(kind: str, surface: str, loc: dict[str, Any], digest: str) -> None:
        key = " ".join(surface.split()).casefold()
        entry = raw_entries.setdefault(
            key, {"term": " ".join(surface.split()), "kind": kind, "hits": []})
        if kind == "topic":
            entry["kind"] = "topic"
            if human_source(surface):
                entry["term"] = " ".join(surface.split())
        entry["hits"].append((digest, loc))

    for _, group in groups:
        ids = [str(c["chunk_id"]) for c in group]
        current: dict[str, Any] | None = None
        section_of: dict[str, str] = {}
        for index, chunk in enumerate(group):
            cid = ids[index]
            headings = _headings(str(chunk.get("text") or ""))
            if not headings:
                if current is not None and cid not in current["chunk_ids"]:
                    current["chunk_ids"].append(cid)
            else:
                if current is not None and cid not in current["chunk_ids"]:
                    current["chunk_ids"].append(cid)
                for title in headings:
                    current = {
                        "title": title,
                        "doc_id": _doc_of(chunk),
                        "source": str(chunk.get("source_file") or _doc_of(chunk)),
                        "chunk_ids": [cid],
                        "digest": "",
                    }
                    raw_threads.append(current)
            section_of[cid] = current["title"] if current else ""
        for thread in raw_threads:
            if thread["doc_id"] == _doc_of(group[0]) and not thread["digest"]:
                body = "\n".join(str(c.get("text") or "") for c in group
                                 if str(c["chunk_id"]) in thread["chunk_ids"])
                thread["digest"] = _digest(thread["title"] + "\n" + body)

        source = str(group[0].get("source_file") or "")
        for index, chunk in enumerate(group):
            cid = ids[index]
            digest = _digest(str(chunk.get("text") or ""))
            if primary_of[digest]["chunk_id"] != chunk["chunk_id"]:
                continue
            prev_id = ids[index - 1] if index else ""
            next_id = ids[index + 1] if index + 1 < len(ids) else ""
            section = section_of.get(cid, "")
            loc = _location(chunk, section, prev_id, next_id, len(ids))
            copies = []
            for other in text_groups[digest]:
                if str(other["chunk_id"]) == cid:
                    continue
                copies.append({
                    "chunk_id": str(other["chunk_id"]),
                    "doc_id": _doc_of(other),
                    "source": str(other.get("source_file") or _doc_of(other)),
                })
            copies.sort(key=lambda item: (item["source"], item["chunk_id"]))
            loc["copies"] = copies[:MAX_COPIES]
            loc["copies_omitted"] = max(0, len(copies) - MAX_COPIES)
            if index == 0 and human_source(source):
                add_entry("topic", source, loc, digest)
            for title in _headings(str(chunk.get("text") or "")):
                add_entry("topic", title, loc, digest)
            for _, surface in _mentions(str(chunk.get("text") or ""), stop, leads):
                add_entry("entity", surface, loc, digest)

    n_docs = len(groups)
    # A term in this many different texts is on too many pages to be a locator.
    # The chunks still hold it; search those directly.
    limit = max(15, (n_docs * 4 + 99) // 100)
    entries = []
    for entry in raw_entries.values():
        seen_digest: set[str] = set()
        locations = []
        for digest, loc in entry["hits"]:
            if digest in seen_digest:
                continue
            seen_digest.add(digest)
            locations.append(loc)
        if len(locations) > limit:
            continue
        locations.sort(key=lambda loc: (loc["source"], loc["chunk_id"]))
        omitted = max(0, len(locations) - MAX_LOCATIONS)
        entries.append({
            "term": entry["term"],
            "kind": entry["kind"],
            "locations": locations[:MAX_LOCATIONS],
            "omitted": omitted,
        })
    entries.sort(key=lambda entry: (entry["term"].casefold(), entry["kind"]))

    threads_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for thread in raw_threads:
        key = (thread["title"].casefold(), thread["digest"])
        chosen = threads_by_key.get(key)
        if chosen is None or (not human_source(chosen["source"])
                              and human_source(thread["source"])):
            if chosen is not None:
                thread = dict(thread)
                thread["copies"] = list(chosen.get("copies") or []) + [{
                    "doc_id": chosen["doc_id"], "source": chosen["source"],
                    "chunk_ids": chosen["chunk_ids"],
                }]
            else:
                thread = dict(thread)
                thread["copies"] = []
            threads_by_key[key] = thread
        else:
            chosen.setdefault("copies", []).append({
                "doc_id": thread["doc_id"], "source": thread["source"],
                "chunk_ids": thread["chunk_ids"],
            })
    threads = []
    for thread in threads_by_key.values():
        copies = thread.get("copies") or []
        threads.append({
            "title": thread["title"],
            "doc_id": thread["doc_id"],
            "source": thread["source"],
            "chunk_ids": thread["chunk_ids"],
            "copies": copies[:MAX_COPIES],
            "copies_omitted": max(0, len(copies) - MAX_COPIES),
        })
    threads.sort(key=lambda thread: (
        thread["title"].casefold(), thread["source"], thread["doc_id"]))
    return {"entries": entries, "threads": threads, "documents": n_docs}


def load(folder: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The book index in ``folder``; ``None`` when there is none, or when the
    one there was built on another core (it describes a different corpus)."""
    if not built_on_this_core(folder):
        return None
    path = index_path(folder)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find(index: dict[str, Any], query: str) -> dict[str, list]:
    """Entries and threads whose label contains the query, case-insensitive."""
    needle = " ".join(query.split()).casefold()
    if not needle:
        return {"entries": [], "threads": []}
    return {
        "entries": [entry for entry in index.get("entries") or []
                    if needle in entry["term"].casefold()],
        "threads": [thread for thread in index.get("threads") or []
                    if needle in thread["title"].casefold()],
    }


def _place(loc: dict[str, Any]) -> str:
    bits = [loc["source"], f"`{loc['chunk_id']}`"]
    if loc.get("section"):
        bits.append(f"section {loc['section']}")
    if loc.get("prev") or loc.get("next"):
        bits.append(f"prev `{loc.get('prev') or '-'}` next `{loc.get('next') or '-'}`")
    if loc.get("copies"):
        also = ", ".join(f"`{copy['chunk_id']}`" for copy in loc["copies"])
        extra = loc.get("copies_omitted") or 0
        more = f" +{extra}" if extra else ""
        bits.append(f"also {also}{more}")
    return " | ".join(bits)


def _entry_brief(entry: dict[str, Any]) -> str:
    """One line for a letter file: the term and the first chunks to open."""
    shown = entry["locations"][:4]
    total = len(entry["locations"]) + int(entry.get("omitted") or 0)
    places = "; ".join(f"{loc['source']} `{loc['chunk_id']}`" for loc in shown)
    extra = total - len(shown)
    more = f" +{extra}" if extra else ""
    return f"- {entry['term']} | {entry['kind']} | {total} | {places}{more}"


def _entry_lines(entry: dict[str, Any]) -> list[str]:
    total = len(entry["locations"]) + int(entry.get("omitted") or 0)
    lines = [f"- {entry['term']} | {entry['kind']} | {total}"]
    for loc in entry["locations"]:
        lines.append(f"  - {_place(loc)}")
    omitted = int(entry.get("omitted") or 0)
    if omitted:
        lines.append(f"  - +{omitted} more chunks")
    return lines


def _thread_lines(thread: dict[str, Any]) -> list[str]:
    ids = thread["chunk_ids"]
    if len(ids) <= 4:
        span = ", ".join(f"`{cid}`" for cid in ids)
    else:
        span = f"`{ids[0]}` .. `{ids[-1]}` ({len(ids)})"
    line = f"- {thread['title']} | {thread['source']} | {span}"
    if thread.get("copies"):
        line += f" | also {len(thread['copies']) + int(thread.get('copies_omitted') or 0)} copies"
    return [line]


def render_guide(index: dict[str, Any]) -> str:
    counts = index["counts"]
    return "\n".join([
        "# Corpus index",
        "",
        "A book index. A line names a topic, an entity, or a thread and the chunks to open.",
        "It does not copy chunk text. Search a letter file, then open those chunk ids.",
        "Previous and next are the neighboring chunks in the same document.",
        "`also` is the same text in another document.",
        "Names, document titles, and headings are indexed. Ordinary wording is still only",
        "in the chunks.",
        "A label that shows up in too many places is left out; it does not narrow a search.",
        "",
        f"- dataset_id: {index.get('dataset_id')}",
        f"- chunk_inventory_sha256: {index.get('chunk_inventory_sha256')}",
        f"- core_sha256: {index.get('core_sha256')}",
        f"- extra stopwords: {', '.join(index.get('stopwords') or []) or 'none'}",
        f"- built_at: {index.get('built_at')}",
        f"- documents: {counts['documents']}",
        f"- chunks: {counts['chunks']}",
        f"- entries: {counts['entries']}",
        f"- threads: {counts['threads']}",
        "",
        "Terms: `index/terms/<letter>.md`. Threads: `index/threads/<letter>.md`.",
        "A letter file is one line per term. `rag-connector index lookup` prints the",
        "neighbors and the copies.",
        "The full map is `index.json`.",
        "",
    ])


def _write_letters(folder: Path, rows: list[tuple[str, list[str]]]) -> None:
    by_letter: dict[str, list[str]] = {}
    for label, lines in rows:
        by_letter.setdefault(_letter(label), []).extend(lines + [""])
    for letter, lines in by_letter.items():
        title = letter.upper() if letter.isalpha() else "0-9"
        text = f"# {title}\n\n" + "\n".join(lines).rstrip() + "\n"
        (folder / f"{letter}.md").write_text(text, encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _envelope_core(folder: str | os.PathLike[str]) -> str | None:
    path = analysis_dir(folder) / ANALYSIS_ENVELOPE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("core_sha256")
    except (ValueError, AttributeError):
        return None


def built_on_this_core(folder: str | os.PathLike[str]) -> bool:
    """True when the area's envelope records this Dataset's core.

    Reads only the manifest's recorded ``core_sha256``, so a lookup stays
    cheap; :func:`prepare` and :func:`open_area` verify the Dataset in full.
    """
    try:
        manifest = json.loads((Path(folder) / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    core = manifest.get("core_sha256")
    return bool(core) and _envelope_core(folder) == core


def open_area(folder: str | os.PathLike[str]) -> Dataset:
    """Verify the Dataset and make its analysis area ready to write.

    An area built on another core describes a different corpus, so it is
    removed as a whole before anything is written (spec: a tool replaces its
    area as a whole). A legacy folder has no core to record and is refused.
    """
    dataset = read_dataset_folder(folder)
    area = analysis_dir(folder)
    if area.exists() and _envelope_core(folder) != dataset.core_sha256:
        shutil.rmtree(area)
    area.mkdir(parents=True, exist_ok=True)
    return dataset


#: Every file the area may hold that a reader is pointed at, in envelope order.
_ARTIFACTS = (
    (INDEX_MD, "markdown",
     "Book index guide: how terms, threads and letter files are laid out"),
    (INDEX_NAME, "json",
     "Book index: names, titles and headings, each with the chunk ids that hold it"),
    ("index/signal.md", "markdown",
     "Signal index guide: subjects a model labelled, with the chunk span to open"),
    ("index/signal.json", "json",
     "Signal index: model-written labels per document; chunk ids from the Dataset"),
    ("search.sqlite", "sqlite",
     "Search index: every chunk with full-text (word and substring) indexes, "
     "copy groups and headings"),
)


def write_envelope(folder: str | os.PathLike[str]) -> Path:
    """(Re)write ``analysis.json``, listing whichever indexes the area holds."""
    from importlib.metadata import version

    area = analysis_dir(folder)
    artifacts = [
        {"path": rel, "format": fmt, "description": text}
        for rel, fmt, text in _ARTIFACTS if (area / rel).is_file()
    ]
    return write_analysis_envelope(
        folder, ANALYSIS_TOOL, tool_version=version("rag-connector"),
        format_version=FORMAT_VERSION, artifacts=artifacts,
    )


def _write(folder: str | os.PathLike[str], index: dict[str, Any]) -> dict[str, Any]:
    # Only the book index's own files are replaced: index/ also holds the
    # signal index, which costs model calls to rebuild.
    terms = index_dir(folder) / "terms"
    threads = index_dir(folder) / "threads"
    for old in (terms, threads):
        if old.exists():
            shutil.rmtree(old)
        old.mkdir(parents=True)
    index_path(folder).write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path(folder).write_text(render_guide(index), encoding="utf-8")
    _write_letters(terms, [(entry["term"], [_entry_brief(entry)]) for entry in index["entries"]])
    _write_letters(threads, [(thread["title"], _thread_lines(thread))
                             for thread in index["threads"]])
    write_envelope(folder)
    return index


def prepare(folder: str | os.PathLike[str], *,
            stopwords: Iterable[str] = ()) -> dict[str, Any]:
    """Write the book index from the Dataset's chunks and return its counts.

    Replaces any book index already there and leaves a signal index alone.
    ``stopwords`` is passed to :func:`build` and recorded in the index.
    """
    dataset = open_area(folder)
    chunks = list(dataset.chunks)
    extra = sorted({word.casefold() for word in stopwords})
    body = build(chunks, stopwords=extra)
    index = {
        "kind": KIND,
        "version": VERSION,
        "role": "book-index",
        "dataset_id": dataset.dataset_id,
        "chunk_inventory_sha256": dataset.chunk_inventory_sha256,
        "core_sha256": dataset.core_sha256,
        "built_at": _now(),
        "stopwords": extra,
        "counts": {
            "documents": body["documents"],
            "chunks": len(chunks),
            "entries": len(body["entries"]),
            "threads": len(body["threads"]),
        },
        "entries": body["entries"],
        "threads": body["threads"],
    }
    _write(folder, index)
    return index["counts"]


def format_find(found: dict[str, list]) -> str:
    lines: list[str] = []
    for entry in found["entries"]:
        lines.extend(_entry_lines(entry))
    for thread in found["threads"]:
        lines.extend(_thread_lines(thread))
    if not lines:
        return "no index entry\n"
    return "\n".join(lines) + "\n"
