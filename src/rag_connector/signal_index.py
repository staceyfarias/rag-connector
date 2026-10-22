"""A signal index: a few high-signal words and the chunk span that holds them.

:mod:`rag_connector.corpus_index` builds the pattern index and calls no model.
This module plans a model pass over a Dataset and reads what it returns. The
model sees one document, or one slice of a long one, and does not see the rest
of the corpus. It returns words and chunk indexes. Chunk ids are attached here,
from the Dataset's own rows, so a model cannot invent one.

Identical copies of a document are not sent. The titled copy is the one that
is labelled, and the others are named on its line.

The words are model output, and are recorded as such: every labelled document
names the model that labelled it. The model is called through a *labeler*,
``(system, prompt) -> (text, input_tokens, output_tokens)``. The default,
:func:`anthropic_labeler`, needs the optional ``llm`` extra
(``pip install "rag-connector[llm]"``) and ``ANTHROPIC_API_KEY``; nothing in
the core imports it. Progress is kept in this package's private area,
``.rag-connector/index/signal_progress.jsonl``, so a later :func:`build`
sends only documents not labelled yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .audit import PRIVATE_AREA, _ensure_private_area
from .corpus_index import _doc_of, _now, _rows, human_source, index_dir, open_area, write_envelope

__all__ = [
    "MODEL",
    "Labeler",
    "anthropic_labeler",
    "build",
    "find",
    "format_hits",
    "load",
    "progress_path",
    "signal_markdown",
    "signal_path",
]

#: The default labelling model.
MODEL = "claude-haiku-5-5"
MAX_CHARS = 180_000
MAX_DOCS = 8
MAX_WORDS = 8
KIND = "rag-connector-signal-index"
PROGRESS_NAME = "signal_progress.jsonl"

#: ``(system, prompt) -> (response text, input tokens, output tokens)``.
Labeler = Callable[[str, str], "tuple[str, int, int]"]


_SYSTEM = """You write index lines for one document at a time. Another model will search these lines to decide which chunk span to open. You are not summarizing for a person, and you are not relating this document to any other.

Return JSON only, no markdown:
{"documents":[{"doc_id":"<the id you were given>","subjects":[{"words":["..."],"start":0,"end":0}]}]}

A subject is one thing the span is about.
- words: 3 to 8 items. No sentences. Include names as the text spells them, and 2 to 4 single words a searcher would use for that subject. Prefer one word that carries the subject by itself.
- start and end: the chunk indexes shown in brackets, inclusive. The span is only the chunks that hold that subject.
- One subject per distinct topic. Do not emit a subject per paragraph when they are the same topic.
- A roster, directory, or schedule is not one subject. Each named post, place, person, or meeting a searcher would look up is its own subject, on the chunks where that name appears.
- Use indexes only from the document you are labeling. Do not invent a doc_id.
"""  # noqa: E501


def progress_path(folder: str | os.PathLike[str]) -> Path:
    """Working state: one line per labelled document. Private, not shared."""
    return Path(folder) / PRIVATE_AREA / "index" / PROGRESS_NAME


def signal_path(folder: str | os.PathLike[str]) -> Path:
    return index_dir(folder) / "signal.json"


def signal_markdown(folder: str | os.PathLike[str]) -> Path:
    return index_dir(folder) / "signal.md"


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _chunk_digest(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()[:16]


def _groups(chunks: Iterable[Any]) -> list[list[dict]]:
    by_doc: dict[str, list[dict]] = {}
    for chunk in _rows(chunks):
        by_doc.setdefault(_doc_of(chunk), []).append(chunk)
    groups = []
    for group in by_doc.values():
        group.sort(key=lambda c: (int(c.get("doc_chunk_index") or 0), str(c["chunk_id"])))
        groups.append(group)
    groups.sort(key=lambda group: (str(group[0].get("source_file") or ""), _doc_of(group[0])))
    return groups


def _primary(groups: list[list[dict]]) -> list[dict]:
    return min(groups, key=lambda group: (
        0 if human_source(str(group[0].get("source_file") or "")) else 1,
        str(group[0].get("source_file") or ""),
        _doc_of(group[0]),
    ))


def documents(chunks: Iterable[Any]) -> list[dict[str, Any]]:
    """One record per distinct text. Copies keep their own chunk ids."""
    buckets: dict[tuple[str, ...], list[list[dict]]] = {}
    for group in _groups(chunks):
        key = tuple(_chunk_digest(str(c.get("text") or "")) for c in group)
        buckets.setdefault(key, []).append(group)
    records = []
    for groups in buckets.values():
        main = _primary(groups)
        copies = []
        for group in groups:
            if _doc_of(group[0]) == _doc_of(main[0]):
                continue
            copies.append({
                "doc_id": _doc_of(group[0]),
                "source": str(group[0].get("source_file") or _doc_of(group[0])),
                "chunk_ids": [str(c["chunk_id"]) for c in group],
            })
        records.append({
            "doc_id": _doc_of(main[0]),
            "source": str(main[0].get("source_file") or _doc_of(main[0])),
            "chunks": main,
            "copies": copies,
        })
    records.sort(key=lambda record: (record["source"], record["doc_id"]))
    return records


def slices_of(record: dict[str, Any], max_chars: int = MAX_CHARS) -> list[dict[str, Any]]:
    """Split a long document so one call stays under the token band."""
    slices: list[dict[str, Any]] = []
    current: list[dict] = []
    size = 0
    start = 0
    for position, chunk in enumerate(record["chunks"]):
        n = len(str(chunk.get("text") or "")) + 48
        if current and size + n > max_chars:
            slices.append(_slice(record, current, start, position - 1))
            current = []
            size = 0
            start = position
        current.append(chunk)
        size += n
    if current:
        slices.append(_slice(record, current, start, start + len(current) - 1))
    return slices


def _slice(record: dict[str, Any], chunks: list[dict], start: int, end: int) -> dict[str, Any]:
    return {
        "doc_id": record["doc_id"],
        "source": record["source"],
        "chunks": chunks,
        "start": start,
        "end": end,
    }


def calls_for(records: Iterable[dict[str, Any]], max_chars: int = MAX_CHARS,
              max_docs: int = MAX_DOCS) -> list[list[dict[str, Any]]]:
    """Pack slices into calls. A slice is never split across a call."""
    packed: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    docs: set[str] = set()
    for record in records:
        for piece in slices_of(record, max_chars):
            n = sum(len(str(c.get("text") or "")) for c in piece["chunks"])
            if current and (size + n > max_chars or len(docs | {piece["doc_id"]}) > max_docs):
                packed.append(current)
                current = []
                size = 0
                docs = set()
            current.append(piece)
            size += n
            docs.add(piece["doc_id"])
    if current:
        packed.append(current)
    return packed


def prompt_for(pieces: list[dict[str, Any]]) -> str:
    blocks = []
    for piece in pieces:
        lines = ["## DOCUMENT", f"doc_id: {piece['doc_id']}", f"source: {piece['source']}"]
        for offset, chunk in enumerate(piece["chunks"]):
            index = piece["start"] + offset
            lines.append(f"[{index}]")
            lines.append(str(chunk.get("text") or "").strip())
        blocks.append("\n".join(lines))
    return (
        "Label every document below. Return JSON only.\n\n" + "\n\n".join(blocks)
    )


def system_prompt() -> str:
    return _SYSTEM


def _words(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;]", value)
    if not isinstance(value, list):
        return []
    words = []
    for item in value:
        word = " ".join(str(item).split())
        if not word or len(word) > 60 or word in words:
            continue
        words.append(word)
        if len(words) >= MAX_WORDS:
            break
    return words


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def parse_payload(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON was not an object")
    return payload


def subjects_in(payload: dict[str, Any], piece: dict[str, Any]) -> list[dict[str, Any]]:
    """Subjects for one slice. Indexes outside the slice are dropped."""
    docs = payload.get("documents")
    if not isinstance(docs, list):
        raise ValueError("documents is not a list")
    raw = None
    for doc in docs:
        if isinstance(doc, dict) and str(doc.get("doc_id") or "") == piece["doc_id"]:
            raw = doc.get("subjects")
            break
    if raw is None and len(docs) == 1 and isinstance(docs[0], dict):
        raw = docs[0].get("subjects")
    if not isinstance(raw, list):
        raise ValueError(f"no subjects for {piece['doc_id']}")
    last = piece["start"] + len(piece["chunks"]) - 1
    subjects = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _int(item.get("start"))
        end = _int(item.get("end"))
        words = _words(item.get("words"))
        if start is None or end is None or not words:
            continue
        if end < start:
            start, end = end, start
        if start < piece["start"] or end > last:
            continue
        ids = [str(piece["chunks"][index - piece["start"]]["chunk_id"])
               for index in range(start, end + 1)]
        subjects.append({"words": words, "start": start, "end": end, "chunk_ids": ids})
    return subjects


def merge_subjects(subjects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlap of the same words becomes one span."""
    merged: list[dict[str, Any]] = []
    for subject in subjects:
        key = tuple(word.casefold() for word in subject["words"])
        placed = False
        for earlier in merged:
            if tuple(word.casefold() for word in earlier["words"]) != key:
                continue
            if subject["start"] <= earlier["end"] + 1 and earlier["start"] <= subject["end"] + 1:
                earlier["start"] = min(earlier["start"], subject["start"])
                earlier["end"] = max(earlier["end"], subject["end"])
                ids = list(dict.fromkeys(earlier["chunk_ids"] + subject["chunk_ids"]))
                earlier["chunk_ids"] = ids
                placed = True
                break
        if not placed:
            merged.append({
                "words": list(subject["words"]),
                "start": subject["start"],
                "end": subject["end"],
                "chunk_ids": list(subject["chunk_ids"]),
            })
    merged.sort(key=lambda subject: (subject["start"], subject["words"][0].casefold()))
    return merged


def load_progress(folder: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    path = progress_path(folder)
    if not path.is_file():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found[str(record["doc_id"])] = record
    return found


def _letter(word: str) -> str:
    ch = word.casefold()[:1]
    return ch if ch.isalpha() else "0"


def _line(record: dict[str, Any], subject: dict[str, Any]) -> str:
    words = "; ".join(subject["words"])
    span = f"{subject['start']}-{subject['end']}"
    ids = subject["chunk_ids"]
    if len(ids) == 1:
        where = f"`{ids[0]}`"
    else:
        where = f"`{ids[0]}` .. `{ids[-1]}` ({len(ids)})"
    copies = record.get("copies") or []
    also = ""
    if copies:
        shown = ", ".join(copy["source"] for copy in copies[:3])
        extra = len(copies) - min(3, len(copies))
        more = f" +{extra}" if extra else ""
        also = f" | also {shown}{more}"
    return f"- {words} | {record['source']} | {span} | {where}{also}"


def render(records: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """The guide, and one markdown body per letter."""
    letters: dict[str, list[str]] = {}
    subjects = 0
    for record in records:
        for subject in record.get("subjects") or []:
            subjects += 1
            letters.setdefault(_letter(subject["words"][0]), []).append(_line(record, subject))
    guide = "\n".join([
        "# Signal index",
        "",
        "Each line is a subject, the document it is in, and the chunk span to open.",
        "The span is inclusive chunk indexes in that document. The ids are the first and",
        "last chunk.",
        "Words are labels for a search, not a statement that the chunk text was copied here.",
        "`also` is the same text in another document.",
        "",
        f"- documents: {len(records)}",
        f"- subjects: {subjects}",
        "",
        "Search `index/signal/<letter>.md`, or `rag-connector index signal-lookup`.",
        "",
    ])
    bodies = {letter: "\n".join(lines) + "\n" for letter, lines in sorted(letters.items())}
    return guide, bodies


def write_index(folder: str | os.PathLike[str], records: list[dict[str, Any]],
                built_at: str, dataset_id: str, core_sha256: str) -> dict[str, Any]:
    guide, bodies = render(records)
    out = index_dir(folder) / "signal"
    out.mkdir(parents=True, exist_ok=True)
    for child in out.glob("*.md"):
        child.unlink()
    for letter, text in bodies.items():
        title = letter.upper() if letter.isalpha() else "0-9"
        (out / f"{letter}.md").write_text(f"# {title}\n\n{text}", encoding="utf-8")
    payload = {
        "kind": KIND,
        "models": sorted({str(r.get("model") or "") for r in records} - {""}),
        "dataset_id": dataset_id,
        "core_sha256": core_sha256,
        "built_at": built_at,
        "documents": records,
    }
    signal_path(folder).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    signal_markdown(folder).write_text(guide, encoding="utf-8")
    write_envelope(folder)
    return {"documents": len(records),
            "subjects": sum(len(r.get("subjects") or []) for r in records)}


def find(payload: dict[str, Any], query: str) -> list[dict[str, Any]]:
    needle = " ".join(query.split()).casefold()
    if not needle:
        return []
    hits = []
    for record in payload.get("documents") or []:
        for subject in record.get("subjects") or []:
            if any(needle in word.casefold() for word in subject["words"]):
                hits.append({"source": record["source"], "doc_id": record["doc_id"], **subject,
                             "copies": record.get("copies") or []})
    return hits


def format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "no signal entry\n"
    lines = []
    for hit in hits:
        record = {"source": hit["source"], "copies": hit.get("copies") or []}
        lines.append(_line(record, hit))
        if len(hit["chunk_ids"]) > 2:
            lines.append("  ids: " + ", ".join(f"`{cid}`" for cid in hit["chunk_ids"]))
    return "\n".join(lines) + "\n"


def load(folder: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The signal index in ``folder``; ``None`` when there is none, or when the
    one there was built on another core."""
    from .corpus_index import built_on_this_core

    path = signal_path(folder)
    if not built_on_this_core(folder) or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The model call
# ---------------------------------------------------------------------------

def anthropic_labeler(model: str = MODEL, *, max_tokens: int = 8192) -> Labeler:
    """A labeler that calls the Anthropic Messages API.

    Needs the ``llm`` extra and ``ANTHROPIC_API_KEY``; the key is read by the
    SDK and never printed. Transient API errors are retried with backoff. A
    response cut off at ``max_tokens`` is an error, not a partial label.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    try:
        from anthropic import Anthropic, APIStatusError
    except ImportError as exc:
        raise ImportError(
            'the signal index calls a model: pip install "rag-connector[llm]"'
        ) from exc
    client = Anthropic()

    def label(system: str, prompt: str) -> tuple[str, int, int]:
        delay = 2.0
        last: Exception | None = None
        for _ in range(4):
            try:
                response = client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            except APIStatusError as exc:
                last = exc
                if exc.status_code not in (408, 409, 429, 500, 529):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise RuntimeError("response hit max_tokens")
            text = "".join(getattr(block, "text", "") for block in response.content)
            usage = response.usage
            return text, int(usage.input_tokens or 0), int(usage.output_tokens or 0)
        raise RuntimeError(f"model call failed after retries: {last}")

    return label


def _label_batch(label: Labeler, pieces: list[dict]) -> tuple[dict, int, int]:
    """Map (doc_id, slice start) to subjects. A bad batch is retried per slice."""
    def one(group: list[dict]) -> tuple[dict, int, int]:
        text, inn, out = label(system_prompt(), prompt_for(group))
        payload = parse_payload(text)
        return ({(p["doc_id"], p["start"]): subjects_in(payload, p) for p in group},
                inn, out)

    try:
        return one(pieces)
    except Exception:
        if len(pieces) == 1:
            raise
    found: dict = {}
    inn = out = 0
    for piece in pieces:
        part, one_in, one_out = one([piece])
        found.update(part)
        inn += one_in
        out += one_out
    return found, inn, out


def _records(folder: str | os.PathLike[str], planned: dict[str, dict]) -> list[dict]:
    """Labelled documents for the index. A subject citing a chunk id that is
    not in its document is dropped: the index never names a chunk the Dataset
    does not hold under that document."""
    records = []
    for doc_id, saved in load_progress(folder).items():
        plan = planned.get(doc_id)
        if plan is None:
            continue
        ids = {str(c["chunk_id"]) for c in plan["chunks"]}
        subjects = [s for s in saved.get("subjects") or [] if set(s["chunk_ids"]) <= ids]
        records.append({
            "doc_id": doc_id,
            "source": plan["source"],
            "model": saved.get("model"),
            "copies": plan["copies"],
            "subjects": subjects,
        })
    records.sort(key=lambda record: (record["source"], record["doc_id"]))
    return records


def build(
    folder: str | os.PathLike[str],
    *,
    model: str = MODEL,
    label: Labeler | None = None,
    limit: int | None = None,
    workers: int = 6,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Label the documents not labelled yet, then (re)write the signal index.

    ``label`` defaults to :func:`anthropic_labeler` for ``model``. A caller
    passing its own labeler names the model it calls in ``model``, which is
    what each document records. Returns counts, token totals and failures; a
    failed call leaves its documents unlabelled for the next run.
    """
    dataset = open_area(folder)
    planned = {r["doc_id"]: r for r in documents(dataset.chunks)}
    done = load_progress(folder)
    pending = sorted((planned[d] for d in planned if d not in done),
                     key=lambda record: (record["source"], record["doc_id"]))
    if limit is not None:
        pending = pending[:limit]
    batches = calls_for(pending)
    log(f"{len(planned)} distinct documents, {len(done)} already labelled, "
        f"{len(pending)} to send in {len(batches)} calls, model {model}")
    result: dict[str, Any] = {"labelled": 0, "tokens_in": 0, "tokens_out": 0,
                              "failures": []}
    if pending:
        label = label or anthropic_labeler(model)
        needed = {r["doc_id"]: {(p["start"], p["end"]) for p in slices_of(r)}
                  for r in pending}
        got_slices: dict[str, set] = {r["doc_id"]: set() for r in pending}
        got_subjects: dict[str, list] = {r["doc_id"]: [] for r in pending}
        _ensure_private_area(Path(folder), dataset)
        path = progress_path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)

        def work(batch: list[dict]):
            try:
                return batch, _label_batch(label, batch), ""
            except Exception as exc:
                names = ", ".join(sorted({p["source"] for p in batch})[:4])
                return batch, ({}, 0, 0), f"{type(exc).__name__}: {exc} ({names})"[:300]

        with path.open("a", encoding="utf-8") as progress, \
                ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in as_completed([pool.submit(work, b) for b in batches]):
                batch, (found, inn, out), error = future.result()
                result["tokens_in"] += inn
                result["tokens_out"] += out
                if error:
                    result["failures"].append(error)
                    continue
                for piece in batch:
                    doc_id, key = piece["doc_id"], (piece["start"], piece["end"])
                    if key in got_slices[doc_id]:
                        continue
                    got_slices[doc_id].add(key)
                    got_subjects[doc_id].extend(found.get((doc_id, piece["start"]), []))
                    if got_slices[doc_id] != needed[doc_id]:
                        continue
                    progress.write(json.dumps({
                        "doc_id": doc_id,
                        "source": planned[doc_id]["source"],
                        "model": model,
                        "subjects": merge_subjects(got_subjects[doc_id]),
                    }, ensure_ascii=False) + "\n")
                    progress.flush()
                    result["labelled"] += 1
                    if result["labelled"] % 25 == 0:
                        log(f"  labelled {result['labelled']}")
    counts = write_index(folder, _records(folder, planned), _now(),
                         dataset.dataset_id, dataset.core_sha256)
    result.update(counts)
    return result
