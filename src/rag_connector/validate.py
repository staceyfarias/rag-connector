
"""Standalone RAG-connector validator: prove the contract works BEFORE a host.

For the developer writing a connector (in this repo or as a separate project):
run this script against your connector and it exercises every method of the
:class:`RagPipeline` contract -- construction, ``info``, ``pull_all_chunks``,
the chunk metadata contract, ``fingerprint`` stability, ``query``, id-format
consistency, score direction, ``get_chunks`` (including whether an override is
real or the base default is answering), paged reads (``list_chunks``),
indexed-vector reads (``get_chunk_vectors``), optional ``generate``, and (in
registry mode) the ``connect``/``build`` reconstruction round-trip -- and prints
a developer-facing PASS/WARN/FAIL report saying exactly what is required, what
is working, and how to fix what isn't. No host application, no corpus
snapshot, and no LLM key needed.

Two ways to point it at a connector::

    # 1. Direct import -- pre-registration development (separate project OK):
    python -m rag_connector.validate --import my_pkg.my_connector:MyRagConnector \
        --kwargs "{\"endpoint\": \"https://rag.internal\", \"collection\": \"docs\"}"

    # 2. Registry mode -- after you register a ConnectorSpec; also validates the
    #    connect() -> connection dict -> build() reconstruction round-trip:
    python -m rag_connector.validate --connector-type my-rag \
        --params "{\"endpoint\": \"https://rag.internal\", \"collection\": \"docs\"}"

Add ``--query "a question your corpus can answer"`` for a realistic retrieval
probe (otherwise one is derived from corpus text), ``--top-k N`` (default 5),
and ``--skip-stability`` to skip the second full corpus pull on huge corpora.

Exit codes: 0 = READY (no failures) ; 1 = NOT READY ; 2 = validator crashed.

This is the *pre-flight* validator, distinct from (and stricter than) whatever
connect-time gate a host applies. A host typically enforces the chunk contract
when it ingests a corpus, but only this script checks what connect time cannot
see -- that ``query()`` returns ids in the same format ``pull_all_chunks()``
produces, that scores run in the declared direction, that retrieval is
deterministic, and that ids are stable between pulls. Direct-import mode
deliberately imports nothing beyond ``rag_connector.base``, so it runs in a
connector project's own environment without installing a host application.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .base import (
    RETRIEVAL_MODES,
    ChunkRecord,
    GeneratedAnswer,
    RagPipeline,
    RetrievedChunk,
)
from .capabilities import (
    DECLARED_PROMPT_METHODS,
    declared_prompt_methods,
    supports,
    supports_declared_prompts,
)
from .errors import UnsupportedCapability
from .prompts import (
    PromptTemplate,
    prompt_template_problems,
    render_prompt,
)
from .run_snapshot import run_snapshot_fingerprint, validate_run_snapshot

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# The score convention is canonical (rag_connector/base.py): RetrievedChunk.score
# is ALWAYS higher-is-better, whatever the store's native metric. The declared
# ``distance`` is informational (what the number means), never a direction.

# Key-name fragments that mean "this is a secret and must NOT be persisted in
# the connection dict" (the docs rule: read credentials from the environment).
_SECRET_KEY_HINTS = ("password", "secret", "token", "api_key", "apikey",
                     "credential")

_MISSING_ID_SENTINEL = "__rag_connector_validator_no_such_chunk__"

# Contract methods a subclass may leave to a concrete base default. A stray
# public method whose name closely resembles one of these -- while the default
# still answers -- is the renamed-override bug: a consumer shipped
# ``fetch_chunks`` against ``get_chunks`` and stayed green for the life of the
# project because the base default produced the same observable results.
_DEFAULTED_CONTRACT_METHODS = (
    "pull_all_chunks", "list_chunks", "get_chunks", "get_chunk_vectors",
    "list_datasets", "fingerprint", "generate", "info",
    "prompt_templates", "render_rag_block", "generate_from_prompt",
)

# Names that legitimately live on a connector without being contract methods:
# the optional capability protocols and registry/connection conventions.
_EXPECTED_EXTRA_METHODS = frozenset({
    "query", "validate_chunks",
    "health", "describe_space", "embed_queries", "upsert", "delete",
    "read_queries", "list_indexes", "run_snapshot", "connection", "connect", "build",
})


@dataclass
class Check:
    """One validator check: verdict + what happened + how to fix it."""

    name: str
    status: str
    details: list[str] = field(default_factory=list)
    fix: str = ""


@dataclass
class Report:
    """All checks for one connector, plus the overall verdict."""

    target: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, details: list[str] | None = None,
            fix: str = "") -> Check:
        check = Check(name, status, list(details or []), fix)
        self.checks.append(check)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:

        return [c for c in self.checks if c.status == WARN]

    @property
    def ready(self) -> bool:
        return not self.failures

    def render(self) -> str:
        icon = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[skip]"}
        lines = [
            "=" * 64,
            "RAG Connector contract validator",
            f"Target: {self.target}",
            "=" * 64,
        ]
        for i, c in enumerate(self.checks, start=1):
            lines.append(f"{icon[c.status]} {i:2}. {c.name}")
            for d in c.details:
                lines.append(f"          {d}")
            if c.fix and c.status in (FAIL, WARN):
                lines.append(f"          FIX: {c.fix}")
        lines.append("-" * 64)
        n_pass = sum(1 for c in self.checks if c.status == PASS)
        if self.ready:
            lines.append(
                f"VERDICT: READY -- {n_pass} passed, "
                f"{len(self.warnings)} warning(s), 0 failures."
            )
            lines.append(
                "The connector satisfies the RagPipeline contract. Next: "
                "publish a ConnectorSpec through the "
                "'rag_connector.connectors' entry point (see "
                "examples/connector_template.py), restart your host "
                "application, and connect a corpus through it."
            )
            if self.warnings:
                lines.append(
                    "Warnings are not blockers, but read them -- each one is a "
                    "way the numbers can quietly mean less than they appear to."
                )
        else:
            lines.append(
                f"VERDICT: NOT READY -- {len(self.failures)} blocking "
                f"failure(s), {len(self.warnings)} warning(s)."
            )
            lines.append(
                "Evaluation results from this connector cannot be trusted "
                "until every FAIL above is fixed. Fix the first failure first; "
                "later checks often fail as a consequence of an earlier one."
            )
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks. Each takes what it needs and writes into the report.
# Ordering matters: later checks reuse earlier results (pulled chunks, hits).
# ---------------------------------------------------------------------------

def _check_info(report: Report, pipeline: RagPipeline) -> dict:
    try:
        info = pipeline.info()
    except Exception as exc:
        report.add("info() returns run metadata", FAIL,
                   [f"info() raised: {exc!r}"],
                   "info() must return a plain dict describing the connector; "
                   "the base-class default works if you have nothing to add.")
        return {}
    if not isinstance(info, dict):
        report.add("info() returns run metadata", FAIL,
                   [f"info() returned {type(info).__name__}, not dict"],
                   "Return a dict; it is persisted as run metadata.")
        return {}
    details = [f"name={info.get('name')!r}, type={info.get('type')!r}"]
    if info.get("embedding_fingerprint"):
        details.append(f"embedding_fingerprint={info['embedding_fingerprint']!r}")
        report.add("info() returns run metadata", PASS, details)
    else:
        report.add(
            "info() returns run metadata", WARN, details,
            "Include 'embedding_fingerprint' (provider + model + dimensions) "
            "if you can: hosts persist it with the frozen corpus so they can "
            "detect that the embedding model changed under you.")
    return info


def _check_run_snapshot(report: Report, pipeline: RagPipeline) -> None:
    """Validate the optional, persisted connector-state disclosure."""
    name = "run_snapshot() is bounded and safe to persist"
    if not supports(pipeline, "run_snapshot"):
        report.add(name, SKIP, ["optional capability not implemented"])
        return
    try:
        snapshot = pipeline.run_snapshot()
    except Exception as exc:
        report.add(
            name, FAIL, [f"run_snapshot() raised: {exc!r}"],
            "Return cheap, read-only state. A snapshot failure must not be "
            "hidden as an empty mapping.")
        return

    try:
        normalized = validate_run_snapshot(snapshot)
        fingerprint = run_snapshot_fingerprint(normalized)
    except Exception as exc:
        report.add(
            name, FAIL, [str(exc)],
            "Return {'schema': 'vendor.run-snapshot.v1', 'settings': {...}, "
            "'observations': {...}} using strict JSON values only. Keep it "
            "under 64 KiB and remove credentials, record dumps, and NaN/Infinity.")
    else:
        report.add(
            name, PASS,
            [f"schema={normalized['schema']!r}; "
             f"{len(normalized['settings'])} setting(s), "
             f"{len(normalized['observations'])} observation(s); "
             f"sha256={fingerprint}"])


def _check_mode(report: Report, info: dict) -> str:
    """Validate the declared C2 retrieval mode and echo internal-filter
    disclosure. Returns the resolved mode so later checks can relax per mode.

    An unknown mode value is a FAIL (the harness can't score a run whose
    convention it doesn't recognize); a missing declaration is the legacy
    default ``scored``.
    """
    mode = info.get("retrieval_mode", "scored") if isinstance(info, dict) else "scored"
    name = "retrieval_mode is a recognized value"
    filters = info.get("internal_filters") if isinstance(info, dict) else None
    detail_filters: list[str] = []
    if filters:
        detail_filters = [
            "internal_filters disclosed (operator awareness only -- hosts "
            "never re-apply these):",
            *[f"  - {f.get('name', '?')}: {f.get('description', '')}"

              + ("" if f.get("value") is None else f" (value={f.get('value')!r})")
              for f in filters if isinstance(f, dict)],
        ]
    if mode not in RETRIEVAL_MODES:
        report.add(name, FAIL, [f"declared retrieval_mode={mode!r}"] + detail_filters,
                   f"Declare one of {RETRIEVAL_MODES} via the retrieval_mode "
                   "class attribute (surfaced through info()). The default "
                   "'scored' is the pre-C2 behavior.")
        return "scored"
    meaning = {
        "scored": "harness top_k bounds the list; scores meaningful; threshold OK",
        "ordered": "harness top_k bounds the list; scores evidence-only; "
                   "threshold refused",
        "complete_set": "the SYSTEM decides cardinality; top_k is a pass-through "
                        "breadth knob; scores evidence-only; @set metrics",
    }[mode]
    report.add(name, PASS, [f"retrieval_mode = {mode!r} -- {meaning}"] + detail_filters)
    return mode


def _check_cardinality_probe(report: Report, pipeline: RagPipeline,
                             sample_query: str, top_k: int) -> None:
    """complete_set only: does the returned cardinality respond to top_k?

    Purely informational -- it distinguishes a breadth-parameterized system (the
    work RAG's neighbor expansion widens with top_k) from a top_k-inert one.
    Neither is a failure; under complete_set the SYSTEM owns cardinality.
    """
    name = "complete_set: returned cardinality vs two top_k values (informational)"
    k2 = (top_k * 2) if top_k else 10
    try:
        n1 = len(pipeline.query(sample_query, top_k=top_k))
        n2 = len(pipeline.query(sample_query, top_k=k2))
    except Exception as exc:
        report.add(name, SKIP, [f"probe raised: {exc!r}"])
        return
    if n1 != n2:
        report.add(name, PASS,
                   [f"returned {n1} at top_k={top_k}, {n2} at top_k={k2} -- "
                    "top_k widens the returned set (breadth-parameterized); "
                    "neither is a failure"])
    else:
        report.add(name, PASS,
                   [f"returned {n1} at both top_k={top_k} and {k2} -- the "
                    "system's own cutoff is top_k-inert; neither is a failure"])


def _check_pull(report: Report, pipeline: RagPipeline) -> list[ChunkRecord]:
    try:
        chunks = pipeline.pull_all_chunks()
    except NotImplementedError:
        report.add("pull_all_chunks() enumerates the corpus", FAIL,
                   ["pull_all_chunks() is not implemented"],
                   "This is one of the two REQUIRED methods: return every "
                   "chunk in the corpus as ChunkRecord objects.")
        return []
    except UnsupportedCapability as exc:
        # The base defaults derive from each other; implementing neither is
        # refused there. Surface that refusal with its ways out, not a generic
        # "an exception happened".
        report.add("pull_all_chunks() enumerates the corpus", FAIL,
                   [str(exc)],
                   "Implement list_chunks (preferred -- bulk pull and paging "
                   "both derive from it) or pull_all_chunks. A corpus that "
                   "cannot be read cannot be frozen as evidence.")
        return []
    except Exception as exc:
        report.add("pull_all_chunks() enumerates the corpus", FAIL,
                   [f"pull_all_chunks() raised: {exc!r}"],
                   "Fix the exception; a host freezes a corpus by calling this "
                   "exact method and would fail the same way.")
        return []
    if not isinstance(chunks, list) or not chunks:
        report.add("pull_all_chunks() enumerates the corpus", FAIL,
                   [f"returned {type(chunks).__name__} "
                    f"with {len(chunks) if isinstance(chunks, list) else '?'} items"],
                   "Return a non-empty list; a connector with zero chunks is "
                   "rejected when a host freezes a corpus.")
        return []
    not_records = [c for c in chunks[:50] if not isinstance(c, ChunkRecord)]
    if not_records:
        report.add("pull_all_chunks() enumerates the corpus", FAIL,
                   [f"items are {type(not_records[0]).__name__}, "
                    "not ChunkRecord"],
                   "Wrap each row in rag_connector.base.ChunkRecord -- the harness "
                   "reads its fields by name.")
        return []
    report.add("pull_all_chunks() enumerates the corpus", PASS,
               [f"{len(chunks)} chunks, "
                f"{len({c.doc_id for c in chunks})} documents"])
    return chunks


def _check_contract(report: Report, chunks: list[ChunkRecord]) -> None:
    problems: list[str] = []
    try:
        RagPipeline.validate_chunks(chunks)
    except ValueError as exc:
        problems.append(str(exc))
    seen: set = set()
    dupes = 0
    empty_text = 0
    for c in chunks:
        if c.chunk_id in seen:
            dupes += 1
        seen.add(c.chunk_id)
        if not (c.text or "").strip():
            empty_text += 1
    if dupes:
        problems.append(f"{dupes} duplicate chunk_id value(s)")

    if empty_text:
        problems.append(f"{empty_text} chunk(s) with blank text")
    if problems:
        report.add(
            "Chunk metadata contract (required fields, unique ids, text)",
            FAIL, problems,
            "Every chunk needs chunk_id / doc_id / source_file / "
            "doc_chunk_index / non-blank text, and chunk_ids must be unique "
            "(global_index is optional -- pass None and the host assigns "
            "ordinals at ingest). Hosts enforce exactly this when freezing a "
            "corpus and will refuse the connect.")
    else:
        report.add("Chunk metadata contract (required fields, unique ids, text)",
                   PASS, ["all required fields present on every chunk"])


def _check_indexes(report: Report, chunks: list[ChunkRecord]) -> None:
    by_doc: dict[str, list[int]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c.doc_chunk_index)
    broken = {doc: sorted(idx) for doc, idx in by_doc.items()
              if sorted(idx) != list(range(len(idx)))}
    if broken:
        doc, idx = next(iter(broken.items()))
        report.add(
            "doc_chunk_index counts 0,1,2,... within each document", FAIL,
            [f"{len(broken)} document(s) with gaps or duplicates, "
             f"e.g. {doc!r}: {idx[:8]}"],
            "Neighbor-window answer-key enrichment walks +/-N by this ordinal; "
            "gaps or repeats grade the wrong neighbors. Renumber per document.")
    else:
        report.add("doc_chunk_index counts 0,1,2,... within each document",
                   PASS)
    # ``global_index`` is legally None (a corpus-wide ordinal is a host
    # concern; see ChunkRecord's docstring) -- so uniqueness applies only to
    # the values a connector actually supplies.
    name = "global_index (when supplied) is unique across the corpus"
    supplied = [c.global_index for c in chunks if c.global_index is not None]
    if len(set(supplied)) != len(supplied):
        report.add(name, FAIL, ["duplicate global_index values"],
                   "Either assign one corpus-wide ordinal per chunk, or pass "
                   "global_index=None and let the host assign ordinals at "
                   "ingest. Duplicates are the one dishonest option.")
    elif not supplied:
        report.add(name, PASS,
                   ["global_index is None on every chunk -- legal; a host "
                    "that needs ordinals assigns them at ingest"])
    elif len(supplied) < len(chunks):
        report.add(name, WARN,
                   [f"{len(chunks) - len(supplied)} of {len(chunks)} chunks "
                    "have global_index=None while the rest carry ordinals"],
                   "Mixed ordinals are ambiguous for a host that sorts by "
                   "them: supply global_index for every chunk or for none.")
    else:
        report.add(name, PASS)


def _check_metadata_keys(report: Report, chunks: list[ChunkRecord]) -> None:
    with_fp = sum(1 for c in chunks
                  if (c.metadata or {}).get("source_fingerprint"))
    if with_fp == 0:
        report.add(
            "Recommended metadata (source_fingerprint for change detection)",
            WARN, ["no chunk carries metadata['source_fingerprint']"],
            "Optional, but without it a host's ability to detect that your "
            "source documents changed is weakened. Supply source_identity, "
            "source_fingerprint, relative_path, file_type if your system "
            "knows them.")
    else:
        report.add(
            "Recommended metadata (source_fingerprint for change detection)",
            PASS, [f"{with_fp}/{len(chunks)} chunks carry a source_fingerprint"])


def _check_stability(report: Report, pipeline: RagPipeline,
                     chunks: list[ChunkRecord], skip: bool) -> None:
    if skip:
        report.add("fingerprint() + chunk_id stability across pulls", SKIP,
                   ["--skip-stability given (large corpus?)"])
        return
    try:
        fp1 = pipeline.fingerprint()
        fp2 = pipeline.fingerprint()
        second = pipeline.pull_all_chunks()
    except Exception as exc:
        report.add("fingerprint() + chunk_id stability across pulls", FAIL,
                   [f"raised: {exc!r}"],
                   "fingerprint() and repeated pulls must work; the baseline "
                   "cache and drift detection call them.")
        return
    problems: list[str] = []
    if not isinstance(fp1, str) or not fp1:
        problems.append(f"fingerprint() returned {fp1!r}, not a non-empty str")
    elif fp1 != fp2:
        problems.append(
            f"fingerprint() changed between two calls ({fp1!r} -> {fp2!r}) "
            "with no corpus change")
    ids1 = {c.chunk_id for c in chunks}
    ids2 = {c.chunk_id for c in second}
    if ids1 != ids2:
        gone = list(ids1 - ids2)[:3]
        new = list(ids2 - ids1)[:3]
        problems.append(
            f"chunk_ids differ between two pulls (e.g. gone={gone!r}, "
            f"new={new!r})")
    if problems:
        report.add(
            "fingerprint() + chunk_id stability across pulls", FAIL, problems,
            "Ids and fingerprint must be stable while the corpus is "
            "unchanged: the frozen answer key is keyed by chunk_id, so "
            "session-scoped or random ids make every future run score zero. "
            "Derive ids from durable store keys, never from enumeration "
            "order or uuid4().")
    else:
        report.add("fingerprint() + chunk_id stability across pulls", PASS,

                   [f"fingerprint={fp1!r}"])


def _check_query(report: Report, pipeline: RagPipeline, sample_query: str,
                 top_k: int, pulled_ids: set,
                 retrieval_mode: str = "scored") -> list[RetrievedChunk]:
    try:
        hits = pipeline.query(sample_query, top_k=top_k)
    except NotImplementedError:
        report.add("query() retrieves ranked chunks", FAIL,
                   ["query() is not implemented"],
                   "This is one of the two REQUIRED methods: send the text to "
                   "your retrieval endpoint and return ranked RetrievedChunks.")
        return []
    except Exception as exc:
        report.add("query() retrieves ranked chunks", FAIL,
                   [f"query({sample_query!r:.60}) raised: {exc!r}"],
                   "Every Test Set question goes through this call; an "
                   "exception here is recorded as a retrieval error for the "
                   "question (never a zero), but a connector that always "
                   "raises cannot be evaluated.")
        return []
    if not isinstance(hits, list):
        report.add("query() retrieves ranked chunks", FAIL,
                   [f"returned {type(hits).__name__}, not list"],
                   "Return a list of rag_connector.base.RetrievedChunk.")
        return []
    if not hits:
        report.add(
            "query() retrieves ranked chunks", WARN,
            [f"no results for the probe query {sample_query!r:.60}"],
            "An empty result is legal (it scores as a genuine zero), but the "
            "validator cannot exercise id matching without hits -- re-run "
            "with --query \"a question your corpus can answer\".")
        return []
    bad_type = [h for h in hits if not isinstance(h, RetrievedChunk)]
    if bad_type:
        report.add("query() retrieves ranked chunks", FAIL,
                   [f"items are {type(bad_type[0]).__name__}, "
                    "not RetrievedChunk"],
                   "Map each raw result into rag_connector.base.RetrievedChunk.")
        return []
    details = [f"{len(hits)} hits for probe {sample_query!r:.60}"]
    if len(hits) > top_k and retrieval_mode != "complete_set":
        report.add("query() retrieves ranked chunks", FAIL,
                   details + [f"returned {len(hits)} hits for top_k={top_k}"],
                   "Respect top_k: the harness treats the list as the "
                   "retrieval cutoff under test.")
        return hits
    if len(hits) > top_k:
        # complete_set: the system decides cardinality (e.g. neighbor
        # expansion), so more than top_k is legal -- informational, never a FAIL.
        details.append(f"returned {len(hits)} hits for top_k={top_k} -- legal "
                       "under complete_set (system-decided cardinality; top_k "
                       "is a breadth knob)")
    ranks = [h.rank for h in hits]
    if ranks != list(range(len(hits))):
        report.add("query() retrieves ranked chunks", WARN,
                   details + [f"rank fields are {ranks[:8]}, expected 0..n-1 "
                              "in list order"],
                   "The harness scores by LIST ORDER (position 0 = best); "
                   "keep the rank field consistent with it to avoid confusing "
                   "human readers of the run sheet.")
    else:
        report.add("query() retrieves ranked chunks", PASS, details)

    # The single most common connector bug gets its own check + verdict line.
    unknown = _unknown_scored_ids(hits, pulled_ids)
    if unknown:
        example_pulled = next(iter(pulled_ids)) if pulled_ids else "?"
        report.add(
            "query() ids match pull_all_chunks() ids (THE classic bug)", FAIL,
            [f"{len(unknown)} scored id(s) across {len(hits)} hits never "
             f"appear in the pulled corpus",
             f"e.g. scored {unknown[0]!r} vs pulled ids like "
             f"{example_pulled!r}"],
            "Return the SAME logical chunk_id from query() that "
            "pull_all_chunks() returns (not a raw store id, not a different "
            "join of the same parts). With mismatched formats every hit "
            "scores as a miss and recall flatlines at zero while looking "
            "like a working connector. A hit standing for a covering item "
            "(a parent document, a summary node, a curated answer) may "
            "instead declare metadata['item_covered_ids'] — the corpus ids "
            "an id-keyed harness credits it through — and is then validated "
            "through THOSE; its own row id is identity, not provenance.")
    else:
        report.add("query() ids match pull_all_chunks() ids (THE classic bug)",
                   PASS, ["every scored id exists in the pulled corpus"])
    return hits


def _unknown_scored_ids(hits: list[RetrievedChunk], pulled_ids: set) -> list:
    """The ids the harness would score that the pulled corpus does not hold.

    Each hit is validated through the ids it asks to be SCORED by. A hit that
    declares ``metadata["item_covered_ids"]`` is one returned item standing
    for the chunks it covers (a parent-document retriever's parent, a summary
    node, a curated answer): id-keyed scoring credits it through those covered
    ids, its own row id is synthetic identity, and so the covered ids are what
    must exist in the corpus. Declared-but-empty is a legal zero-coverage
    claim and asks for nothing; only an ABSENT declaration falls back to the
    row's own ``chunk_id``.
    """
    unknown = []
    for h in hits:
        meta = h.metadata if isinstance(h.metadata, dict) else {}
        covered = meta.get("item_covered_ids")
        if isinstance(covered, (list, tuple)):
            unknown.extend(c for c in covered if str(c) not in pulled_ids)
        elif h.chunk_id is None or str(h.chunk_id) not in pulled_ids:
            unknown.append(h.chunk_id)
    return unknown


def _check_derived_results(report: Report, hits: list[RetrievedChunk],
                           pulled_ids: set) -> None:
    """The derived-result declarations follow docs/contract.md, "Derived results".

    Whether declared covered ids exist in the corpus is the classic-bug check's
    job (it validates a declaring hit through them); this check covers the rest
    of the rules a host relies on to read a derived result correctly.
    """
    from .derived import (
        ITEM_COVERED_IDS_KEY,
        ITEM_INDEX_KEY,
        ITEM_KEYS,
        ITEM_KIND_KEY,
        derived_result_problems,
        read_derived_result,
    )

    name = "Derived-result declarations are well-formed"
    declaring = [h for h in hits
                 if isinstance(h.metadata, dict)
                 and any(key in h.metadata for key in ITEM_KEYS)]
    if not declaring:
        report.add(name, SKIP,
                   ["no hit declares item_* metadata: a plain chunk retriever, "
                    "credited through each row's own chunk_id"])
        return

    failures: list[str] = []
    warnings: list[str] = []
    for hit in declaring:
        for problem in derived_result_problems(hit):
            failures.append(f"rank {hit.rank} ({hit.chunk_id!r}): {problem}")

    # Rows sharing an item_index are ONE item; a host reads its covered ids and
    # kind once, so rows that disagree leave the item's meaning to row order.
    by_index: dict = {}
    for hit in declaring:
        index = hit.metadata.get(ITEM_INDEX_KEY)
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        key = (repr(hit.metadata.get(ITEM_COVERED_IDS_KEY)),
               hit.metadata.get(ITEM_KIND_KEY))
        first = by_index.setdefault(index, (hit.rank, key))
        if first[1] != key:
            failures.append(
                f"rows at rank {first[0]} and {hit.rank} share item_index "
                f"{index} but declare different item_covered_ids or item_kind")

    if not failures:
        for hit in declaring:
            result = read_derived_result(hit)
            if (result.covered_ids == () and not result.is_gap):
                warnings.append(
                    f"rank {hit.rank} ({hit.chunk_id!r}) declares zero covered "
                    f"chunks as kind {result.kind!r}: a host credits it with "
                    "nothing and does NOT read it as a verified 'no answer' -- "
                    "declare item_kind='gap' if that is what it is")
            if (result.kind != "chunk" and hit.chunk_id is not None
                    and str(hit.chunk_id) in pulled_ids):
                warnings.append(
                    f"rank {hit.rank}: a {result.kind} result carries the real "
                    f"corpus id {hit.chunk_id!r} as its own chunk_id -- give it "
                    "None or an id in a namespace of its own")
            if not result.kind_declared:
                warnings.append(
                    f"rank {hit.rank} ({hit.chunk_id!r}) declares no item_kind; "
                    f"read as {result.kind!r} by default -- declare it")

    details = [f"{len(declaring)} of {len(hits)} hit(s) declare a derived result"]
    if failures:
        report.add(name, FAIL, details + failures[:8],
                   "Follow docs/contract.md, 'Derived results': item_covered_ids "
                   "is a list of distinct chunk-id strings, item_kind is one of "
                   "chunk / summary / gap ('group' accepted as summary), a gap "
                   "covers no chunks, a chunk covers exactly itself, and rows "
                   "sharing an item_index agree. A host that has to guess what a "
                   "result stands for grades the wrong thing.")
    elif warnings:
        report.add(name, WARN, details + warnings[:8],
                   "Legal, but a host may read these results differently from "
                   "what you mean; see docs/contract.md, 'Derived results'.")
    else:
        report.add(name, PASS, details)


def _check_determinism(report: Report, pipeline: RagPipeline,
                       sample_query: str, top_k: int,
                       hits: list[RetrievedChunk]) -> None:
    if not hits:
        report.add("query() is deterministic for a repeated query", SKIP,
                   ["no hits to compare"])
        return
    try:
        again = pipeline.query(sample_query, top_k=top_k)
    except Exception as exc:
        report.add("query() is deterministic for a repeated query", WARN,
                   [f"second call raised: {exc!r}"],

                   "Intermittent failures show up as retrieval errors in "
                   "runs; investigate flakiness before trusting comparisons.")
        return
    if [h.chunk_id for h in hits] != [h.chunk_id for h in again]:
        report.add(
            "query() is deterministic for a repeated query", WARN,
            ["the same query returned a different ranking on a second call"],
            "Nondeterministic retrieval makes run-to-run comparison noisy: "
            "identical configurations will differ for reasons that are not "
            "the change under test. If your system has a stable-sort or "
            "seed option, use it for evaluation.")
    else:
        report.add("query() is deterministic for a repeated query", PASS)


def _check_score_direction(report: Report, hits: list[RetrievedChunk],
                           connection: dict | None, info: dict,
                           retrieval_mode: str = "scored") -> None:
    name = "Scores are canonical: higher = better"
    if retrieval_mode in ("ordered", "complete_set"):
        # Scores are declared evidence-only in these modes -- captured but never
        # used for ranking or thresholding -- so the direction/monotonicity
        # check is skipped (design section 3.9). None scores are legal here.
        report.add(name, SKIP,
                   [f"skipped for retrieval_mode={retrieval_mode!r}: scores are "
                    "captured as evidence, never used for ranking or a "
                    "threshold, so their direction is not required"])
        return
    if not hits:
        # Reported as a skipped row rather than omitted: an absent check reads
        # as "nothing to say about the scores", when what happened is that the
        # probe returned nothing to inspect them on.
        report.add(name, SKIP,
                   ["the probe query returned no hits, so there are no scores "
                    "to inspect -- this check did NOT run"],
                   "Re-run with --query \"a question your corpus can answer\" "
                   "to exercise the score convention.")
        return
    declared = (connection or {}).get("distance") or info.get("distance")
    metric_note = (f"store metric declared: {declared!r} (informational -- "
                   "direction is always higher-is-better)"
                   if declared else
                   "no store metric declared (optional; the direction rule "
                   "is universal)")
    none_hits = sum(1 for h in hits if h.score is None)
    if none_hits:
        # Under 'scored' a None score is a contract violation (design §3.9) —
        # the threshold filter and score display trust these values.
        report.add(
            name, FAIL,
            [f"{none_hits} of {len(hits)} hits carry score=None", metric_note],
            "Under the 'scored' retrieval mode every hit must carry a "
            "canonical higher-is-better float score. If your scores are not "
            "meaningful, declare retrieval_mode = 'ordered' (or "
            "'complete_set' if the system decides cardinality) instead of "
            "returning None.")
        return
    if len(hits) < 3:
        report.add(name, SKIP, [metric_note,
                                "too few hits to test score ordering"])
        return
    scores = [float(h.score) for h in hits]
    descending = all(a >= b for a, b in zip(scores, scores[1:]))
    strictly_ascending = all(a < b for a, b in zip(scores, scores[1:]))
    if descending:
        details = [f"scores run non-increasing down the ranking: {scores[:5]}",
                   metric_note]
        if any(getattr(h, "raw_score", None) is not None for h in hits):
            details.append("store-native values preserved in raw_score -- "
                           "good evidence practice")
        report.add(name, PASS, details)
    elif strictly_ascending:
        report.add(
            name, FAIL,
            [f"scores INCREASE down the ranking: {scores[:5]}",
             "best-first results must carry their best (highest) score at "
             "rank 0 -- these look like unconverted store distances "
             "(lower = closer)", metric_note],
            "RetrievedChunk.score is canonical higher-is-better. Convert "
            "store distances with the tested utilities in rag_connector.base -- "
            "cosine_distance_to_similarity (1 - d) or l2_distance_to_score "
            "(-d) -- and keep the native value in raw_score. A user's score "
            "threshold keeps score >= threshold; unconverted distances would "
            "keep exactly the wrong half.")
    else:
        report.add(
            name, WARN,
            [f"scores are not monotonic down the ranking: {scores[:5]}",
             metric_note],
            "Legal when a reranker or business prioritization reorders "
            "results (ranking is scored by list position), but confirm the "
            "scores really are higher-is-better -- the threshold filter "
            "(score >= threshold) trusts them.")


def _check_fetch(report: Report, pipeline: RagPipeline,
                 chunks: list[ChunkRecord]) -> None:
    name = "get_chunks() returns known ids, omits unknown ones"
    wanted = [c.chunk_id for c in chunks[:3]] + [_MISSING_ID_SENTINEL]
    overridden = supports(pipeline, "get_chunks")

    # For a real override, count full corpus pulls made while answering a
    # 4-id fetch: an override that internally calls pull_all_chunks() is the
    # base default's O(corpus) cost wearing a "custom implementation" label.
    pull_count = 0
    absent = object()
    saved = getattr(pipeline, "__dict__", {}).get("pull_all_chunks", absent)
    instrumented = False
    if overridden:
        real_pull = pipeline.pull_all_chunks

        def _counting_pull(*args, **kwargs):
            nonlocal pull_count
            pull_count += 1
            return real_pull(*args, **kwargs)

        try:
            pipeline.pull_all_chunks = _counting_pull
            instrumented = True
        except AttributeError:  # __slots__ etc.: skip the cost accounting
            pass
    try:
        fetched = pipeline.get_chunks(wanted)
    except Exception as exc:
        report.add(name, FAIL, [f"raised: {exc!r}"],
                   "Hosts use this to hydrate ids that fall outside a frozen "
                   "corpus snapshot; the base-class default (filter a full "
                   "pull) works if you don't override it.")
        return
    finally:
        if instrumented:
            if saved is absent:
                del pipeline.pull_all_chunks
            else:
                pipeline.pull_all_chunks = saved
    problems: list[str] = []
    if not isinstance(fetched, dict):
        problems.append(f"returned {type(fetched).__name__}, not dict")

    else:
        missing_known = [cid for cid in wanted[:-1] if cid not in fetched]
        if missing_known:
            problems.append(f"known ids not returned: {missing_known!r}")
        if _MISSING_ID_SENTINEL in fetched:
            problems.append(
                "invented a record for an id that does not exist "
                f"({_MISSING_ID_SENTINEL!r})")
        mis_keyed = [k for k, v in fetched.items()
                     if isinstance(v, ChunkRecord) and v.chunk_id != k]
        if mis_keyed:
            problems.append(f"dict keys don't match record chunk_ids: "
                            f"{mis_keyed[:3]!r}")
    if problems:
        report.add(name, FAIL, problems,
                   "Return {chunk_id: ChunkRecord} for the ids you can find "
                   "and simply omit the rest -- never invent placeholders.")
        return
    if not overridden:
        report.add(
            name, WARN,
            ["using the base-class default: every by-id fetch filters a FULL "
             "corpus pull -- correct results, O(corpus) cost, and it inherits "
             "pull_all_chunks()'s failure modes (one bad unrelated chunk can "
             "break a by-id read of a valid id)"],
            "Override get_chunks with a direct by-id lookup so integrity "
            "checks can sample and inspection can hydrate ids without "
            "downloading the whole corpus.")
        return
    if pull_count:
        report.add(
            name, WARN,
            [f"custom get_chunks() pulled the full corpus {pull_count} "
             f"time(s) to answer a {len(wanted)}-id fetch -- the base "
             "default's O(corpus) cost wearing an override's label"],
            "Answer by-id fetches from the store's by-id read, not from "
            "pull_all_chunks().")
        return
    report.add(name, PASS, ["custom by-id implementation"])


def _check_shadow_names(report: Report, pipeline: RagPipeline) -> None:
    """The renamed-override bug: a public method whose name closely resembles a
    contract method the class does NOT override. The base default answers, the
    results look right, and the author's implementation is dead code."""
    name = "No stray method shadows an inherited contract default"
    cls = type(pipeline)
    inherited = [m for m in _DEFAULTED_CONTRACT_METHODS
                 if getattr(cls, m, None) is getattr(RagPipeline, m, None)]
    if not inherited:
        report.add(name, PASS,
                   ["every defaulted contract method is overridden"])
        return
    defined: set[str] = set()
    for klass in cls.__mro__:
        if klass is RagPipeline:
            break
        for attr_name, value in vars(klass).items():
            if callable(value) or isinstance(value, (staticmethod, classmethod)):
                defined.add(attr_name)
    known = set(_DEFAULTED_CONTRACT_METHODS) | _EXPECTED_EXTRA_METHODS
    suspects = [
        f"{candidate!r} resembles {match[0]!r}, which is NOT overridden -- "
        "the base default is answering for it"
        for candidate in sorted(defined)
        if not candidate.startswith("_") and candidate not in known
        for match in [difflib.get_close_matches(candidate, inherited,
                                                n=1, cutoff=0.6)]
        if match
    ]
    if suspects:
        report.add(
            name, WARN, suspects,
            "If this method was meant to implement the contract, rename it to "
            "the contract name: right now the base default silently answers "
            "in its place (correct results, O(corpus) cost, your code never "
            "runs). If it is intentional extra API, ignore this warning.")
    else:
        report.add(name, PASS)


# Consecutive empty-but-not-final pages tolerated before paging counts as
# stalled. A few are legal (a store may skip a filtered-out segment); an
# unbounded run is a pager that never advances.
_PAGING_STALL_LIMIT = 3


def _check_paging(report: Report, pipeline: RagPipeline,
                  chunks: list[ChunkRecord]) -> None:
    """Exercise an overridden list_chunks: cursor discipline + corpus parity.

    Skipped when inherited -- the base derives paging from pull_all_chunks, so
    there is no separate mechanism to test.
    """
    name = "list_chunks() pages consistently with pull_all_chunks()"
    if not supports(pipeline, "list_chunks"):
        report.add(name, SKIP,
                   ["not overridden -- the base class derives paging from "
                    "pull_all_chunks(), so there is no separate path to test"])
        return
    limit = max(1, (len(chunks) + 4) // 5)   # aim for ~5 pages
    # Termination is judged by progress, not by a page budget: the contract
    # lets a page hold fewer than `limit` items (a backend cap such as a
    # store's list endpoint maximum), so a budget that assumes full pages
    # fails honest small-page pagers. Instead, three things end the loop:
    #   - a repeated cursor is a loop, full stop;
    #   - an empty page that still hands back a cursor makes no progress, and
    #     more than _PAGING_STALL_LIMIT of those in a row is a stalled pager;
    #   - every other page yields at least one item, so collecting more than
    #     the corpus plus one requested page means the pager is re-serving
    #     items and would never reach None.
    # Together these bound the loop for any pager, whatever its page size.
    overrun_at = len(chunks) + limit
    seen_cursors: set[str] = set()
    collected: list = []
    cursor: str | None = None
    pages = 0
    empty_streak = 0
    largest_page = 0
    tripped: str | None = None
    try:
        while True:
            page = pipeline.list_chunks(cursor=cursor, limit=limit)
            items = list(getattr(page, "items", None) or [])
            collected.extend(items)
            pages += 1
            largest_page = max(largest_page, len(items))
            cursor = getattr(page, "next_cursor", None)
            if cursor is None:
                break
            empty_streak = empty_streak + 1 if not items else 0
            if cursor in seen_cursors:
                tripped = (f"cursor {cursor!r} repeated on page {pages} -- "
                           "paging would loop forever")
            elif empty_streak > _PAGING_STALL_LIMIT:
                tripped = (f"{empty_streak} consecutive empty pages still "
                           f"returned a next_cursor (last {cursor!r}) -- "
                           "paging stalled and would never terminate")
            elif len(collected) > overrun_at:
                tripped = (f"paging overran the corpus: {len(collected)} "
                           f"items over {pages} pages for a {len(chunks)}-chunk "
                           "corpus and next_cursor is still not None -- the "
                           "pager is re-serving items and would never "
                           "terminate")
            if tripped:
                report.add(
                    name, FAIL, [tripped],
                    "next_cursor must advance every page and become None on "
                    "the last one; pull_all_chunks() derives from this loop "
                    "and would hang the same way.")
                return
            seen_cursors.add(cursor)
    except Exception as exc:
        report.add(name, FAIL, [f"raised: {exc!r}"],
                   "Hosts that prefer paged reads call this exact method; fix "
                   "the exception or remove the override to use the derived "
                   "default.")
        return
    problems: list[str] = []
    non_records = [c for c in collected if not isinstance(c, ChunkRecord)]
    if non_records:
        problems.append(
            f"page items are {type(non_records[0]).__name__}, not ChunkRecord")
    else:
        paged_ids = {str(c.chunk_id) for c in collected}
        pulled_ids = {str(c.chunk_id) for c in chunks}
        if len(collected) != len(chunks):
            problems.append(
                f"paged read returned {len(collected)} chunks; "
                f"pull_all_chunks() returned {len(chunks)}")
        missing = list(pulled_ids - paged_ids)[:3]
        extra = list(paged_ids - pulled_ids)[:3]
        if missing or extra:
            problems.append(
                f"paged ids diverge from pulled ids (e.g. missing={missing!r}, "
                f"extra={extra!r})")
    if problems:
        report.add(name, FAIL, problems,
                   "The two corpus reads are the same question at two "
                   "granularities and must agree chunk-for-chunk; a host that "
                   "freezes via one and inspects via the other would see two "
                   "different corpora.")
    else:
        details = [f"{pages} page(s) at limit={limit} reproduce the full "
                   "corpus"]
        if pages > 1 and largest_page < limit:
            details.append(
                f"pages capped at {largest_page} items (< requested "
                f"limit={limit}) -- legal: the contract does not require "
                "full pages")
        report.add(name, PASS, details)


def _check_chunk_vectors(report: Report, pipeline: RagPipeline,
                         chunks: list[ChunkRecord]) -> None:
    """Exercise an overridden get_chunk_vectors: honesty rules from the base
    docstring -- no fabricated vectors, gaps stay visible, empty input is {}."""
    name = "get_chunk_vectors() reads honest indexed vectors"
    if not supports(pipeline, "get_chunk_vectors"):
        report.add(name, SKIP,
                   ["not implemented -- scores can only be trusted as "
                    "declared, which is a supported configuration"])
        return
    try:
        if pipeline.get_chunk_vectors([]) != {}:
            report.add(name, FAIL,
                       ["empty input did not return an empty dict"],
                       "Nothing was asked for, so nothing can be missing: {} "
                       "is the complete answer, without a backend call.")
            return
    except Exception as exc:
        report.add(name, FAIL, [f"empty input raised: {exc!r}"],
                   "Empty input returns {} without touching the backend.")
        return
    wanted = [c.chunk_id for c in chunks[:3]]
    try:
        vectors = pipeline.get_chunk_vectors(wanted + [_MISSING_ID_SENTINEL])
    except Exception as exc:
        report.add(name, FAIL, [f"raised: {exc!r}"],
                   "Return the vectors the index holds for the ids it knows "
                   "and omit the rest; raise ConnectorOperationalError only "
                   "for a real backend failure.")
        return
    problems: list[str] = []
    if not isinstance(vectors, Mapping):
        problems.append(f"returned {type(vectors).__name__}, not a mapping")
        vectors = {}
    if _MISSING_ID_SENTINEL in vectors:
        problems.append(
            "fabricated a vector for an id that does not exist -- a "
            "fabricated vector is worse than a gap because nothing "
            "downstream can detect it")
    bad_values = []
    for key, value in vectors.items():
        try:
            [float(component) for component in list(value)[:8]]
        except (TypeError, ValueError):
            bad_values.append(key)
    if bad_values:
        problems.append(f"non-numeric vector values for ids {bad_values[:3]!r}")
    if problems:
        report.add(name, FAIL, problems,
                   "Return {chunk_id: [float, ...]} for the ids the index "
                   "holds, straight out of the index -- never re-embedded, "
                   "never padded, never invented.")
        return
    missing_known = [cid for cid in wanted if cid not in vectors]
    if missing_known:
        report.add(name, WARN,
                   [f"known ids returned no vector: {missing_known!r}"],
                   "A gap is legal and visibly honest, but these chunks came "
                   "straight from pull_all_chunks() -- confirm the index "
                   "really holds no vector for them.")
        return
    dimension = len(list(next(iter(vectors.values()))))
    report.add(name, PASS,
               [f"{len(vectors)} vector(s) of dimension {dimension}; the "
                "missing-id probe stayed absent"])


def _check_generate(report: Report, pipeline: RagPipeline,
                    sample_query: str, top_k: int) -> None:
    name = "generate() -- optional answer-generation path"
    if not supports(pipeline, "generate"):
        report.add(name, SKIP,
                   ["not implemented -- connector is retrieval-only; any "
                    "host feature that evaluates generated answers is simply "
                    "unavailable, which is a supported configuration"])
        return
    try:
        answer = pipeline.generate(sample_query, top_k=top_k)
    except Exception as exc:
        report.add(
            name, WARN, [f"generate() raised at call time: {exc!r}"],
            "If this is a construction-time requirement (an LLM or service "
            "handle the validator can't supply), configure it and re-run. "
            "But RUNTIME failures during generation must be returned as "
            "GeneratedAnswer(error=str(exc)), never raised -- the run records "
            "the failure instead of dying.")
        return
    if not isinstance(answer, GeneratedAnswer):
        report.add(name, FAIL,
                   [f"returned {type(answer).__name__}, not GeneratedAnswer"],
                   "Return rag_connector.base.GeneratedAnswer (query, answer, "
                   "contexts, citations, hydrated_prompt, error).")
        return
    details = []
    if answer.error:
        details.append(f"generation reported a captured error: {answer.error!r:.80}")
        details.append("(error capture is the CORRECT failure behavior)")
    else:
        details.append(f"answer of {len(answer.answer)} chars, "
                       f"{len(answer.citations)} citation(s), "
                       f"{len(answer.contexts)} context(s)")
    non_str = [c for c in answer.citations if not isinstance(c, str)]
    if non_str:
        report.add(name, WARN, details +
                   [f"citations contain non-string values: {non_str[:3]!r}"],
                   "Citations are compared as strings against context "
                   "chunk_ids; keep them str. Keep unmappable (hallucinated) "
                   "citations verbatim so they score zero support.")
    else:
        report.add(name, PASS, details)


def _check_prompt_templates(report: Report, pipeline: RagPipeline,
                            chunks: list[ChunkRecord], sample_query: str,
                            top_k: int) -> None:
    """Exercise the declared-prompt-template capability, end to end.

    Enumerate -> render a block from chunks the VALIDATOR supplies -> render
    each declared template -> generate from one of them, then check the trace
    that comes back. Skipped entirely when the capability is not declared,
    which is a supported configuration: a connector keeps building its prompt
    inside ``generate`` and reporting it afterwards as a string.
    """
    name = "Declared prompt templates -- optional fixed-context generation path"
    present = declared_prompt_methods(pipeline)

    if not present:
        report.add(name, SKIP,
                   ["not implemented -- this connector declares no prompt "
                    "templates, so a host can only capture its prompt after "
                    "the fact through generate(). Fixed-context groundedness "
                    "evaluation is unavailable, which is supported"])
        return

    if not supports_declared_prompts(pipeline):
        missing = [m for m in DECLARED_PROMPT_METHODS if m not in present]
        report.add(
            name, FAIL,
            [f"implements {list(present)} but not {missing}"],
            "This capability is declared as a UNIT: enumerate a variant "
            "(prompt_templates), render a block from supplied chunks "
            "(render_rag_block), generate from the rendered prompt "
            "(generate_from_prompt). Two out of three cannot be used by a "
            "host -- it would hold templates nothing can run, or accept a "
            "prompt nobody can enumerate a template for. Implement all three "
            "or none.")
        return

    # -- enumerate ----------------------------------------------------------
    try:
        templates = list(pipeline.prompt_templates())
    except Exception as exc:
        report.add(name, FAIL, [f"prompt_templates() raised: {exc!r}"],
                   "prompt_templates() must answer from a declaration, not "
                   "from the backend: a host reads it to render a picker "
                   "before anything is connected.")
        return
    if not templates:
        report.add(name, FAIL, ["prompt_templates() returned nothing"],
                   "Return at least one PromptTemplate, or do not implement "
                   "the capability at all. An empty list says 'this connector "
                   "declared zero prompts', which is a different fact from "
                   "'this connector has no declared prompt'.")
        return

    details = [f"{len(templates)} declared variant(s): "
               + ", ".join(repr(getattr(t, "id", "?")) for t in templates)]
    problems: list[str] = []

    seen_ids: dict[str, int] = {}
    for index, template in enumerate(templates):
        if not isinstance(template, PromptTemplate):
            problems.append(f"entry {index} is a {type(template).__name__}, "
                            "not a PromptTemplate")
            continue
        label = template.id or f"<entry {index}>"
        for problem in prompt_template_problems(template):
            problems.append(f"{label}: {problem}")
        if template.id in seen_ids:
            problems.append(
                f"{label}: id declared twice (entries {seen_ids[template.id]} "
                f"and {index}). Ids address a variant, so a Test Set naming "
                "this id could not say which prompt it meant")
        seen_ids[template.id] = index

    if problems:
        report.add(
            name, FAIL, details + problems,
            "Every declared template must be a legal ONE-SHOT RAG prompt: "
            "exactly one {rag_block}, at least one {query}, no other "
            "placeholder, no conversation history, and a cache breakpoint "
            "that is actually reachable. History is REFUSED rather than "
            "warned about, because a host reports these prompts as "
            "single-turn -- see rag_connector.prompts.")
        return

    # -- render a block from chunks the HOST supplies ------------------------
    supplied = [
        RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=None, rank=i,
                       doc_id=c.doc_id)
        for i, c in enumerate(chunks[:max(1, top_k)])
    ]
    try:
        block = pipeline.render_rag_block(supplied)
    except Exception as exc:
        report.add(name, FAIL, details + [f"render_rag_block() raised: {exc!r}"],
                   "render_rag_block(chunks) takes chunks the HOST chose and "
                   "renders them in this connector's own block format. It "
                   "must not retrieve, and must not depend on anything but "
                   "the chunks it was handed.")
        return
    if not isinstance(block, str) or not block.strip():
        report.add(name, FAIL, details +
                   [f"render_rag_block() returned {type(block).__name__} "
                    f"{block!r:.60}"],
                   "Return the block as a non-empty string -- it is what gets "
                   "substituted for the template's rag_block placeholder.")
        return
    details.append(f"render_rag_block() produced {len(block)} chars from "
                   f"{len(supplied)} supplied chunk(s)")

    carried = [c.chunk_id for c in supplied
               if (c.text or "")[:24] and (c.text or "")[:24] in block]
    if not carried:
        report.add(
            name, WARN, details +
            ["none of the supplied chunk texts appears in the rendered block"],
            "The block is the context an answer is supposed to be grounded "
            "in. If the supplied text does not reach it, a groundedness score "
            "measures something else. (Deliberate truncation can trip this "
            "heuristic -- if that is what happened, it is not a bug.)")
        return

    # -- render each declared template --------------------------------------
    first_rendered = None
    for template in templates:
        try:
            rendered = render_prompt(
                template, query=sample_query, rag_block=block,
                chunk_ids=[str(c.chunk_id) for c in supplied])
        except Exception as exc:
            report.add(name, FAIL, details +
                       [f"{template.id!r} failed to render: {exc!r}"],
                       "A declared template must render into a one-shot "
                       "prompt through rag_connector.prompts.render_prompt.")
            return
        user = rendered.user_message.content
        if block not in user or sample_query not in user:
            missing_part = "block" if block not in user else "query"
            report.add(name, FAIL, details +
                       [f"{template.id!r} rendered without its {missing_part}"],
                       "Both the rag_block and the query must survive "
                       "rendering into the single user message.")
            return
        breakpoint_ = rendered.cache_breakpoint
        if breakpoint_ is None:
            details.append(f"{template.id!r}: no cache breakpoint declared "
                           "(supported, and the correct declaration for a "
                           "prompt that is not cacheable as written)")
            first_rendered = first_rendered or rendered
            continue
        # Prefix stability, proved rather than sniffed. Re-render the SAME
        # template and the SAME block under a different question: if the
        # declared prefix is genuinely query-independent it comes back byte
        # for byte, and if it does not, the breakpoint would miss on every
        # call. Checking whether the query STRING appears in the prefix is
        # the wrong test -- a retrieved block routinely contains the words of
        # the question that retrieved it, which is a coincidence of content,
        # not a dependency.
        prefix = rendered.cacheable_prefix()
        other = render_prompt(
            template, query=sample_query + " (prefix-stability probe)",
            rag_block=block,
            chunk_ids=[str(c.chunk_id) for c in supplied]).cacheable_prefix()
        if prefix is None or prefix != other:
            report.add(
                name, FAIL, details +
                [f"{template.id!r} declares cache_breakpoint "
                 f"{breakpoint_.position!r}, but the prefix before it changes "
                 "when the question changes"],
                "Caching matches on an exact prefix, so a prefix that varies "
                "with the question is never reused. Nothing reorders your "
                "template for you -- fidelity to the prompt you actually ship "
                "beats a cache saving. Declare a separate variant whose "
                "template genuinely puts the block first, or declare 'none'.")
            return
        details.append(
            f"{template.id!r}: cache breakpoint {breakpoint_.position!r} at "
            f"message {breakpoint_.message_index}, offset "
            f"{breakpoint_.char_offset} ({len(prefix)}-char stable prefix)")
        first_rendered = first_rendered or rendered

    # -- generate from a rendered prompt ------------------------------------
    try:
        answer = pipeline.generate_from_prompt(first_rendered,
                                               contexts=supplied)
    except Exception as exc:
        report.add(
            name, WARN, details +
            [f"generate_from_prompt() raised at call time: {exc!r}"],
            "If this is a construction-time requirement the validator cannot "
            "supply (a generation service handle), configure it and re-run. "
            "But RUNTIME failures must be returned as "
            "GeneratedAnswer(error=str(exc)) with the trace still attached, "
            "never raised -- a failed generation whose prompt is recorded is "
            "diagnosable; one whose prompt is lost is not.")
        return

    if not isinstance(answer, GeneratedAnswer):
        report.add(name, FAIL, details +
                   [f"returned {type(answer).__name__}, not GeneratedAnswer"],
                   "generate_from_prompt() returns the same GeneratedAnswer "
                   "shape generate() does, with .trace populated -- so every "
                   "existing consumer keeps working unchanged.")
        return

    trace = answer.trace
    trace_problems: list[str] = []
    if trace is None:
        trace_problems.append("GeneratedAnswer.trace is None")
    else:
        if trace.template_sha256 != first_rendered.template_sha256:
            trace_problems.append(
                "trace records a different template hash than the prompt it "
                f"was given ({trace.template_sha256[:12]} vs "
                f"{first_rendered.template_sha256[:12]})")
        if trace.block_sha256 != first_rendered.block_sha256:
            trace_problems.append(
                "trace records a different RAG-block hash than the prompt it "
                "was given")
        if not trace.messages:
            trace_problems.append("trace carries no messages")
        try:
            json.dumps(trace.to_dict())
        except (TypeError, ValueError) as exc:
            trace_problems.append(f"trace.to_dict() is not JSON: {exc}")
        details.append(
            f"trace: model={trace.model!r}, {len(trace.messages)} message(s), "
            f"{len(trace.response)} response chars, "
            f"usage={trace.usage.to_dict()}")
    if not answer.hydrated_prompt:
        trace_problems.append(
            "hydrated_prompt is empty. The trace SUPPLEMENTS it; it does not "
            "replace it, and existing consumers still read it")
    if answer.error:
        details.append(
            f"generation reported a captured error: {answer.error!r:.80}")
        details.append("(error capture is the CORRECT failure behavior)")

    if trace_problems:
        report.add(name, FAIL, details + trace_problems,
                   "generate_from_prompt() must return a GeneratedAnswer with "
                   "a JSON-serializable GenerationTrace describing the call it "
                   "actually made, AND with hydrated_prompt still populated.")
    else:
        report.add(name, PASS, details)


def _check_spec_prompt_templates(report: Report, spec,
                                 pipeline: RagPipeline) -> None:
    """The registry declaration must be readable BEFORE anything is bound.

    A host renders a prompt-variant picker from a registry listing, where no
    params, credentials or backend exist yet -- the same reason
    ``connector_info`` is zero-arg. A connector that only answers off a bound
    instance forces an operator to connect something before they can see what
    they are choosing between.
    """
    name = "ConnectorSpec.prompt_templates (readable before binding)"
    declares_on_instance = supports(pipeline, "prompt_templates")

    if not spec.supports_prompt_templates:
        if declares_on_instance:
            report.add(
                name, WARN,
                ["the bound instance declares prompt templates, but the "
                 "registered ConnectorSpec does not"],
                "Set prompt_templates=<zero-arg callable> on your "
                "ConnectorSpec, returning the same declared set the instance "
                "returns. Without it a host cannot show the variants until "
                "after a connector is bound -- and binding is exactly what an "
                "operator is trying to configure.")
        else:
            report.add(name, SKIP, ["no prompt templates declared"])
        return

    try:
        declared = list(spec.prompt_templates())
    except Exception as exc:
        report.add(name, FAIL, [f"spec.prompt_templates() raised: {exc!r}"],
                   "It is called with no arguments from a registry listing: "
                   "no params, no credentials, no backend.")
        return

    problems = [f"{t.id!r}: {p}"
                for t in declared if isinstance(t, PromptTemplate)
                for p in prompt_template_problems(t)]
    if problems:
        report.add(name, FAIL, problems,
                   "The pre-binding declaration is held to the same contract "
                   "as the instance one -- it is what an operator chooses "
                   "from.")
        return

    spec_ids = [getattr(t, "id", "?") for t in declared]
    if declares_on_instance:
        instance_ids = [t.id for t in pipeline.prompt_templates()]
        if spec_ids != instance_ids:
            report.add(
                name, FAIL,
                [f"registry declares {spec_ids}, instance declares "
                 f"{instance_ids}"],
                "Serve ONE declared set through both surfaces (a module-level "
                "constant returned by each). If they disagree, the variant an "
                "operator picked from the listing is not the variant that "
                "runs, and the result is labelled with an id it was not "
                "measured on.")
            return
    report.add(name, PASS,
               [f"declared before binding: {spec_ids}"]
               + ([] if declares_on_instance else
                  ["(the bound instance does not implement prompt_templates; "
                   "a host can list the variants but not run them)"]))


def _secret_key_paths(value, prefix: str = "") -> list[str]:
    """Every secret-looking key in ``value``, at ANY depth, as dotted paths.

    Top-level-only scanning is how a credential hides in plain sight: a
    connection dict with ``{"auth": {"api_key": ...}}`` is exactly as persisted
    and exactly as leaked as one with ``api_key`` at the root.
    """
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(hint in str(key).lower() for hint in _SECRET_KEY_HINTS):
                found.append(path)
            found.extend(_secret_key_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found.extend(_secret_key_paths(item, f"{prefix}[{i}]"))
    return found


def _check_connection(report: Report, connection: dict) -> None:
    problems: list[str] = []
    try:
        json.dumps(connection)
    except (TypeError, ValueError) as exc:
        problems.append(f"connection dict is not JSON-serializable: {exc}")
    secret_keys = _secret_key_paths(connection)
    if secret_keys:
        problems.append(f"secret-looking keys persisted: {secret_keys!r}")
    if problems:
        report.add(
            "Connection dict is persistable and secret-free", FAIL, problems,
            "The connection dict is stored in plaintext by the host and "
            "must fully reconstruct the connector: JSON-serializable values "
            "only, and NO credentials -- read those from the environment in "
            "__init__ instead.")
    else:
        report.add("Connection dict is persistable and secret-free", PASS,
                   [f"keys: {sorted(connection)}"])


def _check_rebuild(report: Report, spec, connection: dict,
                   sample_query: str, top_k: int, pulled_ids: set) -> None:

    """Registry round-trip: the persisted dict alone must rebuild the connector."""
    name = "build(connection) reconstructs the connector (registry round-trip)"
    try:
        rebuilt = spec.build(dict(connection))
        rebuilt_hits = rebuilt.query(sample_query, top_k=top_k)
    except Exception as exc:
        report.add(
            name, FAIL, [f"raised: {exc!r}"],
            "The evaluation loop reconstructs your connector in a FRESH "
            "process from nothing but the persisted connection dict. "
            "Everything __init__ needs must be in that dict (or the "
            "environment).")
        return
    # Same §-unit rule as the classic-bug check: a hit declaring
    # item_covered_ids is validated through those, not through its row id.
    unknown = _unknown_scored_ids(rebuilt_hits, pulled_ids)
    if unknown:
        report.add(name, FAIL,
                   [f"rebuilt connector returned unknown ids: {unknown[:3]!r}"],
                   "The rebuilt connector must address the same corpus with "
                   "the same id format as the one that froze the corpus.")
    else:
        report.add(name, PASS,
                   [f"rebuilt connector answered the probe with "
                    f"{len(rebuilt_hits)} known-id hit(s)"])


def _derive_probe_query(chunks: list[ChunkRecord]) -> str:
    """A retrieval probe from corpus text when the user supplies none."""
    middle = chunks[len(chunks) // 2]
    return " ".join((middle.text or "").split()[:8]) or "test"


def validate_pipeline(pipeline: RagPipeline, *, target: str,
                      sample_query: str | None = None, top_k: int = 5,
                      connection: dict | None = None, spec=None,
                      skip_stability: bool = False) -> Report:
    """Run every contract check against a constructed pipeline. Pure library
    entry point -- the CLI below and any future in-app pre-flight both call
    this, so the two surfaces can never drift."""
    report = Report(target=target)
    info = _check_info(report, pipeline)
    _check_run_snapshot(report, pipeline)
    retrieval_mode = _check_mode(report, info)
    _check_shadow_names(report, pipeline)
    chunks = _check_pull(report, pipeline)
    if not chunks:
        return report   # everything downstream needs the corpus
    _check_contract(report, chunks)
    _check_indexes(report, chunks)
    _check_metadata_keys(report, chunks)
    _check_stability(report, pipeline, chunks, skip_stability)

    probe = (sample_query or "").strip() or _derive_probe_query(chunks)
    if not sample_query:
        report.add("Probe query", WARN,
                   [f"derived from corpus text: {probe!r:.60}"],
                   "Pass --query \"a question your corpus can answer\" for a "
                   "realistic retrieval probe; a text-fragment probe can "
                   "under-exercise your query path.")
    pulled_ids = {str(c.chunk_id) for c in chunks}
    hits = _check_query(report, pipeline, probe, top_k, pulled_ids,
                        retrieval_mode)
    _check_derived_results(report, hits, pulled_ids)
    _check_determinism(report, pipeline, probe, top_k, hits)
    if retrieval_mode == "complete_set":
        _check_cardinality_probe(report, pipeline, probe, top_k)
    _check_score_direction(report, hits, connection, info, retrieval_mode)
    _check_fetch(report, pipeline, chunks)
    _check_paging(report, pipeline, chunks)
    _check_chunk_vectors(report, pipeline, chunks)
    _check_generate(report, pipeline, probe, top_k)
    _check_prompt_templates(report, pipeline, chunks, probe, top_k)
    if spec is not None:
        _check_spec_prompt_templates(report, spec, pipeline)
    if connection is not None:
        _check_connection(report, connection)
    if spec is not None and connection is not None:
        _check_rebuild(report, spec, connection, probe, top_k, pulled_ids)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_target(import_target: str, kwargs: dict) -> RagPipeline:
    """Resolve ``module.path:AttrName`` and construct/call it with kwargs."""
    module_name, _, attr_name = import_target.partition(":")
    if not module_name or not attr_name:
        raise SystemExit(
            f"--import expects 'module.path:ClassOrFactory', got {import_target!r}")
    module = importlib.import_module(module_name)
    attr: Callable = getattr(module, attr_name)
    pipeline = attr(**kwargs)
    if not isinstance(pipeline, RagPipeline):
        raise SystemExit(
            f"{import_target} produced {type(pipeline).__name__}, which is not "
            "a RagPipeline subclass -- subclass rag_connector.base.RagPipeline.")
    return pipeline


def main(argv: list[str] | None = None, *,
         prog: str = "python -m rag_connector.validate") -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate a RAG connector against the RagPipeline "
                    "contract without running a host application.")
    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument("--import", dest="import_target", metavar="MOD:ATTR",
                        help="Import path to your connector class or factory, "
                             "e.g. my_pkg.my_connector:MyRagConnector")
    source.add_argument("--connector-type", metavar="TYPE",
                        help="A registered connector_type; exercises the full "
                             "connect()/build() registry round-trip")
    parser.add_argument("--kwargs", default="{}", metavar="JSON",
                        help="JSON kwargs for the --import class/factory")
    parser.add_argument("--params", default="{}", metavar="JSON",
                        help="JSON form params for --connector-type connect()")
    parser.add_argument("--query", default=None,
                        help="A question your corpus can answer (recommended)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip the second full corpus pull (large corpora)")
    args = parser.parse_args(argv)

    # Registry mode reconstructs a real connector that may need cloud
    # credentials (Bedrock, Pinecone, ...), so load a .env from the directory
    # the operator ran this from. Best-effort and registry-only: direct-import
    # mode keeps its promise to import nothing beyond the connector's own
    # environment.
    #
    # The working directory, explicitly -- NOT this file's location, which is
    # inside site-packages for an installed wheel, and NOT a bare
    # load_dotenv(), whose find_dotenv() walks up and can stop at a stray
    # shadowing .env that lacks the cloud credentials.
    if args.connector_type:
        from pathlib import Path

        env_path = Path.cwd() / ".env"
        try:
            from dotenv import load_dotenv
        except ImportError:
            # python-dotenv is deliberately not a runtime dependency (the core
            # stays dependency-light), which makes a silent skip here the
            # ambient-package trap: .env loading works wherever some OTHER
            # project installed dotenv and silently dies on a clean install,
            # surfacing as credential errors that point nowhere near the
            # cause. If there is a .env to load, say out loud that it wasn't.
            if env_path.exists():
                print(
                    "WARNING: ./.env exists but python-dotenv is not "
                    "installed, so it will NOT be loaded. Install "
                    "python-dotenv, or export the variables directly.",
                    file=sys.stderr,
                )
        else:
            try:
                load_dotenv(env_path)
            except Exception:
                pass

    try:
        if args.import_target:
            pipeline = _load_target(args.import_target, json.loads(args.kwargs))
            report = validate_pipeline(
                pipeline, target=args.import_target, sample_query=args.query,
                top_k=args.top_k, skip_stability=args.skip_stability)
        else:
            from .registry import get_connector, registered_types
            spec = get_connector(args.connector_type)
            if spec is None:
                print(f"Connector type {args.connector_type!r} is not "
                      f"registered. Registered types: {registered_types()}",
                      file=sys.stderr)
                return 1
            if spec.connect is None:
                print(f"Connector {args.connector_type!r} registers no "
                      "connect(); validate it with --import instead.",
                      file=sys.stderr)
                return 1
            pipeline, connection = spec.connect(json.loads(args.params))
            report = validate_pipeline(
                pipeline, target=f"registered connector {args.connector_type!r}",
                sample_query=args.query, top_k=args.top_k,
                connection=connection, spec=spec,
                skip_stability=args.skip_stability)
    except SystemExit:
        raise
    except Exception as exc:  # construction/e.g. import errors: report cleanly
        print(f"Validator could not run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    print(report.render())
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
