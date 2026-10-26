"""Command-line tools for discovery, validation, Datasets, and the Reference RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import all_specs


def _add_reference_commands(subparsers: argparse._SubParsersAction) -> None:
    reference = subparsers.add_parser(
        "reference",
        help="provision the built-in FastEmbed + Chroma Reference RAG",
    )
    commands = reference.add_subparsers(dest="reference_command", required=True)
    provision = commands.add_parser(
        "provision",
        help="ingest a document folder into a reusable local Reference RAG",
    )
    provision.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="folder containing supported documents; omit to provision the "
             "Pelorus Space corpus bundled with this package",
    )
    provision.add_argument("--persist-dir", default="./rag_connector_data/reference")
    provision.add_argument("--collection", required=True)
    provision.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    provision.add_argument("--chunk-size", type=int, default=1000)
    provision.add_argument("--chunk-overlap", type=int, default=200)
    provision.add_argument(
        "--append",
        action="store_true",
        help="append to an existing collection instead of resetting it",
    )
    provision.add_argument(
        "--normalized",
        choices=("true", "false"),
        default=None,
        help="declare whether this space's vectors are L2-normalized; omit "
             "when you do not know (hosts read that as unverifiable, which is "
             "not the same as false)",
    )


def _add_dataset_commands(subparsers: argparse._SubParsersAction) -> None:
    dataset = subparsers.add_parser(
        "dataset",
        help="write a Dataset folder (docs/dataset-spec.md)",
    )
    commands = dataset.add_subparsers(dest="dataset_command", required=True)
    build = commands.add_parser(
        "build",
        help="chunk a folder of documents into a new Dataset (no vector store)",
    )
    build.add_argument("--folder", required=True,
                       help="folder containing supported documents")
    build.add_argument("--out", default=None,
                       help="new Dataset folder; must not exist or be empty "
                            "(default: a folder named for the Dataset inside "
                            "the Datasets directory, see `rag-connector "
                            "datasets-dir`)")
    build.add_argument("--chunk-size", type=int, default=1000)
    build.add_argument("--chunk-overlap", type=int, default=200)
    build.add_argument("--name", default=None,
                       help="Dataset name (default: the folder's name)")
    export = commands.add_parser(
        "export",
        help="freeze a connector's corpus, as its system chunked it, into a "
             "new Dataset",
    )
    export.add_argument("--connector", required=True, metavar="TYPE",
                        help="a registered connector_type (see `rag-connector list`)")
    export.add_argument("--params", default="{}", metavar="JSON",
                        help="JSON params for the connector's connect() "
                             "(or its connection document, if it has no connect)")
    export.add_argument("--out", default=None,
                        help="new Dataset folder; must not exist or be empty "
                             "(default: inside the Datasets directory)")
    export.add_argument("--name", default=None,
                        help="Dataset name (default: the connector type)")
    audit = commands.add_parser(
        "audit",
        help="compare a Dataset with a connector's corpus as it is now; "
             "changes nothing, records the audit in the Dataset",
    )
    audit.add_argument("--dataset", required=True, metavar="PATH",
                       help="the Dataset folder to audit")
    audit.add_argument("--connector", required=True, metavar="TYPE",
                       help="a registered connector_type (see `rag-connector list`)")
    audit.add_argument("--params", default="{}", metavar="JSON",
                       help="JSON params for the connector's connect() "
                            "(or its connection document, if it has no connect)")
    audit.add_argument("--label", default=None,
                       help="a name for this connector instance, e.g. 'staging'")
    audit.add_argument("--host", default=None,
                       help="the application auditing, e.g. 'ragauge'")
    audit.add_argument("--no-record", action="store_true",
                       help="print the audit without storing it in the Dataset")
    audits = commands.add_parser(
        "audits",
        help="list the audits recorded in a Dataset",
    )
    audits.add_argument("--dataset", required=True, metavar="PATH")
    sync = commands.add_parser(
        "sync",
        help="take a NEW Dataset from a connector's corpus as it is now, "
             "recording which Dataset (and audit) it follows; the source is "
             "never changed",
    )
    sync.add_argument("--dataset", required=True, metavar="PATH",
                      help="the source Dataset folder (read, never written)")
    sync.add_argument("--connector", required=True, metavar="TYPE",
                      help="a registered connector_type (see `rag-connector list`)")
    sync.add_argument("--params", default="{}", metavar="JSON",
                      help="JSON params for the connector's connect() "
                           "(or its connection document, if it has no connect)")
    sync.add_argument("--out", required=True, metavar="PATH",
                      help="the new Dataset folder; must not exist or be empty")
    sync.add_argument("--audit", default=None, metavar="AUDIT_ID",
                      help="an audit recorded in the source, for this same "
                           "connector instance, that motivated the copy")
    sync.add_argument("--name", default=None,
                      help="the new Dataset's name (default: the source's)")
    sync.add_argument("--label", default=None,
                      help="a name for this connector instance, e.g. 'staging'")
    sync.add_argument("--host", default=None,
                      help="the application syncing, e.g. 'ragauge'")
    listing = commands.add_parser(
        "list",
        help="list the Dataset folders in the Datasets directory "
             "(reads each manifest only; does not verify)",
    )
    listing.add_argument("--dir", default=None,
                         help="list this directory instead of the Datasets "
                              "directory")


def _add_index_commands(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser(
        "index",
        help="build or search a Dataset's corpus index, in analysis/rag-connector/",
    )
    commands = index.add_subparsers(dest="index_command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="write the book index (names, titles, headings -> chunk ids); "
             "calls no model",
    )
    prepare.add_argument("--stopword", action="append", default=[], metavar="WORD",
                         help="a capitalized word that names nothing in this corpus; "
                              "repeatable, recorded in the index")
    prepare.add_argument("--no-search-db", action="store_true",
                         help="write only the book index, not search.sqlite")
    status = commands.add_parser("status", help="how big the indexes are")
    lookup = commands.add_parser(
        "lookup", help="print book-index entries and threads whose label contains QUERY",
    )
    lookup.add_argument("query")
    signal = commands.add_parser(
        "signal",
        help="label each document with a model (needs the llm extra and "
             "ANTHROPIC_API_KEY); skips documents already labelled",
    )
    signal.add_argument("--model", default=None,
                        help="the labelling model (default: claude-haiku-5-5)")
    signal.add_argument("--limit", type=int, default=None,
                        help="label at most this many documents this run")
    signal.add_argument("--workers", type=int, default=6)
    signal_lookup = commands.add_parser(
        "signal-lookup", help="print signal-index subjects whose words contain QUERY",
    )
    signal_lookup.add_argument("query")
    search_db = commands.add_parser(
        "search-db",
        help="build the full-text search index (search.sqlite); calls no model",
    )
    search_db.add_argument("--out", default=None, metavar="PATH",
                           help="write the file here (a file, or a directory to hold "
                                "search.sqlite) instead of the Dataset's analysis area; "
                                "nothing in the Dataset folder is written")
    search = commands.add_parser(
        "search",
        help="full-text search: words, phrases, NEAR, substrings, regex, filters",
    )
    search.add_argument("--term", action="append", default=[], metavar="TERM",
                        help="every --term must match (AND); several words are a "
                             "phrase, a trailing * a prefix")
    search.add_argument("--any", action="append", default=[], metavar="TERM",
                        help="at least one --any must match (OR)")
    search.add_argument("--not", dest="not_", action="append", default=[], metavar="TERM",
                        help="no --not may match")
    search.add_argument("--phrase", action="append", default=[], metavar="TEXT",
                        help="an exact word sequence")
    search.add_argument("--near", nargs="+", default=None, metavar="TERM",
                        help="two or more terms within --within words of each other")
    search.add_argument("--within", type=int, default=10, metavar="N",
                        help="the word distance for --near (default 10)")
    search.add_argument("--regex", action="append", default=[], metavar="PATTERN",
                        help="a Python regular expression, case-insensitive "
                             "unless it says (?-i)")
    search.add_argument("--substring", action="append", default=[], metavar="TEXT",
                        help="a case-insensitive substring, punctuation included "
                             "($18, 41.5, BE-FAST)")
    search.add_argument("--substring-terms", action="store_true",
                        help="match --term/--any/--not as substrings, not whole words")
    search.add_argument("--titles", action="store_true",
                        help="let word terms match the document title too")
    search.add_argument("--module", action="append", default=[], metavar="ID",
                        help="only this module (task_id, or an unambiguous prefix)")
    search.add_argument("--general", action="store_true",
                        help="only chunks with no module (channel general info); "
                             "with --module, those modules plus general")
    search.add_argument("--doc", action="append", default=[], metavar="DOC",
                        help="only this document (doc_id, prefix, or exact title)")
    search.add_argument("--exclude-doc", action="append", default=[], metavar="DOC",
                        help="leave this document out")
    search.add_argument("--type", action="append", default=[], metavar="TYPE",
                        help="only this resource_type or type (video, url, "
                             "general_info, ...)")
    search.add_argument("--limit", type=int, default=None,
                        help="show at most N hits (the total is still reported)")
    neighbors = commands.add_parser(
        "neighbors", aliases=["neighbours"],
        help="a chunk with the chunks around it in its document, full text",
    )
    neighbors.add_argument("chunk", help="a chunk id or an unambiguous prefix")
    neighbors.add_argument("--radius", type=int, default=1)
    outline = commands.add_parser(
        "outline", help="a document's chunks in order, with their headings",
    )
    outline.add_argument("doc", help="a doc id, prefix, exact title, or a chunk id in it")
    copies = commands.add_parser(
        "copies", help="a chunk's identical, near and overlapping copies",
    )
    copies.add_argument("chunk", help="a chunk id or an unambiguous prefix")
    for command in (search, neighbors, outline, copies):
        command.add_argument("--db", default=None, metavar="PATH",
                             help="a search index built with --out (default: the "
                                  "Dataset's analysis/rag-connector/search.sqlite)")
        command.add_argument("--json", action="store_true", help="print JSON")
    for command in (prepare, status, lookup, signal, signal_lookup, search_db, search,
                    neighbors, outline, copies):
        command.add_argument("--dataset", required=True, metavar="PATH",
                             help="the Dataset folder")


def _emit(text: str) -> None:
    """Write to stdout without failing on a console that cannot show a character."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        pass
    sys.stdout.write(text)


def _emit_json(payload) -> None:
    """JSON, escaped to ASCII unless stdout is UTF-8, so it always parses."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    _emit(json.dumps(payload, indent=2, ensure_ascii=encoding != "utf8") + "\n")


def _short(value) -> str:
    return "(general)" if value is None else str(value)[:8]


def _format_search(result) -> str:
    lines = [f"{result.total} chunks match, in {len(result.by_doc)} documents"
             + (f" (showing {len(result.hits)})" if result.truncated else "")]
    if result.by_module:
        lines.append("modules: " + ", ".join(f"{_short(m['task_id'])} {m['hits']}"
                                             for m in result.by_module))
    for number, hit in enumerate(result.hits, 1):
        bits = [hit["chunk_id"], hit["source"],
                f"chunk {hit['doc_chunk_index']}"
                + (f" ({hit['chunk_label']})" if hit["chunk_label"] else ""),
                f"module {_short(hit['task_id'])}"]
        if hit["resource_type"] or hit["type"]:
            bits.append(hit["resource_type"] or hit["type"])
        if hit["score"] is not None:
            bits.append(f"score {hit['score']}")
        lines.append("")
        lines.append(f"{number}. " + " | ".join(bits))
        if hit["heading"]:
            lines.append(f"   under: {hit['heading']}")
        lines.append(f"   {hit['snippet']}")
        if hit["copies"]:
            lines.append("   copies: " + ", ".join(
                f"{kind} {info['group']} ({info['size']})"
                for kind, info in sorted(hit["copies"].items())))
    if result.by_doc:
        lines += ["", "by document:"]
        lines += [f"  {d['hits']:>5}  {d['source']}  [{d['doc_id']}]" for d in result.by_doc]
    return "\n".join(lines) + "\n"


def _format_neighbors(items) -> str:
    lines = []
    for item in items:
        mark = "  <== asked for" if item["target"] else ""
        label = f" ({item['chunk_label']})" if item["chunk_label"] else ""
        lines.append(f"=== {item['chunk_id']} | {item['source']} | chunk "
                     f"{item['doc_chunk_index']}{label}{mark}")
        if item["heading_before"]:
            lines.append(f"[under: {item['heading_before']}]")
        lines.append(item["plain"].rstrip())
        lines.append("")
    return "\n".join(lines)


def _format_outline(items) -> str:
    if not items:
        return "no chunks\n"
    lines = [f"{items[0]['source']}  [{items[0]['doc_id']}]  module "
             f"{_short(items[0]['task_id'])}  {len(items)} chunks"]
    for item in items:
        heading = item["heading"] or (f"(under: {item['heading_before']})"
                                      if item["heading_before"] else "")
        lines.append(f"  {item['doc_chunk_index']!s:>4}  {item['chunk_label'] or '-':<8} "
                     f"{item['chunk_id']}  {item['chars']:>6}  {heading}")
    return "\n".join(lines) + "\n"


def _format_copies(found) -> str:
    lines = [found["chunk_id"]]
    for kind in ("identical", "near", "overlap"):
        items = found[kind]
        if not items:
            lines.append(f"  {kind}: none")
            continue
        group = found.get(kind + "_group")
        lines.append(f"  {kind}" + (f" ({group})" if group else "") + f": {len(items)}")
        for item in items:
            extra = ""
            if kind == "overlap":
                c = item.get("containment")
                extra = (f"  containment {c} / larger {item['containment_larger']}"
                         if c is not None else "  (linked through another chunk)")
                if not item["grouped"]:
                    extra += "  (contained, not grouped)"
            lines.append(f"    {item['chunk_id']}  {item['source']}  chunk "
                         f"{item['doc_chunk_index']}{extra}")
    return "\n".join(lines) + "\n"


def _run_search_command(args: argparse.Namespace) -> int:
    from . import search_index

    folder = Path(args.dataset)
    if args.index_command == "search-db":
        path = search_index.build(folder, out=args.out)
        with search_index.SearchIndex(path) as index:
            meta = index.meta
        print(f"search index: {meta['chunk_count']} chunks, {meta['document_count']} "
              f"documents, {path.stat().st_size / 1e6:.1f} MB")
        print(f"  {path}")
        return 0
    with search_index.open_index(folder, path=args.db) as index:
        if args.index_command == "search":
            near = (*args.near, args.within) if args.near else None
            result = index.search(
                all=args.term, any=args.any, not_=args.not_, phrase=args.phrase,
                near=near, regex=args.regex, substring=args.substring,
                word=not args.substring_terms, modules=args.module,
                general_only=args.general, docs=args.doc, exclude_docs=args.exclude_doc,
                types=args.type, titles=args.titles, limit=args.limit)
            if args.json:
                _emit_json(result.to_dict())
            else:
                _emit(_format_search(result))
        elif args.index_command in ("neighbors", "neighbours"):
            items = index.neighbors(args.chunk, radius=args.radius)
            _emit_json(items) if args.json else _emit(_format_neighbors(items))
        elif args.index_command == "outline":
            items = index.outline(args.doc)
            _emit_json(items) if args.json else _emit(_format_outline(items))
        else:
            found = index.copies(args.chunk)
            _emit_json(found) if args.json else _emit(_format_copies(found))
    return 0


def _run_index_command(args: argparse.Namespace) -> int:
    from . import corpus_index, signal_index

    folder = Path(args.dataset)
    if args.index_command in ("search-db", "search", "neighbors", "neighbours",
                              "outline", "copies"):
        return _run_search_command(args)
    if args.index_command == "prepare":
        counts = corpus_index.prepare(folder, stopwords=args.stopword)
        print(f"indexed {counts['documents']} documents, {counts['chunks']} chunks, "
              f"{counts['entries']} entries, {counts['threads']} threads")
        print(f"  {corpus_index.markdown_path(folder)}")
        if not args.no_search_db:
            from . import search_index

            path = search_index.build(folder)
            print(f"search index: {path.stat().st_size / 1e6:.1f} MB")
            print(f"  {path}")
        return 0
    if args.index_command == "signal":
        result = signal_index.build(folder, model=args.model or signal_index.MODEL,
                                    limit=args.limit, workers=args.workers)
        print(f"labelled {result['labelled']} this run; tokens in "
              f"{result['tokens_in']} out {result['tokens_out']}")
        print(f"wrote {result['documents']} documents, {result['subjects']} subjects")
        print(f"  {signal_index.signal_markdown(folder)}")
        for item in result["failures"][:15]:
            print(f"  failed: {item}", file=sys.stderr)
        return 1 if result["failures"] else 0
    if args.index_command == "signal-lookup":
        payload = signal_index.load(folder)
        if payload is None:
            print("no signal index for this Dataset; `rag-connector index signal` builds one")
            return 1
        print(signal_index.format_hits(signal_index.find(payload, args.query)), end="")
        return 0
    index = corpus_index.load(folder)
    if index is None:
        print("no book index for this Dataset; `rag-connector index prepare` builds one")
        return 1
    if args.index_command == "status":
        counts = index["counts"]
        print(f"book index: {counts['documents']} documents, {counts['chunks']} chunks, "
              f"{counts['entries']} entries, {counts['threads']} threads "
              f"(built {index['built_at']})")
        signal = signal_index.load(folder)
        if signal is not None:
            subjects = sum(len(d.get("subjects") or []) for d in signal["documents"])
            print(f"signal index: {len(signal['documents'])} documents, {subjects} subjects "
                  f"(built {signal['built_at']}, {', '.join(signal['models'])})")
        print(_search_status(folder))
        print(f"  {corpus_index.analysis_dir(folder)}")
        return 0
    print(corpus_index.format_find(corpus_index.find(index, args.query)), end="")
    return 0


def _search_status(folder: Path) -> str:
    from . import search_index

    try:
        with search_index.open_index(folder) as index:
            meta = index.meta
    except search_index.SearchIndexMissing:
        return "search index: none (`rag-connector index search-db` builds one)"
    except search_index.SearchIndexError as exc:
        return f"search index: not usable: {exc}"
    return (f"search index: {meta['chunk_count']} chunks (built {meta['built_at']}, "
            f"{search_index.search_db_path(folder).stat().st_size / 1e6:.1f} MB)")


def _add_datasets_dir_command(subparsers: argparse._SubParsersAction) -> None:
    datasets_dir = subparsers.add_parser(
        "datasets-dir",
        help="show, set or clear the shared Datasets directory",
    )
    group = datasets_dir.add_mutually_exclusive_group()
    group.add_argument("--set", metavar="PATH", default=None,
                       help="save PATH as the Datasets directory "
                            "(~/.rag-connector/config.json)")
    group.add_argument("--unset", action="store_true",
                       help="clear the saved setting and use the default")
    datasets_dir.add_argument("--json", action="store_true",
                              help="print the directory, where it came from "
                                   "and whether it exists, as JSON")


def _dataset_summary(dataset) -> dict:
    return {
        "path": str(dataset.path),
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "chunk_count": dataset.chunk_count,
        "source_count": dataset.source_summary.get("source_count"),
        "chunking": dict(dataset.chunking or {}),
        "chunk_inventory_sha256": dataset.chunk_inventory_sha256,
        "core_sha256": dataset.core_sha256,
        "corpus_fingerprint": dataset.corpus_fingerprint,
    }


def _audit_brief(summary) -> dict:
    keys = ("audit_id", "created_at", "path", "problem", "in_sync",
            "connector", "backend", "chunk_counts", "document_counts")
    brief = {key: summary[key] for key in keys if key in summary}
    if "signals" in summary:
        brief["signals"] = {s["code"]: s["count"] for s in summary["signals"]}
    return brief


def _run_audit_command(args: argparse.Namespace) -> int:
    from .audit import audit_dataset, record_audit
    from .dataset import read_dataset_folder
    from .registry import get_connector, registered_types

    dataset = read_dataset_folder(args.dataset)
    spec = get_connector(args.connector)
    if spec is None:
        print(f"Connector type {args.connector!r} is not registered. "
              f"Registered types: {registered_types()}", file=sys.stderr)
        return 1
    params = json.loads(args.params)
    if spec.connect is not None:
        pipeline, connection = spec.connect(params)
    else:
        pipeline, connection = spec.build(params), params
    try:
        result = audit_dataset(
            dataset, pipeline,
            connector_type=spec.connector_type, connection=connection,
            instance_label=args.label, host=args.host,
        )
    finally:
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
    if not args.no_record:
        result = record_audit(args.dataset, result)
    summary = dict(result.summary)
    if result.path is not None:
        summary["path"] = str(result.path)
    print(json.dumps(_audit_brief(summary), indent=2, sort_keys=True))
    return 0


def _run_sync_command(args: argparse.Namespace) -> int:
    from .registry import get_connector, registered_types
    from .sync import sync_dataset

    spec = get_connector(args.connector)
    if spec is None:
        print(f"Connector type {args.connector!r} is not registered. "
              f"Registered types: {registered_types()}", file=sys.stderr)
        return 1
    params = json.loads(args.params)
    if spec.connect is not None:
        pipeline, connection = spec.connect(params)
    else:
        pipeline, connection = spec.build(params), params
    try:
        result = sync_dataset(
            args.dataset, pipeline, args.out,
            connector_type=spec.connector_type, connection=connection,
            audit_id=args.audit, instance_label=args.label, host=args.host,
            name=args.name,
        )
    finally:
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
    printed = _dataset_summary(result.dataset)
    printed["derived_from"] = dict(result.dataset.derived_from or {})
    printed["sync"] = {
        "sync_id": result.record["sync_id"],
        "path": str(result.path),
        "same_chunk_inventory_as_source":
            result.record["same_chunk_inventory_as_source"],
        "audit": result.record["audit"],
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0


def _run_dataset_command(args: argparse.Namespace) -> int:
    from . import dataset as ds
    from .datasets_dir import datasets_dir, folder_name_for, list_dataset_folders

    if args.dataset_command == "list":
        for entry in list_dataset_folders(args.dir):
            print(json.dumps({
                "path": str(entry.path),
                "dataset_id": entry.dataset_id,
                "name": entry.name,
                "chunk_count": entry.chunk_count,
                "legacy": entry.legacy,
                "problem": entry.problem,
            }, sort_keys=True))
        return 0

    if args.dataset_command == "audits":
        from .audit import list_audits

        for summary in list_audits(args.dataset):
            print(json.dumps(_audit_brief(summary), sort_keys=True))
        return 0

    if args.dataset_command == "audit":
        return _run_audit_command(args)

    if args.dataset_command == "sync":
        return _run_sync_command(args)

    if args.dataset_command == "build":
        name = args.name or Path(args.folder).resolve().name
        out = args.out or datasets_dir(create=True) / folder_name_for(name)
        result = ds.build_dataset_from_folder(
            args.folder,
            out,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            name=args.name,
        )
    else:
        from .registry import get_connector, registered_types

        spec = get_connector(args.connector)
        if spec is None:
            print(f"Connector type {args.connector!r} is not registered. "
                  f"Registered types: {registered_types()}", file=sys.stderr)
            return 1
        params = json.loads(args.params)
        if spec.connect is not None:
            pipeline, _connection = spec.connect(params)
        else:
            pipeline = spec.build(params)
        try:
            name = args.name or spec.connector_type
            out = args.out or datasets_dir(create=True) / folder_name_for(name)
            result = ds.export_dataset_from_connector(
                pipeline,
                out,
                connector_type=spec.connector_type,
                name=name,
            )
        finally:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
    print(json.dumps(_dataset_summary(result), indent=2, sort_keys=True))
    return 0


def _run_datasets_dir_command(args: argparse.Namespace) -> int:
    from .datasets_dir import config_path, resolve_datasets_dir, set_datasets_dir

    try:
        if args.set is not None:
            set_datasets_dir(args.set)
        elif args.unset:
            set_datasets_dir(None)
        path, source = resolve_datasets_dir()
    except (OSError, ValueError) as exc:
        print(f"rag-connector datasets-dir: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({
            "path": str(path),
            "source": source,
            "exists": path.is_dir(),
            "config_file": str(config_path()),
        }, indent=2, sort_keys=True))
    else:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list[:1] == ["validate"]:
        from .validate import main as validate_main

        return validate_main(args_list[1:], prog="rag-connector validate")

    parser = argparse.ArgumentParser(prog="rag-connector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list installed connector plugins")
    _add_reference_commands(subparsers)
    _add_dataset_commands(subparsers)
    _add_datasets_dir_command(subparsers)
    _add_index_commands(subparsers)
    subparsers.add_parser(
        "validate",
        help="validate a connector (run with --help for validator options)",
    )
    args = parser.parse_args(args_list)

    if args.command == "dataset":
        from .errors import ConnectorError

        try:
            return _run_dataset_command(args)
        except (OSError, ValueError, ConnectorError) as exc:
            # A refusal (non-empty output folder, a chunk that breaks the
            # contract, a connector that cannot read its corpus) is an answer
            # for the operator, not a crash: name it and exit non-zero.
            print(f"rag-connector dataset: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
    if args.command == "index":
        try:
            return _run_index_command(args)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            # A Dataset that does not verify, a missing key or extra: name it.
            print(f"rag-connector index: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.command == "datasets-dir":
        return _run_datasets_dir_command(args)
    if args.command == "list":
        for spec in all_specs():
            print(f"{spec.connector_type}\t{spec.label}")
    elif args.command == "reference" and args.reference_command == "provision":
        # Provisioning is the demo system's own build path, not a connector
        # capability — see rag_connector.reference_provision.
        from .reference_provision import provision_reference_rag

        connector = provision_reference_rag(
            args.folder,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            replace_existing=not args.append,
            normalized=None if args.normalized is None else args.normalized == "true",
        )
        payload = {
            "connection": connector.connection(),
            "info": connector.info(),
        }
        # This command IS a handoff: it writes a store and exits, and whoever
        # reads it next is a different process. Chroma has no flush, so the
        # barrier available is to stop owning the directory before we go --
        # otherwise the engine keeps the store open for the rest of this
        # process. See ReferenceRagConnector.close().
        connector.close()
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


# Without this, ``python -m rag_connector.cli list`` imports the module, calls
# nothing, and exits 0 -- printing no connectors. The author guide diagnoses a
# registration problem by a connector being ABSENT from this listing, so a
# silent empty success is the one wrong answer it can give. ``validate.py``
# has carried the same guard all along.
if __name__ == "__main__":
    sys.exit(main())
