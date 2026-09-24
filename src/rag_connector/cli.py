"""Command-line tools for discovery, validation, Datasets, and the Reference RAG."""

from __future__ import annotations

import argparse
import json
import sys

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
    build.add_argument("--out", required=True,
                       help="new Dataset folder; must not exist or be empty")
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
    export.add_argument("--out", required=True,
                        help="new Dataset folder; must not exist or be empty")
    export.add_argument("--name", default=None,
                        help="Dataset name (default: the connector type)")


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


def _run_dataset_command(args: argparse.Namespace) -> int:
    from . import dataset as ds

    if args.dataset_command == "build":
        result = ds.build_dataset_from_folder(
            args.folder,
            args.out,
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
            result = ds.export_dataset_from_connector(
                pipeline,
                args.out,
                connector_type=spec.connector_type,
                name=args.name or spec.connector_type,
            )
        finally:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
    print(json.dumps(_dataset_summary(result), indent=2, sort_keys=True))
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
