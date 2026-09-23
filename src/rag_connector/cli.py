"""Command-line tools for discovery, validation, and the Reference RAG."""

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


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list[:1] == ["validate"]:
        from .validate import main as validate_main

        return validate_main(args_list[1:], prog="rag-connector validate")

    parser = argparse.ArgumentParser(prog="rag-connector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list installed connector plugins")
    _add_reference_commands(subparsers)
    subparsers.add_parser(
        "validate",
        help="validate a connector (run with --help for validator options)",
    )
    args = parser.parse_args(args_list)

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
