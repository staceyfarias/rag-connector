# Changelog

Notable changes to rag-connector. Dates are the day the work landed.

## Unreleased

Additive.

- **Dataset spec 1.1, draft** (`docs/dataset-spec.md`). Documentation only;
  the core and the library are unchanged and still write `spec_version`
  `"1.0"`. The draft defines every top-level entry in a Dataset folder: the
  core; shared areas `sources/`, `testsets/`, `analysis/<tool>/` and `docs/`;
  tool-private areas `.<tool>/` (with `extensions/<tool>/` still valid
  throughout 1.x); and free entries owned by whoever keeps the folder. It adds
  the rule that only `data/chunks.jsonl` is corpus, and reserves
  `.rag-connector/` for connector attachments. Sections the library does not
  implement yet are marked as such.

- **A shared Datasets directory** (`rag_connector.datasets_dir`). One place
  every product looks for Datasets: an explicit path, else
  `RAG_CONNECTOR_DATASETS_DIR`, else `datasets_dir` in
  `~/.rag-connector/config.json`, else the default: `datasets/` in the repository
  when running from a source checkout (gitignored), else
  `~/rag-connector/datasets`. A checkout also ships the prechunked Pelorus Space
  v2.1 Dataset (`ds-900cf7746b744988`) in `examples/pelorus-space-v2.1/`, to be
  copied into the directory. API:
  `datasets_dir`, `resolve_datasets_dir` (also says which rule chose it),
  `set_datasets_dir`, `list_dataset_folders` (manifest only; does not verify),
  `default_datasets_dir`, `config_path`. CLI: `rag-connector datasets-dir
  [--set PATH | --unset] [--json]` and `rag-connector dataset list`;
  `dataset build` and `dataset export` now default `--out` to a folder inside
  the directory (an explicit `--out` behaves as before).

## 0.2.0a2 — 2026-09-28

Additive, except the bundled demo corpus, which changes.

- **Changed: the bundled Pelorus Space corpus is version 2** (2026-09-27):
  28 documents and 426 frozen chunks at 1000/200, replacing 92 documents and
  605 chunks. Every chunk id and the bundled Dataset id change, so anything
  that cited version-1 chunk ids must be rebuilt. Version 1 was two layers,
  20 guides plus the 72 short documents they were written from; version 2
  keeps one realistic set (formal terms, guides, brochures, an FAQ, a dated
  notice), fixes accidental contradictions, and adds deliberate test
  material (a dated conflicting notice; two scope-in-another-chunk cases).
  See `src/rag_connector/corpus/README.md`.

- **Dataset spec, `rag-connector-dataset` 1.0** (`docs/dataset-spec.md`,
  versioned separately from this package) and `rag_connector.dataset`. A
  Dataset folder has a write-once core (`dataset.json`, `data/chunks.jsonl`,
  `data/sources.json`), a reserved `testsets/` area defined by the Test Set
  spec, and one `extensions/<tool>/` area per tool behind an `extension.json`
  envelope. Readers verify `chunk_inventory_sha256` (testset-kit's
  `chunk-inventory-v1`, adopted exactly) and `core_sha256` and refuse a
  mismatch; the Dataset id is derived from the chunk inventory, so a re-chunk
  is a new Dataset. API: `write_dataset_folder`, `read_dataset_folder`,
  `chunk_inventory_sha256`, `source_manifest_from_chunks`, `list_extensions`,
  `write_extension_envelope`. RAGauge's pre-spec Dataset folders stay readable
  through `read_legacy_dataset_folder` (read-only, unverified, never migrated).
- `CHUNKER_METHOD` / `CHUNKER_VERSION` in the ingest kit name the splitting
  behaviour a Dataset records.
- **`rag-connector dataset build`** (chunk a documents folder, no vector store)
  and **`rag-connector dataset export`** (freeze a connector's corpus, without
  embeddings), with library forms `build_dataset_from_folder` and
  `export_dataset_from_connector`.
- **Derived results** in the contract: the four-field model (covered chunk ids,
  served text, kind `chunk`/`summary`/`gap` with `group` accepted as
  `summary`, optional curator notes) plus `item_index` / `item_label`, with the
  key constants and helpers in `rag_connector.derived`. The validator checks
  these declarations when a connector makes them.

## 0.2.0a1 — first public release

The first release published to PyPI, and the first from this public repository.
Development before it happened in a private repository whose history is not
carried here; its `v0.1.0a0` and `v0.2.0a0` tags were never published, and this
version is numbered after them so no version goes backwards.

What it contains:

- The black-box retrieval contract (`RagPipeline`, `ChunkRecord`,
  `RetrievedChunk`), retrieval-mode and score semantics, and stable chunk
  identity.
- Optional capabilities: corpus reading, query embedding, indexed chunk vectors,
  generation, dataset listing, run snapshots, publication and telemetry.
- The connector registry (`rag_connector.connectors` entry points) with
  reconstructable, secret-free connection documents.
- The conformance validator and reusable connector test kit
  (`rag-connector validate`).
- The Reference RAG, a local FastEmbed + Chroma implementation (`reference`
  extra).
- The ingest kit (`rag_connector.ingest`) and the bundled Pelorus Space demo
  corpus: 92 documents and the frozen 605-chunk split that RAGauge and Pelorus
  Query both cite.
