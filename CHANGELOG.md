# Changelog

Notable changes to rag-connector. Dates are the day the work landed.

## Unreleased

Additive.

- **A corpus index over a Dataset** (`rag_connector.corpus_index`,
  `rag_connector.signal_index`; `rag-connector index prepare|status|lookup|
  signal|signal-lookup`). Moved from testset-kit so every product reads one
  index. The book index maps names, document titles and section headings to
  the chunk ids that hold them, with neighbors and identical copies, and calls
  no model. The signal index has a model label each document with a few words
  and the chunk span that holds them; each document records the model, chunk
  ids come from the Dataset, never the model, and progress resumes. Both are
  written to the shared analysis area `analysis/rag-connector/`. Rebuilding
  the book index no longer deletes the signal index, as it did in testset-kit.
  The model call needs the new optional `llm` extra (`anthropic`); nothing in
  the core imports it.

- **The book index's extraction rules are corpus-neutral.** testset-kit's
  draft was tuned to one corpus: only `Post`, `District` and `Sec.` could
  lead a numbered name, and words such as `commander`, `chairman` and
  `accused` were stopwords. Now any capitalized word and a number is a
  numbered name (`Route 66`, `Building 4`), except dates, a postal code after
  an address comma, and a list number after a sentence; a line-initial or
  year-like one counts only when the corpus itself numbers things with that
  word. Corpus-specific stopwords are passed in (`stopwords=`,
  `--stopword`) and recorded in `index.json` (format version 2).

- **Analysis areas** (`write_analysis_envelope`, `list_analyses`,
  `AnalysisEnvelope`). The spec's `analysis/<tool>/analysis.json` envelope,
  until now documented only: it records the core it was built on and lists
  its artifacts, and an artifact it lists must be in the area.

- **Dataset spec 1.1, draft** (`docs/dataset-spec.md`). Documentation only;
  the core and the library are unchanged and still write `spec_version`
  `"1.0"`. The draft defines every top-level entry in a Dataset folder: the
  core; shared areas `sources/`, `testsets/`, `analysis/<tool>/` and `docs/`;
  tool-private areas `.<tool>/` (with `extensions/<tool>/` still valid
  throughout 1.x); and free entries owned by whoever keeps the folder. It adds
  the rule that only `data/chunks.jsonl` is corpus, and reserves
  `.rag-connector/` for connector attachments. Sections the library does not
  implement yet are marked as such.

- **`FINDINGS.md`, a convention for findings about a Dataset** (draft, in
  `docs/dataset-spec.md`; documentation only, no code). A top-level shared
  file where any tool or person records what is learned about the Dataset.
  Each finding is its own top-level entry starting with a `# ` heading;
  programs only append, people edit freely. Added to the layout, the shared
  areas, the rule on who writes where (rule 6 gains this one exception) and the
  rule on shared content (rule 8).

- **Dataset audits** (`rag_connector.audit`; `rag-connector dataset audit`
  and `dataset audits`). Compare a Dataset with one connector instance's
  corpus as it is now, and record what differs, chunk by chunk and document by
  document: unchanged, text changed, id changed (same text under a new id, as
  an index rebuild does), ambiguous, missing and added chunks; removed, added
  and changed documents; metadata keys added and removed; and an observed
  chunking profile of each side. An audit changes nothing. Any number are kept
  under `.rag-connector/audits/<audit-id>/`, each naming the instance it
  describes by `connection_sha256`, never by its configuration.

- **Sync into a copy** (`rag_connector.sync`; `rag-connector dataset sync`).
  Writes a **new** Dataset from a connector instance's corpus as it is now,
  never touching the Dataset it follows (whose Test Sets cite its chunk ids).
  Lineage between the two Datasets goes in the new core as the manifest key
  `derived_from` (`dataset_id`, `core_sha256`), written once and covered by
  `core_sha256`; the instance (`connection_sha256`) and the audit that
  motivated the copy go in `.rag-connector/syncs/<sync-id>/sync.json`, since
  they describe one backend. A cited audit must be of the source's core and of
  the same connector instance, or the sync is refused; a backend that moved on
  since the audit is recorded as `backend_matches_audit: false`.
  `write_dataset_folder` and `export_dataset_from_connector` gain
  `derived_from=`, and `Dataset` a `derived_from` property.

- **Dataset spec: rule 9, "a Dataset is static; the backend is not".**
  Anything that describes a backend (a connection, an id map, a "connected"
  status) is recorded per connector instance, never as one slot per Dataset,
  and is never shared between instances, even of one connector type over
  identical chunks.

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
