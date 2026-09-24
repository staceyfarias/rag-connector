# Dataset spec — `rag-connector-dataset` 1.0

A **Dataset** is a folder holding a frozen, chunked corpus: the chunks, the
list of source documents they came from, and a small manifest saying what the
folder holds and how it was split. Any tool can read one, and any tool can add
its own work to one without disturbing the others.

This page is normative. It is implemented by `rag_connector.dataset`, which
imports no optional extras. The spec is versioned **independently** of the
`rag-connector` package and of the [connector contract](contract.md): spec id
`rag-connector-dataset`, version `1.0`.

Building a Dataset is optional tooling, not part of the connector contract. A
connector is a read-only view of a system someone else built; a Dataset is
something a host writes. `rag-connector dataset build` and
`rag-connector dataset export` ([below](#command-line)) are two ways to make
one, but any tool that writes the files described here makes a conforming
Dataset.

## Folder layout

```
<dataset>/
  dataset.json          CORE — the manifest
  data/chunks.jsonl     CORE — one chunk row per line
  data/sources.json     CORE — the source list
  testsets/<name>/      reserved, shared area for Test Sets
  extensions/<tool>/    one area per tool
    extension.json      the envelope (this spec)
    ...                 anything else, in that tool's own format
```

- **The core** is the three files marked CORE. They are written together, once,
  and never changed.
- **`testsets/`** is reserved for Test Sets. Its contents are defined by the
  Test Set spec (testset-kit), not here; this spec only reserves the name.
- **`extensions/<tool>/`** belongs to one tool. The spec defines only the
  envelope file inside it. What else goes in the area is the tool's business,
  and nothing here defines, reads or validates it.

## Rules

1. **The core is write-once.** Nothing — not the tool that wrote it, not a user
   edit, not a migration — changes `dataset.json`, `data/chunks.jsonl` or
   `data/sources.json` after they are written. A human description, a context
   note, a scan result: none of these is core. They live in the tool's own
   extension area.
2. **Readers verify, and refuse a mismatch.** A reader recomputes
   `chunk_inventory_sha256` and `core_sha256` (and `chunk_count`,
   `dataset_id` and `corpus_fingerprint`) from the files and refuses the
   Dataset if any differs from the manifest. A Dataset whose recorded identity
   no longer matches its contents is not an older corpus; it is an unknown one.
3. **A re-chunk is a new Dataset.** Chunk ids are positions (`doc.md:chunk-3`)
   that a different split refills with different text. The id is derived from
   the chunk inventory, so a different split always has a different id.
4. **Readers ignore what they do not know**: unknown manifest keys, and
   extension areas belonging to other tools.
5. **No tool writes another tool's area.** A tool writes only
   `extensions/<its name>/`, and Test Sets in `testsets/` under the rules of the
   Test Set spec.
6. **The manifest does not list extensions.** Listing them would mean rewriting
   the write-once core whenever a tool arrived. Discover them by listing
   `extensions/*/extension.json`.

## `dataset.json` — the manifest

```json
{
  "spec": "rag-connector-dataset",
  "spec_version": "1.0",
  "dataset_id": "ds-bfc47256c82fc7b2",
  "name": "synthetic",
  "created_at": "2026-09-24T00:00:00Z",
  "chunking": {
    "method": "rag-connector-chunker",
    "chunker_version": "1",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "sources": {"source_count": 2, "file_types": {"md": 1, "txt": 1}},
  "chunk_count": 3,
  "chunk_inventory_sha256": "bfc47256c82fc7b2a2b0a487d7f2e4603fb41dfc7a2c360c72b24321fc6a8272",
  "corpus_fingerprint": "…64 hex…",
  "core_sha256": "…64 hex…"
}
```

| Key | Meaning |
| --- | --- |
| `spec`, `spec_version` | Always `rag-connector-dataset`, and the version the writer followed. A reader reads any `1.x` and refuses another major version. |
| `dataset_id` | `ds-` followed by the first 16 hex digits of `chunk_inventory_sha256`. Content-derived: identical chunk ids and texts always get the same id, whatever the name, time or tool; a re-chunk always gets a new one. |
| `name` | Human-readable name. Not part of the id. |
| `created_at` | When the core was written, ISO 8601, UTC. |
| `chunking` | How the corpus was split — see below. Required. |
| `sources` | A summary of the source list: `source_count`, and `file_types` (count per file type). |
| `chunk_count` | Number of rows in `data/chunks.jsonl`. |
| `chunk_inventory_sha256` | The chunk-inventory fingerprint — see below. |
| `corpus_fingerprint` | The existing source-level fingerprint (`rag_connector.fingerprints.corpus_fingerprint`) over `data/sources.json`. It changes when a source document changes, and is blind to how the documents were split — which is why it is not the Dataset's identity. |
| `core_sha256` | The digest over all three core files — see below. |

Unknown keys are allowed and ignored by readers. Note what that does and does
not permit: a writer (for example a later `1.x` writer) may include extra keys
**when it writes the core**, and they are covered by `core_sha256` like every
other key. Adding a key afterwards is an edit to the core, and readers refuse it.

### The `chunking` block

`method` is required. Two methods are defined:

| `method` | Other keys | Means |
| --- | --- | --- |
| `rag-connector-chunker` | `chunker_version`, `chunk_size`, `chunk_overlap` | Split by `rag_connector.ingest.chunk_loaded_documents` at those settings. `chunker_version` names the splitting behaviour (`rag_connector.ingest.CHUNKER_VERSION`, currently `"1"`); the same settings reproduce the same split only under the same version. |
| `exported-from-connector` | `connector_type` | Read out of a connector's corpus exactly as its system chunked it. Claims nothing about how. |

Another tool may use its own `method` string with whatever keys it needs.

### The chunk inventory — `chunk_inventory_sha256`

Adopted **exactly** from testset-kit (`testset_kit/core.py`,
`chunk_inventory_sha256`, method `chunk-inventory-v1`), so a Test Set and the
Dataset it was built on record the same number:

1. For every chunk, the line `<chunk_id>` TAB `<sha256 of the chunk text, UTF-8,
   lowercase hex>` LF.
2. Sort the chunks by `chunk_id` in plain string order (Unicode code point —
   `x:chunk-10` sorts before `x:chunk-2`).
3. The fingerprint is the sha256 of the concatenated lines, UTF-8, full
   lowercase hex.

It covers chunk ids and texts, and nothing else — those are what an answer key
cites.

### The core digest — `core_sha256`

`dataset.json` carries this value, so it cannot hash its own bytes. The manifest
is hashed in canonical form, and the two data files byte for byte:

```
canonical manifest = the manifest WITHOUT "core_sha256", serialised as JSON with
                     keys sorted, no whitespace (separators "," and ":"),
                     non-ASCII characters kept as UTF-8

core_sha256 = sha256 of the UTF-8 text:
  "rag-connector-dataset-core/1\n"
  "dataset.json\t"      + sha256(canonical manifest) + "\n"
  "data/chunks.jsonl\t" + sha256(file bytes)         + "\n"
  "data/sources.json\t" + sha256(file bytes)         + "\n"
```

Re-indenting `dataset.json` does not change it; changing, adding or removing any
key does.

The data files are hashed as bytes, so anything that rewrites line endings —
`git` with `core.autocrlf`, an editor — changes the core. Keep a Dataset out of
line-ending conversion (in git, mark its files `-text` in `.gitattributes`).

## `data/chunks.jsonl`

One JSON object per line, UTF-8, LF line endings, with exactly these keys — the
row RAGauge already writes and the bundled frozen chunk set already uses:

`chunk_id`, `doc_id`, `source_file`, `doc_chunk_index`, `global_index`,
`char_start`, `char_end`, `content_module_id`, `text`, `metadata`.

The rules for each field are the [chunk identity rules](contract.md#chunk-identity)
of the connector contract: `chunk_id` unique and stable, `text` non-blank,
`global_index` legally `null`. Anything else about a chunk goes inside
`metadata`; a row with an unknown top-level key is refused rather than silently
dropped. **No embeddings** — a Dataset is a chunking, not an index. **No
absolute paths**: the writer removes `metadata.document_filepath`, which records
the machine the corpus was read on.

## `data/sources.json`

A JSON array, one object per source document, sorted by `source_identity`:

| Key | Meaning |
| --- | --- |
| `source_identity` | The source's identity (for the bundled chunker, its corpus-relative path, casefolded). |
| `source_fingerprint` | sha256 of file type + parsed text. `""` when the producer could not report one: **unknown**, not a fingerprint of nothing. |
| `relative_path` | Corpus-relative path. |
| `filename` | The chunks' `source_file`. |
| `file_type` | File type without the dot. |

`rag_connector.dataset.source_manifest_from_chunks` derives this list from the
chunks; it is the builder RAGauge has used for its own Datasets, moved here so
every tool shares one definition.

## `extensions/<tool>/extension.json` — the envelope

```json
{
  "tool": "ragauge",
  "tool_version": "0.9",
  "format_version": "1",
  "core_sha256": "…the core_sha256 of the Dataset it was built on…",
  "created_at": "2026-09-24T01:00:00Z"
}
```

| Key | Meaning |
| --- | --- |
| `tool` | The tool's name, equal to the directory name: lowercase letters, digits, `.`, `_`, `-`. |
| `tool_version` | The version of the tool that wrote the area. |
| `format_version` | The version of the tool's own format for what it keeps in the area. |
| `core_sha256` | The core this extension was built on. |
| `created_at` | ISO 8601, UTC. |

A directory under `extensions/` without an envelope is not an extension.

## Legacy RAGauge folders ("v0 layout")

RAGauge Datasets written before this spec have the same three files and the same
chunk row, but RAGauge's own manifest (`id` instead of `dataset_id`, no `spec`,
no inventory or core hash). They stay readable, **read-only and never migrated
in place**: `read_legacy_dataset_folder` reads one without writing anything and
returns it marked unverified, with the chunk inventory computed at read time.
The verifying reader refuses one unless asked (`allow_legacy=True`). To bring a
legacy Dataset under this spec, write a **new** Dataset from its chunks; it gets
a new, content-derived id.

## API

```python
from rag_connector.dataset import (
    write_dataset_folder, read_dataset_folder, read_legacy_dataset_folder,
    chunk_inventory_sha256, source_manifest_from_chunks,
    list_extensions, write_extension_envelope,
    chunker_chunking, exported_chunking,
)
```

| Function | Does |
| --- | --- |
| `write_dataset_folder(out_dir, chunks, *, name, chunking, sources=None, created_at=None)` | Writes a new core, refusing an existing non-empty directory, and returns the verified `Dataset`. |
| `read_dataset_folder(path, *, allow_legacy=False)` | Reads and verifies; raises `DatasetIntegrityError` on any mismatch. |
| `read_legacy_dataset_folder(path)` | Reads a RAGauge v0 folder, unverified, writing nothing. |
| `chunk_inventory_sha256(rows)` | The inventory fingerprint over `ChunkRecord`s or `{"chunk_id", "text"}` mappings. |
| `source_manifest_from_chunks(chunks)` | The source list. |
| `list_extensions(folder)` | Every envelope, sorted by tool, each with any `problems`. |
| `write_extension_envelope(folder, tool, *, tool_version, format_version, created_at=None, overwrite=False)` | Creates `extensions/<tool>/extension.json` recording the verified core. |
| `chunker_chunking(...)`, `exported_chunking(connector_type)` | The two defined `chunking` blocks. |

## Command line

```bash
# From a folder of documents, with this package's chunker (no vector store):
rag-connector dataset build --folder ./my-documents --out ./my-dataset \
  [--chunk-size 1000] [--chunk-overlap 200] [--name "My documents"]

# From a connector's own corpus, as its system chunked it:
rag-connector dataset export --connector my-rag --params '{"endpoint": "..."}' \
  --out ./my-dataset [--name "..."]
```

`export` reads the whole corpus through the connector's corpus read
(`pull_all_chunks`, which pages through `list_chunks` when that is what the
connector implements) and drops any embeddings. Both commands refuse an output
directory that already holds anything, and print the new Dataset's id, chunk
count and hashes as JSON. The same two operations are library functions:
`build_dataset_from_folder(folder, out_dir, *, chunk_size, chunk_overlap, name)`
and `export_dataset_from_connector(pipeline, out_dir, *, connector_type, name)`.

## About the demo corpus

The corpus bundled with this package, `pelorus_space` ("Pelorus Space"), is an
MIT-licensed demo corpus, like the rest of this package. Nothing in this spec or
in the Dataset format depends on it or on any product.
