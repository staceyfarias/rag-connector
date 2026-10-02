# Dataset spec — `rag-connector-dataset` 1.1 (draft)

A **Dataset** is a folder holding a frozen, chunked corpus: the chunks, the
list of source documents they came from, and a small manifest saying what the
folder holds and how it was split. Any tool can read one, and any tool can add
its own work to one without disturbing the others.

This page is normative. It is implemented by `rag_connector.dataset`, which
imports no optional extras. The spec is versioned **independently** of the
`rag-connector` package and of the [connector contract](contract.md): spec id
`rag-connector-dataset`, version `1.1`, in draft.

Building a Dataset is optional tooling, not part of the connector contract. A
connector is a read-only view of a system someone else built; a Dataset is
something a host writes. `rag-connector dataset build` and
`rag-connector dataset export` ([below](#command-line)) are two ways to make
one, but any tool that writes the files described here makes a conforming
Dataset.

> **Status — draft.** 1.1 changes nothing in the core: every existing 1.0
> Dataset folder is a 1.1 Dataset, with its `dataset_id` and `core_sha256`
> unchanged. (A core *written* with `spec_version` `"1.1"` would have a
> different `core_sha256` from the same corpus written as `"1.0"`, because the
> digest covers every manifest key; `dataset_id` would not change.) What 1.1
> adds is the layout *around* the core: shared areas, tool-private
> dot-folders, and the rule that only the core is corpus. Everything a tool
> conformantly writes under 1.0 stays conformant: `extensions/<tool>/` remains
> a valid tool-private location throughout 1.x. `rag_connector.dataset`
> implements the core, the envelope, and `extensions/<tool>/` discovery. It
> does **not yet** implement anything marked *not yet implemented* below, and
> it still writes `spec_version` `"1.0"`.

## Folder layout

```
<dataset>/
  dataset.json            CORE — the manifest
  data/chunks.jsonl       CORE — one chunk row per line
  data/sources.json       CORE — the source list

  sources/                shared — the source documents, optional
  testsets/<name>/        shared — Test Sets
  analysis/<tool>/        shared — one tool's analysis of the corpus
    analysis.json           its envelope
  docs/                   shared — documentation about the Dataset

  .<tool>/                private — one tool's own state
    extension.json          its envelope
  extensions/<tool>/      private — the 1.0 location, still valid
    extension.json          its envelope

  README.md, AGENTS.md,   free — the owner's: notes, pages, agent
  index.html, .claude/ …  configuration, anything else
```

Every top-level entry is in exactly one of four classes:

| Class | Entries | Owned by |
| --- | --- | --- |
| **Core** | `dataset.json` and the whole `data/` directory | Nobody — written once, never changed |
| **Shared areas** | `sources/`, `testsets/`, `analysis/`, `docs/` | Divided between tools and the owner, as each section below says |
| **Tool-private areas** | `.<tool>/` or `extensions/<tool>/`, holding an `extension.json` whose `tool` is `<tool>` | That one tool |
| **Free entries** | Everything else, including other dot-entries such as `.claude/` and `.git/` | The Dataset's owner — the person or project that keeps the folder |

Shared areas use plain names because they are meant to be read: by other
tools, by people browsing the folder, and by anyone the Dataset is handed to.
A dot-folder means **one tool's, not shared**. It does not mean temporary: a
tool-private area may hold state that lasts for months, such as a connection
used for regression testing.

## Rules

1. **The core is write-once.** Nothing — not the tool that wrote it, not a user
   edit, not a migration — changes `dataset.json`, `data/chunks.jsonl` or
   `data/sources.json` after they are written, and nothing else is written
   under `data/`. A human description, a context note, a scan result: none of
   these is core. A reader ignores any other file it finds under `data/`; it
   neither refuses the Dataset nor reads the file.
2. **Readers verify, and refuse a mismatch.** A reader recomputes
   `chunk_inventory_sha256` and `core_sha256` (and `chunk_count`,
   `dataset_id` and `corpus_fingerprint`) from the files and refuses the
   Dataset if any differs from the manifest. A Dataset whose recorded identity
   no longer matches its contents is not an older corpus; it is an unknown one.
3. **A re-chunk is a new Dataset.** Chunk ids are positions (`doc.md:chunk-3`)
   that a different split refills with different text. The id is derived from
   the chunk inventory, so a different split always has a different id.
4. **Only the core is corpus.** The corpus is exactly the rows of
   `data/chunks.jsonl`. No other file in the folder is corpus content —
   whatever its name, format or location, including the documents in
   `sources/`. A `README.md` or `AGENTS.md` at the top level is never chunked,
   retrieved or cited. This is an allowlist on purpose: an ignore list fails
   open, and one forgotten entry would make a file evidence.
5. **Readers ignore what they do not know**: unknown manifest keys, other
   tools' areas, and free entries.
6. **Each tool writes only its own areas**: its private area (`.<its name>/`,
   or `extensions/<its name>/`), `analysis/<its name>/`, `docs/<its name>/`,
   and Test Sets in `testsets/` under the rules of the Test Set spec. Free entries and the rest of `docs/`
   belong to the owner; a tool writes there only when the owner asks it to,
   never on its own initiative.
7. **The manifest lists nothing outside the core.** Listing an area would mean
   rewriting the write-once core whenever a tool arrived. Areas are discovered
   by their envelopes.
8. **The shared areas this spec defines hold nothing secret or
   machine-local**: no credentials, no absolute paths, no machine names in
   `sources/`, `analysis/` or `docs/`. A shared area travels with every copy
   of the Dataset; a tool-private area does not
   ([below](#sharing-a-dataset)). What a Test Set may record is the Test Set
   spec's to say; a Test Set manifest's `dataset_path` is currently an
   absolute path.

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
key does. The core's file names are part of the digest, which is why the core
keeps the paths it has.

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

The chunks are one flat collection in one file. A document's place in a folder
hierarchy is carried by its `source_file` and its source's `relative_path`, so
a reader can present the corpus as folders without the core being split into
them.

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

## Shared areas

### `sources/` — the source documents

*Not yet implemented.* Optional. When present, it holds the documents the
corpus was chunked from, each at its `relative_path` from `data/sources.json`.
It lets a Dataset be re-chunked, loaded into a system that takes documents
rather than chunks, or read by a person in its original form.

- It is **not core and not corpus.** It is not covered by `core_sha256`, and
  rule 4 applies to it: nothing in it is retrieved or cited. The chunks remain
  the corpus; re-chunking `sources/` produces a new Dataset.
- **A file is part of the Dataset only if `data/sources.json` lists it.**
  Anything else in `sources/` is ignored.
- **It is written with the core and not edited afterwards.** A document can
  be checked only when its `source_fingerprint` was computed by this
  package's builder (`chunking.method` is `rag-connector-chunker`) under the
  same parsers: the fingerprint covers *parsed* text, so a different parser
  version changes it without the document changing. A reader that checks a
  document recomputes its fingerprint and reports a mismatch for that
  document. Every other document — an exported Dataset's fingerprint comes
  from the backend, by an algorithm this spec does not know, and `""` is
  unknown — is reported as **unchecked**, never as matching or changed.
- A Dataset exported from a connector usually has no `sources/`: a connector
  reads chunks, not documents.

### `testsets/<name>/` — Test Sets

Reserved for Test Sets. Their contents are defined by the Test Set spec
(testset-kit), not here; this spec only reserves the name.

### `analysis/<tool>/` — shared analysis

*Not yet implemented.* One area per tool, holding that tool's findings about
the corpus that are meant for other tools and for people: a chunking report,
per-chunk observations, a survey of what the corpus contains. The area is
recognized by its envelope, `analysis/<tool>/analysis.json`
([below](#envelopes)).

- Analysis is **about one core.** Its envelope records the `core_sha256` it
  was computed from, and a reader uses it only when that matches the Dataset
  it is reading. An area recording another core describes a different corpus,
  for example after the folder was copied around a re-chunk.
- A tool **replaces its area as a whole** when it re-runs, and records when
  it did in the envelope's `created_at`. A reader that relies on an earlier
  version keeps its own copy.
- Analysis is **evidence about the corpus, not corpus** (rule 4). A
  per-chunk finding cites its chunk by `chunk_id`.
- A tool's intermediate and working state goes in its private area, not here.

### `docs/` — documentation

*Not yet implemented.* Documentation about the Dataset for people: what it
covers, how it was built, known problems. It belongs to the owner, except
`docs/<tool>/`, where a tool may write documentation it generates. Like
everything outside the core, none of it is corpus.

## Tool-private areas — `.<tool>/`

*The `.<tool>/` location is not yet implemented; `extensions/<tool>/` is.* A
tool keeps its own state in its private area, `.<tool>/` or
`extensions/<tool>/`, recognized by its envelope, `extension.json`, whose
`tool` equals the directory name (without the dot). What else the tool keeps there is its business, in its own format, and
nothing in this spec defines, reads or validates it.

- **Private means not shared**, not temporary. The area may hold durable state
  (a stored connection, a mapping that is used for months) as well as working
  files a tool deletes itself.
- **A dot-entry without a matching envelope is not a tool area.** `.git/`,
  `.claude/`, `.DS_Store` and the like are free entries.
- **`.rag-connector/` is this package's.** It is reserved for the record of
  attaching this Dataset to a connector instance (`.rag-connector/attachments/`),
  whose format a later section of this spec will define.

**`extensions/<tool>/` remains valid.** The 1.0 location is a tool-private
area throughout 1.x, with the same envelope and the same rules. A reader
discovers tool areas at both locations. `.<tool>/` is the recommended location
for a tool's next format; a tool **may** move its area there, when it chooses,
and nothing requires it to. One name in both locations at once is reported as a
problem rather than one being chosen silently.

## Envelopes

A tool-private area and an analysis area each carry an envelope recording which
tool wrote it and which core it was built on:

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
| `tool` | The tool's name, equal to the area's directory name (without the leading `.` for a private area): lowercase letters, digits, `.`, `_`, `-`. |
| `tool_version` | The version of the tool that wrote the area. |
| `format_version` | The version of the tool's own format for what it keeps in the area. |
| `core_sha256` | The core this area was built on. |
| `created_at` | ISO 8601, UTC. |

The private envelope is `extension.json`. The analysis envelope is
`analysis.json`, with the same keys and one more, `artifacts`, so a reader can
find what is in the area without knowing the tool:

```json
"artifacts": [
  {"path": "chunking_report.md", "format": "markdown",
   "description": "How the corpus was split, and the damage found"},
  {"path": "chunk_observations.jsonl", "format": "jsonl",
   "description": "One finding per line, citing chunk_id"}
]
```

Each `path` is relative to the area. `format` is a short name (`markdown`,
`json`, `jsonl`, `csv`, `html`) and `description` is one line for a person.

## Sharing a Dataset

*Not yet implemented.* A **shared copy** of a Dataset — a zip, a clone, a
folder handed to someone — holds the core and the shared areas. It leaves out
every tool-private area: what a tool keeps privately can refer to a machine, a
backend or a session that the recipient does not have. Free entries are the
owner's to include or leave out.

A recipient verifies the core on arrival (rule 2), and uses an analysis area
only if its envelope matches the core.

Leaving private areas out also leaves out any evidence kept there. A Test Set
travels, but the build records that justify it may sit in its tool's private
area and do not. A tool whose shared output depends on such records should
publish what a recipient needs in a shared area.

## Legacy RAGauge folders ("v0 layout")

RAGauge Datasets written before this spec have the same three files and the same
chunk row, but RAGauge's own manifest (`id` instead of `dataset_id`, no `spec`,
no inventory or core hash). They stay readable, **read-only and never migrated
in place**: `read_legacy_dataset_folder` reads one without writing anything and
returns it marked unverified, with the chunk inventory computed at read time.
The verifying reader refuses one unless asked (`allow_legacy=True`). To bring a
legacy Dataset under this spec, write a **new** Dataset from its chunks; it gets
a new, content-derived id. A v0 folder may hold scan files under `data/`; rule 1
forbids that in a spec Dataset.

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
| `list_extensions(folder)` | Every envelope under `extensions/`, sorted by tool, each with any `problems`. Does not yet discover `.<tool>/` areas. |
| `write_extension_envelope(folder, tool, *, tool_version, format_version, created_at=None, overwrite=False)` | Creates `extensions/<tool>/extension.json` recording the verified core. Still writes the 1.0 location. |
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

## The Datasets directory

Not part of the spec's format, but how tools find Datasets. A Dataset folder
can live anywhere; so that products share them, `rag_connector.datasets_dir`
names one directory where they are looked for and written by default. Resolved
in order: an explicit path, the `RAG_CONNECTOR_DATASETS_DIR` environment
variable, the `datasets_dir` setting in `~/.rag-connector/config.json`, then
the default: `datasets/` in the repository when running from a source checkout
(gitignored; the pristine demo Dataset is under `examples/`), else
`~/rag-connector/datasets`. A Dataset is a direct child folder holding a
`dataset.json`; folders starting with `.` or `_` are skipped. Listing reads the
manifests only and does not verify; verification belongs to opening a Dataset
(`read_dataset_folder`).

```
rag-connector datasets-dir [--set PATH | --unset] [--json]
rag-connector dataset list [--dir PATH]
```

`dataset build` and `dataset export` write to `<Datasets directory>/<name>` when
`--out` is omitted, and still refuse a non-empty target.

## Known tool areas

*Informative.* Tool names already taken, so a new tool does not collide with
them. Each tool defines and documents the contents of its own areas; this spec
defines only the envelopes.

| Tool name | Written today | Reserved | Contents defined by |
| --- | --- | --- | --- |
| `rag-connector` | — | `.rag-connector/` | This spec |
| `testset-kit` | `testsets/`, `extensions/testset-kit/` | `analysis/testset-kit/` | testset-kit: the Test Set spec (`contract/CORE.md`) and its output contract (`contract/OUTPUT.md`) |
| `ragauge` | `extensions/ragauge/` | `analysis/ragauge/` | RAGauge |
| `pelorus` | `extensions/pelorus/` (on export) | — | Pelorus |

Pelorus also keeps a `.pelorus/` sidecar in document folders it serves. It
carries no envelope, so it is not a tool area under this spec; in a Dataset
folder it is a free entry.

## About the demo corpus

The corpus bundled with this package, `pelorus_space` ("Pelorus Space"), is an
MIT-licensed demo corpus, like the rest of this package. Nothing in this spec or
in the Dataset format depends on it or on any product.
