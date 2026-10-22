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
  FINDINGS.md             shared — what is learned about the Dataset: one
                          entry per top-level heading, from any tool or person

  .<tool>/                private — one tool's own state
    extension.json          its envelope
  .rag-connector/         private — this package's: audits, attachments
    audits/<audit-id>/      one audit of the Dataset against one connector
                            instance
    syncs/<sync-id>/        in a Dataset made by a sync: the instance and
                            audit it was taken from
  extensions/<tool>/      private — the 1.0 location, still valid
    extension.json          its envelope

  README.md, AGENTS.md,   free — the owner's: notes, pages, agent
  index.html, .claude/ …  configuration, anything else
```

Every top-level entry is in exactly one of four classes:

| Class | Entries | Owned by |
| --- | --- | --- |
| **Core** | `dataset.json` and the whole `data/` directory | Nobody — written once, never changed |
| **Shared areas** | `sources/`, `testsets/`, `analysis/`, `docs/`, `FINDINGS.md` | Divided between tools and the owner, as each section below says |
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
   never on its own initiative. The one exception is `FINDINGS.md`
   ([below](#findingsmd--findings-about-the-dataset)): any tool may **append**
   an entry to it, and only append.
7. **The manifest lists nothing outside the core.** Listing an area would mean
   rewriting the write-once core whenever a tool arrived. Areas are discovered
   by their envelopes.
8. **The shared areas this spec defines hold nothing secret or
   machine-local**: no credentials, no absolute paths, no machine names in
   `sources/`, `analysis/`, `docs/` or `FINDINGS.md`. A shared area travels with every copy
   of the Dataset; a tool-private area does not
   ([below](#sharing-a-dataset)). What a Test Set may record is the Test Set
   spec's to say; a Test Set manifest's `dataset_path` is currently an
   absolute path.
9. **A Dataset is static; the systems it is used with are not.** Anything
   that describes a backend — a connection, a chunk-id map, a "connected"
   status, a backend fingerprint — belongs to **one connector instance**, and
   is never kept as a single slot per Dataset
   ([below](#the-dataset-is-static-the-backend-is-not)).

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
| `derived_from` | Optional. The Dataset this one was made from, as `{"dataset_id", "core_sha256"}` — written by a sync ([below](#syncing-into-a-copy--rag-connectorsyncs)). Lineage between Datasets only: nothing about a backend. |

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

*Implemented: `write_analysis_envelope`, `list_analyses`.* One area per tool, holding that tool's findings about
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

#### `analysis/rag-connector/` — the corpus index

This package's own analysis area holds two indexes over the chunks, built
separately from freezing them and replaceable at any time:

- **The book index** (`rag-connector index prepare`,
  `rag_connector.corpus_index`). Names, document titles and section headings,
  each with the chunk ids that hold it, the previous and next chunk in that
  document, and identical copies elsewhere. Taken from the chunk text by
  pattern; no model is called. The patterns are general (capitalized
  phrases, headings, and numbered names such as `Route 66` or `Sec. 12`,
  with dates and postal codes left out); a corpus whose own capitalized
  words name nothing passes them as extra stopwords
  (`prepare --stopword WORD`), and the index records the list it used. `index.json` is the full map, `index.md` says
  how it is laid out, and `index/terms/<letter>.md` and
  `index/threads/<letter>.md` are the slices to read. A label found in too
  many places to narrow a search is left out, and ordinary wording is never
  indexed: the chunks remain the only place to search text.
- **The signal index** (`rag-connector index signal`,
  `rag_connector.signal_index`). A model labels each distinct document, on its
  own, with a few high-signal words and the chunk-index span that holds them.
  The words are model output and each labelled document records the model
  that wrote them; chunk ids are attached from the Dataset's rows, never taken
  from the model, and a saved subject naming a chunk outside its document is
  dropped. `index/signal.json`, `index/signal.md`, `index/signal/<letter>.md`.
  Labelling progress is working state, kept privately in
  `.rag-connector/index/signal_progress.jsonl`, so a later run sends only the
  documents not labelled yet. It needs the optional `llm` extra.

Rebuilding the book index leaves the signal index in place. Both name the
core they were built on; a reader given an area built on another core gets no
index, and the next build replaces the area as a whole.

### `docs/` — documentation

*Not yet implemented.* Documentation about the Dataset for people: what it
covers, how it was built, known problems. It belongs to the owner, except
`docs/<tool>/`, where a tool may write documentation it generates. Like
everything outside the core, none of it is corpus.

### `FINDINGS.md` — findings about the Dataset

*A convention; no code is required to use it.* One top-level file where
anything learned about the Dataset itself is collected, from any tool or any
person: damage found in the chunking, repeated or conflicting content,
ambiguities, judgement calls a tool had to make. A person or a coding agent
opens one known file instead of looking through each tool's area.

The only requirement:

- **Each finding is its own top-level entry, and each entry starts with a `# `
  heading.** An entry runs from its heading to the next top-level heading. A
  line starting with `# ` inside a fenced code block is not a heading.
- Everything else is free. Use any sub-headings, lists, tables, quotes or
  links inside an entry. A summary or overview is just another top-level entry.
- **Programs only append** new entries at the end. A person may add, change or
  remove anything.

What this file is, and is not:

- It holds findings **about the Dataset** (its chunks, documents and corpus),
  from any tool. It is not about a Test Set or a run. A Test Set or a run may
  mention a finding by its heading; a finding does not point back.
- A finding that concerns a chunk names it by `chunk_id` in its text, so
  "what is known about this chunk?" is a search for the id.
- It is **evidence about the corpus, not corpus** (rule 4), like everything
  outside the core.
- It is a shared file (rule 8): nothing secret and nothing machine-local. It
  travels with every copy of the Dataset, which already carries every chunk, so
  quoting a chunk in an entry exposes nothing the copy does not hold.
- It is not tied to one core the way `analysis/<tool>/` is. A finding about a
  chunk is only meaningful for the Dataset that has that chunk id, so say in
  the entry which Dataset (or `core_sha256`) it was made on if that matters.

Suggested, never required. Keep entries short and clear, in three parts:

- **An id and a description, in the heading**: `# F07: Statements that
  conflict`. The description is short and says what the issue is.
- **A source line**, in words a person understands, for example "Source:
  testset-kit Scanner, confirmed by reading", "Source: RAGauge TEST 2 grader",
  "Source: person". A reader can filter on it; an entry without one shows
  under "unknown source".
- **The details**: a few lines, with the chunk ids and a short quote where it
  helps.

An entry states what was observed. It is a fact, not a decision, a
recommendation or a task. There is no status or kind field. Anything more
belongs in the details.

Scale: one file is meant for tens or hundreds of entries. A tool with many
records (a chunk-by-chunk survey, say) should write a summary entry and keep
the records in its own `analysis/<tool>/` area, citing it.

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
- **`.rag-connector/` is this package's.** It holds the audits of this
  Dataset (`.rag-connector/audits/`, [below](#audits--rag-connectoraudits)),
  the record of the sync that made it, if one did (`.rag-connector/syncs/`,
  [below](#syncing-into-a-copy--rag-connectorsyncs)), and is reserved for the record of attaching it to a connector instance
  (`.rag-connector/attachments/`), whose format a later section of this spec
  will define. `list_extensions` does not yet discover it.

**`extensions/<tool>/` remains valid.** The 1.0 location is a tool-private
area throughout 1.x, with the same envelope and the same rules. A reader
discovers tool areas at both locations. `.<tool>/` is the recommended location
for a tool's next format; a tool **may** move its area there, when it chooses,
and nothing requires it to. One name in both locations at once is reported as a
problem rather than one being chosen silently.

## The Dataset is static; the backend is not

A Dataset's core never changes. The RAG system it was exported from, or is
attached to, changes all the time: its index is rebuilt and hands out new chunk
ids, its corpus is re-ingested with other chunking settings or metadata fields,
documents are added and removed. And there is rarely one such system: the same
connector type may be configured by several applications, or by one
application for several environments (development, staging, production), and
those systems need not be in the same state.

So the relationship between a Dataset and a backend is a fact about **one
connector instance at one time**, never about the Dataset:

- **Keyed by instance, never by Dataset alone.** Whatever describes a backend
  — its connection, a map from Dataset chunk ids to its native ids, whether it
  is "connected", its fingerprint — is recorded per connector instance. A tool
  must not keep a single "current connection" per Dataset: when a second
  application, or a second instance in the same application, attaches the
  same folder, one slot means the last writer silently replaces the other's
  connection, and each then runs against a backend it did not check.
- **Never shared across instances.** A record made for one instance is not
  used for another, even of the same connector type, even over what looks
  like the same backend, even when both held identical chunks when the record
  was made. Two instances may diverge at any moment afterwards.
- **One record per act.** Attaching produces its own record, identified by its
  own id that the application keeps; an audit likewise
  ([below](#audits--rag-connectoraudits)). Several applications can therefore
  use one Dataset folder at once, each finding its own records by id.
- **An instance is identified by its connection.** `connection_sha256` (sha256
  over the connection document as canonical JSON) tells instances of one
  connector type apart without storing their configuration in the Dataset.
  Connection documents are secret-free by contract, but they can still name
  hosts and paths, which a Dataset does not need to carry.
- **The Dataset is never updated to follow the backend.** When a backend has
  moved on, the choice is the application's: reattach (new ids for the same
  text), take a **new** Dataset from the backend's current corpus as a copy (a
  re-chunk is a new Dataset, rule 3), or keep using the old one where it still
  matches. Test Sets cite the old Dataset's chunk ids, so migrating them is a
  separate decision, and nothing here makes it. Taking the copy is a
  **sync** ([below](#syncing-into-a-copy--rag-connectorsyncs)).

## Audits — `.rag-connector/audits/`

An audit compares a Dataset with one connector instance's whole corpus as it is
now, and records what differs. **It changes nothing**: not the core, not an
attachment, not the backend. A Dataset may hold any number of audits, from
any application and any instance, each in its own directory:

```
.rag-connector/
  extension.json                the area's envelope
  audits/<audit-id>/
    audit.json                  the summary
    chunks.jsonl                one row per Dataset chunk, plus one per
                                backend chunk no Dataset chunk matched
```

`<audit-id>` is `audit-<UTC time>-<8 hex>`, so audits sort by time. An audit is
written once and never edited. Like the rest of a tool-private area, audits do
not travel with a shared copy ([below](#sharing-a-dataset)): they describe a
backend the recipient may not have.

**Reading the corpus.** The audit reads the corpus once through the corpus read
(`pull_all_chunks`). A connector that cannot read its corpus, and a backend
failure, raise: neither is ever recorded as "every chunk missing". A corpus
read that returns two chunks under one id is a broken read and refuses the
audit. An empty corpus read is recorded as what it is.

**Matching** is by id, then by exact text, and **never by position**:

1. A backend chunk with the same `chunk_id`.
2. Otherwise a backend chunk with the same text (sha256), only when exactly
   one unmatched Dataset chunk and exactly one unmatched backend chunk carry
   that text. Anything else is reported as ambiguous, not assigned.

Each Dataset chunk gets one status; each unmatched backend chunk is `added`:

| Status | Means |
| --- | --- |
| `unchanged` | Same id, same text. Metadata may still differ (below). |
| `text_changed` | Same id, different text — an edit, or a re-chunk that reused positional ids. |
| `id_changed` | Same text under another id — what an index rebuild does. Any id map made before is stale for this chunk. |
| `ambiguous` | Its text matches several unmatched backend chunks. The candidates are listed. |
| `missing` | Neither its id nor its text is in the backend. |
| `added` | A backend chunk that matched no Dataset chunk. |

For every matched chunk the row also lists what else differs:
`fields_changed` (`doc_id`, `source_file`, `doc_chunk_index`, `char_start`,
`char_end`, `content_module_id`), and `metadata_keys_added`,
`metadata_keys_removed`, `metadata_keys_changed`. `metadata.document_filepath`
is ignored on both sides, since the writer strips it from the core.
`global_index` is not compared: it is host-assigned.

**Documents** are grouped by `doc_id`: `unchanged` (every chunk unchanged,
nothing added under the doc), `removed` (no chunk found by id or text, and no
backend chunk has the `doc_id`), `added` (a backend `doc_id` with no Dataset
chunk behind it) and `changed` (anything else). Each row gives the chunk count
on both sides and the chunk statuses.

**Chunking profiles.** For each side: chunk count, document count, chunk length
in characters (min, median, mean, p90, max), chunks per document, the metadata
keys and how many chunks carry each, and the exact repeated text between
neighbouring chunks of a document (`neighbour_overlap_chars`). These are
**observed**, not declared settings: a chunker that trims at boundaries shows
less overlap than it was configured with. Compare the two sides; do not read
either as a recovered `chunk_size` or `chunk_overlap`.

**Signals** name what was observed, each with a count; they are never a
verdict. `chunk_ids_changed`, `chunk_text_changed`, `chunks_missing`,
`chunks_added`, `chunks_ambiguous`, `documents_removed`, `documents_added`,
`documents_chunk_count_changed` (present on both sides with a different number
of chunks, as a re-chunk does), `metadata_keys_added`, `metadata_keys_removed`,
`chunk_metadata_changed`, `chunk_fields_changed`.

**`in_sync`** is true only when the backend's chunk inventory (same definition
as the Dataset's) equals the Dataset's and no matched chunk's metadata or
fields differ.

**`audit.json`** records, besides the counts, signals, profiles and documents:
`format_version` (`"1"`), `created_at`, the Dataset (`dataset_id`,
`core_sha256`, `chunk_inventory_sha256`, `chunk_count`, `chunking`), the
instance (`connector_type`, `connection_sha256`, and the application's own
`instance_label` and `host`, any of which may be `null`), and the backend
(`chunk_count`, `chunk_inventory_sha256`). An audit is recorded only in the
Dataset whose core it was made against.

## Syncing into a copy — `.rag-connector/syncs/`

A **sync** follows a backend that has moved on without changing the Dataset
that fell behind: it writes a **new** Dataset from one connector instance's
corpus as it is now (exactly as `export` does), and records which Dataset it
follows and, optionally, which audit of that Dataset motivated it. The source
Dataset is read and verified, and **never written**: not its core, not its
private area. A legacy folder records no core hash to cite, and is refused.
The source's Test Sets keep citing the source's chunk ids; whether, and how, to
move them to the copy is the application's decision.

Lineage is recorded in two places, split by what kind of fact it is:

- **In the new core**, the manifest key `derived_from`: the source's
  `dataset_id` and `core_sha256`. That one Dataset was made from another is a
  fact about the two Datasets, true for as long as both exist, so it is
  written once with the core, covered by `core_sha256`, and travels with
  every shared copy. It is written **only at creation**: like any manifest
  key, adding it to an existing core is an edit readers refuse. It changes no
  identity rule: `dataset_id` is still the chunk inventory's, so a sync of an
  unchanged backend has the source's `dataset_id` and a different
  `core_sha256` (its `created_at` and `derived_from` differ).
- **In the new Dataset's private area**, `.rag-connector/syncs/<sync-id>/sync.json`:
  the connector instance the copy was taken from and the audit that motivated
  it. Both describe one backend (rule 9), and the audit lives in the
  **source's** private area, which a shared copy does not carry, so this
  record stays private too.

```
.rag-connector/
  extension.json            the area's envelope (core_sha256 of the NEW core)
  syncs/<sync-id>/
    sync.json
```

`<sync-id>` is `sync-<UTC time>-<8 hex>`. A Dataset made by a sync holds one,
written in the same step as the core: the core and the record are staged
together and moved into place at once, so a backend failure or an empty corpus
leaves nothing behind.

**Citing an audit.** The audit must be recorded in the source Dataset, must
have been made against the source's core, by the same connector type, and for
the **same connector instance**: both the audit and the sync must carry a
`connection_sha256`, and they must be equal. Otherwise the sync is refused —
an audit of one instance says nothing about another (rule 9). The backend may
still have moved on between the audit and the sync; that is recorded, not
refused, as `backend_matches_audit`.

**`sync.json`** records `sync_id`, `format_version` (`"1"`), `created_at`;
`source` and `dataset` (each `dataset_id`, `core_sha256`,
`chunk_inventory_sha256`, `chunk_count`); `same_chunk_inventory_as_source`
(same ids and texts — metadata may still differ); the instance (`connector`:
`connector_type`, `connection_sha256`, `instance_label`, `host`,
`retrieval_mode`); and `audit`: `null`, or the cited `audit_id`, its
`created_at` and `in_sync`, the backend chunk inventory it saw, and
`backend_matches_audit` (whether that equals the copy's inventory).

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
    list_analyses, write_analysis_envelope,
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
| `list_analyses(folder)` | Every `analysis/*/analysis.json`, sorted by tool, as `AnalysisEnvelope`s (with `artifacts`); a listed artifact the area does not hold is a problem. |
| `write_analysis_envelope(folder, tool, *, tool_version, format_version, artifacts, created_at=None)` | Writes `analysis/<tool>/analysis.json` recording the verified core, over artifacts already in the area; replaced on every call. |
| `chunker_chunking(...)`, `exported_chunking(connector_type)` | The two defined `chunking` blocks. |

```python
from rag_connector import corpus_index, signal_index
```

| Function | Does |
| --- | --- |
| `corpus_index.prepare(folder, *, stopwords=())` | Writes the book index into `analysis/rag-connector/` and returns its counts. |
| `corpus_index.build(chunks, *, stopwords=())` | The book index body from `ChunkRecord`s or chunk-row mappings; writes nothing. |
| `corpus_index.load(folder)`, `corpus_index.find(index, query)` | The index on this core, or `None`; the entries and threads whose label contains `query`. |
| `signal_index.build(folder, *, model=MODEL, label=None, limit=None, workers=6)` | Labels the documents not labelled yet and rewrites the signal index. `label` is `(system, prompt) -> (text, input_tokens, output_tokens)`; the default calls the Anthropic API (`llm` extra). |
| `signal_index.load(folder)`, `signal_index.find(payload, query)` | The signal index on this core, or `None`; the subjects whose words contain `query`. |

```python
from rag_connector.audit import (
    audit_dataset, record_audit, read_audit, list_audits, connection_sha256,
)
```

| Function | Does |
| --- | --- |
| `audit_dataset(dataset, pipeline, *, connector_type, connection=None, instance_label=None, host=None)` | Compares a read `Dataset` with the connector's corpus and returns a `DatasetAudit`. Writes nothing. |
| `record_audit(folder, audit)` | Stores it under `.rag-connector/audits/<audit-id>/`, after checking the folder holds the audited core. |
| `list_audits(folder)`, `read_audit(path)` | Every recorded `audit.json`, oldest first; one audit with its chunk rows. |
| `connection_sha256(connection)` | The instance identity recorded in an audit. |

```python
from rag_connector.sync import sync_dataset, list_syncs
```

| Function | Does |
| --- | --- |
| `sync_dataset(source, pipeline, out_dir, *, connector_type, connection=None, audit_id=None, instance_label=None, host=None, name=None)` | Writes a new Dataset from the connector's corpus with `derived_from` naming `source`, and its `sync.json`; returns a `DatasetSync` (`dataset`, `record`, `path`). Never writes `source`. `name` defaults to the source's. |
| `list_syncs(folder)` | Every `sync.json` in a Dataset folder, oldest first. |
| `Dataset.derived_from` | The manifest's `derived_from`, or `None`. |

## Command line

```bash
# The corpus index, in analysis/rag-connector/ (no model call):
rag-connector index prepare --dataset ./my-dataset [--stopword WORD ...]
rag-connector index status --dataset ./my-dataset
rag-connector index lookup --dataset ./my-dataset "Port St. Joe"

# The signal index (llm extra and ANTHROPIC_API_KEY; resumes where it stopped):
rag-connector index signal --dataset ./my-dataset [--model ...] [--limit N] [--workers N]
rag-connector index signal-lookup --dataset ./my-dataset "roster"
```

```bash
# From a folder of documents, with this package's chunker (no vector store):
rag-connector dataset build --folder ./my-documents --out ./my-dataset \
  [--chunk-size 1000] [--chunk-overlap 200] [--name "My documents"]

# From a connector's own corpus, as its system chunked it:
rag-connector dataset export --connector my-rag --params '{"endpoint": "..."}' \
  --out ./my-dataset [--name "..."]
```

```bash
# Compare a Dataset with a connector instance's corpus; records the audit:
rag-connector dataset audit --dataset ./my-dataset --connector my-rag \
  --params '{"endpoint": "..."}' [--label staging] [--host my-app] [--no-record]
rag-connector dataset audits --dataset ./my-dataset   # one JSON line per audit

# Take a NEW Dataset from the instance's corpus as it is now; the source is
# never changed. --audit cites an audit of the source for the same instance:
rag-connector dataset sync --dataset ./my-dataset --connector my-rag \
  --params '{"endpoint": "..."}' --out ./my-dataset-2 [--audit <audit-id>] \
  [--name "..."] [--label staging] [--host my-app]
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
| `rag-connector` | `.rag-connector/audits/`, `.rag-connector/syncs/`, `.rag-connector/index/`, `analysis/rag-connector/` | `.rag-connector/attachments/` | This spec |
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
