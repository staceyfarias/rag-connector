# RAG Connector

RAG Connector is the shared contract that lets **RAGauge** and **Pelorus Query**
work with the same Retrieval-Augmented Generation pipeline as a black box —
without either of them knowing how that pipeline is built.

- **RAGauge** evaluates a RAG system: whether it retrieves the right evidence
  and whether its answers are grounded in it, measured against a frozen corpus
  and test set.
- **Pelorus Query** sits on top of a RAG system and serves curated,
  source-grounded answers (Evidence Extracts), improving them as queries repeat.

Both need the same things from the pipeline underneath: ask it a question and
get ranked chunks back, read its corpus, know which embedding space and
retrieval mode produced a score, and cite a chunk by an id that means the same
thing to both. RAG Connector defines that once. A pipeline that implements the
contract — or is wrapped by a connector that does — can be measured by RAGauge
and curated by Pelorus Query, and a chunk one of them cites resolves to the
same text in the other. The bundled demo corpus and its frozen chunk set exist
for exactly that: both products are built against the same 605 chunk ids.

Interoperability between those two products is why this library exists, and it
is why the contract is narrow. Nothing here evaluates or curates: evaluation
policy belongs to RAGauge, curation behavior to Pelorus Query. Any other tool
that needs to treat a RAG pipeline as a black box can build on the same
contract. (RAGauge and Pelorus Query are not public yet.)

It defines:

- the retrieval contract and portable result models;
- stable chunk identity, score, and retrieval-mode semantics;
- optional capabilities such as corpus reading, query embedding, indexed chunk
  vectors, generation, dataset listing, run snapshots, publication, and telemetry;
- reconstructable, secret-free connector registration;
- a conformance validator and reusable connector test kit;
- an optional FastEmbed + Chroma Reference RAG that can be provisioned from a
  document folder, or from the 92-document demo corpus bundled with the package;
- an ingest kit (`rag_connector.ingest`) for hosts that build their own corpus —
  deliberately outside the read-only connector contract.

The bundled Reference RAG is a known-good development and learning
implementation, not a production service.

## Status

Pre-1.0, and used in production by its own authors. The retrieval contract,
chunk identity, the registry, the validator, document loading, fingerprinting
and the Reference RAG are all implemented and covered by the test suite; they
are what this package is for, and they are stable in practice.

What that means for depending on it: pin a version. The core contract —
`RagPipeline`, `ChunkRecord`, `RetrievedChunk`, retrieval modes, the registry
entry point — is the part least likely to move, and a change to it would break
the authors' own connectors first. The newer optional capabilities are younger
and may still gain fields. Anything listed under
[Reserved surface](https://github.com/staceyfarias/rag-connector/blob/main/docs/contract.md#reserved-surface) is exported but unwired:
nothing produces it, the validator does not check it, and it is not something
to build against yet.

## Reference RAG

Install the optional implementation, provision a corpus, and validate the
result through the same black-box contract an external connector is held to:

```bash
python -m pip install "rag-connector[reference]"
rag-connector reference provision \
  --collection example \
  --persist-dir ./rag_connector_data/reference
rag-connector validate \
  --connector-type reference \
  --params "{\"collection_name\":\"example\",\"persist_dir\":\"./rag_connector_data/reference\"}" \
  --query "a question your documents can answer"
```

With no folder named, `provision` uses the bundled **Pelorus Space** corpus — 92
short documents about a fictional interplanetary travel operator, MIT-licensed
like the rest of the package. Your own documents go in its place, as the first
positional argument:

```bash
rag-connector reference provision ./my-documents \
  --collection example \
  --persist-dir ./rag_connector_data/reference
```

Add `--normalized true|false` to declare whether the embedding space's vectors
are L2-normalized. Omit it and the space reports "not stated", which a host
reads as unverifiable — deliberately distinct from an asserted `false`. See
[the embedding space descriptor](https://github.com/staceyfarias/rag-connector/blob/main/docs/contract.md#the-embedding-space-descriptor).

The provision command prints the secret-free connection document needed to
reopen the same instance. Supported source formats are `.txt`, `.md`, `.pdf`,
`.docx`, and `.rtf`.

Two things to expect on a first run. The embedding model
(`BAAI/bge-small-en-v1.5`) is **downloaded on first use** into fastembed's
cache, so the first provision needs network and later ones do not. And **on
Windows**, install the `reference` extra into a virtualenv on a short path, or
enable long-path support first: onnxruntime, which fastembed depends on, ships
files whose full paths exceed the 260-character `MAX_PATH` limit, and pip
cannot unpack them under a deep directory.

## Ingest kit

Loading a folder of documents, chunking it, and fingerprinting the result is
**not** part of the connector contract — a connector is a read interface onto a
system someone else already built, and nothing in that contract writes a corpus.
That work lives in `rag_connector.ingest`, which is declared API for the other
audience: a host building a corpus it owns.

```python
from rag_connector.ingest import chunk_loaded_documents, load_documents

documents = load_documents("./my-documents")
chunks = chunk_loaded_documents(documents, chunk_size=1000, chunk_overlap=200)
```

Each chunk's `char_start`/`char_end` are an invariant, not a hint:
`document.content[char_start:char_end] == chunk.text` exactly, so a stored
citation resolves back to its source. They index that **normalized text** — the
document as loaded, line endings read as `\n` — and not the bytes of the file,
so a CRLF source resolves by the same offsets a LF one does. A chunk id is
`<relative path>:chunk-<n>` — `booking_cancellation_policy.md:chunk-0` — and
subfolders are walked while dot-directories (`.git`, and any tool's state
folder) are not. The kit imports with no optional extras installed; a parser is
imported only when a file needs one.

### The frozen chunk set

The bundled corpus also ships **pre-chunked**: 605 records at 1000/200, with
the ids anything downstream cites. Read it instead of re-chunking whenever a
stored artifact refers to a chunk — re-chunking is a re-derivation, and a
re-derivation that lands one character differently regrounds every citation.

```python
from rag_connector.ingest import load_bundled_chunks
from rag_connector.reference_provision import provision_reference_rag_from_chunks

chunks = load_bundled_chunks()                        # 605 ChunkRecords
connector = provision_reference_rag_from_chunks(collection_name="frozen")
```

The file is one JSON object per line in the shape RAGauge already writes, so
both products load the same corpus, the same split, and the same ids with no
adapter between them. Regenerating it is one library call —
`chunks_to_jsonl(build_bundled_chunks())`, written to
`rag_connector/corpus/<name>.chunks.jsonl` — and the suite fails if the
committed file and the kit ever disagree. (The repository keeps that call as
`tools/freeze_bundled_chunks.py`, which is development tooling and is not part
of the installed package.)

#### Reproducing it

Below is everything `build_bundled_chunks()` applies to the bundled documents.
That call **re-derives** the split; `load_bundled_chunks()` reads the frozen
file instead of re-deriving it, and is what anything citing a chunk id should
use. The settings are written down because a chunk set is only trustworthy if
someone else can arrive at the same one.

| Setting | Value |
|---|---|
| `chunk_size` | `1000` characters |
| `chunk_overlap` | `200` characters |
| Boundary search | last 20% of the window, preferring `

` → `
` → `. ` → `? ` → `! ` → space |
| Trailing chunk | dropped when its span lies wholly inside its predecessor |
| Whitespace | each chunk is stripped, and `char_start`/`char_end` record the **stripped** span |
| Source identity | the corpus-relative path, forward slashes, casefolded |
| Chunk id | `<source identity>:chunk-<n>`, `<n>` dense over emitted chunks |
| File discovery | recursive, sorted by relative POSIX path, dot-directories skipped, `.txt .md .pdf .docx .rtf` |
| Corpus layout | the 92 bundled documents, flat |

Verify a reproduction byte-for-byte. This re-derives the split from the
bundled documents and digests it, and needs no optional extras:

```bash
python -c "import hashlib; from rag_connector.ingest import build_bundled_chunks, chunks_to_jsonl; b = chunks_to_jsonl(build_bundled_chunks()).encode('utf-8'); print(hashlib.sha256(b).hexdigest(), len(b), b.count(b'\n'))"
```

```
290ca86076d9b878e54bdd2d5c1a1c870c4ab20c57edd1a013983e8782ba11fc 857219 605
```

That is the sha256, the byte count and the line count of the shipped file
itself (LF; it is pinned to LF in `.gitattributes`), so digesting the file
rather than a re-derivation gives the same three numbers.

Two things that are **not** in the table, deliberately:

- **The embedding model does not affect the chunk set.** It decides the vectors,
  not the split. The bundled datasets happen to use `BAAI/bge-small-en-v1.5`
  (384-dim, cosine), and a different model over these same 605 chunks is a
  different index of the same substrate.
- **Everything above is version-bound.** The boundary rule and the trailing-chunk
  rule are behaviour, not configuration, so reproducing the hash needs the same
  `rag-connector` release as well as the same parameters. A change to either
  moves the frozen set, which is why the drift test exists.

One caveat if you reproduce this recipe on **your own** corpus rather than the
bundled one: text and Markdown are deterministic, but PDF, DOCX and RTF go
through third-party parsers whose output can change between their own releases.
A corpus of `.txt`/`.md` reproduces exactly; one full of PDFs reproduces only
against a pinned parser.

## Datasets

A **Dataset** is a portable folder holding a frozen, chunked corpus — the chunk
rows, the source list, and a manifest recording how it was split and the hashes
that let any reader verify it. The format is specified in
[the Dataset spec](https://github.com/staceyfarias/rag-connector/blob/main/docs/dataset-spec.md)
(`rag-connector-dataset` 1.0, versioned separately from this package) and
implemented by `rag_connector.dataset`. Two optional ways to make one:

```bash
# chunk a folder of documents (no vector store, no model download)
rag-connector dataset build --folder ./my-documents --out ./my-dataset
# freeze an existing system's corpus, through its connector
rag-connector dataset export --connector my-rag --params "{...}" --out ./my-dataset
```

## Installed connectors

List every connector registered in the current environment — the bundled
Reference RAG plus any independently installed connector package:

```bash
rag-connector list
```

A connector that does not appear here is not installed in this environment, or
its package publishes no `rag_connector.connectors` entry point.

## Development

```bash
python -m pip install -e ".[dev,reference]"
python -m pytest
python -m ruff check .
```

`reference` is in there because the Reference RAG's own tests need chromadb and
fastembed. Without it the suite still runs clean — those tests skip, naming the
install that turns them back on.

See [Contract](https://github.com/staceyfarias/rag-connector/blob/main/docs/contract.md) and the
[Connector author guide](https://github.com/staceyfarias/rag-connector/blob/main/docs/connector-author-guide.md).

RAG Connector is licensed under the [MIT License](https://github.com/staceyfarias/rag-connector/blob/main/LICENSE).
