# Reference RAG (local)

Reference RAG is the connector bundled inside the `rag-connector` package
itself. It is not something the host application builds — it ships with the
contract as the one known-good implementation every host can rely on, for
development, learning, and baselining. It is deliberately not a production RAG.

It is entirely local and self-contained: [FastEmbed](https://github.com/qdrant/fastembed)
computes embeddings and [Chroma](https://www.trychroma.com/) stores them on
disk. No external service, no API keys, no credentials — you provision it from
a folder of documents and query it offline.

## Connection parameters

- **`persist_dir`** — the directory on disk holding the Chroma database. All
  collections for this connector live under it. Point two connections at the
  same directory and they share one store; point them at different directories
  and they are separate worlds. Default: `./rag_connector_data/reference`.
- **`collection_name`** — the Chroma collection to read. One collection is one
  corpus. This is the field that selects *which* provisioned corpus you are
  connecting to, so it is required and has no default.
- **`embedding_model`** — the FastEmbed model id used to embed your query.
  Default: `BAAI/bge-small-en-v1.5`. It must match the model the collection was
  provisioned with; a query embedded by a different model is being compared in
  a different vector space, and the scores will be meaningless rather than
  merely worse. The model is downloaded and cached on first use.

## Worth knowing

- **This is the one connector meant to be rebuilt and reconfigured directly.**
  Every other connector is a read interface onto an index someone else already
  built, so its chunking and its answer generation are not yours to change.
  Here the pipeline is owned end-to-end, which is why a host can offer things
  like re-chunking a corpus or assigning an answer-generation model for this
  connector and not for the others.
- **The connector itself is read-only by construction.** Provisioning — the
  chunk-and-embed step that stands in for a real system's ingest — is a
  separate module (`rag_connector.reference_provision`), not a method on the
  connector. Nothing reachable through the connector interface can create,
  clear, or overwrite a corpus.
- **Chroma has no flush.** Close a writing connection before a separate process
  reads the same `persist_dir`, or the reader may see a cold or partial store.
- **First query is slow.** The embedding model has to be downloaded and loaded
  before anything is retrieved. Later queries are fast; do not read the first
  one as a latency measurement.
