# Changelog

Notable changes to rag-connector. Dates are the day the work landed.

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
