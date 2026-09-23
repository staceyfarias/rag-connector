# Bundled corpora — MIT licensed, and they ship

Everything under this directory is **package data**, not documentation: it is
read at runtime from inside the installed package, so it has to be in the wheel
as well as the sdist (same invisibility trap as `py.typed` and
`CONNECTOR-INFO.md` — an editable install reads it off `src/` and stays green
whether or not it ships). It is declared in `[tool.setuptools.package-data]`
and in `MANIFEST.in`.

## License

These documents are licensed **MIT**, the same as the rest of this package.
They are original synthetic material written for testing retrieval; nothing in
them is quoted from a third party, and they describe no real company, product,
or person. Reuse them freely.

## `pelorus_space/`

92 short documents (`.txt` and `.md`) for a fictional interplanetary travel
operator — booking and cancellation policy, medical clearance, baggage rules,
loyalty tiers, destination guides. They exist because a retrieval demo needs a
corpus with real structure: overlapping topics, near-duplicate policy wording,
and questions whose answer is split across documents. A handful of paragraphs
of lorem ipsum demonstrates nothing.

`provision_reference_rag()` with no folder argument provisions this corpus, so
`pip install "rag-connector[reference]"` gives you a Reference RAG with
something to point it at — see `rag_connector.ingest.bundled_corpus_path`.

The names inside these documents, "Pelorus Voyages" included, are deliberate
and stay as they are. Do not rename anything in the document text: the corpus
is byte-identical to the copy other repositories measure against, and a rename
would silently change every source fingerprint and every chunk id derived from
it.
