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

28 documents (`.md` and one `.txt` notice) for a fictional interplanetary
travel operator, shaped like what a real operator publishes: formal booking
terms and conditions, a passenger handbook, a medical and fitness guide, a
loyalty programme guide, packing, onboard and departure guides, itinerary and
destination guides, and deliberately overlapping marketing material (excursion
flyers, itinerary fact sheets, a family age-limits flyer, a muster card and an
FAQ page). Each carries a publisher, edition and issue date. They exist because
a retrieval demo needs a corpus with real structure: long documents whose
tables and rules are cut by chunking, overlapping topics, near-duplicate
wording between a policy and its brochure, and questions whose answer is split
across documents. A handful of paragraphs of lorem ipsum demonstrates nothing.

Some features are deliberate test material, not errors: loyalty tiers share
their names with cabin classes; a notice issued 15 November 2042 raises the
companion discount from 15 to 20 percent for bookings from 1 January 2043
while the older documents still say 15; the Outer Reaches Tour charges sheet
and Section 11 of the medical guide state their itinerary scope once, chunks
away from the values it governs.

This is corpus version 2 (28 documents, 426 chunks). Version 1 (92
documents, 605 chunks) had 72 short documents that duplicated the guides; its
chunk ids are not those of version 2.

`provision_reference_rag()` with no folder argument provisions this corpus, so
`pip install "rag-connector[reference]"` gives you a Reference RAG with
something to point it at — see `rag_connector.ingest.bundled_corpus_path`.

The names inside these documents, "Pelorus Voyages" included, are deliberate
and stay as they are. Do not rename anything in the document text: the corpus
is byte-identical to the copy other repositories measure against, and a rename
would silently change every source fingerprint and every chunk id derived from
it.
