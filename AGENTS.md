# Agent guidance

RAG Connector is the product-neutral boundary for treating a RAG pipeline as a
black box. Do not add RAGauge evaluation policy or Pelorus curation behavior to
the core contract.

## Contract priorities

1. Evidence integrity outranks convenience. Every declaration a connector
   makes is read as truth by measurement code downstream, so a value that
   is guessed, synthesized, or quietly defaulted is worse than an absent
   one.
2. Stable chunk IDs must agree across retrieval and corpus-reading paths.
3. A backend error is not an empty retrieval result.
4. Canonical scores are higher-is-better; preserve backend-native scores.
5. Retrieval mode controls whether scores and `top_k` have metric meaning.
6. Fingerprints identify separate domains: connector configuration, corpus,
   embedding space, and retrieval policy.
7. Optional capabilities must remain independently optional.
8. Persisted connection dictionaries must be JSON-serializable and secret-free.

## Verification

```bash
python -m pytest
python -m ruff check .
```

Keep the core dependency-light. Connector-specific SDKs belong in separately
installed connector packages.

## Packaging: purge `egg-info` before building a release

```bash
rm -rf src/*.egg-info build dist && python -m build
```

**A green suite says nothing about what ships.** Two things are only observable
from a built artifact, and every consumer editable-installs this package — an
editable install resolves to `src/` on disk, so it always sees `py.typed` and
never builds an sdist.

- `py.typed` must be in the **wheel**, or consumers silently lose PEP 561 type
  information while the classifiers still advertise `Typing :: Typed`. It ships
  because `[tool.setuptools.package-data]` declares it explicitly, not by
  default.
- **Development notes must not be in the sdist.** Status write-ups, plans, and
  roadmaps are written for whoever picks the work up next, not for someone
  installing the package. Keep them untracked (there is a gitignored path for
  exactly this), because a public repo publishes every tracked file whatever
  `MANIFEST.in` says. What stays under `docs/` is reference a reader of the
  package needs. Note what this cuts both ways on: `tests/` **does** ship, by an
  explicit `MANIFEST.in` decision, so a comment or docstring in the suite is
  published prose — hold it to the same standard as the docs.

⚠ **`MANIFEST.in` changes are masked by a stale `SOURCES.txt`.** setuptools
reuses `src/rag_connector.egg-info/SOURCES.txt` when building an sdist, so
removing an entry from `MANIFEST.in` has no effect until that file is deleted.
This is not hypothetical: it hid the removal of an internal status file on the
first verified build (2026-07-28), and `egg-info/` is gitignored, so a dirty
working tree carries the stale list silently into a release. Verify a removal
by listing the built artifact, never by reading `MANIFEST.in`.

