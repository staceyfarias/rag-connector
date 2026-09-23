# Contributing

RAG Connector is a small, product-neutral library that defines one thing: the
contract for treating a RAG pipeline as a black box. Its value is that the
contract stays narrow and that a connector written against it keeps working, so
changes are weighed against compatibility first — a redesign that is cleaner but
moves the surface is usually the wrong trade here.

Before opening a change:

```bash
python -m pip install -e ".[dev,reference]"
python -m pytest
python -m ruff check .
```

Connector-specific SDK dependencies do not belong in the core package. Add
concrete connectors through optional extras or independently installed packages.

All contributions are made under the MIT License.
