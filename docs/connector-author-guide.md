# Connector author guide

A connector translates a concrete RAG system into the portable contract. Keep
backend SDK dependencies in the connector package; the optional Reference RAG
is the only concrete backend bundled with `rag-connector`.

The Reference RAG is worth reading before you write anything. It is the
conformance suite's known-good implementation — the validator is run against it
— so it is a worked answer to every question below rather than a sketch, and
anything it does is by construction a legal way to do it.

## Before you write any code

Most of a connector's difficulty is not in the translation; it is in finding
out what the backend actually does. Establish these from **the production
code and configuration**, with file-and-line evidence, before designing
anything — an assumption recorded here as a fact becomes a declaration a host
reads as truth:

- the retrieval entry point, and what embeds the query;
- the store query it issues, and any filter or threshold applied to every
  request — those are `internal_filters`, and they must be disclosed;
- neighbor expansion, re-ordering, or prioritization after the store returns;
- whether the final order still carries score meaning, or is a policy order
  that only looks like a ranking;
- whether `top_k` is a hard limit or a hint the system may overrule;
- how a failure differs, observably, from a successful empty result — and
  whether the backend can return nothing at all. A system that always returns
  its best *k* matches cannot abstain on an unanswerable question, and that is
  a finding about the system worth reporting, not a gap to paper over;
- the stable id carried on every returned chunk;
- whether the text it returns is the frozen chunk text or presentation text
  assembled for display.

Then **prove corpus identity**, because retrieval without a trustworthy corpus
enumeration is not a usable connector: every id retrieval returns must exist in
the corpus read, and two reads must agree. The validator checks both, but it
can only check them *within one session* — so satisfy the harder rule it cannot
see, that ids are stable across re-ingestion. Positional ids that a re-chunk
refills with different text are the classic silent corruption: everything
passes, and every citation anyone stored now points at the wrong words.

Finally, decide the **runtime boundary** — in-process against the backend's own
libraries, or across its internal API — from the real deployment topology
rather than from convenience, and write down why. It is the decision most
expensive to revisit later.

## Required implementation

Subclass `RagPipeline` and implement **two** things: `query()`, and **one**
corpus read — either `list_chunks()` (preferred) or `pull_all_chunks()`. Each
derives from the other, so implement whichever your store actually offers.

Prefer `list_chunks`. Pagination is what real stores natively expose, and
deriving the bulk pull from it is free, whereas deriving paging from a bulk pull
makes every page O(corpus). If you find yourself writing a paging loop inside
`pull_all_chunks`, you are hand-building the adapter the base class already
provides — and a host that prefers paged reads has nothing to call.

**The corpus read returns everything in the collection.** Apply only the
filters that define the search space — the static collection boundary sent with
every search, such as the partition key of an index shared by many customers —
and never the filters that narrow results within it (current section or module,
delivery mode, access, thresholds, top-k); those belong in `query()`. A corpus
filtered by result rules hides exactly the content retrieval cannot reach, so
the tests can never find that defect. Disclose the boundary in
`info()["internal_filters"]`; anything else you exclude is a declared deviation
named in `info()`. See [contract](contract.md), "The corpus read returns the
whole collection".

`info()` is **not** required: the base default returns `name`, `type` and
`retrieval_mode`. Override it to add anything a host should freeze as run
metadata, such as `embedding_fingerprint`.

Retrieval IDs must be stable across query and corpus pulls, ranks must be
zero-based, and scores must follow the declared retrieval mode.

If the system can return content that is **not** in the corpus — a curated
answer, a synthesized summary, a template response — give it `chunk_id=None`,
or an id in a namespace of its own, plus metadata saying where it came from.
Never borrow a real chunk's id for it. `RetrievedChunk.chunk_id` is optional
precisely so that this case has an honest representation; a borrowed id makes
evidence that was never in the corpus indistinguishable from evidence that was.

If such a result stands for corpus chunks — a parent document, a summary
written from passages, a curated answer — or is a verified "the corpus does not
answer this", declare it with the four-field derived-result model: the chunk
ids it covers, the text you served, its kind (`chunk`, `summary` or `gap`), and
optionally curator notes kept apart from the text. Build the keys with the
helper rather than by hand:

```python
from rag_connector.derived import derived_result_metadata

RetrievedChunk(
    chunk_id="answer:42",                       # its own namespace
    text="A summary of the refund rules...",
    score=0.91, rank=0,
    metadata={
        **derived_result_metadata(
            kind="summary",
            covered_ids=["refunds.md:chunk-0", "refunds.md:chunk-3"],
            curated="Checked against the 2026 policy.",
        ),
        "acme_answer_id": 42,                   # your own extras: acme_*
    },
)
```

Two mistakes the validator will catch: covered ids the corpus does not hold,
and a "no answer" result that covers nothing but is not declared
`kind="gap"` — a host will not read it as an abstention. See
[Derived results](contract.md#derived-results).

See [the contract](contract.md) for the normative rules behind each of these.

## Optional capabilities

Beyond the required pair, `generate()` and a cheap `get_chunks()` override are
the two a host is most likely to use. Implement only what the backend supports
honestly — never synthesize embeddings, scores, corpus completeness or
generation evidence to unlock a host feature. An unimplemented capability is a
supported configuration; a faked one silently corrupts every measurement taken
through it.

If you want a host to be able to measure whether your generator stays grounded
in the context it was given, declare your prompt templates. `generate()` reports
its prompt only afterwards, as one opaque string, which is enough to see what
was sent and not enough to hold it constant or vary it deliberately.
`prompt_templates()` publishes the template text itself (with `{rag_block}` and
`{query}` placeholders), `render_rag_block(chunks)` renders context the *host*
supplies in your real block format, and `generate_from_prompt()` runs the
result and returns a structured trace beside the usual `GeneratedAnswer`.
Publish the same declaration as a zero-arg `ConnectorSpec.prompt_templates` so
an operator can see the variants before binding anything. Two rules will bite if
you ignore them: a template implying conversation history is **rejected**, not
warned about, and a cache breakpoint declared on a prefix that changes with the
question is refused rather than fixed by reordering your prompt for you — if
your realistic prompt puts the question first, declare `cache_breakpoint="none"`
and keep the prompt you actually ship.

If your backend holds more than one bindable unit — collections, indexes,
namespaces — implement dataset listing so a host can offer the choice instead
of asking an operator to retype a name from memory. Supply
`ConnectorSpec.list_datasets` (a function of *partial* params, callable before
anything is bound) and/or override `list_datasets()` on the connector itself.
See [Dataset listing](contract.md#dataset-listing) for the split, and note the
one rule that matters: listing reports, it never rebinds.

Implement `run_snapshot()` when the backend can cheaply expose configuration or
state that would help explain differences between evaluation runs. Return
bounded, strict JSON with separate `settings` and `observations` objects; these
values describe the system and are not evaluation scores. The validator screens
nested keys against a best-effort denylist of secret-looking names — values are
never inspected — so keeping actual secrets out is the connector's
responsibility, not the denylist's. Hosts may capture the snapshot before and
after every run and treat a mid-run difference as breaking run comparability,
so include the settings that genuinely affect what retrieval returns, and —
where served content can change out-of-band (index updates, curated or managed
content) — a last-modified marker for the most recent change that affects
results. See [Run snapshots](contract.md#run-snapshots).

Beyond those, query embedding (`describe_space` / `embed_queries`), indexed
chunk vectors (`get_chunk_vectors`), and `health()` are implemented by the
bundled `ReferenceRagConnector` and safe to build against — see
[Optional capabilities](contract.md#optional-capabilities), which names the
reference implementation for each and says which the validator now checks. A
smaller [Reserved surface](contract.md#reserved-surface) of types nothing
produces yet is listed separately; do not build against those.

## Configuration and secrets

Publish a `ConnectorSpec` through the `rag_connector.connectors` package entry
point. Its `build(connection)` function must reconstruct the connector in a
fresh process using only the persisted JSON document and external secret
references. Tokens, passwords, and API keys never belong in the connection
document.

That constraint is not arbitrary caution, and knowing why keeps you from
designing around it: **this library owns no storage and no secret resolution.**
Where a connection document is kept, and how a credential reference turns into
a credential, are host policies behind narrow interfaces, so that connector
management can be shared without anyone committing to a particular database,
keychain, or deployment model. The consequence for you is that there is no
library-side place to stash state between `connect()` and `build()`. Whatever
`__init__` needs is in the document you returned, or in the environment.

Start from [`examples/connector_template.py`](../examples/connector_template.py).
In an independently packaged connector, expose its `CONNECTOR_SPEC` with:

```toml
[project.entry-points."rag_connector.connectors"]
my-rag = "my_package.connector:CONNECTOR_SPEC"
```

## Validation

`rag-connector validate` is the conformance suite. It reports PASS / WARN /
FAIL per check, with the fix for anything that failed, and verifies:

- `info()` returns usable run metadata, and `retrieval_mode` is recognized;
- the corpus read enumerates a non-empty corpus of `ChunkRecord` objects;
- the chunk metadata contract: required fields, unique `chunk_id`, non-blank
  text, gapless per-document `doc_chunk_index`, and `global_index` unique
  *where supplied* (all-`None` is legal and passes);
- `fingerprint()` and `chunk_id` values are stable across two full pulls;
- `query()` returns ranked `RetrievedChunk` objects and respects `top_k`
  (except under `complete_set`, where the system owns cardinality);
- **retrieved IDs all exist in the pulled corpus** — the classic bug;
- derived-result declarations (`item_covered_ids`, `item_kind`,
  `item_curated`, `item_index`, `item_label`), where a hit makes one, follow
  [the rules](contract.md#derived-results) — a hit that declares covered ids
  is checked through those ids in the classic-bug check above;
- retrieval is deterministic for a repeated query;
- scores run higher-is-better down the ranking (skipped under `ordered` and
  `complete_set`, where scores are evidence only, and skipped-with-a-reason when
  the probe returned no hits to inspect);
- `get_chunks()` returns known IDs and omits unknown ones without inventing
  placeholders — **and whether your override is real**: inheriting the base
  default warns, and so does an "override" that answers a by-ID fetch by pulling
  the whole corpus;
- **no stray public method shadows an unoverridden contract default** — the
  renamed-override bug, where a misspelled method leaves the base default
  quietly answering and your implementation dead;
- `list_chunks()`, when overridden, pages with an advancing cursor that
  terminates, and reproduces `pull_all_chunks()` chunk for chunk. Pages may
  hold fewer items than `limit` (a backend cap is legal and is noted, not
  failed); what fails is a repeated cursor, a run of empty pages that still
  return a cursor, or a pager that keeps serving items past the corpus size;
- `get_chunk_vectors()`, when overridden, returns honest indexed vectors —
  empty input yields `{}` without a backend call, and unknown IDs stay absent
  rather than being fabricated;
- `generate()`, if implemented, returns a `GeneratedAnswer`;
- declared prompt templates, if implemented, are implemented as a **unit** —
  all three of `prompt_templates()`, `render_rag_block()` and
  `generate_from_prompt()` — every declared template is a legal one-shot RAG
  prompt with no conversation history, every declared cache breakpoint sits on
  a prefix that is genuinely stable across questions, and the returned
  `GeneratedAnswer` carries both a JSON-serializable `trace` and a still-populated
  `hydrated_prompt`; the registry-level declaration is checked to match the
  instance one;
- `run_snapshot()`, if implemented, returns bounded, strict JSON (nested keys
  screened against a secret-name denylist; values are not inspected) with
  separate `settings` and `observations` objects;
- the connection document is JSON-serializable and free of secret-looking keys;
- in registry mode, `build(connection)` reconstructs a working connector.

It does **not** yet check query embedding (`describe_space` / `embed_queries`)
or `health()`, even though both are implemented and safe to build against. Nor
does it check the unwired [reserved surface](contract.md#reserved-surface).

Run it before integrating with a host. There are two modes.

**Direct import** works before your connector is registered, and imports nothing
beyond `rag_connector.base` — so it runs in your own project's environment:

```bash
rag-connector validate \
  --import my_package.connector:MyConnector \
  --kwargs "{\"endpoint\":\"http://localhost:8000\"}" \
  --query "a question the corpus can answer"
```

**Registry mode** runs after your package is installed and additionally proves
the `connect()` → secret-free connection document → fresh `build(connection)`
reconstruction round-trip that a host depends on:

```bash
rag-connector list          # confirm your connector_type is discovered
rag-connector validate \
  --connector-type my-rag \
  --params "{\"endpoint\":\"http://localhost:8000\"}" \
  --query "a question the corpus can answer"
```

If `rag-connector list` does not show your `connector_type`, the package is not
installed in this environment or its entry point is missing or misspelled.

It exits `0` for READY, `1` for NOT READY, and `2` if the validator itself could
not run.

Warnings are not blockers, but read them — each one is a way the numbers can
quietly mean less than they appear to.

## Integrating with a host

Hosts discover connectors through installed package metadata, not by scanning
source directories. The sequence is the same for any host:

1. **Install the host and your connector into the same Python environment.**
   This is the single most common integration failure — a connector installed
   into a different interpreter than the one the host runs is invisible to it.
   Use `python -m pip install -e .` while developing.
2. **Confirm registration** with `rag-connector list`. Your `connector_type`
   must appear. If it does not, the package is not installed here, the entry
   point is missing or misspelled, or importing your module raises.
3. **Validate in registry mode** before opening the host, so a contract failure
   surfaces as a readable report rather than as bad numbers later.
4. **Restart the host.** Entry points are read at process start; installing a
   package or changing its metadata while the host is running has no effect.
5. **Connect a corpus** through the host's own UI or API, then confirm the
   corpus size is plausible, the persisted `connector_type` is yours, and
   reopening it reconstructs the connector.

The boundary that sequence runs along is worth stating, because it tells you
which side to take a problem to. **This library owns** the contract and its
portable types, capability and retrieval-mode declarations, score and
fingerprint semantics, discovery and registration, connection parameter
schemas, secret-free connection documents and their reconstruction, validation
and the reusable conformance tests, and the author CLI, template and Reference
RAG. **A host owns** authentication and authorization, its own administrative
UI, where connection documents and secret references are stored, binding a
connector to whatever it calls a dataset or tenant, product-specific lifecycle
and jobs, gating features on your declared capabilities, and whatever it does
with the evidence you return. A host route may expose connector management, but
it delegates to this library rather than keeping a second registry or factory
of its own — so a discovery or reconstruction bug is this library's, and a
storage or permissions bug is the host's.

### Troubleshooting a connector that does not appear

| Symptom | Cause |
| --- | --- |
| Absent from `rag-connector list` | Not installed in this environment; or no `rag_connector.connectors` entry point; or the entry-point target path is wrong. |
| Listed by the CLI, absent in the host | The host runs a different interpreter, or has not been restarted. |
| Registration raises on start-up | Another installed package already claims that `connector_type`. |
| `ImportError` during discovery | A backend SDK is missing — declare it in your package's dependencies, not the host's. |
| Appears, but connecting fails | Run registry-mode validation; the report names the failing contract rule. |
