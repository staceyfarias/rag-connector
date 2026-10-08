# Contract

RAG Connector models a RAG pipeline as a black box. A host supplies a query and
receives ranked retrieval evidence. Everything else is independently optional.

This page is normative: every rule below is enforced by `rag-connector
validate` or by the library itself. Beyond the enforced core there are
[Optional capabilities](#optional-capabilities) — real, implemented, and safe to
build against, some now validator-checked and some not (the table says which) —
and a smaller [Reserved surface](#reserved-surface) of types nothing produces
yet.

## Universal rules

1. Retrieval results are best-first and ranks are zero-based.
2. Stable chunk IDs agree across retrieval and corpus-reading paths.
3. Canonical scores are higher-is-better when scores are comparable.
4. Backend-native scores remain available as `raw_score` when transformed.
5. A backend failure raises `ConnectorOperationalError`; it is never represented
   by an empty result.
6. An empty result means the backend successfully found nothing.
7. Unsupported optional operations raise `UnsupportedCapability`.
8. Persisted connection documents are JSON-serializable and contain no secrets.

## The pipeline

Subclass `RagPipeline`. **`query()` is the only abstract method.** You must also
supply **one** of the two corpus reads:

| Method | Purpose |
| --- | --- |
| `query(text, top_k=None) -> list[RetrievedChunk]` | Run retrieval, best-first. Abstract. `top_k` is unset by default: retrieve however the connector is configured (see [top_k](#top_k-is-unset-by-default)). |
| `list_chunks(*, cursor=None, limit=1000) -> ChunkPage` | One page of the corpus, plus the cursor for the next. **Preferred.** |
| `pull_all_chunks() -> list[ChunkRecord]` | Enumerate the entire corpus. |

**Prefer `list_chunks`.** Pagination is what real stores natively expose, and
deriving the bulk pull from it costs nothing — whereas deriving paging from a
bulk pull makes every page, by-ID read and sample O(corpus). Each derives from
the other, so implement whichever your store actually offers; neither is
abstract, because making one abstract would force the other's implementers to
write a stub. Implementing *neither* raises `UnsupportedCapability` at call
time, naming both ways out.

If you implement both, put per-page invariants in `list_chunks` and corpus-wide
ones in `pull_all_chunks` — the latter can only be judged once paging has
finished.

These have working defaults you may override:

| Method | Default behavior |
| --- | --- |
| `list_chunks` / `pull_all_chunks` | Each derives from the other, as above. |
| `fingerprint() -> str` | SHA-256 over sorted `(chunk_id, text)` pairs, truncated to 16 chars. Override with a cheaper identity if the backend can compute one without a full pull. |
| `get_chunks(ids) -> dict[str, ChunkRecord]` | Filters a full `pull_all_chunks()`. Correct but expensive; override with a direct by-ID lookup. Missing IDs are simply absent from the result — never invent a placeholder. |
| `get_chunk_vectors(ids) -> dict[str, list[float]]` | Raises `UnsupportedCapability`. Override to return the vectors your index actually holds — see [Indexed chunk vectors](#indexed-chunk-vectors). |
| `generate(text, *, top_k=None, llm=None) -> GeneratedAnswer` | Raises `NotImplementedError`. Retrieval-only connectors leave it alone. |
| `info() -> dict` | Returns `name`, `type`, and `retrieval_mode`. Add anything a host should freeze as run metadata. |

A corpus read must return a *complete* snapshot. A partial pull is an
operational failure, not a smaller successful corpus. A `list_chunks` cursor
that repeats is refused rather than looped forever: a connector can be wrong,
and hanging is the worst way to find out.

### The corpus read returns the whole collection — never filter results

The corpus read returns **every chunk inside the collection boundary** the
connector is configured for. It never reproduces the system's retrieval rules.

Two kinds of metadata filter must be kept apart:

| | Defines the search space (collection boundary) | Narrows results within the space |
|---|---|---|
| What | Static filters sent with **every** search: the partition a shared index is divided by, e.g. a customer or channel key | Filters that depend on the query or its context: which section or module the user is in, presentation or delivery mode, access rules, score thresholds, top-k |
| Corpus read | **Applies exactly these** | **Never applies these** |
| `query()` | Applies them | Applies them |

Many customers sharing one index, partitioned by metadata, is a valid design.
The partition filters are the collection boundary: whatever that space admits
is the corpus, including content shared across the space (for example documents
every partition's search can see). Disclose the boundary filters in
`info()["internal_filters"]` — the filters the system applies to every query —
and apply exactly those in the corpus read, as explicit connection parameters,
so a host freezes them with the Dataset.

**Why.** The corpus is the reference a test measures retrieval against. If the
corpus read applies a filter that narrows results, content retrieval cannot
reach is silently missing from the corpus, so no test can ever ask about it and
no failure is ever seen: the defect the tests exist to find is hidden by the
fixture. A complete corpus turns the same defect into a visible miss.

**Any other exclusion is a declared deviation**, never a silent filter: name
it in `info()` (what is excluded and why), so the Dataset records it and a
reader can see the corpus is not the whole collection. Records that are not
servable content at all (internal bookkeeping a query can never return) may be
excluded on the same terms.

### `info()` keys hosts read

`retrieval_mode` is required (the base default supplies it). `embedding_fingerprint`
— provider, model and dimensions — is strongly recommended: hosts persist it with
the frozen corpus to detect that the embedding model changed underneath an
existing answer key. `distance` may name the backend metric; it is informational
only and never implies a score direction. `internal_filters` may disclose filters
the system applies to every query, for operator awareness; hosts never re-apply
them.

## Chunk identity

`ChunkRecord` carries five required fields. `validate_chunks()` raises
`ValueError` if any is `None` or an empty string:

| Field | Rule |
| --- | --- |
| `chunk_id` | Stable, unique across the corpus. |
| `doc_id` | Document identity. |
| `source_file` | Human-readable document name. |
| `doc_chunk_index` | Ordinal within its document, counting `0, 1, 2, …` with no gaps or repeats. |
| `text` | Non-blank chunk text. |

`global_index` — the corpus-wide ordinal — is **legally `None`** and is *not* in
the required set. A total order over the whole corpus is something a paged or
sharded store may not have and a hosted retrieval API will not expose, so
requiring it put one product's need into shared vocabulary. A host that needs
ordinals assigns them at ingest, where it knows the order it read things in.
Supply it for every chunk or for none; duplicates are the one dishonest option.

It has **no default**, deliberately — pass `global_index=None` explicitly. (It
precedes `text` in the field order, so adding a default would force one on
`text`, which is never absent, or reorder a dataclass whose consumers may
construct positionally.)

> **If you pass `None`, do not then sort on it.** `validate_chunks()` accepts an
> absent ordinal, so `sort(key=lambda c: c.global_index)` raises a `TypeError`
> from inside the sort key — far from the actual problem. This has bitten a
> shipped connector; sort on a `None`-tolerant key instead.

`doc_chunk_index` stays required: a store that can return a chunk can almost
always say where in its document it sits, and no host backfills it.

Optional: `embedding`, `char_start`, `char_end`, `content_module_id`, and a
`metadata` passthrough. Put `source_fingerprint` in `metadata` where you can —
without it a host cannot tell that your source documents changed. `source_identity`,
`relative_path` and `file_type` are recognized too.

**Stability is the load-bearing requirement.** A host freezes an answer key
against `chunk_id` values. Derive IDs from durable backend keys — never from
enumeration order, insertion sequence, or `uuid4()`. IDs that change between
pulls make every future run score zero against a key that is still perfectly
valid.

`doc_chunk_index` gaps are equally consequential: neighbor-window logic walks
±N by this ordinal, so a gap silently grades the wrong neighbors.

**The classic bug:** `query()` returning IDs in a different format from
`pull_all_chunks()`. Every hit then scores as a miss, recall flatlines at zero,
and the connector looks like it is working. The validator checks this
explicitly.

## Retrieval results and scores

`RetrievedChunk` carries `chunk_id`, `text`, `score`, `rank`, and optional
`doc_id`, `raw_score`, `metadata`. `rank` is zero-based and matches list order;
list order is what a host scores.

`score` is **canonical: higher is always better**, in the score's native scale.
No normalization is invented — L2 has no principled `[0, 1]` mapping, and cosine
similarity may legitimately be negative. Convert with the tested utilities and
keep the backend-native value in `raw_score`:

| Utility | Conversion |
| --- | --- |
| `cosine_distance_to_similarity(d)` | `1 - d` |
| `l2_distance_to_score(d)` | `-d` (a labeled direction flip, not a similarity) |
| `similarity_passthrough(s)` | identity, for scores already higher-is-better |

Leave `raw_score` as `None` when no conversion happened. `SCORE_CONVENTION`
(`"higher_is_better"`) is the stamp hosts record so stored scores stay readable
later.

A host's score threshold keeps `score >= threshold`. Unconverted distances would
therefore keep exactly the wrong half of the results — which is why the
direction rule is universal rather than per-backend.

## Derived results

Most retrievers return chunks: a returned row *is* the chunk its `chunk_id`
names. Some return something else — a parent document standing for the chunks
inside it, a summary written from several passages, a curated answer, or a
verified "the corpus does not answer this". A host still needs to know which
corpus chunks such a result stands for, what text was served, and what kind of
thing it was. A connector says so through `RetrievedChunk.metadata`, using four
fields:

| Field | Where | Meaning |
| --- | --- | --- |
| covered chunk ids | `metadata["item_covered_ids"]` | The corpus chunk ids the result stands for. Id-keyed scoring (which chunks were retrieved) credits the result through these, never through its own row id. |
| text | `RetrievedChunk.text` | What was served: the chunk's text, or a summary representing the covered chunks. Content-based judgement reads this. |
| kind | `metadata["item_kind"]` | `chunk`, `summary` or `gap`. `group` is accepted as a synonym of `summary`, because that is what existing connectors emit. |
| curated (optional) | `metadata["item_curated"]` | Curator notes about the result, kept separate from `text` and never part of it, so a host can report them without grading them as corpus content. |

Two grouping keys already in use are part of the same vocabulary:

| Key | Meaning |
| --- | --- |
| `item_index` | Optional non-negative integer: which returned item the row belongs to, 0-based. Rows sharing an index are **one** item occupying one ranked slot, and must declare the same `item_covered_ids` and `item_kind`. Without it, each row is its own item. |
| `item_label` | Optional string: the connector's own name for what the item is. Recorded and shown verbatim; nothing interprets it. |

The rules:

1. **Declare nothing and nothing changes.** A row without `item_*` keys is a
   chunk, credited through its own `chunk_id`. Plain chunk retrievers need no
   changes.
2. **`item_covered_ids` is a list of distinct chunk-id strings**, every one of
   which exists in the corpus read — an id the corpus does not hold is left
   out, not declared. It is a read set, not a ranking. An **empty** list is a
   declaration of zero coverage (for example, a summary whose citations were
   all lost); an **absent** key means "credit my own `chunk_id`". The two are
   different statements.
3. **Kinds.** A `chunk` row is the corpus chunk it names: its `chunk_id` is a
   corpus id and, if it declares covered ids, they are exactly `[chunk_id]`. A
   `summary` row stands for its covered chunks; give it `chunk_id=None` or an id
   in a namespace of its own, never a real chunk's id. A `gap` is a verified
   "the corpus does not answer this": it covers no chunks (`item_covered_ids`
   absent or empty) and its `text` is what was served in place of an answer.
4. **A gap is read from `item_kind == "gap"`, never from zero coverage.** A
   result that covers nothing for any other reason is not an abstention.
5. **Default kind.** When `item_kind` is absent, a row declaring
   `item_covered_ids` is read as a `summary`, and any other row as a `chunk`.
   Declare the kind rather than relying on this.
6. **An unknown kind is refused**, not guessed.
7. **Vendor extras** go in `<vendor>_*` keys (`acme_extract_id`). This
   contract never defines or reads them.

`rag_connector.derived` holds the key constants, `derived_result_metadata(...)`
to build a row's keys, `read_derived_result(hit)` to read one with the defaults
applied, and `derived_result_problems(hit)` for the structural checks.
`rag-connector validate` checks every declaring row: malformed declarations
fail; zero coverage without `gap`, a summary carrying a real corpus id, and an
undeclared kind warn.

## Retrieval modes

Declare `retrieval_mode` as a class attribute; it surfaces through `info()` and
a host freezes it. The default is `scored`, the only behavior that existed
before the convention.

| Mode | Cardinality | Scores | Threshold |
| --- | --- | --- | --- |
| `scored` | The connector's configured depth bounds the list (a caller's `top_k` overrides it) | Meaningful; every hit must carry a float | Allowed |
| `ordered` | As `scored` | Evidence only; `None` is legal | Refused |
| `complete_set` | **The system decides**; `top_k` is at most a breadth hint | Evidence only; `None` is legal | Refused |

Under `scored`, a `None` score is a contract violation — declare `ordered`
instead of returning `None`. Under `complete_set`, returning more than `top_k`
items is legal and expected (neighbor expansion, for instance).

### top_k is unset by default

`query(text)` and `generate(text)` take `top_k=None` by default (2026-10-08):
**retrieve however the connector is configured.** How deep a system retrieves
is part of the system, so it belongs in the connector's connection document
(the reference connector's `top_k` parameter, default 5), not in a caller's
argument.

- **An evaluation host does not pass `top_k`.** It observes the configured
  system as a black box. Its own metrics cutoff (the k of @k metrics, for
  example RAGauge's `metrics_k`) is a host setting and never reaches the
  connector; nor does the host cut what comes back before grading it.
- A caller that genuinely wants a depth (a debugging tool, an application)
  may pass a positive int; a `scored` or `ordered` connector honours it.
- Compatibility: a connector written with `top_k: int = 5` still works,
  because a host that omits the argument gets that default. Never pass
  `top_k=None` explicitly to a connector you did not write.

`resolve_retrieval_mode(live_mode, dataset_override)` returns
`(resolved_mode, source)`. The **only** legal override is `scored → ordered`
("I distrust this connector's scores"). Meaning cannot be minted into scores,
and cardinality is a behavioral fact of the system rather than an operator
opinion, so every other override raises `RetrievalModeError`.

`refuse_threshold_for_mode(mode, threshold)` raises rather than silently
dropping a threshold requested against a non-`scored` mode. Apply-or-reject:
swallowing a stored default is the exact bug class the refusal exists to
prevent.

## Errors versus empty results

All boundary failures derive from `ConnectorError` (itself a `RuntimeError`):

| Exception | Meaning |
| --- | --- |
| `ConnectorOperationalError` | The operation is supported, but the backend failed. **Missing evidence, not evidence of nothing.** |
| `UnsupportedCapability` | The connector does not implement a requested optional capability. |
| `ConnectorContractError` | Returned data violates the portable contract. |
| `ConnectorRegistrationError` | A connector could not be registered or reconstructed safely. Also a `ValueError`, so hosts that caught a duplicate registration that way before the typed class existed keep working. |

Returning `[]` for a backend failure is the single most damaging thing a
connector can do: it converts an outage into a confident zero score.

Generation is the deliberate exception. A **runtime** generation failure is
returned as `GeneratedAnswer(error=...)` rather than raised, so a run records
the failure instead of dying.

## Connection documents and registration

A `ConnectorSpec` is everything a host needs to offer and reconstruct a
connector:

```python
ConnectorSpec(
    connector_type="my-rag",   # unique across the environment
    label="My RAG",
    description="...",
    build=_build,              # (connection: dict) -> RagPipeline
    connect=_connect,          # optional: (params: dict) -> (pipeline, connection)
    params=[...],              # form fields a host renders
)
```

`connect()` is optional; a spec without it is not user-connectable and must be
built from a connection document directly.

`build(connection)` must reconstruct the connector **in a fresh process** from
nothing but the persisted JSON document plus external secret references.
Everything `__init__` needs lives in that document or in the environment.

The connection document is stored in plaintext by the host. It must be
JSON-serializable and free of credentials — read tokens, passwords and API keys
from the environment inside `__init__`. The validator fails any connection whose
keys look like secrets (`password`, `secret`, `token`, `api_key`, `apikey`,
`credential`).

Register by publishing `CONNECTOR_SPEC` through the package entry-point group
`rag_connector.connectors`:

```toml
[project.entry-points."rag_connector.connectors"]
my-rag = "my_package.connector:CONNECTOR_SPEC"
```

Discovery is per-environment and cached after the first **successful** call. An
entry point that raises therefore raises again on the next call, rather than
marking discovery done and serving a silently partial registry from then on —
the shape where a connector is simply missing and nobody is told why.
Successfully loaded entry points are tracked individually, so a retry does not
re-register a spec that a factory entry point already produced.

`connector_type` values must be unique; registering a second, different spec
under an existing type raises. Confirm registration with `rag-connector list`.

## Fingerprint domains

Connector configuration, corpus, embedding space and retrieval policy are
**separate identities**. Hosts decide which must match for a given operation —
comparing two runs over one corpus has different requirements from detecting
that an embedding model changed. Do not collapse them into a single hash.

Today `RagPipeline.fingerprint()` covers the corpus domain, and
`info()["embedding_fingerprint"]` covers the embedding space.

## Optional capabilities

Beyond the two abstract methods, a connector may implement any of the following.
Each is **independently optional**: no capability implies another. They are real
— the bundled `ReferenceRagConnector` satisfies every one in this table, so
each has a working reference to copy. (One nuance: for paged corpus reading the
reference deliberately carries no override — `base.py` derives pages from
`pull_all_chunks`, and that derived default is the implementation it uses.)

| Capability | Method | Returns | Reference impl | Validator |
| --- | --- | --- | --- | --- |
| Paged corpus reading | `list_chunks(*, cursor=None, limit=1000)` | `ChunkPage` | derived default (`base.py`) | checked when overridden |
| Query embedding | `describe_space()` / `embed_queries(texts)` | `EmbeddingSpaceDescriptor` / `EmbeddingBatch` | `reference.py` | not checked |
| Indexed chunk vectors | `get_chunk_vectors(ids)` | `dict[str, list[float]]` | `reference.py` | checked when overridden |
| Health | `health()` | `ConnectorHealth` | `reference.py` | not checked |
| Generation | `generate(text, *, top_k=5, llm=None)` | `GeneratedAnswer` | `reference.py` | checked when overridden |
| Dataset listing | `list_datasets()` | `list[DatasetRef]` | `reference.py` | not checked |
| Run snapshot | `run_snapshot()` | JSON object with `settings` and `observations` | `reference.py` | checked when implemented |
| Declared prompt templates | `prompt_templates()` / `render_rag_block(chunks)` / `generate_from_prompt(prompt, *, contexts=(), llm=None, decoding=None)` | `Sequence[PromptTemplate]` / `str` / `GeneratedAnswer` (with `.trace`) | `reference.py` | checked when declared |

An override is what gets checked, not mere structural presence: a capability
left to its base default is reported as skipped, which is a supported
configuration rather than a failure.

### Declared prompt templates

`generate()` builds its prompt internally and reports it afterwards as the
opaque `GeneratedAnswer.hydrated_prompt` string. That is enough to see what was
sent and not enough to *control* it: the template cannot be separated from the
data poured into it, hashed on its own, held constant, or varied deliberately.
This capability adds the controlled path alongside it. It is additive — an
existing connector implements none of it and behaves exactly as before.

Three methods, declared **as a unit** (`supports_declared_prompts(pipeline)`;
`declared_prompt_methods(pipeline)` names which halves are present). Two out of
three is a failure the validator reports, not a degraded mode: it leaves a host
holding templates nothing can run, or accepting a prompt nobody can enumerate a
template for.

1. **`prompt_templates()`** enumerates `PromptTemplate` values — a stable `id`,
   a human `description`, `template` text carrying `{rag_block}` and `{query}`,
   an optional `system` preamble, and a declared `cache_breakpoint` position.
   `ConnectorSpec.prompt_templates` is the same declaration as a **zero-arg
   callable**, readable from a registry listing before any params, credentials
   or backend exist — the `connector_info` shape, for the same reason: an
   operator choosing a prompt has to see the choice before binding anything.
   Serve one module-level constant through both surfaces, or the variant that
   was picked is not the variant that runs.
2. **`render_rag_block(chunks)`** renders chunks *the host supplies* in the
   connector's own production block format. The host fixing the context is what
   makes groundedness measurable; the connector owning the format is what keeps
   the prompt realistic. `render_numbered_rag_block` is available for the
   conventional `[n] chunk_id=...` shape.
3. **`generate_from_prompt(prompt, ...)`** runs a `RenderedPrompt` (built by
   the shared `render_prompt`) and returns a `GeneratedAnswer` whose `.trace` is
   a `GenerationTrace`: every message with its role, the model, the decoding
   params, the response, token usage, and the template and block hashes.

**The trace supplements `hydrated_prompt`; it does not replace it.** Both are
populated, `trace.prompt.text` is the value to use for the string so the two
cannot disagree, and `trace` is `None` — absent, never an empty trace — on a
connector that does not declare the capability.

#### One-shot is enforced, not assumed

A host documents these prompts as single-turn with no conversation history, so
the contract makes that true rather than hoping for it. `PromptTemplate` has no
field that can hold a prior turn, `render_prompt` emits at most one system
message and exactly one user message, and `assert_one_shot` re-checks that on
the way out. History smuggled into the template *string* — `{chat_history}`, an
`Assistant:` turn label, `[INST]`, "the conversation so far", "previous turns"
— is **rejected** in `validate.py` and raised by `render_prompt`, not accepted
with a warning. An answer grounded in an earlier turn nobody recorded is
indistinguishable, afterwards, from an answer grounded in nothing.

Substitution is literal replacement of the two placeholders, not `str.format`,
so a template may show a JSON shape in its instructions without escaping every
brace. Any *other* `{placeholder}` is refused, because it would reach the model
verbatim.

#### Cache breakpoints are declared, never manufactured

`cache_breakpoint` is one of `none`, `system_end`, `before_rag_block`,
`after_rag_block`. It names where a prompt-cache breakpoint **may** go;
`render_prompt` resolves it to a `(message_index, char_offset)` a caller can
split on, and `cacheable_prefix()` returns the content before it.

**Nothing reorders a template to make it cacheable.** Caching matches on an
exact prefix, so a breakpoint whose prefix contains `{query}` would never be
reused — that declaration is refused, and the author is told the two honest ways
out: declare a separate variant whose template genuinely puts the block first,
or declare `none`. Hoisting the block above the query would produce a prompt the
connector does not ship, and a measurement of a prompt nobody runs is worth less
than no measurement. `none` is the correct declaration for a realistic prompt
that is not cacheable in the order it is written — the Reference RAG's own
shipped variant declares exactly that.

A declared breakpoint is a permission, not a promise: providers ignore a cached
prefix below a model-dependent minimum, silently. Apply it, then verify against
`trace.usage.cache_read_input_tokens`. `TokenUsage` fields are all legally
`None`, meaning *not reported* — never zero, because a zero cache read is an
actionable observation and an unknown one is not. `GenerationTrace` also records
`cache_breakpoint_applied`, so "declared but not applied" is distinguishable
from "applied and missed".

### The embedding space descriptor

`describe_space()` returns an `EmbeddingSpaceDescriptor`, whose `metric` and
`normalized` fields are **tri-state**. A value asserts the property; `None`
means the connector cannot or does not report it. The distinction is
load-bearing: a host's similarity thresholds are calibrated in the units of one
specific geometry, so a connector reporting a *different* metric declares those
numbers invalid for its space, while one reporting `None` declares only that it
does not know. Unverifiable is not the same as incompatible, and a connector
that guesses a plausible default turns the first into the second silently.

Assert a value only when you control or have measured it — the bundled
Reference RAG takes `normalized` as a constructor argument, defaulting to
`None`, rather than inferring it from a model name, because its embedding
client is swappable.

`fingerprint` is the stable identity of the space itself, binding every derived
artifact (cached vectors, centroids, saved clusterings) to the space it was
computed in. Reuse an existing identity rather than deriving a second one: two
fingerprints for one space is how drift detection stops working.

The matching protocols — `PagedCorpusReader`, `QueryEmbedder`,
`HealthCapability`, `ChunkVectorReader`, `AnswerGenerator`,
`DeclaredPromptGenerator` — are `runtime_checkable`, so a host can discover
support with `isinstance`.
`RagConnector` is the structural form of the enforced contract and is exercised
by `tests/test_contract.py`.

`RunSnapshotProvider` is also runtime-checkable. Unlike methods with a concrete
base default, a connector satisfies it only when it actually implements
`run_snapshot()`.

Any method with a concrete `RagPipeline` default is an exception to
`isinstance` discovery: `get_chunk_vectors`, `generate`, and all three
declared-prompt methods ship raising defaults and `list_chunks` ships a derived
one, so *every* pipeline satisfies `ChunkVectorReader`, `AnswerGenerator`,
`DeclaredPromptGenerator`, and `PagedCorpusReader` structurally.
To ask whether a connector answers for itself rather than inheriting the
default, use `supports(pipeline, "method_name")` (or the original
`supports_chunk_vectors(pipeline)` shorthand for chunk vectors, and
`supports_declared_prompts(pipeline)` for the all-three prompt capability). Either way,
calling and catching is always correct.

### Run snapshots

`run_snapshot()` is an optional, read-only description of connector state that
may explain differences between evaluation runs. It returns a
JSON-serializable object (screened against a best-effort denylist of
secret-looking key names — values are never inspected, so the connector
remains responsible for not returning secrets) with this shape:

```json
{
  "schema": "my-rag.run-snapshot.v1",
  "settings": {"retrieval_strategy": "hybrid", "reranker": "model-v2"},
  "observations": {"indexed_chunks": 1200, "active_extracts": 83}
}
```

`settings` are configured behavior. `observations` are cheap facts about the
bound system at capture time. Neither is an evaluation score: hosts keep these
values separate from TEST metrics, preserve the raw JSON, and may calculate a
canonical fingerprint or a key-by-key diff. A connector may add arbitrary
nested JSON values under either section, but must not include credentials,
large record dumps, timestamps that make an otherwise stable snapshot change,
or values it cannot establish honestly. One timestamp is the deliberate
exception: where served content can change out-of-band (index updates,
curated or managed content), a **last-modified marker for the most recent
change that affects what retrieval returns** belongs in `observations` — it
moves only when the system actually changed, which is exactly the drift the
snapshot exists to disclose. The conformance validator requires
strict JSON (no NaN or infinity), scans nested keys for credentials, and caps
the encoded snapshot at 64 KiB.

Capture is deliberately per evaluation result rather than only when a corpus
is connected. A mutable RAG may change between two tests in one higher-level
run; recording each observation is how a host can disclose that drift instead
of combining unlike states into one headline.

### Dataset listing

A backend usually holds more than one of whatever unit a connector binds to — a
Chroma collection, a Pinecone index or namespace, a hosted knowledge base. A
connector instance reads exactly one. `list_datasets()` answers the other
question: what else is there to bind to, so a host can offer the choice instead
of making an operator retype a name from memory.

There are two surfaces, and which one a host uses is decided by whether
anything is bound yet.

| Surface | Signature | For |
| --- | --- | --- |
| `ConnectorSpec.list_datasets` | `(params: dict) -> list[DatasetRef]` | Enumerating **before** anything is bound. Guarded by `spec.supports_dataset_listing`. |
| `RagPipeline.list_datasets` | `() -> list[DatasetRef]` | A connector already bound, showing what else its backend holds. Guarded by `supports(pipeline, "list_datasets")`. |

The spec-level one takes **partial** params: enough to reach the backend (a
persist directory, an endpoint, credentials) but not the field that selects a
dataset, since that is the field the operator is being helped to choose. That
asymmetry is the whole reason it exists — a host cannot reach the instance
method without first binding, and binding takes the very parameter in question.
The library therefore does **not** offer a generic "bind a throwaway instance
and enumerate" fallback: it would fail precisely when it was needed. Implement
either surface, or both; the Reference RAG implements both, with the instance
method delegating to the spec-level function.

`DatasetRef` carries:

| Field | Rule |
| --- | --- |
| `id` | Identifies the dataset in the backend's own terms. |
| `label` | Human-readable name; may equal `id`. |
| `item_count` | **Best-effort, legally `None`.** Report it only when the backend answers cheaply. `None` means "not reported" — never `0`, which would make a populated dataset read as a mistake. |
| `connect_params` | The fragment of this connector's `params` that selects this dataset. This is what makes a listing actionable: a host merges it into the params it already collected and calls `connect` again, without knowing which param name any given connector selects on. |
| `metadata` | Whatever else is cheap and honest (the backend's own noun for the unit, an error that prevented a count). |

**Listing never rebinds.** `list_datasets()` returns information; it does not
change what the instance reads. Rebinding is the host's act, and it produces a
*new* instance through `connect`/`build`, because run identity and every issued
fingerprint are anchored to a binding that must not move underneath them.

The default raises `UnsupportedCapability` rather than returning an empty list,
for the same reason `get_chunk_vectors` does: "cannot enumerate" and "holds
nothing" are opposite facts, and an empty list states the second while meaning
the first.

### Indexed chunk vectors

`get_chunk_vectors(ids)` returns the vectors the index **actually holds** —
never a re-embedding of the chunk text. A caller recomputes
`dot(unit_query, unit_chunk)` and compares it to the score the connector
reported, which is how a connector that declares cosine but returns a rescaled
score (`(1 + cos) / 2`, say) gets caught. A rescale preserves ranking, so
retrieval looks perfect while every calibrated threshold quietly stops meaning
what it was calibrated to mean. Re-embedding the text would test the embedder
against itself and pass even when the index is stale or was written in a
different embedding space.

- Missing ids are simply **absent** from the returned dict — never a placeholder
  and never a zero vector. A fabricated vector is worse than a gap, because
  nothing downstream can detect it.
- Empty input returns an empty dict without hitting the backend.
- A connector that cannot do this raises `UnsupportedCapability` (the inherited
  default), so a caller skips verification and reports the connector as
  *unverified* rather than *verified clean*. Silence must never look like
  success.

**What is true of this table and not of [Reserved surface](#reserved-surface):**
these methods exist, return constructed values, and are safe to implement
against. Paged reading, indexed chunk vectors and generation are now exercised
by `rag-connector validate` when you override them; query embedding and health
are not yet, so for those treat the signatures as stable and the verification as
your own responsibility.

Implement only what the backend supports honestly — never synthesize embeddings,
scores, corpus completeness or generation evidence to unlock a host feature. An
unimplemented capability is a supported configuration; a faked one silently
corrupts every measurement taken through it. Unsupported optional operations
raise `UnsupportedCapability`.

## Reserved surface

The following are exported from `rag_connector` but are **not yet produced,
consumed, or validated by any code path** in this library. They record intended
direction. Treat them as unstable: nothing constructs them, the validator does
not check them, and a host cannot rely on receiving them.

- **Descriptor types** — `ConnectorDescriptor`, `ScoreSemantics`,
  `RetrievalResult`. `ConnectorDescriptor` is the intended future home of the
  four separate fingerprint domains described above.
- **Capability protocols nothing implements yet** — `ContentPublisher`,
  `QueryTelemetryReader`, and `IndexIntrospector`.

The intended capability profiles, once these are wired, are:

| Profile | Protocols |
| --- | --- |
| Retrieval | `RagConnector` |
| Corpus inspection | `PagedCorpusReader` |
| Query geometry | `QueryEmbedder` |
| Corpus geometry | `QueryEmbedder` + `ChunkVectorReader` |
| Answer evaluation | `AnswerGenerator` |
| Curation publication | `ContentPublisher` |
| Operational inspection | `HealthCapability`, `QueryTelemetryReader`, `IndexIntrospector` |
