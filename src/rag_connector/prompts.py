"""Declared prompt templates and structured generation traces.

Why this exists
---------------
``RagPipeline.generate`` builds its prompt internally and hands back an opaque
``GeneratedAnswer.hydrated_prompt`` STRING *after the fact*. A host therefore
learns the prompt only post-hoc and cannot separate the **template** from the
**data** poured into it. That makes the prompt uninspectable, unhashable
independently of the query it happened to carry, and impossible to hold
constant or vary deliberately -- which is exactly what a groundedness
measurement needs, because "did the generator stay inside the context it was
given" is only answerable when you know what else it was told.

This module is the fix, at the contract layer:

* :class:`PromptTemplate` -- what a connector *declares*: a stable id, a human
  description, template text carrying ``{rag_block}`` and ``{query}``, and the
  one position where a prompt-cache breakpoint MAY go.
* :func:`render_prompt` -- the single, shared substitution. Every connector
  renders identically, so a rendered prompt is reproducible from
  ``(template, rag_block, query)`` by anyone holding those three.
* :class:`RenderedPrompt` -- the messages actually sent, with the template hash
  and the block hash carried separately so a run can key on either.
* :class:`GenerationTrace` -- what came back: model, decoding params, response,
  token usage, and the hashes. It **supplements** ``GeneratedAnswer``; the
  legacy ``hydrated_prompt`` string stays populated.

Three rules are enforced here rather than documented and hoped for.

**One-shot, structurally.** The type system cannot express conversation
history: a template has one body and an optional system preamble, and there is
no field for a prior turn. :func:`render_prompt` then produces at most one
system message and exactly one user message, and :func:`assert_one_shot`
re-checks that on the way out. A template that *implies* history in its prose
(``Assistant:`` turn labels, ``{chat_history}``, "the conversation so far") is
**rejected**, not warned about -- see :func:`prompt_template_problems`.

Be precise about what those two halves buy, because they are not equal. The
STRUCTURAL half is a fact about the contract: no rendered prompt can carry a
prior assistant turn, because nothing can hold one. The PROSE half is a
denylist, and denylists leak -- an adversarial sweep got 10 of 11 fabricated
transcripts past it, including near-misses of listed patterns (``AI assistant:``,
``### Assistant``) and a zero-width space before a colon. So a host may state
single-turn-with-no-history as a structural fact, and must NOT state that a
connector author cannot hand-write a fake transcript into the body text. That
one stays an assumption about connectors, and the validator only raises its
cost.

**Fidelity beats cache savings.** A connector *declares* where a breakpoint may
go; nothing here rearranges its prompt to make one possible. If the declared
position sits after the query -- so the "stable prefix" is not stable and the
breakpoint would save nothing -- the declaration is refused and the author is
told to fix the template or declare ``"none"``. Silently hoisting the RAG block
above the query would produce a prompt the connector does not ship, and
measuring a prompt nobody runs is worse than measuring nothing.

**A declared breakpoint is a permission, not a promise.** Providers cache on an
exact prefix match and ignore prefixes below a model-dependent minimum
(roughly 512-4096 tokens), silently. So a caller applies the breakpoint and
then *verifies* against reported usage -- :class:`TokenUsage` keeps
``cache_read_input_tokens`` and ``cache_creation_input_tokens`` for exactly
that -- rather than assuming a declared position took effect.

This module is provider-neutral: nothing here builds a provider request. It
names the position, and the caller's own client places whatever its provider
calls a cache breakpoint there. The token-usage field names follow the
Anthropic API's spelling because it is the most explicit of the common shapes
(it distinguishes a cache *write* from a cache *read*);
:meth:`TokenUsage.from_mapping` accepts the common aliases from other shapes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import RetrievedChunk
from .errors import ConnectorContractError

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

#: The RAG context goes here. Exactly one occurrence, so the block's position
#: (and therefore a breakpoint offset either side of it) is unambiguous.
RAG_BLOCK_PLACEHOLDER = "{rag_block}"

#: The question goes here. One or more occurrences: repeating the query after a
#: long block is real practice, not a mistake, so it is allowed. Offsets are
#: measured against the FIRST occurrence, which is the one that bounds any
#: query-independent prefix.
QUERY_PLACEHOLDER = "{query}"

PROMPT_TEMPLATE_PLACEHOLDERS = (RAG_BLOCK_PLACEHOLDER, QUERY_PLACEHOLDER)

_PLACEHOLDER_NAMES = frozenset({"rag_block", "query"})

#: ``{like_this}`` -- what a reader (and ``str.format``) would take for a
#: substitution slot. JSON braces in prompt prose (``{"answer": ...}``) do not
#: match, so an instruction block showing a JSON shape passes untouched.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: A stable id: it is stored in a host's Test Set, copied into a Test Run, and
#: compared across runs, so it has to survive a round trip through a filename,
#: a JSON key and a URL query string without being re-spelled.
_TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ---------------------------------------------------------------------------
# Cache-breakpoint positions
# ---------------------------------------------------------------------------

#: No breakpoint is declared. Always legal, and the honest answer for a prompt
#: whose realistic ordering is not cacheable -- which is common, because
#: putting the question first is a perfectly ordinary thing for a production
#: prompt to do.
CACHE_BREAKPOINT_NONE = "none"

#: End of the system message. Requires a non-empty ``system``.
CACHE_BREAKPOINT_SYSTEM_END = "system_end"

#: Immediately before the rendered RAG block, i.e. after the static instruction
#: prefix. Caches the instructions across every query AND every block.
CACHE_BREAKPOINT_BEFORE_RAG_BLOCK = "before_rag_block"

#: Immediately after the rendered RAG block. Caches instructions + block, so it
#: pays off when many queries run against ONE frozen block -- which is exactly
#: the fixed-context shape a groundedness fixture has.
CACHE_BREAKPOINT_AFTER_RAG_BLOCK = "after_rag_block"

CACHE_BREAKPOINT_POSITIONS = (
    CACHE_BREAKPOINT_NONE,
    CACHE_BREAKPOINT_SYSTEM_END,
    CACHE_BREAKPOINT_BEFORE_RAG_BLOCK,
    CACHE_BREAKPOINT_AFTER_RAG_BLOCK,
)


# ---------------------------------------------------------------------------
# History markers: what makes a template multi-turn in disguise
# ---------------------------------------------------------------------------
#
# The dataclass cannot hold a prior turn, so the only way history arrives is
# smuggled inside the template STRING -- either as a substitution slot a host
# is expected to fill, or as transcript scaffolding the model reads as a
# conversation. Both are rejected.
#
# Each entry is (compiled pattern, what it means), so the refusal names the
# thing it found instead of saying "invalid template".

#: Placeholder names that are requests for conversation history.
_HISTORY_PLACEHOLDER_NAMES = frozenset({
    "history", "chat_history", "conversation", "conversation_history",
    "messages", "message_history", "turns", "previous_turns", "prior_turns",
    "transcript", "dialogue", "dialog", "dialogue_history", "context_history",
    "previous_messages", "prior_messages", "past_messages", "memory",
})

_HISTORY_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A turn label for a speaker that is not the user: the template is a
    # transcript. ``Question:`` / ``User:`` alone is just a field label and is
    # deliberately NOT matched -- a one-shot prompt may well label its query.
    (re.compile(r"^[ \t>*#-]*(assistant|ai|bot|chatbot|agent|model)\s*:",
                re.IGNORECASE | re.MULTILINE),
     "an assistant/AI turn label, which only exists in a transcript"),
    (re.compile(r"</?(assistant|ai_message|assistant_message)\s*>",
                re.IGNORECASE),
     "an assistant turn tag"),
    (re.compile(r"<\|im_(start|end)\|>|\[/?INST\]|<\|(start|end)_header_id\|>|"
                r"<\|eot_id\|>"),
     "chat-template turn delimiters"),
    # Prose that tells the model prior turns exist.
    (re.compile(r"\b(chat|conversation|message|dialogue|dialog)\s+history\b",
                re.IGNORECASE),
     "a reference to conversation history"),
    (re.compile(r"\bconversation\s+so\s+far\b", re.IGNORECASE),
     "a reference to the conversation so far"),
    (re.compile(r"\b(previous|prior|earlier|preceding|past)\s+"
                r"(turn|turns|message|messages|exchange|exchanges|"
                r"conversation|reply|replies|response|responses)\b",
                re.IGNORECASE),
     "a reference to earlier turns"),
    (re.compile(r"\bearlier\s+in\s+(this|the)\s+conversation\b", re.IGNORECASE),
     "a reference to earlier in the conversation"),
    (re.compile(r"\b(the\s+)?conversation\s+above\b", re.IGNORECASE),
     "a reference to a conversation above the prompt"),
    (re.compile(r"\bfollow[\s-]?up\s+question\b", re.IGNORECASE),
     "a follow-up question, which presupposes a preceding turn"),
)


class PromptTemplateError(ConnectorContractError):
    """A declared prompt template violates the contract.

    A subclass of :class:`~rag_connector.errors.ConnectorContractError` because
    that is what it is: the connector declared something the contract does not
    permit. It is raised, not warned, so a history-implying template cannot
    reach a run and be reported as a single-turn measurement.
    """


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def block_sha256(text: str) -> str:
    """Identity of a rendered RAG block: plain SHA-256 of its UTF-8 bytes.

    Deliberately plain, with no envelope, so anyone holding the block text can
    reproduce the value with ``sha256sum`` and check a stored run against it.
    A block is one string; there is no structure to canonicalize, and wrapping
    it would buy nothing but a value only this library can compute.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def template_sha256(*, system: str | None, template: str) -> str:
    """Identity of the prompt TEXT: SHA-256 over a versioned canonical payload.

    Two fields, so unlike a block this needs a canonical form -- otherwise
    ``system=None, template="ab"`` and ``system="a", template="b"`` would
    collide.

    Covers exactly the text that will be sent. Not the ``id``: two templates
    with the same text ARE the same prompt path, and a rename must not read
    downstream as a prompt change (id and hash travel together, so the rename
    is still visible). Not the declared cache breakpoint either: a breakpoint
    changes how a request is transported, not one byte of what the model reads,
    and folding it in would make a purely operational change look like a
    different prompt and break comparability for no gain. ``version`` is here
    so a future field can be added without silently re-hashing every stored
    template.
    """
    payload = {"version": 1, "system": system, "template": template}
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# The declared template
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One prompt path a connector declares it can be measured on.

    Frozen: a template is evidence. A host hashes it, stores the hash beside a
    run, and later compares runs on it -- a template that could be mutated
    after rendering would invalidate that hash without changing it.

    ``template`` must contain ``{rag_block}`` exactly once and ``{query}`` at
    least once, and no other ``{placeholder}``. Substitution is literal
    replacement, NOT ``str.format``, so a template may show a JSON shape in its
    instructions without escaping every brace -- which is worth having, because
    a prompt that asks for JSON back is the common case and ``str.format``
    would explode on it.

    There is no field for prior turns, and that absence is the enforcement: see
    the module docstring.

    ``decoding`` is the connector's *declared* default decoding params
    (temperature, max tokens, whatever its generation service takes). It is
    advisory -- a caller may override it -- but it is recorded in the trace, so
    a run always says which values were actually used rather than leaving them
    to be inferred from a connector's source.
    """

    id: str
    description: str
    template: str
    system: str | None = None
    cache_breakpoint: str = CACHE_BREAKPOINT_NONE
    decoding: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def template_sha256(self) -> str:
        """Stable identity of this template's text (see :func:`template_sha256`)."""
        return template_sha256(system=self.system, template=self.template)

    def validate(self) -> None:
        """Raise :class:`PromptTemplateError` if this template is not legal."""
        problems = prompt_template_problems(self)
        if problems:
            raise PromptTemplateError(
                f"Prompt template {self.id!r} is not a legal one-shot RAG "
                "template: " + "; ".join(problems)
            )

    def to_dict(self) -> dict:
        """JSON-serializable form, including the derived hash.

        The hash travels with the text so a host that persists this dict can
        detect an edited template without re-deriving the hash itself (and
        without needing this library to read its own stored artifact).
        """
        return {
            "id": self.id,
            "description": self.description,
            "template": self.template,
            "system": self.system,
            "cache_breakpoint": self.cache_breakpoint,
            "decoding": dict(self.decoding),
            "metadata": dict(self.metadata),
            "template_sha256": self.template_sha256,
        }


def prompt_template_problems(template: PromptTemplate) -> list[str]:
    """Every way ``template`` violates the contract, as fixable sentences.

    Returns a list rather than raising, because the conformance validator wants
    to report ALL the problems in one pass instead of making an author fix them
    one round trip at a time. :meth:`PromptTemplate.validate` is the raising
    form, and :func:`render_prompt` calls it -- so a template that was never
    put through the validator still cannot be rendered into a prompt.
    """
    problems: list[str] = []

    if not isinstance(template, PromptTemplate):
        return [f"expected a PromptTemplate, got {type(template).__name__}"]

    tid = template.id or ""
    if not tid.strip():
        problems.append("'id' is empty; a prompt variant needs a stable id "
                        "that a host can store and address it by")
    elif not _TEMPLATE_ID_RE.match(tid):
        problems.append(
            f"'id' {tid!r} is not a stable id; use letters, digits, '.', '_' "
            "and '-' only (max 64 chars, starting with a letter or digit) so "
            "it survives a filename, a JSON key and a URL unchanged")

    if not (template.description or "").strip():
        problems.append("'description' is empty; an operator choosing between "
                        "prompt variants needs to know what this one does")

    body = template.template or ""
    if not body.strip():
        problems.append("'template' is empty")
        return problems

    block_count = body.count(RAG_BLOCK_PLACEHOLDER)
    if block_count != 1:
        problems.append(
            f"'template' contains {RAG_BLOCK_PLACEHOLDER} {block_count} "
            "time(s); it must appear exactly once, so the block's position -- "
            "and any cache breakpoint either side of it -- is unambiguous")
    if QUERY_PLACEHOLDER not in body:
        problems.append(
            f"'template' never uses {QUERY_PLACEHOLDER}; the question would "
            "not reach the model")

    # An unknown ``{placeholder}`` is an author expecting a substitution that
    # will never happen: the literal braces would be sent to the model.
    for name in sorted({m.group(1) for m in _PLACEHOLDER_RE.finditer(body)}):
        if name in _PLACEHOLDER_NAMES:
            continue
        if name in _HISTORY_PLACEHOLDER_NAMES:
            problems.append(
                f"'template' asks for {{{name}}}, which is conversation "
                "history. These prompts are one-shot by contract: exactly one "
                "user message and no prior assistant turns. Remove it -- there "
                "is no supported way to supply history here")
        else:
            problems.append(
                f"'template' contains an unknown placeholder {{{name}}}; only "
                f"{RAG_BLOCK_PLACEHOLDER} and {QUERY_PLACEHOLDER} are "
                "substituted, so this would be sent to the model verbatim")

    # History smuggled in as prose or transcript scaffolding.
    for where, text in (("template", body), ("system", template.system or "")):
        for pattern, meaning in _HISTORY_MARKERS:
            match = pattern.search(text)
            if match is None:
                continue
            problems.append(
                f"{where!r} contains {match.group(0).strip()!r} -- {meaning}. "
                "These prompts are one-shot by contract (exactly one user "
                "message, no prior assistant turns); a host reports them as "
                "single-turn, so a template implying history is refused "
                "rather than accepted with a warning")

    problems.extend(_cache_breakpoint_problems(template))
    return problems


def _cache_breakpoint_problems(template: PromptTemplate) -> list[str]:
    """Whether the declared breakpoint position is real and worth anything.

    Never proposes a different position, and never reorders anything. A
    breakpoint that cannot work is refused with the two honest ways out: change
    the template on purpose (a different declared variant), or declare
    ``"none"``.
    """
    position = template.cache_breakpoint
    if position not in CACHE_BREAKPOINT_POSITIONS:
        return [f"'cache_breakpoint' {position!r} is not one of "
                f"{CACHE_BREAKPOINT_POSITIONS}"]
    if position == CACHE_BREAKPOINT_NONE:
        return []

    if position == CACHE_BREAKPOINT_SYSTEM_END:
        if not (template.system or "").strip():
            return ["'cache_breakpoint' is 'system_end' but there is no "
                    "system message for it to sit at the end of"]
        return []

    body = template.template or ""
    block_at = body.find(RAG_BLOCK_PLACEHOLDER)
    if block_at < 0:
        return []          # already reported: the block placeholder is missing
    boundary = (block_at if position == CACHE_BREAKPOINT_BEFORE_RAG_BLOCK
                else block_at + len(RAG_BLOCK_PLACEHOLDER))

    # Caching is a PREFIX match: everything before the breakpoint must be
    # byte-identical from call to call, or the cache misses every time. A query
    # ahead of the breakpoint changes on every question, so the declaration
    # would promise a saving that can never occur.
    if QUERY_PLACEHOLDER in body[:boundary]:
        return [
            f"'cache_breakpoint' is {position!r}, but {QUERY_PLACEHOLDER} "
            "appears before that point. Caching matches on an exact prefix, so "
            "a prefix containing the question changes with every question and "
            "would never be reused. This is NOT reordered for you: moving the "
            "block above the query would change the prompt you actually ship, "
            "and a measurement of a prompt you do not run is worth less than no "
            "measurement. Either declare a SEPARATE variant whose template "
            "genuinely puts the block first, or set cache_breakpoint='none' -- "
            "'none' is the honest answer for a prompt that is not cacheable in "
            "the order it is written."
        ]
    return []


def validate_prompt_template(template: PromptTemplate) -> None:
    """Raise :class:`PromptTemplateError` unless ``template`` is legal."""
    template.validate()


def validate_prompt_templates(
    templates: Sequence[PromptTemplate],
) -> tuple[PromptTemplate, ...]:
    """Validate a whole declared set, including id uniqueness across it.

    Duplicate ids are checked here rather than on the template because they are
    a property of the SET: a host addresses a variant by id, so two variants
    sharing one id means a Test Set naming that id cannot say which prompt it
    meant, and whichever one enumeration happened to yield first would answer.
    """
    if not templates:
        raise PromptTemplateError(
            "prompt_templates() returned nothing. A connector that declares "
            "the prompt-template capability must declare at least one variant; "
            "a connector with no declared prompt simply leaves the capability "
            "unimplemented instead."
        )
    seen: dict[str, int] = {}
    for index, template in enumerate(templates):
        if not isinstance(template, PromptTemplate):
            raise PromptTemplateError(
                f"prompt_templates()[{index}] is a "
                f"{type(template).__name__}, not a PromptTemplate"
            )
        template.validate()
        if template.id in seen:
            raise PromptTemplateError(
                f"prompt_templates() declares id {template.id!r} twice "
                f"(entries {seen[template.id]} and {index}). Ids address a "
                "variant, so they must be unique."
            )
        seen[template.id] = index
    return tuple(templates)


# ---------------------------------------------------------------------------
# Rendering a RAG block
# ---------------------------------------------------------------------------

def render_numbered_rag_block(
    chunks: Iterable[RetrievedChunk],
    *,
    include_chunk_ids: bool = True,
) -> str:
    """A conventional numbered RAG block: ``[n] chunk_id=...`` then the text.

    A convenience, not the contract. ``render_rag_block`` on a connector is the
    contract, precisely so each connector can render the format it *actually*
    ships -- that realism is the whole justification for measuring a declared
    prompt instead of a synthetic one. This is here because the numbered form
    is common enough to be worth not rewriting, and because the Reference RAG
    uses it.
    """
    lines = []
    for index, chunk in enumerate(chunks):
        head = (f"[{index + 1}] chunk_id={chunk.chunk_id}"
                if include_chunk_ids else f"[{index + 1}]")
        lines.append(f"{head}\n{chunk.text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# The rendered prompt
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PromptMessage:
    """One message as sent: a role and its content.

    Roles are ``"system"`` and ``"user"`` only. ``"assistant"`` is absent from
    the legal set because a prior assistant turn is precisely the thing being
    ruled out -- see :func:`assert_one_shot`.
    """

    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PromptCacheBreakpoint:
    """Where a cache breakpoint MAY be placed in a rendered prompt.

    ``message_index`` indexes :attr:`RenderedPrompt.messages`; ``char_offset``
    is a character offset into that message's ``content``, so a caller can
    split the content there and mark the first part however its provider spells
    a cache breakpoint. Resolved from the template's declared position, never
    inferred from the text and never moved to a "better" spot.

    Applying it is the caller's choice, and it may not take effect: providers
    ignore a cached prefix below a model-dependent minimum, silently. Verify
    against ``TokenUsage.cache_read_input_tokens`` rather than assuming.
    """

    position: str
    message_index: int
    char_offset: int

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "message_index": self.message_index,
            "char_offset": self.char_offset,
        }


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A fully rendered, one-shot prompt plus the identity of its parts.

    The point of the type is that the template and the data are separable
    *after* rendering: ``template_sha256`` identifies the instructions,
    ``block_sha256`` identifies the context, and ``query`` is the question. A
    host can hold any one constant and vary another, and say afterwards which
    it did -- none of which is possible from a single hydrated string.
    """

    template_id: str
    template_sha256: str
    messages: tuple[PromptMessage, ...]
    query: str
    rag_block: str
    block_sha256: str
    chunk_ids: tuple[str, ...] = ()
    #: The decoding params the TEMPLATE declared, carried through so a caller
    #: can honor them without re-fetching the template, and so a caller that
    #: overrides them has something explicit to override. Declared, not
    #: applied: what was actually sent is recorded on the trace.
    decoding: Mapping[str, Any] = field(default_factory=dict)
    cache_breakpoint: PromptCacheBreakpoint | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def system_message(self) -> PromptMessage | None:
        for message in self.messages:
            if message.role == "system":
                return message
        return None

    @property
    def user_message(self) -> PromptMessage:
        for message in self.messages:
            if message.role == "user":
                return message
        raise PromptTemplateError("rendered prompt has no user message")

    @property
    def text(self) -> str:
        """All message content joined -- the legacy ``hydrated_prompt`` value.

        Kept so a connector implementing the new path can populate
        ``GeneratedAnswer.hydrated_prompt`` from one property rather than
        re-joining by hand, and so the two never disagree. Lossy on purpose
        (roles are dropped): read :attr:`messages` for anything structural.
        """
        return "\n\n".join(message.content for message in self.messages)

    def cacheable_prefix(self) -> str | None:
        """The content preceding the declared breakpoint, or ``None``.

        Content in render order, not wire bytes -- for inspection and for
        hashing the thing you expect to be reused. If this string changes
        between two calls that were supposed to share a cache, that is the bug.
        """
        breakpoint_ = self.cache_breakpoint
        if breakpoint_ is None:
            return None
        parts = [m.content for m in self.messages[:breakpoint_.message_index]]
        parts.append(
            self.messages[breakpoint_.message_index]
            .content[:breakpoint_.char_offset]
        )
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "messages": [m.to_dict() for m in self.messages],
            "query": self.query,
            "rag_block": self.rag_block,
            "block_sha256": self.block_sha256,
            "chunk_ids": list(self.chunk_ids),
            "decoding": dict(self.decoding),
            "cache_breakpoint": (self.cache_breakpoint.to_dict()
                                 if self.cache_breakpoint else None),
            "metadata": dict(self.metadata),
        }


def assert_one_shot(messages: Sequence[PromptMessage]) -> None:
    """Raise unless ``messages`` is one optional system + exactly one user turn.

    The invariant a host's documentation depends on. It is checked on every
    render rather than trusted, because "no conversation history" is not a
    hopeful description of these prompts -- it is what makes a groundedness
    number mean anything. An answer grounded in an earlier turn nobody recorded
    is indistinguishable, after the fact, from an answer grounded in nothing.
    """
    roles = [m.role for m in messages]
    assistant = [i for i, role in enumerate(roles) if role == "assistant"]
    if assistant:
        raise PromptTemplateError(
            f"prompt carries assistant turn(s) at {assistant}; these prompts "
            "are one-shot -- no prior assistant turns."
        )
    users = [i for i, role in enumerate(roles) if role == "user"]
    if len(users) != 1:
        raise PromptTemplateError(
            f"prompt carries {len(users)} user message(s); exactly one is "
            f"required (roles seen: {roles})"
        )
    systems = [i for i, role in enumerate(roles) if role == "system"]
    if len(systems) > 1:
        raise PromptTemplateError(
            f"prompt carries {len(systems)} system messages; at most one is "
            "allowed."
        )
    if systems and systems[0] > users[0]:
        raise PromptTemplateError(
            "the system message follows the user message; a system preamble "
            "comes first."
        )
    unknown = sorted({r for r in roles} - {"system", "user"})
    if unknown:
        raise PromptTemplateError(
            f"prompt carries unsupported role(s) {unknown}; only 'system' and "
            "'user' are legal in a one-shot RAG prompt."
        )


def render_prompt(
    template: PromptTemplate,
    *,
    query: str,
    rag_block: str,
    chunk_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RenderedPrompt:
    """Render ``template`` with ``rag_block`` and ``query`` into one-shot messages.

    The single shared substitution: every connector renders through here, so a
    rendered prompt is reproducible from its three inputs by anyone, and a
    per-connector substitution bug cannot make one connector's "same template"
    mean something different.

    Substitution is literal replacement of the two placeholders, not
    ``str.format`` -- an instruction block containing JSON braces is left
    exactly as the author wrote it.

    Validates the template first (so an unvalidated template cannot become a
    prompt) and asserts the one-shot invariant on the way out.
    """
    template.validate()

    body = template.template
    block_at = body.find(RAG_BLOCK_PLACEHOLDER)
    before = body[:block_at].replace(QUERY_PLACEHOLDER, query)
    after = body[block_at + len(RAG_BLOCK_PLACEHOLDER):].replace(
        QUERY_PLACEHOLDER, query)
    user_content = f"{before}{rag_block}{after}"

    messages: list[PromptMessage] = []
    if (template.system or "").strip():
        messages.append(PromptMessage("system", template.system))
    user_index = len(messages)
    messages.append(PromptMessage("user", user_content))

    position = template.cache_breakpoint
    breakpoint_: PromptCacheBreakpoint | None = None
    if position == CACHE_BREAKPOINT_SYSTEM_END:
        breakpoint_ = PromptCacheBreakpoint(position, 0, len(messages[0].content))
    elif position == CACHE_BREAKPOINT_BEFORE_RAG_BLOCK:
        breakpoint_ = PromptCacheBreakpoint(position, user_index, len(before))
    elif position == CACHE_BREAKPOINT_AFTER_RAG_BLOCK:
        breakpoint_ = PromptCacheBreakpoint(
            position, user_index, len(before) + len(rag_block))

    rendered = RenderedPrompt(
        template_id=template.id,
        template_sha256=template.template_sha256,
        messages=tuple(messages),
        query=query,
        rag_block=rag_block,
        block_sha256=block_sha256(rag_block),
        chunk_ids=tuple(str(cid) for cid in chunk_ids),
        decoding=dict(template.decoding),
        cache_breakpoint=breakpoint_,
        metadata=dict(metadata or {}),
    )
    assert_one_shot(rendered.messages)
    return rendered


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

_USAGE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens", "promptTokens",
                     "inputTokens"),
    "output_tokens": ("output_tokens", "completion_tokens",
                      "completionTokens", "outputTokens"),
    "cache_creation_input_tokens": (
        "cache_creation_input_tokens", "cacheCreationInputTokens",
        "cache_creation", "cache_write_tokens", "cache_creation_tokens"),
    "cache_read_input_tokens": (
        "cache_read_input_tokens", "cacheReadInputTokens", "cache_read",
        "cached_tokens", "cache_read_tokens"),
}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens a generation call reported, or ``None`` where it reported none.

    Every field is legally ``None``, and ``None`` means "not reported", never
    zero -- the same rule the rest of this contract follows for unreported
    facts. A zero cache read is a real, actionable observation (the breakpoint
    did not take effect); an unknown cache read is not, and conflating them
    would turn "we cannot tell" into "caching is broken".

    The two cache fields are why this type exists at all. A declared cache
    breakpoint is a permission, not a promise -- prefixes below a
    model-dependent minimum are ignored silently -- so a run that wants to
    claim caching worked has to show ``cache_read_input_tokens > 0``.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> TokenUsage:
        """Best-effort read of a provider usage object or dict.

        Tolerant by design: this library is provider-neutral, connectors bring
        their own generation service, and a usage shape nobody anticipated must
        degrade to "not reported" rather than raise inside a generation call
        that otherwise succeeded. The canonical spellings are the Anthropic
        API's; common aliases from other shapes are accepted.
        """
        if raw is None:
            return cls()
        getter = (raw.get if isinstance(raw, Mapping)
                  else lambda key, default=None: getattr(raw, key, default))
        values: dict[str, int | None] = {}
        for canonical, aliases in _USAGE_ALIASES.items():
            values[canonical] = None
            for alias in aliases:
                candidate = getter(alias)
                if isinstance(candidate, bool) or not isinstance(
                        candidate, (int, float)):
                    continue
                values[canonical] = int(candidate)
                break
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    """What a generation call was given and what it returned, in full.

    Supplements :class:`~rag_connector.base.GeneratedAnswer` -- it does not
    replace it, and ``hydrated_prompt`` stays populated alongside. The answer
    text and citations remain on ``GeneratedAnswer``, so nothing is stored
    twice and existing consumers keep reading the field they always read; the
    trace adds what that dataclass never carried: the messages with their
    roles, the model, the decoding params, the raw response, token usage, and
    the template and block hashes (through :attr:`prompt`).

    ``error`` follows the generation convention in ``docs/contract.md``: a
    RUNTIME failure is captured, not raised, so a run records the failure
    instead of dying. A trace with an error is still evidence -- it says what
    was sent, which is what makes the failure diagnosable.
    """

    prompt: RenderedPrompt
    response: str = ""
    model: str | None = None
    decoding: Mapping[str, Any] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: Whether the caller ACTUALLY placed the declared breakpoint on the
    #: request. Distinct from ``prompt.cache_breakpoint``, which only says one
    #: was declared -- "declared but not applied" and "applied" are different
    #: runs, and a cache-read of zero means different things in each.
    cache_breakpoint_applied: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> tuple[PromptMessage, ...]:
        return self.prompt.messages

    @property
    def template_id(self) -> str:
        return self.prompt.template_id

    @property
    def template_sha256(self) -> str:
        return self.prompt.template_sha256

    @property
    def block_sha256(self) -> str:
        return self.prompt.block_sha256

    def to_dict(self) -> dict:
        """JSON-serializable trace, for a host to persist beside a run."""
        return {
            "schema": "rag-connector.generation-trace.v1",
            "prompt": self.prompt.to_dict(),
            "response": self.response,
            "model": self.model,
            "decoding": dict(self.decoding),
            "usage": self.usage.to_dict(),
            "cache_breakpoint_applied": self.cache_breakpoint_applied,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


def token_usage_from_response(response: Any) -> TokenUsage:
    """Pull token usage out of a generation response, best-effort.

    Looks in the places the common client libraries put it -- a
    ``usage_metadata`` mapping, a ``usage`` object, or a ``token_usage`` entry
    under ``response_metadata`` -- and returns an all-``None``
    :class:`TokenUsage` when none of them is there. Never raises: a connector
    calling this inside a successful generation must not have the call fail
    over telemetry it was only ever hoping for.
    """
    if response is None:
        return TokenUsage()
    for attribute in ("usage_metadata", "usage"):
        candidate = getattr(response, attribute, None)
        if candidate is None and isinstance(response, Mapping):
            candidate = response.get(attribute)
        if candidate is not None:
            usage = TokenUsage.from_mapping(candidate)
            if usage != TokenUsage():
                return usage
    meta = getattr(response, "response_metadata", None)
    if meta is None and isinstance(response, Mapping):
        meta = response.get("response_metadata")
    if isinstance(meta, Mapping):
        for key in ("token_usage", "usage"):
            if key in meta:
                return TokenUsage.from_mapping(meta[key])
    return TokenUsage()


__all__ = [
    "CACHE_BREAKPOINT_AFTER_RAG_BLOCK",
    "CACHE_BREAKPOINT_BEFORE_RAG_BLOCK",
    "CACHE_BREAKPOINT_NONE",
    "CACHE_BREAKPOINT_POSITIONS",
    "CACHE_BREAKPOINT_SYSTEM_END",
    "PROMPT_TEMPLATE_PLACEHOLDERS",
    "QUERY_PLACEHOLDER",
    "RAG_BLOCK_PLACEHOLDER",
    "GenerationTrace",
    "PromptCacheBreakpoint",
    "PromptMessage",
    "PromptTemplate",
    "PromptTemplateError",
    "RenderedPrompt",
    "TokenUsage",
    "assert_one_shot",
    "block_sha256",
    "prompt_template_problems",
    "render_numbered_rag_block",
    "render_prompt",
    "template_sha256",
    "token_usage_from_response",
    "validate_prompt_template",
    "validate_prompt_templates",
]
