"""Contract tests for declared prompt templates and generation traces.

The capability is additive and independently optional, so these tests carry two
burdens at once: prove the new surface behaves, and prove that a connector
which knows nothing about it is completely unaffected.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from rag_connector.base import GeneratedAnswer, RagPipeline, RetrievedChunk
from rag_connector.capabilities import (
    DECLARED_PROMPT_METHODS,
    declared_prompt_methods,
    supports,
    supports_declared_prompts,
)
from rag_connector.errors import UnsupportedCapability
from rag_connector.prompts import (
    CACHE_BREAKPOINT_AFTER_RAG_BLOCK,
    CACHE_BREAKPOINT_BEFORE_RAG_BLOCK,
    CACHE_BREAKPOINT_NONE,
    CACHE_BREAKPOINT_SYSTEM_END,
    GenerationTrace,
    PromptMessage,
    PromptTemplate,
    PromptTemplateError,
    TokenUsage,
    assert_one_shot,
    block_sha256,
    prompt_template_problems,
    render_numbered_rag_block,
    render_prompt,
    template_sha256,
    token_usage_from_response,
    validate_prompt_templates,
)
from rag_connector.reference import (
    DEFAULT_REFERENCE_PROMPT_ID,
    REFERENCE_PROMPT_TEMPLATES,
    ReferenceRagConnector,
    reference_prompt_template,
)
from rag_connector.registry import ConnectorSpec, get_connector
from rag_connector.validate import FAIL, PASS, SKIP, WARN, validate_pipeline

GOOD = PromptTemplate(
    id="good-v1",
    description="A legal one-shot RAG prompt.",
    template="Use only the context.\n\nContext:\n{rag_block}\n\nQ: {query}",
    cache_breakpoint=CACHE_BREAKPOINT_AFTER_RAG_BLOCK,
)


def _chunk(cid, text, rank=0):
    return RetrievedChunk(chunk_id=cid, text=text, score=None, rank=rank)


def _problem_text(template: PromptTemplate) -> str:
    return " | ".join(prompt_template_problems(template))


# ---------------------------------------------------------------------------
# Template shape
# ---------------------------------------------------------------------------

def test_a_legal_template_has_no_problems():
    assert prompt_template_problems(GOOD) == []
    GOOD.validate()


@pytest.mark.parametrize("template, expected", [
    ("Context:\n{rag_block}", "{query}"),
    ("Q: {query}", "{rag_block}"),
    ("{rag_block}{rag_block} {query}", "{rag_block}"),
])
def test_both_placeholders_are_required(template, expected):
    problems = _problem_text(PromptTemplate("t", "d", template))
    assert expected in problems


def test_the_query_may_legally_appear_twice():
    # Repeating the question after a long block is real practice, not a
    # mistake: it is a standard mitigation for a model losing the question in
    # the middle of a large context.
    template = PromptTemplate(
        "repeat-v1", "query top and bottom",
        "Q: {query}\n\n{rag_block}\n\nAgain, the question: {query}")
    assert prompt_template_problems(template) == []
    rendered = render_prompt(template, query="who?", rag_block="B")
    assert rendered.user_message.content.count("who?") == 2


def test_an_unknown_placeholder_is_refused():
    problems = _problem_text(
        PromptTemplate("t", "d", "{tone} {rag_block} {query}"))
    assert "unknown placeholder {tone}" in problems


def test_json_braces_in_instructions_are_not_mistaken_for_placeholders():
    # Asking for JSON back is the common case, and this is exactly why
    # substitution is literal replacement rather than str.format -- which would
    # not merely mis-scan this template but raise on it.
    template = PromptTemplate(
        "json-v1", "asks for JSON",
        'Return {"answer": string, "citations": []}.\n'
        "{rag_block}\n{query}")
    assert prompt_template_problems(template) == []
    rendered = render_prompt(template, query="q", rag_block="b")
    assert '{"answer": string, "citations": []}' in rendered.user_message.content


def test_a_stable_id_is_required():
    assert "'id' is empty" in _problem_text(PromptTemplate("", "d", "{rag_block}{query}"))
    assert "not a stable id" in _problem_text(
        PromptTemplate("has spaces", "d", "{rag_block}{query}"))


def test_a_description_is_required():
    assert "'description' is empty" in _problem_text(
        PromptTemplate("t", "  ", "{rag_block}{query}"))


# ---------------------------------------------------------------------------
# One-shot enforcement -- rejected, never merely warned about
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    "{rag_block}\nAssistant: sure\nUser: {query}",
    "{rag_block}\nAI: earlier answer\n{query}",
    "<|im_start|>assistant\n{rag_block}{query}",
    "[INST] {rag_block} {query} [/INST]",
    "Given the chat history below.\n{rag_block}\n{query}",
    "Use the conversation so far.\n{rag_block}\n{query}",
    "Refer to previous turns.\n{rag_block}\n{query}",
    "Answer this follow-up question.\n{rag_block}\n{query}",
    "Consider the conversation above.\n{rag_block}\n{query}",
])
def test_a_template_implying_history_is_rejected(body):
    template = PromptTemplate("hist-v1", "multi-turn in disguise", body)
    problems = _problem_text(template)
    assert "one-shot" in problems
    with pytest.raises(PromptTemplateError):
        template.validate()


def test_history_in_the_system_message_is_rejected_too():
    template = PromptTemplate(
        "sys-hist-v1", "history hidden in the preamble",
        "{rag_block}\n{query}",
        system="You continue a conversation. Use the chat history.")
    assert "'system' contains" in _problem_text(template)


def test_a_history_placeholder_is_named_as_history_not_as_a_typo():
    # The diagnosis matters: an author told "unknown placeholder" looks for a
    # spelling mistake; an author told "these prompts are one-shot" learns the
    # rule.
    problems = _problem_text(
        PromptTemplate("t", "d", "{chat_history}\n{rag_block}\n{query}"))
    assert "conversation history" in problems
    assert "one-shot" in problems


def test_a_history_template_cannot_even_be_rendered():
    # Validation runs inside render_prompt, so a template that never went
    # through the conformance validator still cannot become a prompt.
    template = PromptTemplate("t", "d", "Assistant: hi\n{rag_block}{query}")
    with pytest.raises(PromptTemplateError):
        render_prompt(template, query="q", rag_block="b")


def test_a_single_user_facing_label_is_not_history():
    # "Question:" and "User:" label a field. Flagging them would refuse
    # perfectly ordinary one-shot prompts.
    for label in ("Question", "User", "Human"):
        template = PromptTemplate(
            "t", "d", f"{{rag_block}}\n\n{label}: {{query}}")
        assert prompt_template_problems(template) == [], label


def test_assert_one_shot_rejects_prior_assistant_turns():
    with pytest.raises(PromptTemplateError, match="assistant turn"):
        assert_one_shot([PromptMessage("assistant", "earlier"),
                         PromptMessage("user", "now")])


def test_assert_one_shot_rejects_two_user_messages():
    with pytest.raises(PromptTemplateError, match="2 user message"):
        assert_one_shot([PromptMessage("user", "a"), PromptMessage("user", "b")])


def test_assert_one_shot_rejects_a_system_message_after_the_user_turn():
    with pytest.raises(PromptTemplateError, match="system message follows"):
        assert_one_shot([PromptMessage("user", "a"),
                         PromptMessage("system", "b")])


def test_rendering_produces_exactly_one_optional_system_and_one_user_turn():
    template = PromptTemplate("t", "d", "{rag_block}\n{query}",
                              system="You cite your sources.")
    rendered = render_prompt(template, query="q", rag_block="b")
    assert [m.role for m in rendered.messages] == ["system", "user"]
    assert rendered.system_message.content == "You cite your sources."
    assert_one_shot(rendered.messages)


def test_a_blank_system_string_produces_no_system_message():
    rendered = render_prompt(
        PromptTemplate("t", "d", "{rag_block}{query}", system="   "),
        query="q", rag_block="b")
    assert [m.role for m in rendered.messages] == ["user"]


# ---------------------------------------------------------------------------
# Cache breakpoints -- declared, checked, never silently repaired
# ---------------------------------------------------------------------------

def test_an_unknown_breakpoint_position_is_refused():
    assert "is not one of" in _problem_text(
        PromptTemplate("t", "d", "{rag_block}{query}",
                       cache_breakpoint="somewhere_nice"))


def test_system_end_requires_a_system_message():
    assert "no system message" in _problem_text(
        PromptTemplate("t", "d", "{rag_block}{query}",
                       cache_breakpoint=CACHE_BREAKPOINT_SYSTEM_END))


def test_a_breakpoint_after_the_query_is_refused_and_nothing_is_reordered():
    # THE rule: fidelity beats cache savings. The refusal names the two honest
    # ways out and does not take either on the author's behalf.
    template = PromptTemplate(
        "q-first", "question before context",
        "Q: {query}\n\nContext:\n{rag_block}",
        cache_breakpoint=CACHE_BREAKPOINT_BEFORE_RAG_BLOCK)
    problems = _problem_text(template)
    assert "exact prefix" in problems
    assert "NOT reordered for you" in problems
    assert "cache_breakpoint='none'" in problems
    # The template itself is untouched: no repair, no reordering.
    assert template.template.startswith("Q: {query}")


def test_the_same_ordering_is_legal_when_no_breakpoint_is_claimed():
    # A question-first prompt is a perfectly good prompt. Only the cache CLAIM
    # was wrong, so dropping the claim -- not rewriting the prompt -- fixes it.
    template = PromptTemplate(
        "q-first", "question before context",
        "Q: {query}\n\nContext:\n{rag_block}",
        cache_breakpoint=CACHE_BREAKPOINT_NONE)
    assert prompt_template_problems(template) == []
    assert render_prompt(template, query="q", rag_block="b").cache_breakpoint is None


def test_a_resolved_breakpoint_points_at_the_declared_boundary():
    rendered = render_prompt(GOOD, query="who?", rag_block="BLOCKTEXT")
    breakpoint_ = rendered.cache_breakpoint
    assert breakpoint_.position == CACHE_BREAKPOINT_AFTER_RAG_BLOCK
    content = rendered.messages[breakpoint_.message_index].content
    assert content[:breakpoint_.char_offset].endswith("BLOCKTEXT")
    assert rendered.cacheable_prefix() == content[:breakpoint_.char_offset]


def test_before_and_after_bracket_the_block_exactly():
    before = render_prompt(
        PromptTemplate("b", "d", "I.\n{rag_block}\nQ: {query}",
                       cache_breakpoint=CACHE_BREAKPOINT_BEFORE_RAG_BLOCK),
        query="q", rag_block="BLOCK")
    after = render_prompt(
        PromptTemplate("a", "d", "I.\n{rag_block}\nQ: {query}",
                       cache_breakpoint=CACHE_BREAKPOINT_AFTER_RAG_BLOCK),
        query="q", rag_block="BLOCK")
    assert after.cache_breakpoint.char_offset - before.cache_breakpoint.char_offset \
        == len("BLOCK")
    assert before.cacheable_prefix() == "I.\n"


def test_a_system_end_breakpoint_lands_at_the_end_of_the_system_message():
    rendered = render_prompt(
        PromptTemplate("s", "d", "{rag_block}\n{query}", system="PREAMBLE",
                       cache_breakpoint=CACHE_BREAKPOINT_SYSTEM_END),
        query="q", rag_block="b")
    assert rendered.cache_breakpoint.message_index == 0
    assert rendered.cacheable_prefix() == "PREAMBLE"


def test_a_declared_prefix_is_byte_stable_across_different_questions():
    # The property a breakpoint actually depends on. Substring-checking the
    # query against the prefix is the wrong test -- a retrieved block routinely
    # contains the words of the question that retrieved it.
    block = "The apple orchard was planted in 1998."
    first = render_prompt(GOOD, query="apple orchard", rag_block=block)
    second = render_prompt(GOOD, query="when planted?", rag_block=block)
    assert first.cacheable_prefix() == second.cacheable_prefix()
    assert "apple orchard" in first.cacheable_prefix()   # content, not dependency


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def test_a_block_hash_is_a_plain_sha256_anyone_can_reproduce():
    assert block_sha256("hello") == hashlib.sha256(b"hello").hexdigest()


def test_a_template_hash_covers_the_text_and_not_the_id():
    renamed = PromptTemplate("other-id", "different words entirely", GOOD.template,
                             cache_breakpoint=GOOD.cache_breakpoint)
    assert renamed.template_sha256 == GOOD.template_sha256


def test_a_template_hash_separates_the_system_from_the_body():
    assert template_sha256(system=None, template="ab") != \
        template_sha256(system="a", template="b")


def test_editing_the_template_moves_its_hash():
    edited = PromptTemplate(GOOD.id, GOOD.description, GOOD.template + " ")
    assert edited.template_sha256 != GOOD.template_sha256


def test_a_cache_breakpoint_change_does_not_move_the_template_hash():
    # It changes how a request is transported, not one byte of what the model
    # reads; folding it in would break comparability over an operational tweak.
    recast = PromptTemplate(GOOD.id, GOOD.description, GOOD.template,
                            cache_breakpoint=CACHE_BREAKPOINT_NONE)
    assert recast.template_sha256 == GOOD.template_sha256


def test_a_rendered_prompt_carries_both_hashes_separately():
    rendered = render_prompt(GOOD, query="q", rag_block="BLOCK")
    assert rendered.template_sha256 == GOOD.template_sha256
    assert rendered.block_sha256 == block_sha256("BLOCK")


# ---------------------------------------------------------------------------
# Declared sets
# ---------------------------------------------------------------------------

def test_duplicate_ids_across_a_declared_set_are_refused():
    with pytest.raises(PromptTemplateError, match="twice"):
        validate_prompt_templates((GOOD, PromptTemplate(
            GOOD.id, "a different prompt", "{rag_block} {query}")))


def test_an_empty_declared_set_is_refused():
    with pytest.raises(PromptTemplateError, match="at least one"):
        validate_prompt_templates(())


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

def test_unreported_usage_is_none_and_never_zero():
    # Zero cache reads is an actionable observation (the breakpoint did not
    # take); unknown cache reads is not. Conflating them would report "caching
    # is broken" whenever a provider simply said nothing.
    usage = TokenUsage.from_mapping(None)
    assert usage.input_tokens is None
    assert usage.cache_read_input_tokens is None


def test_usage_reads_the_canonical_cache_fields():
    usage = TokenUsage.from_mapping({
        "input_tokens": 10, "output_tokens": 3,
        "cache_creation_input_tokens": 900, "cache_read_input_tokens": 800})
    assert usage.cache_creation_input_tokens == 900
    assert usage.cache_read_input_tokens == 800
    assert usage.to_dict()["input_tokens"] == 10


def test_usage_accepts_common_aliases_from_other_shapes():
    usage = TokenUsage.from_mapping(
        {"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 5})
    assert (usage.input_tokens, usage.output_tokens) == (7, 2)
    assert usage.cache_read_input_tokens == 5


def test_usage_extraction_never_raises_on_an_unfamiliar_response():
    assert token_usage_from_response(SimpleNamespace(content="hi")) == TokenUsage()
    assert token_usage_from_response(None) == TokenUsage()


def test_usage_is_read_from_response_metadata_when_that_is_where_it_lives():
    response = SimpleNamespace(
        content="hi", response_metadata={"token_usage": {"prompt_tokens": 4}})
    assert token_usage_from_response(response).input_tokens == 4


# ---------------------------------------------------------------------------
# Capability detection and defaults
# ---------------------------------------------------------------------------

class PlainConnector(RagPipeline):
    """An existing connector that knows nothing about prompt templates."""

    name = "plain"

    def query(self, text, top_k=5):
        return [_chunk("a", "alpha beta gamma")]

    def pull_all_chunks(self):
        return []


class PartialConnector(PlainConnector):
    """Enumerates variants with nothing able to run them."""

    def prompt_templates(self):
        return [GOOD]


def test_the_defaults_leave_an_existing_connector_untouched():
    pipeline = PlainConnector()
    assert supports_declared_prompts(pipeline) is False
    assert declared_prompt_methods(pipeline) == ()
    for method in DECLARED_PROMPT_METHODS:
        assert supports(pipeline, method) is False


@pytest.mark.parametrize("call", [
    lambda p: p.prompt_templates(),
    lambda p: p.render_rag_block([]),
    lambda p: p.generate_from_prompt(render_prompt(GOOD, query="q", rag_block="b")),
])
def test_the_defaults_refuse_loudly_rather_than_answering_emptily(call):
    # An empty template list would read as "declared zero prompts", which is a
    # different fact from "has no declared prompt" -- the same reason
    # get_chunk_vectors raises instead of returning {}.
    with pytest.raises(UnsupportedCapability):
        call(PlainConnector())


def test_a_partial_declaration_is_not_the_capability():
    pipeline = PartialConnector()
    assert declared_prompt_methods(pipeline) == ("prompt_templates",)
    assert supports_declared_prompts(pipeline) is False


def test_the_reference_rag_declares_the_whole_capability():
    pipeline = object.__new__(ReferenceRagConnector)
    assert supports_declared_prompts(pipeline) is True


# ---------------------------------------------------------------------------
# GeneratedAnswer back-compatibility
# ---------------------------------------------------------------------------

def test_generated_answer_still_constructs_positionally_without_a_trace():
    # The field was appended LAST precisely so this keeps working: existing
    # code builds this dataclass positionally, and a field inserted anywhere
    # else would misassign arguments silently.
    answer = GeneratedAnswer("q", "a", [], ["c"], "PROMPT", None)
    assert answer.hydrated_prompt == "PROMPT"
    assert answer.trace is None


def test_an_absent_trace_is_none_not_an_empty_trace():
    assert GeneratedAnswer("q", "a", []).trace is None


# ---------------------------------------------------------------------------
# The Reference RAG implementation
# ---------------------------------------------------------------------------

def test_the_reference_declares_two_valid_variants():
    ids = [t.id for t in REFERENCE_PROMPT_TEMPLATES]
    assert ids == ["question-first-json-v1", "context-first-json-v1"]
    for template in REFERENCE_PROMPT_TEMPLATES:
        assert prompt_template_problems(template) == []


def test_the_shipped_reference_prompt_declares_no_breakpoint():
    # Its question precedes its context, so no breakpoint could ever be reused
    # -- and the contract does not reorder the prompt to manufacture one.
    shipped = reference_prompt_template(DEFAULT_REFERENCE_PROMPT_ID)
    assert shipped.cache_breakpoint == CACHE_BREAKPOINT_NONE
    assert shipped.template.index("{query}") < shipped.template.index("{rag_block}")


def test_the_context_first_variant_declares_a_usable_breakpoint():
    variant = reference_prompt_template("context-first-json-v1")
    assert variant.cache_breakpoint == CACHE_BREAKPOINT_AFTER_RAG_BLOCK
    block = "[1] chunk_id=a\nalpha"
    first = render_prompt(variant, query="one", rag_block=block)
    second = render_prompt(variant, query="two", rag_block=block)
    assert first.cacheable_prefix() == second.cacheable_prefix()


def test_an_unknown_reference_template_id_raises_instead_of_defaulting():
    # Silently serving a different prompt would label a result with the id that
    # was asked for and measure it on the prompt that was not.
    with pytest.raises(KeyError, match="declared ids are"):
        reference_prompt_template("no-such-prompt")


def test_the_registry_serves_the_same_declaration_before_binding():
    spec = get_connector("reference")
    assert spec.supports_prompt_templates is True
    assert [t.id for t in spec.prompt_templates()] == \
        [t.id for t in REFERENCE_PROMPT_TEMPLATES]


def test_prompt_templates_stays_out_of_the_public_dict():
    # Same reasoning as connector_info: a callable cannot be JSON, and the
    # consumers of that dict do not want the template text.
    spec = get_connector("reference")
    assert "prompt_templates" not in spec.to_public_dict()
    assert "{rag_block}" not in json.dumps(spec.to_public_dict())


def test_a_spec_declares_no_prompt_templates_by_default():
    spec = ConnectorSpec(connector_type="x", label="X", description="d",
                         build=lambda c: None)
    assert spec.prompt_templates is None
    assert spec.supports_prompt_templates is False


def test_the_reference_renders_its_own_block_format_from_supplied_chunks():
    pipeline = object.__new__(ReferenceRagConnector)
    block = pipeline.render_rag_block([_chunk("a", "alpha", 0),
                                       _chunk("b", "beta", 1)])
    assert block == "[1] chunk_id=a\nalpha\n\n[2] chunk_id=b\nbeta"
    assert block == render_numbered_rag_block([_chunk("a", "alpha", 0),
                                               _chunk("b", "beta", 1)])


#: The prompt the Reference RAG hardcoded before the template refactor,
#: reproduced here as a literal. ``generate`` must keep sending exactly this,
#: byte for byte -- the refactor moved where the prompt is DECLARED, not what
#: gets sent.
_LEGACY_REFERENCE_PROMPT = (
    "Answer the question using only the supplied context. If the context "
    "is insufficient, say so. Cite supporting chunks inline as [1], [2], "
    "etc. Return ONLY JSON with keys answer (string) and citations (a list "
    "of the cited chunk_id values).\n\n"
    "Question: q\n\nContext:\n[1] chunk_id=chunk-a\nsource a"
)


def _stub_reference(content='{"answer":"A [1]","citations":["1"]}'):
    pipeline = object.__new__(ReferenceRagConnector)
    pipeline.query = lambda text, top_k: [_chunk("chunk-a", "source a")]
    llm = SimpleNamespace(
        invoke=lambda prompt: SimpleNamespace(content=content),
        model_name="stub-model")
    return pipeline, llm


def test_generate_still_sends_the_exact_prompt_it_always_sent():
    pipeline, llm = _stub_reference()
    result = pipeline.generate("q", top_k=1, llm=llm)
    assert result.hydrated_prompt == _LEGACY_REFERENCE_PROMPT


def test_generate_now_also_returns_a_trace_alongside_the_legacy_string():
    pipeline, llm = _stub_reference()
    result = pipeline.generate("q", top_k=1, llm=llm)
    assert result.trace is not None
    assert result.trace.template_id == DEFAULT_REFERENCE_PROMPT_ID
    assert result.trace.prompt.text == result.hydrated_prompt
    assert result.answer == "A [1]"
    assert result.citations == ["chunk-a"]


def test_generate_from_prompt_carries_the_whole_call_on_the_trace():
    pipeline, llm = _stub_reference()
    contexts = [_chunk("chunk-a", "source a")]
    prompt = render_prompt(GOOD, query="q",
                           rag_block=pipeline.render_rag_block(contexts),
                           chunk_ids=["chunk-a"])
    result = pipeline.generate_from_prompt(prompt, contexts=contexts, llm=llm,
                                           decoding={"temperature": 0.0})
    trace = result.trace
    assert [m.role for m in trace.messages] == ["user"]
    assert trace.model == "stub-model"
    assert trace.decoding == {"temperature": 0.0}
    assert trace.response == '{"answer":"A [1]","citations":["1"]}'
    assert trace.template_sha256 == GOOD.template_sha256
    assert trace.block_sha256 == block_sha256(prompt.rag_block)
    assert trace.usage == TokenUsage()
    assert trace.cache_breakpoint_applied is False
    assert result.hydrated_prompt == prompt.text
    assert result.contexts == contexts


def test_generate_from_prompt_captures_a_runtime_failure_with_its_prompt():
    pipeline = object.__new__(ReferenceRagConnector)
    boom = SimpleNamespace(invoke=lambda prompt: (_ for _ in ()).throw(
        RuntimeError("service down")))
    prompt = render_prompt(GOOD, query="q", rag_block="B")
    result = pipeline.generate_from_prompt(prompt, llm=boom)
    assert result.error == "service down"
    assert result.answer == ""
    # The prompt survives the failure, which is what makes it diagnosable.
    assert result.trace.prompt.template_sha256 == GOOD.template_sha256
    assert result.hydrated_prompt == prompt.text


def test_generate_from_prompt_captures_a_missing_llm_as_an_error():
    pipeline = object.__new__(ReferenceRagConnector)
    prompt = render_prompt(GOOD, query="q", rag_block="B")
    result = pipeline.generate_from_prompt(prompt)
    assert result.error == "Reference RAG generation requires an LLM"
    assert result.trace is not None


def test_a_trace_is_json_serializable_for_a_host_to_persist():
    pipeline, llm = _stub_reference()
    result = pipeline.generate("q", top_k=1, llm=llm)
    payload = json.dumps(result.trace.to_dict())
    restored = json.loads(payload)
    assert restored["schema"] == "rag-connector.generation-trace.v1"
    assert restored["prompt"]["template_id"] == DEFAULT_REFERENCE_PROMPT_ID
    assert restored["prompt"]["messages"][0]["role"] == "user"
    assert restored["usage"]["cache_read_input_tokens"] is None


def test_a_system_message_reaches_the_llm_as_roles_not_flattened_text():
    seen = {}
    pipeline = object.__new__(ReferenceRagConnector)
    llm = SimpleNamespace(invoke=lambda payload: (
        seen.update(payload=payload),
        SimpleNamespace(content='{"answer":"A","citations":[]}'))[1])
    prompt = render_prompt(
        PromptTemplate("s", "d", "{rag_block}\n{query}", system="PREAMBLE"),
        query="q", rag_block="B")
    pipeline.generate_from_prompt(prompt, llm=llm)
    assert seen["payload"] == [("system", "PREAMBLE"), ("human", "B\nq")]


def test_a_prompt_without_a_system_message_is_passed_as_the_bare_string():
    # Byte-identical to the call this connector has always made.
    seen = {}
    pipeline = object.__new__(ReferenceRagConnector)
    llm = SimpleNamespace(invoke=lambda payload: (
        seen.update(payload=payload),
        SimpleNamespace(content='{"answer":"A","citations":[]}'))[1])
    prompt = render_prompt(GOOD, query="q", rag_block="B")
    pipeline.generate_from_prompt(prompt, llm=llm)
    assert seen["payload"] == prompt.user_message.content


# ---------------------------------------------------------------------------
# The conformance validator
# ---------------------------------------------------------------------------

class ConformantConnector(PlainConnector):
    """A minimal but complete declared-prompt connector."""

    name = "conformant"

    def __init__(self, template=GOOD):
        self._template = template

    def pull_all_chunks(self):
        from rag_connector.base import ChunkRecord
        return [ChunkRecord(f"c{i}", "doc", "doc.md", i, i,
                            f"chunk text number {i} about orchards")
                for i in range(3)]

    def query(self, text, top_k=5):
        return [_chunk(c.chunk_id, c.text, i)
                for i, c in enumerate(self.pull_all_chunks()[:top_k])]

    def prompt_templates(self):
        return [self._template]

    def render_rag_block(self, chunks):
        return render_numbered_rag_block(chunks)

    def generate_from_prompt(self, prompt, *, contexts=(), llm=None,
                             decoding=None):
        return GeneratedAnswer(
            query=prompt.query, answer="an answer", contexts=list(contexts),
            citations=[], hydrated_prompt=prompt.text,
            trace=GenerationTrace(prompt=prompt, response="an answer",
                                  model="fake-model"))


def _check(report, name_fragment):
    return next(c for c in report.checks if name_fragment in c.name)


def _validate(pipeline, **kwargs):
    return validate_pipeline(pipeline, target="t", sample_query="orchards",
                             skip_stability=True, **kwargs)


def test_the_validator_skips_a_connector_that_declares_no_templates():
    # A retrieval-only connector reaching the same checks a declaring one does:
    # same corpus, same query path, and simply no declared prompt.
    class RetrievalOnly(ConformantConnector):
        prompt_templates = RagPipeline.prompt_templates
        render_rag_block = RagPipeline.render_rag_block
        generate_from_prompt = RagPipeline.generate_from_prompt

    check = _check(_validate(RetrievalOnly()), "Declared prompt templates")
    assert check.status == SKIP


def test_the_validator_passes_a_conformant_declared_prompt_connector():
    report = _validate(ConformantConnector())
    check = _check(report, "Declared prompt templates")
    assert check.status == PASS, check.details
    assert any("cache breakpoint" in d for d in check.details)


def test_the_validator_fails_a_half_declared_capability():
    class Half(ConformantConnector):
        render_rag_block = RagPipeline.render_rag_block

    check = _check(_validate(Half()), "Declared prompt templates")
    assert check.status == FAIL
    assert "render_rag_block" in check.details[0]


def test_the_validator_fails_a_template_that_implies_history():
    connector = ConformantConnector(PromptTemplate(
        "hist", "smuggles a transcript",
        "{rag_block}\nAssistant: earlier\nQ: {query}"))
    check = _check(_validate(connector), "Declared prompt templates")
    assert check.status == FAIL
    assert any("one-shot" in d for d in check.details)


def test_the_validator_fails_a_trace_that_misreports_its_template():
    class Liar(ConformantConnector):
        def generate_from_prompt(self, prompt, *, contexts=(), llm=None,
                                 decoding=None):
            other = render_prompt(
                PromptTemplate("other", "a different prompt entirely",
                               "{rag_block} :: {query}"),
                query=prompt.query, rag_block=prompt.rag_block)
            return GeneratedAnswer(prompt.query, "a", [], [], prompt.text,
                                   None, {}, GenerationTrace(prompt=other))

    check = _check(_validate(Liar()), "Declared prompt templates")
    assert check.status == FAIL
    assert any("different template hash" in d for d in check.details)


def test_the_validator_fails_a_connector_that_drops_the_legacy_prompt_string():
    class Dropper(ConformantConnector):
        def generate_from_prompt(self, prompt, *, contexts=(), llm=None,
                                 decoding=None):
            return GeneratedAnswer(prompt.query, "a", [], [], "", None, {},
                                   GenerationTrace(prompt=prompt))

    check = _check(_validate(Dropper()), "Declared prompt templates")
    assert check.status == FAIL
    assert any("hydrated_prompt is empty" in d for d in check.details)


def test_the_validator_fails_a_missing_trace():
    class NoTrace(ConformantConnector):
        def generate_from_prompt(self, prompt, *, contexts=(), llm=None,
                                 decoding=None):
            return GeneratedAnswer(prompt.query, "a", [], [], prompt.text)

    check = _check(_validate(NoTrace()), "Declared prompt templates")
    assert check.status == FAIL
    assert any("trace is None" in d for d in check.details)


def test_the_validator_warns_when_only_the_instance_declares_templates():
    spec = ConnectorSpec(connector_type="x", label="X", description="d",
                         build=lambda c: None)
    check = _check(_validate(ConformantConnector(), spec=spec),
                   "ConnectorSpec.prompt_templates")
    assert check.status == WARN


def test_the_validator_fails_a_registry_declaration_that_disagrees():
    spec = ConnectorSpec(
        connector_type="x", label="X", description="d", build=lambda c: None,
        prompt_templates=lambda: [PromptTemplate(
            "listed-but-not-served", "shown in the picker only",
            "{rag_block} {query}")])
    check = _check(_validate(ConformantConnector(), spec=spec),
                   "ConnectorSpec.prompt_templates")
    assert check.status == FAIL
    assert any("instance declares" in d for d in check.details)


def test_the_validator_passes_matching_registry_and_instance_declarations():
    spec = ConnectorSpec(
        connector_type="x", label="X", description="d", build=lambda c: None,
        prompt_templates=lambda: [GOOD])
    check = _check(_validate(ConformantConnector(), spec=spec),
                   "ConnectorSpec.prompt_templates")
    assert check.status == PASS
