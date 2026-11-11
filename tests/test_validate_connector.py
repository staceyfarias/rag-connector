
"""Contract tests for the standalone connector validator (src/rag/validate.py).

Each fake connector below is broken in exactly one way a real connector
developer gets wrong; the validator must catch each with a FAIL/WARN on the
right check and pass the compliant one with a READY verdict.
"""

from __future__ import annotations

import os
import sys

from rag_connector.base import ChunkRecord, GeneratedAnswer, RagPipeline, RetrievedChunk
from rag_connector.validate import FAIL, PASS, SKIP, WARN, main, validate_pipeline


def _chunk(cid, doc="doc-a", idx=0, gidx=0, text="alpha beta gamma delta"):
    return ChunkRecord(
        chunk_id=cid, doc_id=doc, source_file=f"{doc}.md",
        doc_chunk_index=idx, global_index=gidx, text=text,
        metadata={"source_fingerprint": "fp-1"},
    )


class GoodConnector(RagPipeline):
    """Fully compliant in-memory connector: canonical higher-is-better scores
    (converted from store distances, native value kept in raw_score)."""

    name = "good"

    def __init__(self):
        self._chunks = [
            _chunk("a:0", "doc-a", 0, 0, "alpha beta gamma delta"),
            _chunk("a:1", "doc-a", 1, 1, "epsilon zeta eta theta"),
            _chunk("b:0", "doc-b", 0, 2, "iota kappa lambda mu"),
        ]

    def query(self, text, top_k=5):
        return [RetrievedChunk(chunk_id=c.chunk_id, text=c.text,
                               score=round(1.0 - 0.1 * i, 4), rank=i,
                               doc_id=c.doc_id, raw_score=round(0.1 * i, 4))
                for i, c in enumerate(self._chunks[:top_k])]

    def pull_all_chunks(self):
        return list(self._chunks)


class MismatchedIdConnector(GoodConnector):
    """query() returns raw store ids -- THE classic connector bug."""

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        for h in hits:
            h.chunk_id = f"uuid-{h.chunk_id}"
        return hits


class UnitConnector(GoodConnector):
    """Serves one composed UNIT per covering item (a parent document, a
    summary node, a curated answer): a synthetic row id that is deliberately
    not a corpus id, with the corpus grounding declared as
    ``metadata["item_covered_ids"]`` — the ids an id-keyed harness credits it
    through. Includes a zero-coverage unit (an explicit empty declaration,
    e.g. a served negative outcome), which asks the corpus for nothing.
    """

    def query(self, text, top_k=5):
        return [
            RetrievedChunk(
                chunk_id="unit:parent-1", text="composed summary text",
                score=0.9, rank=0, doc_id=None,
                metadata={"item_index": 0, "item_kind": "group",
                          "item_covered_ids": ["a:0", "a:1"]}),
            RetrievedChunk(
                chunk_id="unit:empty-1", text="declared negative outcome",
                score=0.8, rank=1, doc_id=None,
                metadata={"item_index": 1, "item_kind": "group",
                          "item_covered_ids": []}),
            RetrievedChunk(
                chunk_id="b:0", text="iota kappa lambda mu", score=0.7,
                rank=2, doc_id="doc-b",
                metadata={"item_index": 2, "item_kind": "chunk",
                          "item_covered_ids": ["b:0"]}),
        ][:top_k]


class BadUnitConnector(UnitConnector):
    """A unit declaring coverage of an id the corpus does not hold — the
    classic bug reappearing one level up, and it must still be caught."""

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        hits[0].metadata["item_covered_ids"] = ["a:0", "ghost:9"]
        return hits


class DuplicateIdConnector(GoodConnector):
    def pull_all_chunks(self):
        chunks = super().pull_all_chunks()
        chunks[1].chunk_id = chunks[0].chunk_id
        return chunks


class GappyIndexConnector(GoodConnector):
    """doc_chunk_index skips 1 within doc-a (breaks neighbor windows)."""

    def pull_all_chunks(self):
        chunks = super().pull_all_chunks()
        chunks[1].doc_chunk_index = 5
        return chunks


class UnstableIdConnector(GoodConnector):
    """Each pull mints new ids -- the frozen answer key would rot instantly."""

    def __init__(self):
        super().__init__()
        self._pulls = 0

    def pull_all_chunks(self):
        self._pulls += 1
        return [_chunk(f"pull{self._pulls}:{i}", "doc-a", i, i)
                for i in range(3)]

    def query(self, text, top_k=5):   # ids match the *latest* pull
        return [RetrievedChunk(chunk_id=f"pull{self._pulls}:{i}", text="t",
                               score=0.1 * (i + 1), rank=i)
                for i in range(2)]


class UnconvertedDistanceConnector(GoodConnector):
    """Returns raw store distances (ascending down the ranking) -- the exact
    mistake the canonical convention exists to catch."""

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        for i, h in enumerate(hits):
            h.score = 0.1 * (i + 1)   # distance-shaped: lowest (best) first
            h.raw_score = None
        return hits

    def info(self):
        return {"name": self.name, "type": "x", "distance": "cosine"}


class PrioritizedOrderConnector(GoodConnector):
    """Business prioritization reorders results: scores honest but not
    monotonic (e.g. current-module chunks promoted over higher-scored ones)."""

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        for h, s in zip(hits, (0.9, 0.5, 0.7)):
            h.score = s
        return hits


class GeneratingConnector(GoodConnector):
    def generate(self, text, *, top_k=5, llm=None):
        contexts = self.query(text, top_k=top_k)
        return GeneratedAnswer(text, "an answer [1]", contexts,
                               [str(contexts[0].chunk_id)], "prompt")


class CompleteSetConnector(RagPipeline):
    """System-decided cardinality: neighbor expansion returns MORE than top_k,
    some chunks carry no query score (None), order is non-monotonic. All legal
    under complete_set."""

    name = "complete-set"
    retrieval_mode = "complete_set"

    def __init__(self):
        self._chunks = [_chunk(f"c:{i}", "doc-a", i, i, f"word{i} alpha beta")
                        for i in range(6)]

    def query(self, text, top_k=5):
        # Ignore top_k as a hard cap (breadth knob only): return top_k + 2, with
        # a couple of unscored neighbor chunks (score None) and non-monotonic order.
        n = min(top_k + 2, len(self._chunks))
        scored = [0.9, 0.4, None, 0.7, None]
        return [RetrievedChunk(chunk_id=c.chunk_id, text=c.text,
                               score=(scored[i] if i < len(scored) else None),
                               rank=i, doc_id=c.doc_id)
                for i, c in enumerate(self._chunks[:n])]

    def pull_all_chunks(self):
        return list(self._chunks)


class OrderedConnector(GoodConnector):
    """Rank-only API: scores present but declared meaningless (mode 'ordered')."""

    retrieval_mode = "ordered"

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        for h, s in zip(hits, (0.2, 0.9, 0.5)):  # non-monotonic, would else warn
            h.score = s
        return hits


class NoneScoreScoredConnector(GoodConnector):
    """Declares (defaults to) 'scored' but returns score=None hits -- a contract
    violation: the threshold filter and score display trust scored-mode values.
    The right move is declaring 'ordered'/'complete_set', not returning None."""

    def query(self, text, top_k=5):
        hits = super().query(text, top_k)
        for h in hits:
            h.score = None
            h.raw_score = None
        return hits


class UnknownModeConnector(GoodConnector):
    retrieval_mode = "teleport"


class SilentConnector(GoodConnector):
    """Retrieval finds nothing for the probe. Legal -- an empty result scores as
    a genuine zero -- but it means the score convention was never exercised."""

    def query(self, text, top_k=5):
        return []


class SnapshotConnector(GoodConnector):
    def run_snapshot(self):
        return {
            "schema": "test.run-snapshot.v1",
            "settings": {"strategy": "hybrid"},
            "observations": {"indexed_chunks": 3},
        }


class SecretSnapshotConnector(GoodConnector):
    def run_snapshot(self):
        return {
            "schema": "test.run-snapshot.v1",
            "settings": {"auth": {"api_token": "do-not-persist"}},
            "observations": {},
        }


class RequiresTopKConnector(GoodConnector):
    """``query`` demands a top_k: an evaluation host, which passes none
    (2026-10-08), cannot call it."""

    def query(self, text, top_k):
        return super().query(text, top_k)


def _statuses(report):
    return {c.name: c.status for c in report.checks}


def _status_of(report, fragment):
    matches = [c for c in report.checks if fragment in c.name]
    assert matches, f"no check matching {fragment!r} in {list(_statuses(report))}"
    return matches[0].status


def test_compliant_connector_is_ready():
    report = validate_pipeline(GoodConnector(), target="good", sample_query="alpha")
    assert report.ready, [f"{c.name}: {c.details}" for c in report.failures]
    assert _status_of(report, "classic bug") == PASS
    assert _status_of(report, "stability") == PASS
    # Retrieval-only: generation is SKIP (supported config), never a failure.
    assert _status_of(report, "generate()") == SKIP


def test_query_without_top_k_passes_for_a_defaulted_connector():
    report = validate_pipeline(GoodConnector(), target="good", sample_query="alpha")
    assert _status_of(report, "top_k unset") == PASS


def test_a_connector_that_requires_top_k_is_not_ready():
    report = validate_pipeline(RequiresTopKConnector(), target="needs-k",
                               sample_query="alpha")
    assert _status_of(report, "top_k unset") == FAIL
    assert not report.ready


def test_valid_run_snapshot_is_checked():
    report = validate_pipeline(SnapshotConnector(), target="snapshot",
                               sample_query="alpha")

    assert report.ready
    assert _status_of(report, "run_snapshot()") == PASS


def test_secret_in_nested_run_snapshot_is_a_blocking_failure():
    report = validate_pipeline(SecretSnapshotConnector(), target="snapshot",
                               sample_query="alpha")

    assert not report.ready
    assert _status_of(report, "run_snapshot()") == FAIL


def test_no_hits_reports_the_score_check_as_skipped_not_absent():
    """An omitted row reads as 'nothing to say about the scores'. What actually
    happened is that the check never ran, and the report has to say so."""
    report = validate_pipeline(SilentConnector(), target="silent",
                               sample_query="nothing in this corpus matches")

    # _status_of asserts the row exists at all -- that is half the point here.
    assert _status_of(report, "canonical") == SKIP


def test_id_format_mismatch_is_a_blocking_failure():
    report = validate_pipeline(MismatchedIdConnector(), target="x",
                               sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "classic bug") == FAIL


def test_a_unit_hit_is_validated_through_its_declared_covered_ids():
    """A synthetic unit id is identity, not provenance: the hit passes the
    classic-bug check through its declared covered ids — including the
    zero-coverage unit, whose explicit empty declaration asks for nothing."""
    report = validate_pipeline(UnitConnector(), target="unit",
                               sample_query="alpha")
    assert _status_of(report, "classic bug") == PASS


def test_a_unit_covering_an_unknown_id_still_fails_the_classic_bug_check():
    report = validate_pipeline(BadUnitConnector(), target="bad-unit",
                               sample_query="alpha")
    assert _status_of(report, "classic bug") == FAIL


def test_duplicate_ids_fail_the_contract_check():
    report = validate_pipeline(DuplicateIdConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "metadata contract") == FAIL


def test_gappy_doc_chunk_index_fails():
    report = validate_pipeline(GappyIndexConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "doc_chunk_index") == FAIL


def test_unstable_ids_fail_stability():
    report = validate_pipeline(UnstableIdConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "stability") == FAIL


def test_unconverted_distances_are_a_blocking_failure():
    # Ascending scores look like raw store distances: the canonical convention
    # (higher = better) makes this a FAIL with a pointer at the utilities.
    report = validate_pipeline(UnconvertedDistanceConnector(), target="x",
                               sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "canonical") == FAIL


def test_canonical_scores_pass_without_any_metric_declaration():
    # Direction is universal now -- no 'distance' declaration is required.
    report = validate_pipeline(GoodConnector(), target="x", sample_query="alpha")
    assert _status_of(report, "canonical") == PASS
    assert report.ready


def test_none_scores_under_scored_mode_are_a_blocking_failure():
    # Design section 3.9: scored + None score -> FAIL (not a crash). Optional[float]
    # made None representable, so the validator must catch it cleanly.
    report = validate_pipeline(NoneScoreScoredConnector(), target="x",
                               sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "canonical") == FAIL


def test_prioritized_nonmonotonic_scores_warn_not_fail():
    # Business prioritization (e.g. module priority) legally reorders results;
    # ranking is positional, so honest-but-non-monotonic scores are a WARN.
    report = validate_pipeline(PrioritizedOrderConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "canonical") == WARN
    assert report.ready


def test_generate_shape_is_validated_when_implemented():
    report = validate_pipeline(GeneratingConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "generate()") == PASS


def test_secret_keys_in_connection_fail():
    report = validate_pipeline(
        GoodConnector(), target="x", sample_query="alpha",
        connection={"endpoint": "https://x", "api_token": "hunter2"})
    assert _status_of(report, "secret-free") == FAIL


def test_nested_secret_keys_in_connection_fail_too():
    """A credential one level down is exactly as persisted and exactly as
    leaked as one at the root; top-level-only scanning would wave it through."""
    report = validate_pipeline(
        GoodConnector(), target="x", sample_query="alpha",
        connection={"endpoint": "https://x",
                    "auth": {"api_key": "hunter2"},
                    "headers": [{"x-secret-header": "hunter2"}]})
    check = next(c for c in report.checks if "secret-free" in c.name)
    assert check.status == FAIL
    assert "auth.api_key" in "".join(check.details)
    assert "headers[0].x-secret-header" in "".join(check.details)


def test_get_chunks_must_omit_unknown_ids_not_invent_them():
    class InventingConnector(GoodConnector):
        def get_chunks(self, chunk_ids):
            return {cid: _chunk(cid) for cid in chunk_ids}   # invents missing

    report = validate_pipeline(InventingConnector(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "get_chunks") == FAIL


def test_render_states_verdict_and_fix():
    report = validate_pipeline(MismatchedIdConnector(), target="x",
                               sample_query="alpha")
    text = report.render()
    assert "NOT READY" in text
    assert "FIX:" in text
    ready = validate_pipeline(GoodConnector(), target="good",
                              sample_query="alpha").render()
    assert "VERDICT: READY" in ready


# C2 mode-aware validation

def test_scored_is_the_default_mode_and_passes():
    report = validate_pipeline(GoodConnector(), target="good", sample_query="alpha")
    assert _status_of(report, "retrieval_mode is a recognized value") == PASS


def test_complete_set_allows_more_than_top_k_and_skips_direction():
    report = validate_pipeline(CompleteSetConnector(), target="cs",
                               sample_query="alpha", top_k=3)
    assert report.ready, [f"{c.name}: {c.details}" for c in report.failures]
    # more-than-top_k is legal, not a FAIL
    assert _status_of(report, "retrieves ranked chunks") == PASS
    # direction check skipped (scores are evidence-only)
    assert _status_of(report, "canonical") == SKIP
    # the two-k cardinality probe ran (informational)
    assert _status_of(report, "returned cardinality vs two top_k") == PASS


def test_ordered_mode_skips_direction_check():
    report = validate_pipeline(OrderedConnector(), target="ord",
                               sample_query="alpha")
    assert report.ready
    assert _status_of(report, "canonical") == SKIP
    # no cardinality probe for ordered (only complete_set)
    assert not any("returned cardinality vs two top_k" in c.name
                   for c in report.checks)


def test_unknown_mode_is_a_blocking_failure():
    report = validate_pipeline(UnknownModeConnector(), target="x",
                               sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "retrieval_mode is a recognized value") == FAIL


def test_internal_filters_are_echoed_not_reapplied():
    class DisclosingConnector(GoodConnector):
        def info(self):
            base = super().info()
            base["internal_filters"] = [
                {"name": "similarity_floor", "value": 0.7,
                 "description": "server drops matches below 0.7"}]
            return base

    report = validate_pipeline(DisclosingConnector(), target="x",
                               sample_query="alpha")
    mode_check = next(c for c in report.checks
                      if "retrieval_mode is a recognized value" in c.name)
    assert any("similarity_floor" in d for d in mode_check.details)


def test_cli_import_mode_round_trip(capsys):
    exit_code = main([
        "--import", f"{__name__}:GoodConnector",
        "--query", "alpha",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "VERDICT: READY" in out


def test_cli_reports_not_ready_exit_code(capsys):
    exit_code = main([
        "--import", f"{__name__}:MismatchedIdConnector",
        "--query", "alpha",
    ])
    assert exit_code == 1
    assert "NOT READY" in capsys.readouterr().out


# Degraded-path detection: the real mechanism is dead or missing and a base
# default quietly produces the same observable result. Every connector below
# was READY before these checks existed.

class WrongNameConnector(GoodConnector):
    """Implemented the by-id fetch under the WRONG NAME (a bug seen in a real
    connector): the base get_chunks default silently answers; this method is
    dead."""

    def fetch_chunks(self, chunk_ids):
        raise AssertionError("dead code: nothing ever calls this")


def test_a_stray_method_shadowing_a_contract_default_is_called_out():
    report = validate_pipeline(WrongNameConnector(), target="x",
                               sample_query="alpha")
    check = next(c for c in report.checks if "stray method" in c.name)
    assert check.status == WARN
    assert any("fetch_chunks" in d and "get_chunks" in d for d in check.details)


def test_inherited_get_chunks_default_is_a_warning_not_a_buried_pass():
    # The base default answers correctly at O(corpus); an author must SEE that
    # their override (if they thought they wrote one) is not what answered.
    report = validate_pipeline(GoodConnector(), target="x", sample_query="alpha")
    assert _status_of(report, "get_chunks()") == WARN
    assert report.ready   # degraded, not broken


def test_an_o_corpus_get_chunks_override_is_flagged():
    class DisguisedPull(GoodConnector):
        def get_chunks(self, chunk_ids):
            wanted = set(chunk_ids)
            return {c.chunk_id: c
                    for c in self.pull_all_chunks() if c.chunk_id in wanted}

    report = validate_pipeline(DisguisedPull(), target="x", sample_query="alpha")
    check = next(c for c in report.checks if "get_chunks()" in c.name)
    assert check.status == WARN
    assert any("pulled the full corpus" in d for d in check.details)


def test_a_true_by_id_get_chunks_override_passes():
    class ById(GoodConnector):
        def get_chunks(self, chunk_ids):
            by_id = {c.chunk_id: c for c in self._chunks}
            return {cid: by_id[cid] for cid in chunk_ids if cid in by_id}

    report = validate_pipeline(ById(), target="x", sample_query="alpha")
    assert _status_of(report, "get_chunks()") == PASS


def test_all_none_global_index_is_legal_and_ready():
    # The contract made global_index legally None (a host assigns ordinals at
    # ingest); the validator must not fail what the contract allows.
    class NoOrdinals(GoodConnector):
        def pull_all_chunks(self):
            chunks = super().pull_all_chunks()
            for c in chunks:
                c.global_index = None
            return chunks

    report = validate_pipeline(NoOrdinals(), target="x", sample_query="alpha")
    assert report.ready, [f"{c.name}: {c.details}" for c in report.failures]
    assert _status_of(report, "global_index") == PASS


def test_duplicate_supplied_global_index_still_fails():
    class DuplicateOrdinals(GoodConnector):
        def pull_all_chunks(self):
            chunks = super().pull_all_chunks()
            chunks[1].global_index = chunks[0].global_index
            return chunks

    report = validate_pipeline(DuplicateOrdinals(), target="x",
                               sample_query="alpha")
    assert _status_of(report, "global_index") == FAIL


def test_a_broken_list_chunks_override_fails_instead_of_hiding():
    # Before this check the paged path was never exercised: a dead override
    # validated READY because only pull_all_chunks was ever called.
    class StuckPager(GoodConnector):
        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            return ChunkPage(items=[], next_cursor="stuck")

    report = validate_pipeline(StuckPager(), target="x", sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "list_chunks") == FAIL


def test_a_consistent_list_chunks_override_passes():
    class Pager(GoodConnector):
        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            start = int(cursor) if cursor else 0
            page = self._chunks[start:start + limit]
            nxt = start + len(page)
            return ChunkPage(
                items=page,
                next_cursor=str(nxt) if nxt < len(self._chunks) else None)

    report = validate_pipeline(Pager(), target="x", sample_query="alpha")
    assert _status_of(report, "list_chunks") == PASS


class _BigCorpus(GoodConnector):
    """A 200-chunk corpus: large enough that a pager capped well below the
    validator's requested limit needs many more pages than ~5."""

    def __init__(self):
        self._chunks = [
            _chunk(f"c:{i}", f"doc-{i // 10}", i % 10, i,
                   f"alpha beta gamma delta {i}")
            for i in range(200)
        ]


def _paging_check(report):
    matches = [c for c in report.checks if "list_chunks" in c.name]
    assert matches
    return matches[0]


def test_a_list_chunks_override_with_capped_pages_passes():
    # The contract does not require full pages. A backend cap (a store's list
    # endpoint maximum) yields far more pages than the requested limit implies;
    # that is honest paging, not a pager that never terminates.
    class CappedPager(_BigCorpus):
        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            start = int(cursor) if cursor else 0
            page = self._chunks[start:start + min(limit, 7)]
            nxt = start + len(page)
            return ChunkPage(
                items=page,
                next_cursor=str(nxt) if nxt < len(self._chunks) else None)

    report = validate_pipeline(CappedPager(), target="x", sample_query="alpha")
    check = _paging_check(report)
    assert check.status == PASS, check.details
    assert any("capped at 7 items" in d for d in check.details)


def test_a_repeating_list_chunks_cursor_fails_naming_the_repeat():
    class LoopingPager(GoodConnector):
        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            return ChunkPage(items=self._chunks[:1], next_cursor="again")

    report = validate_pipeline(LoopingPager(), target="x", sample_query="alpha")
    check = _paging_check(report)
    assert check.status == FAIL
    assert "repeated" in check.details[0]


def test_empty_pages_with_ever_advancing_cursors_fail_as_a_stall():
    class StallingPager(GoodConnector):
        calls = 0

        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            type(self).calls += 1
            assert type(self).calls < 1000, "validator did not terminate"
            n = int(cursor) if cursor else 0
            return ChunkPage(items=[], next_cursor=str(n + 1))

    report = validate_pipeline(StallingPager(), target="x",
                               sample_query="alpha")
    check = _paging_check(report)
    assert check.status == FAIL
    assert "empty pages" in check.details[0]
    assert "stalled" in check.details[0]


def test_a_pager_reserving_items_forever_fails_as_an_overrun():
    class EndlessPager(GoodConnector):
        calls = 0

        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            type(self).calls += 1
            assert type(self).calls < 1000, "validator did not terminate"
            n = int(cursor) if cursor else 0
            return ChunkPage(items=self._chunks[:1], next_cursor=str(n + 1))

    report = validate_pipeline(EndlessPager(), target="x", sample_query="alpha")
    check = _paging_check(report)
    assert check.status == FAIL
    assert "overran the corpus" in check.details[0]


def test_a_full_page_pager_over_a_big_corpus_passes_without_a_cap_note():
    class FullPager(_BigCorpus):
        def list_chunks(self, *, cursor=None, limit=1000):
            from rag_connector.models import ChunkPage
            start = int(cursor) if cursor else 0
            page = self._chunks[start:start + limit]
            nxt = start + len(page)
            return ChunkPage(
                items=page,
                next_cursor=str(nxt) if nxt < len(self._chunks) else None)

    report = validate_pipeline(FullPager(), target="x", sample_query="alpha")
    check = _paging_check(report)
    assert check.status == PASS, check.details
    assert not any("capped" in d for d in check.details)


def test_fabricated_chunk_vectors_are_a_blocking_failure():
    # Also previously invisible: get_chunk_vectors was never called, so a
    # connector inventing vectors for unknown ids validated READY.
    class LyingVectors(GoodConnector):
        def get_chunk_vectors(self, ids):
            return {i: [0.0, 0.0] for i in ids}

    report = validate_pipeline(LyingVectors(), target="x", sample_query="alpha")
    assert not report.ready
    assert _status_of(report, "get_chunk_vectors") == FAIL


def test_honest_chunk_vectors_pass():
    class HonestVectors(GoodConnector):
        def get_chunk_vectors(self, ids):
            if not ids:
                return {}
            known = {c.chunk_id: [0.1, 0.2] for c in self._chunks}
            return {i: known[i] for i in ids if i in known}

    report = validate_pipeline(HonestVectors(), target="x", sample_query="alpha")
    assert _status_of(report, "get_chunk_vectors") == PASS


def test_unimplemented_chunk_vectors_are_a_supported_skip():
    report = validate_pipeline(GoodConnector(), target="x", sample_query="alpha")
    assert _status_of(report, "get_chunk_vectors") == SKIP


def test_implementing_no_corpus_read_names_both_ways_out():
    # The base raises UnsupportedCapability (not NotImplementedError) when
    # neither read exists; the report must surface its guidance, not a
    # generic "an exception happened".
    class NoCorpus(RagPipeline):
        def query(self, text, top_k=5):
            return []

    report = validate_pipeline(NoCorpus(), target="x", sample_query="alpha")
    assert not report.ready
    pull = next(c for c in report.checks if "pull_all_chunks()" in c.name)
    assert pull.status == FAIL
    assert "list_chunks" in pull.fix


def test_registry_mode_loads_dotenv_from_working_directory(tmp_path, monkeypatch):
    """The validator must read the operator's .env, not the library's own.

    Resolving relative to __file__ pointed inside site-packages for an
    installed wheel, so a connector needing cloud credentials silently got
    none.
    """
    (tmp_path / ".env").write_text(
        "RAG_CONNECTOR_DOTENV_PROBE=loaded\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_CONNECTOR_DOTENV_PROBE", raising=False)

    # An unregistered type exits 1 *after* the .env load, which is all we need.
    assert main(["--connector-type", "definitely-not-registered"]) == 1
    assert os.environ.get("RAG_CONNECTOR_DOTENV_PROBE") == "loaded"


def test_registry_mode_warns_when_dotenv_is_missing(tmp_path, monkeypatch,
                                                    capsys):
    """python-dotenv is not a runtime dependency, so on a clean install the
    .env load used to silently no-op -- credentials skipped, no hint why. With
    a .env present and dotenv absent, the operator must be told out loud."""
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "dotenv", None)  # import raises

    assert main(["--connector-type", "definitely-not-registered"]) == 1
    assert "python-dotenv is not installed" in capsys.readouterr().err
