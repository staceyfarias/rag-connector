import pytest

from rag_connector.base import (
    DEFAULT_RETRIEVAL_MODE,
    RETRIEVAL_MODES,
    RagPipeline,
    RetrievalModeError,
    RetrievedChunk,
    refuse_threshold_for_mode,
    resolve_retrieval_mode,
)


def test_base_pipeline_declares_scored_by_default():
    assert RagPipeline.retrieval_mode == DEFAULT_RETRIEVAL_MODE == "scored"


def test_info_surfaces_retrieval_mode():
    class Pipeline(RagPipeline):
        retrieval_mode = "complete_set"

        def query(self, text, top_k=5):
            return []

        def pull_all_chunks(self):
            return []

    assert Pipeline().info()["retrieval_mode"] == "complete_set"


@pytest.mark.parametrize(
    ("live", "override", "expected"),
    [
        (None, None, ("scored", "connector")),
        ("complete_set", None, ("complete_set", "connector")),
        ("ordered", None, ("ordered", "connector")),
        ("scored", "ordered", ("ordered", "dataset_override")),
        ("ordered", "ordered", ("ordered", "connector")),
    ],
)
def test_retrieval_mode_resolution(live, override, expected):
    assert resolve_retrieval_mode(live, override) == expected


@pytest.mark.parametrize(
    ("live", "override"),
    [
        ("ordered", "scored"),
        ("complete_set", "scored"),
        ("scored", "complete_set"),
        ("ordered", "complete_set"),
        ("complete_set", "ordered"),
        ("nonsense", None),
        ("scored", "nonsense"),
    ],
)
def test_illegal_or_unknown_modes_are_refused(live, override):
    with pytest.raises(RetrievalModeError):
        resolve_retrieval_mode(live, override)


def test_every_mode_name_is_registered():
    assert set(RETRIEVAL_MODES) == {"scored", "ordered", "complete_set"}


def test_threshold_allowed_only_for_scored_mode():
    refuse_threshold_for_mode("scored", 0.5)
    for mode in RETRIEVAL_MODES:
        refuse_threshold_for_mode(mode, None)
    for mode in ("ordered", "complete_set"):
        with pytest.raises(RetrievalModeError):
            refuse_threshold_for_mode(mode, 0.7, dataset_name="Work RAG")


def test_retrieved_chunk_accepts_none_score():
    hit = RetrievedChunk(chunk_id="c", text="t", score=None, rank=0)
    assert hit.score is None

