
"""The canonical score convention: utilities + RetrievedChunk contract."""

import pytest

from rag_connector.base import (
    SCORE_CONVENTION,
    RetrievedChunk,
    cosine_distance_to_similarity,
    l2_distance_to_score,
    similarity_passthrough,
)


def test_cosine_distance_to_similarity_is_exact():
    assert cosine_distance_to_similarity(0.0) == 1.0     # identical
    assert cosine_distance_to_similarity(1.0) == 0.0     # orthogonal
    assert cosine_distance_to_similarity(2.0) == -1.0    # opposite
    assert cosine_distance_to_similarity(0.25) == 0.75


def test_negative_similarity_is_legal_output():
    # Cosine similarity ranges -1..1; the conversion must not clamp.
    assert cosine_distance_to_similarity(1.7) == pytest.approx(-0.7)


def test_l2_is_a_direction_flip_not_a_similarity():
    # Negation only: 0 = exact, more negative = farther. No invented [0,1]
    # normalization -- L2 is unbounded and scale depends on the embedding.
    assert l2_distance_to_score(0.0) == 0.0
    assert l2_distance_to_score(1.5) == -1.5
    assert l2_distance_to_score(0.2) > l2_distance_to_score(0.9)  # closer > farther


def test_similarity_passthrough_is_identity():
    assert similarity_passthrough(0.87) == 0.87
    assert similarity_passthrough(-0.2) == -0.2


def test_retrieved_chunk_raw_score_defaults_none():
    hit = RetrievedChunk(chunk_id="c", text="t", score=0.9, rank=0)
    assert hit.raw_score is None
    converted = RetrievedChunk(chunk_id="c", text="t",
                               score=cosine_distance_to_similarity(0.1),
                               rank=0, raw_score=0.1)
    assert converted.score == 0.9 and converted.raw_score == 0.1


def test_convention_token_is_stable():
    # Stamped into run meta; display logic keys off this exact string.
    assert SCORE_CONVENTION == "higher_is_better"
