"""query()'s optional answer_metadata keyword (docs/contract.md)."""

from rag_connector import query_accepts_answer_metadata


class _Old:
    def query(self, text, top_k=5):
        return []


class _New:
    def query(self, text, top_k=5, *, answer_metadata=None):
        return []


class _Kwargs:
    def query(self, text, **kwargs):
        return []


def test_a_connector_written_before_the_keyword_does_not_accept_it():
    assert not query_accepts_answer_metadata(_Old())


def test_the_keyword_or_kwargs_accepts_it():
    assert query_accepts_answer_metadata(_New())
    assert query_accepts_answer_metadata(_Kwargs())


def test_no_query_does_not_accept_it():
    assert not query_accepts_answer_metadata(object())


def test_the_reference_rag_accepts_it():
    from rag_connector.reference import ReferenceRagConnector
    assert query_accepts_answer_metadata(ReferenceRagConnector)
