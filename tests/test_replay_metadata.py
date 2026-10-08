from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pymupdf
import pytest

from zotero_arxiv_daily import replay_metadata
from zotero_arxiv_daily.protocol import Paper


def paper():
    return Paper("arxiv", "Title", ["Author"], "Abstract", "https://arxiv.org/abs/2609.31603")


def test_fetch_affiliation_text_uses_bounded_front_matter():
    with pymupdf.open() as document:
        for index in range(4):
            page = document.new_page()
            page.insert_text((50, 50), f"Page {index + 1}: Author, Test University. " * 3)
        content = document.tobytes()
    def respond(request):
        assert str(request.url) == "https://arxiv.org/pdf/2609.31603"
        return httpx.Response(200, content=content)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        text = replay_metadata.fetch_affiliation_text(paper(), client)
    assert "Test University" in text
    assert "Page 3" in text
    assert "Page 4" not in text


@pytest.mark.parametrize("content", [b"not a PDF", b"%PDF too large"])
def test_fetch_affiliation_text_rejects_invalid_or_oversized_pdf(monkeypatch, content):
    monkeypatch.setattr(replay_metadata.time, "sleep", lambda _: None)
    monkeypatch.setattr(replay_metadata, "MAX_PDF_BYTES", 10)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))) as client:
        with pytest.raises(ValueError):
            replay_metadata.fetch_affiliation_text(paper(), client)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_score_papers_rejects_missing_or_nonfinite_scores(monkeypatch, value):
    client = Mock()
    item = paper()
    item.score = value
    executor = SimpleNamespace(
        fetch_zotero_corpus=lambda: ["corpus"], filter_corpus=lambda corpus: corpus,
        reranker=SimpleNamespace(rerank=lambda papers, corpus: papers), openai_client=client,
    )
    monkeypatch.setattr(replay_metadata, "Executor", lambda _: executor)
    with pytest.raises(ValueError, match="relevance score"):
        replay_metadata.score_papers(None, [item])
    client.close.assert_called_once()


def test_score_papers_reuses_daily_scorer_without_reordering(monkeypatch):
    items = [paper(), paper()]
    client = Mock()
    def rerank(papers, corpus):
        assert corpus == ["selected corpus"]
        papers[0].score, papers[1].score = 6.5, 9.0
        return list(reversed(papers))
    executor = SimpleNamespace(
        fetch_zotero_corpus=lambda: ["all corpus"], filter_corpus=lambda _: ["selected corpus"],
        reranker=SimpleNamespace(rerank=rerank), openai_client=client,
    )
    monkeypatch.setattr(replay_metadata, "Executor", lambda _: executor)
    replay_metadata.score_papers(None, items)
    assert [item.score for item in items] == [6.5, 9.0]
    client.close.assert_called_once()


def test_score_papers_requires_zotero_reference_papers(monkeypatch):
    client = Mock()
    executor = SimpleNamespace(fetch_zotero_corpus=lambda: [], filter_corpus=lambda _: [], openai_client=client)
    monkeypatch.setattr(replay_metadata, "Executor", lambda _: executor)
    with pytest.raises(ValueError, match="No Zotero reference"):
        replay_metadata.score_papers(None, [paper()])
    client.close.assert_called_once()
