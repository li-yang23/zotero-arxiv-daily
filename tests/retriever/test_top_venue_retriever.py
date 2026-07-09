from datetime import date
from copy import deepcopy

import pytest

import zotero_arxiv_daily.retriever.base as retriever_base_module
import zotero_arxiv_daily.retriever.top_venue_retriever as top_venue_module
from zotero_arxiv_daily.retriever.top_venue_retriever import TopVenueRetriever


class InlineExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, func, values):
        return [func(value) for value in values]


def make_openalex_work(*, title: str, abstract_words: list[str], venue: str = "Nature") -> dict:
    inverted_index = {}
    for position, word in enumerate(abstract_words):
        inverted_index.setdefault(word, []).append(position)
    return {
        "title": title,
        "abstract_inverted_index": inverted_index,
        "publication_date": date.today().isoformat(),
        "doi": "https://doi.org/10.0000/example",
        "id": "https://openalex.org/W1",
        "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ],
        "primary_location": {
            "landing_page_url": "https://example.com/paper",
            "pdf_url": "https://example.com/paper.pdf",
            "source": {"display_name": venue},
        },
    }


def test_top_venue_retriever_filters_and_converts_openalex_works(config, monkeypatch: pytest.MonkeyPatch):
    test_config = deepcopy(config)
    test_config.source.top_venue.venues = ["Nature"]
    test_config.source.top_venue.model_keywords = ["large language model"]
    test_config.source.top_venue.security_keywords = ["privacy"]
    test_config.source.top_venue.lookback_days = 7
    test_config.source.top_venue.request_delay_seconds = 0

    matching_work = make_openalex_work(
        title="Privacy Controls for Large Language Model Agents",
        abstract_words=["large", "language", "model", "privacy", "access", "control"],
    )
    irrelevant_work = make_openalex_work(
        title="A Diffusion Model for Weather Forecasting",
        abstract_words=["diffusion", "model", "forecasting"],
    )
    calls = []

    def fake_resolve(self, venue, mailto):
        calls.append(("resolve", venue, mailto))
        return ["S123"]

    def fake_fetch(self, *, source_ids, from_publication_date, per_page, mailto):
        calls.append(("fetch", source_ids, from_publication_date, per_page, mailto))
        return [matching_work, irrelevant_work]

    monkeypatch.setattr(TopVenueRetriever, "_resolve_source_ids", fake_resolve)
    monkeypatch.setattr(TopVenueRetriever, "_fetch_openalex_works", fake_fetch)
    monkeypatch.setattr(retriever_base_module, "ProcessPoolExecutor", InlineExecutor)

    retriever = TopVenueRetriever(test_config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title == "Privacy Controls for Large Language Model Agents"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].source == "top_venue:Nature"
    assert papers[0].url == "https://example.com/paper"
    assert papers[0].pdf_url == "https://example.com/paper.pdf"
    assert "Venue: Nature." in papers[0].abstract
    assert "large language model privacy access control" in papers[0].abstract
    assert calls[0] == ("resolve", "Nature", None)
    assert calls[1][0] == "fetch"
    assert calls[1][1] == ["S123"]


def test_top_venue_retriever_handles_fetch_failure(config, monkeypatch: pytest.MonkeyPatch):
    test_config = deepcopy(config)
    test_config.source.top_venue.venues = ["Science"]

    class FailingUrlopen:
        def __init__(self, *args, **kwargs):
            raise OSError("network unavailable")

    monkeypatch.setattr(top_venue_module, "urlopen", FailingUrlopen)

    retriever = TopVenueRetriever(test_config)

    assert retriever._fetch_openalex_works(
        source_ids=["S456"],
        from_publication_date=date.today().isoformat(),
        per_page=1,
        mailto=None,
    ) == []
