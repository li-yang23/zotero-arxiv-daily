from types import SimpleNamespace

import feedparser

import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever_module
import zotero_arxiv_daily.retriever.base as retriever_base_module
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


class InlineExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, func, values):
        return [func(value) for value in values]


def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)

    class FakeArxivClient:
        def __init__(self, *args, **kwargs):
            pass

        def results(self, search):
            return [
                SimpleNamespace(
                    title=entry.title,
                    authors=[SimpleNamespace(name="Test Author")],
                    summary=entry.get("summary", entry.title),
                    entry_id=entry.link,
                    pdf_url=None,
                )
                for entry in parsed_result.entries
                if entry.id.removeprefix("oai:arXiv.org:") in search.id_list
            ]

    def fake_convert_to_paper(self, raw_paper):
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[author.name for author in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

    monkeypatch.setattr(arxiv_retriever_module.arxiv, "Client", FakeArxivClient)
    monkeypatch.setattr(retriever_base_module, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(ArxivRetriever, "convert_to_paper", fake_convert_to_paper)
    
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type","new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)
