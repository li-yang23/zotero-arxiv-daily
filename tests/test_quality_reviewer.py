from types import SimpleNamespace

import httpx

from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.quality_reviewer import QualityReviewer


class FakeChatClient:
    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *args, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


def make_paper(title: str) -> Paper:
    return Paper(
        source="arxiv",
        title=title,
        authors=["Author"],
        abstract="An abstract about a new method with experiments.",
        url=f"https://example.com/{title}",
        pdf_url=f"https://example.com/{title}.pdf",
        full_text="Full paper text with method, theory, and experiments.",
    )


def review_json(overall_score: float) -> str:
    return f"""{{
        "problem": "Problem statement",
        "method": "Proposed method",
        "conclusion": "Core conclusion",
        "innovation_score": 7.5,
        "rigor_score": 8.0,
        "significance_score": 7.0,
        "overall_score": {overall_score},
        "rationale": "Review rationale"
    }}"""


def test_quality_reviewer_parses_review_and_filters_by_threshold(config):
    config.quality_filter.min_score = 7.0
    client = FakeChatClient([review_json(8.0), review_json(6.5)])
    reviewer = QualityReviewer(client, config)
    papers = [make_paper("strong"), make_paper("weak")]

    selected = reviewer.filter_high_quality(papers)

    assert selected == [papers[0]]
    assert papers[0].quality_review is not None
    assert papers[0].quality_review.problem == "Problem statement"
    assert papers[0].quality_review.overall_score == 8.0
    assert papers[1].quality_review is not None
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert "senior program committee reviewer" in client.requests[0]["messages"][0]["content"]


def test_quality_reviewer_excludes_invalid_review(config):
    client = FakeChatClient(['{"problem": "missing required fields"}'])
    reviewer = QualityReviewer(client, config)
    papers = [make_paper("invalid")]

    selected = reviewer.filter_high_quality(papers)

    assert selected == []
    assert papers[0].quality_review is None


def test_quality_reviewer_handles_httpx_timeout(config):
    client = FakeChatClient([httpx.ReadTimeout("read timed out")])
    reviewer = QualityReviewer(client, config)
    papers = [make_paper("timeout")]

    selected = reviewer.filter_high_quality(papers)

    assert selected == []
    assert papers[0].quality_review is None
