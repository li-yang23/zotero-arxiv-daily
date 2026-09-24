from types import SimpleNamespace

from zotero_arxiv_daily.protocol import Paper, QualityReview
from zotero_arxiv_daily.request_executor import RequestExecutor
from zotero_arxiv_daily.request_models import PaperRequest, PaperRequestGroup, PaperResolution


class FakeResolver:
    def __init__(self, resolutions):
        self.resolutions = iter(resolutions)

    def resolve(self, _request):
        return next(self.resolutions)


class FakeReviewer:
    def review_paper(self, _paper):
        return QualityReview(
            problem="Problem",
            method="Method",
            conclusion="Conclusion",
            innovation_score=7,
            rigor_score=8,
            significance_score=7,
            overall_score=7.5,
            rationale="Rationale",
        )


def test_request_executor_keeps_all_resolved_papers_without_quality_filtering(config):
    requests = [
        PaperRequest(title="Paper One", url=None, input_index=0),
        PaperRequest(title="Paper Two", url=None, input_index=1),
    ]
    paper = Paper(
        source="test",
        title="Paper One",
        authors=["Author"],
        abstract="Abstract",
        url="https://example.com/one",
    )
    paper.generate_tldr = lambda *_args: setattr(paper, "tldr", "Summary") or "Summary"
    resolver = FakeResolver(
        [
            PaperResolution(request=requests[0], status="matched", paper=paper),
            PaperResolution(request=requests[1], status="not_found", detail="No exact match"),
        ]
    )
    executor = RequestExecutor(
        config,
        resolver=resolver,
        openai_client=SimpleNamespace(),
        quality_reviewer=FakeReviewer(),
    )

    report = executor.process([PaperRequestGroup(label="Venue", papers=requests)])

    assert report.requested_count == 2
    assert report.resolved_count == 1
    assert report.groups[0].papers == [paper]
    assert paper.tldr == "Summary"
    assert paper.quality_review is not None
    assert paper.quality_review.overall_score == 7.5
    assert [(issue.title, issue.status) for issue in report.issues] == [("Paper Two", "not_found")]
