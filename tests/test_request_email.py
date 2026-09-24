from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.request_email import render_request_email
from zotero_arxiv_daily.request_models import RequestIssue, RequestReport
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def test_render_request_email_preserves_groups_and_reports_failures():
    paper = Paper(
        source="test",
        title="Resolved Paper",
        authors=["Author"],
        abstract="Abstract",
        url="https://example.com/paper",
        tldr="中文总结",
    )
    report = RequestReport(
        requested_count=2,
        groups=[PaperGroup(label="NDSS 2026", summary=None, papers=[paper])],
        issues=[
            RequestIssue(
                group="CCS 2026",
                title="Missing <Paper>",
                status="not_found",
                detail="No exact & trusted match",
            )
        ],
    )

    html = render_request_email(report, language="Chinese")

    assert "本次请求共 2 篇，已返回 1 篇" in html
    assert "NDSS 2026" in html
    assert "Resolved Paper" in html
    assert "未完整处理的论文" in html
    assert "Missing &lt;Paper&gt;" in html
    assert "No exact &amp; trusted match" in html
    assert "如需退订" not in html
