from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily.venue_monitor import VenueMonitor, VenueReviewer, candidate, render_digest
from zotero_arxiv_daily.venue_sources import VENUES, VenuePaper, VenueSources
from zotero_arxiv_daily.venue_state import FileStateStore, GitHubStateStore


TODAY = date(2026, 10, 14)
ABSTRACT = "We study privacy leakage in large language models. Our method uses differential privacy to reduce memorization and tests utility on benchmark tasks."


def paper(venue="ICLR", title="Privacy of large language models"):
    return VenuePaper(venue, title, title, "https://openreview.net/forum?id=example", abstract=ABSTRACT,
                      authors=["Author"], published="2026-05-01")


def review():
    return {"relevant": True, "category": "隐私保护与数据泄露", "summary": "从摘要提取的论文总结。",
            "reason": "研究大模型隐私。", "relevance_score": 9.0}


def setup_monitor(tmp_path, papers=None, max_reviews=40):
    sources = Mock()
    sources.fetch.side_effect = lambda venue, *_: [p for p in papers or [paper()] if p.venue == venue.name]
    sources.enrich.side_effect = lambda p: p
    reviewer = Mock()
    reviewer.review.return_value = review()
    store = FileStateStore(tmp_path / "state.json")
    send = Mock()
    monitor = VenueMonitor(sources, reviewer, store, send, max_reviews=max_reviews)
    return monitor, sources, reviewer, store, send


def test_all_14_checked_and_independent_digest_sent_once(tmp_path):
    monitor, sources, reviewer, store, send = setup_monitor(tmp_path)
    assert monitor.run(TODAY)["sent"] == 1
    assert len(sources.fetch.call_args_list) == 14
    assert "顶会顶刊安全专题" in send.call_args.kwargs["subject"]
    assert monitor.run(TODAY)["sent"] == 0
    assert send.call_count == reviewer.review.call_count == 1
    assert next(iter(store.load()["deliveries"].values()))["status"] == "sent"
    assert "paper" not in next(iter(store.load()["papers"].values()))


def test_failed_source_does_not_block_other_venues_or_advance_watermark(tmp_path):
    monitor, sources, _, store, _ = setup_monitor(tmp_path)
    def fetch(venue, *_):
        if venue.name == "Nature":
            raise RuntimeError("unavailable")
        return [paper()] if venue.name == "ICLR" else []
    sources.fetch.side_effect = fetch
    report = monitor.run(TODAY)
    assert report["sent"] == 1
    assert report["venues"]["Nature"]["status"] == "failed"
    assert "Nature" not in store.load()["venues"]
    assert report["errors"]


def test_review_failure_remains_pending_and_retries(tmp_path):
    monitor, _, reviewer, store, send = setup_monitor(tmp_path)
    reviewer.review.side_effect = RuntimeError("API unavailable")
    assert monitor.run(TODAY)["pending"] == 1
    assert not send.called
    assert not store.load()["deliveries"]
    reviewer.review.side_effect = None
    assert monitor.run(date(2026, 10, 15))["sent"] == 1


def test_discovery_needs_no_llm_and_does_not_consume_email_queue(tmp_path):
    monitor, _, reviewer, store, send = setup_monitor(tmp_path)
    assert monitor.run(TODAY, discover_only=True)["pending"] == 1
    assert not reviewer.review.called and not send.called
    assert monitor.run(TODAY)["sent"] == 1


def test_budget_preserves_backlog_and_fairness(tmp_path):
    papers = [paper(title=f"Privacy of LLM {i}") for i in range(4)] + [paper("Nature")]
    monitor, _, _, store, send = setup_monitor(tmp_path, papers, max_reviews=2)
    report = monitor.run(TODAY)
    assert report["sent"] == 2 and report["pending"] == 3
    assert "Nature" in send.call_args.args[0]
    assert monitor.run(date(2026, 10, 15))["sent"] == 2
    assert monitor.run(date(2026, 10, 16))["sent"] == 1


def test_new_arrivals_take_priority_over_bootstrap_backlog(tmp_path):
    papers = [paper(title=f"LLM privacy old {i}") for i in range(3)]
    monitor, _, _, _, send = setup_monitor(tmp_path, papers, max_reviews=1)
    monitor.run(TODAY)
    papers.append(paper(title="LLM privacy newly announced"))
    monitor.run(date(2026, 10, 15))
    assert "newly announced" in send.call_args.args[0]


def test_smtp_uncertainty_is_persisted_before_send_and_never_blindly_retried(tmp_path):
    monitor, _, _, store, send = setup_monitor(tmp_path)
    def smtp(*args, **kwargs):
        assert next(iter(store.load()["deliveries"].values()))["status"] == "sending"
        raise TimeoutError("ambiguous SMTP result")
    send.side_effect = smtp
    with pytest.raises(TimeoutError):
        monitor.run(TODAY)
    send.side_effect = None
    report = monitor.run(TODAY)
    assert send.call_count == 1
    assert report["errors"] and report["sent"] == 0


def test_new_abstract_reconsiders_excluded_title(tmp_path):
    p = paper(title="A new benchmark")
    p.abstract = ""
    monitor, _, _, _, send = setup_monitor(tmp_path, [p])
    assert monitor.run(TODAY)["sent"] == 0
    p.abstract = ABSTRACT
    assert monitor.run(TODAY)["sent"] == 1


def test_title_update_with_same_doi_is_not_resent(tmp_path):
    p = paper()
    p.doi = "10.1/same"
    monitor, _, _, _, send = setup_monitor(tmp_path, [p])
    assert monitor.run(TODAY)["sent"] == 1
    p.title = "Revised title about LLM privacy"
    p.identifier = "another-provider"
    assert monitor.run(TODAY)["sent"] == 0
    assert send.call_count == 1


def test_new_metadata_refreshes_pending_paper(tmp_path):
    p = paper()
    p.abstract = ""
    monitor, _, reviewer, _, _ = setup_monitor(tmp_path, [p])
    reviewer.review.side_effect = ValueError("missing abstract")
    assert monitor.run(TODAY)["sent"] == 0
    p.abstract = ABSTRACT
    reviewer.review.side_effect = None
    assert monitor.run(TODAY)["sent"] == 1


def test_keyword_boundaries_and_no_venue_name_contamination():
    p = paper("USENIX Security", "Reagents in diffusion physics")
    p.abstract = "Chemical reagents diffuse."
    assert not candidate(p)
    p.title, p.abstract = "A better language model", "We improve language model accuracy."
    assert not candidate(p)
    p.title, p.abstract = "Private agent memory", ""
    assert candidate(p)


def test_email_escapes_source_and_has_summary_without_full_analysis():
    p = paper(title="<script>alert(1)</script>")
    html = render_digest([{"paper": p.to_dict(), "review": review()}], TODAY.isoformat(),
                         {"pending": 1, "errors": []}, balance="5 CNY")
    assert "<script>" not in html
    assert "完整分析" not in html
    assert "font-size:17px" in html and "font-size:20px" in html
    assert "5 CNY" in html and "非 Zotero 相似度" in html


def test_reviewer_rejects_malformed_decisions_and_missing_abstracts():
    llm = Mock()
    llm.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"relevant":"true"}'))])
    reviewer = VenueReviewer(llm, OmegaConf.create({"generation_kwargs": {"model": "test"}}))
    with pytest.raises(RuntimeError):
        reviewer.review(paper())
    p = paper()
    p.abstract = ""
    with pytest.raises(ValueError):
        reviewer.review(p)
    assert llm.chat.completions.create.call_count == 1


def source_with(handler):
    return VenueSources(httpx.Client(transport=httpx.MockTransport(handler)), delay=0)


def test_openreview_only_accepts_exact_venue_and_paginates():
    calls = []
    def handler(request):
        calls.append(request)
        offset = int(request.url.params["offset"])
        ids = ["ICLR.cc/2026/Conference", "ICLR.cc/2026/Conference/Rejected_Submission"]
        return httpx.Response(200, json={"count": 2, "notes": [{"id": str(offset), "content": {
            "venueid": {"value": ids[offset]}, "title": {"value": "Privacy of LLM"},
            "abstract": {"value": ABSTRACT}, "authors": {"value": ["Author"]},
        }}]})
    papers = source_with(handler).openreview(VENUES[0], 2026)
    assert len(calls) == 2 and len(papers) == 1
    assert papers[0].abstract == ABSTRACT


def test_dblp_exact_stream_year_and_paginated_results():
    def handler(request):
        assert request.url.params["q"] == "stream:conf/ndss: year:2026:"
        offset = int(request.url.params["f"])
        key = "conf/ndss/test" if offset == 0 else "conf/ndssw/test"
        return httpx.Response(200, json={"result": {"hits": {"@total": "2", "hit": [{"info": {
            "key": key, "year": "2026", "title": "Privacy of LLM", "type": "Conference and Workshop Papers",
            "authors": {"author": {"text": "Author"}}, "ee": "https://doi.org/10.1/test",
        }}]}}})
    papers = source_with(handler).dblp(VENUES[3], 2026)
    assert len(papers) == 1 and papers[0].doi == "10.1/test"


def test_crossref_first_window_then_overlapping_index_window_and_pagination():
    requests = []
    def handler(request):
        requests.append(request)
        items = [{"DOI": f"10.1/{i}", "title": [f"Paper {i}"], "abstract": "<jats:p>Text</jats:p>"}
                 for i in range(200)] if request.url.params["cursor"] == "*" else []
        return httpx.Response(200, json={"message": {"items": items, "next-cursor": "next"}})
    source = source_with(handler)
    assert len(source.fetch(VENUES[7], TODAY)) == 200
    assert "from-index-date:2026-07-16" in requests[0].url.params["filter"]
    assert "from-pub-date:2026-01-01" in requests[0].url.params["filter"]
    source.fetch(VENUES[7], TODAY, "2026-10-13")
    assert "from-index-date:2026-10-10" in requests[2].url.params["filter"]
    assert "until-index-date:2026-10-14" in requests[2].url.params["filter"]


def test_jmlr_uses_paper_title_not_abs_label():
    source = source_with(lambda _: httpx.Response(200, text='''
      <dl><dt>Private Language Models</dt><dd>Author
      <a href="/papers/v27/25-1234.html">[abs]</a><a href="x.pdf">[pdf]</a></dd></dl>
    '''))
    papers = source.fetch(VENUES[10], TODAY)
    assert len(papers) == 1
    assert papers[0].title == "Private Language Models"


def test_official_ai_export_joins_abstracts_and_institutions():
    def handler(request):
        if request.url.path.endswith("abstracts.json"):
            return httpx.Response(200, json={"123": ABSTRACT})
        return httpx.Response(200, json={"count": 1, "next": None, "results": [{
            "id": 123, "name": "LLM privacy", "decision": "Accept (Poster)",
            "virtualsite_url": "/virtual/2026/poster/123", "authors": [
                {"fullname": "Author", "institution": "Example University"}],
        }]})
    p = source_with(handler).fetch(VENUES[0], TODAY)[0]
    assert p.abstract == ABSTRACT and p.affiliations == ["Example University"]
    assert p.url == "https://iclr.cc/virtual/2026/poster/123"


def test_official_security_directories():
    pages = {
        "www.ndss-symposium.org": '<div class="pt-cv-content-item"><h2 class="pt-cv-title"><a href="/ndss-paper/test">LLM privacy</a></h2><div class="pt-cv-ctf-display_authors">Alice (Example University)</div></div>',
        "www.sigsac.org": '<table><tr><th>Title</th><th>Author</th></tr><tr><td>LLM privacy</td><td>Alice (Example University)</td></tr></table>',
        "www.ieee-security.org": '<div class="list-group-item"><b><a data-toggle="collapse" href="#paper">LLM privacy</a></b><div class="authorlist">Alice<sup>1</sup>, Bob<sup>1</sup><br/><sup>1</sup>: Example University</div></div>',
    }
    source = source_with(lambda request: httpx.Response(200, text=pages[request.url.host]))
    for index in (3, 5, 6):
        papers = source.fetch(VENUES[index], TODAY)
        assert len(papers) == 1 and papers[0].title == "LLM privacy"
        assert "Example University" in papers[0].affiliations[0]
    assert source.fetch(VENUES[5], TODAY)[0].authors == ["Alice, Bob"]


def test_usenix_fallback_reports_degraded_coverage_and_paginates():
    def handler(request):
        if request.url.host == "www.usenix.org":
            return httpx.Response(403)
        token = request.url.params.get("token")
        return httpx.Response(200, json={"token": None if token else "next", "data": [{
            "paperId": token or "first", "title": "LLM privacy", "venue": "USENIX Security Symposium",
            "abstract": ABSTRACT, "url": "https://www.semanticscholar.org/paper/test", "authors": [],
        }]})
    source = source_with(handler)
    assert len(source.fetch(VENUES[4], TODAY)) == 2
    assert "Semantic Scholar" in source.warnings[0]


def test_corrupt_local_state_never_resets(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"version":2}')
    with pytest.raises(ValueError):
        FileStateStore(path).load()


def test_missing_github_file_on_existing_branch_is_fatal():
    def handler(request):
        if "/git/ref/" in request.url.path:
            return httpx.Response(200, json={"object": {"sha": "abc"}})
        return httpx.Response(404, json={"message": "Not Found"})
    store = GitHubStateStore(httpx.Client(transport=httpx.MockTransport(handler)), "owner/repo")
    with pytest.raises(httpx.HTTPStatusError):
        store.load()


def test_transient_github_conflict_retries_only_unchanged_state(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.venue_state.sleep", lambda _: None)
    writes = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"sha": "original"})
        writes.append(request)
        return httpx.Response(409) if len(writes) == 1 else httpx.Response(200, json={"content": {"sha": "new"}})
    store = GitHubStateStore(httpx.Client(transport=httpx.MockTransport(handler)), "owner/repo")
    store.sha = "original"
    store.save({"test": True})
    assert store.sha == "new" and len(writes) == 2


def test_real_github_conflict_never_overwrites_another_writer():
    writes = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"sha": "another-writer"})
        writes.append(request)
        return httpx.Response(409)
    store = GitHubStateStore(httpx.Client(transport=httpx.MockTransport(handler)), "owner/repo")
    store.sha = "original"
    with pytest.raises(httpx.HTTPStatusError):
        store.save({"test": True})
    assert len(writes) == 1
