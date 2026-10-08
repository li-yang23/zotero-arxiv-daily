from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from zotero_arxiv_daily import replay_daily as replay_module
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.replay_daily import replay, validate_manifest
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def manifest():
    return {"emails": [{"date": "2026-09-29", "run_id": "123", "urls": [
        "https://arxiv.org/abs/2609.12345", "https://arxiv.org/abs/2609.12346",
    ]}]}


@pytest.fixture
def replay_setup(monkeypatch):
    monkeypatch.setattr(replay_module, "OpenAI", lambda **kwargs: nullcontext())
    monkeypatch.setattr(replay_module.httpx, "Client", lambda **kwargs: nullcontext())
    monkeypatch.setattr(replay_module, "TopicClusterer", lambda *args: SimpleNamespace(
        cluster_papers=lambda papers: [PaperGroup("Topic", None, papers)],
    ))
    monkeypatch.setattr(replay_module, "fetch_api_balance", lambda config: None)
    monkeypatch.setattr(replay_module, "fetch_paper", lambda url, client: Paper(
        source="arxiv", title=url, authors=[], abstract="Original abstract", url=url,
    ))
    def generate(paper, client, params, *, strict=False):
        assert strict
        paper.tldr = "A regenerated summary."
        return paper.tldr
    monkeypatch.setattr(Paper, "generate_tldr", generate)
    sent = []
    monkeypatch.setattr(replay_module, "send_email", lambda config, html, **kwargs: sent.append((html, kwargs)))
    return sent


def test_replay_preserves_email_date_papers_and_skips_delivered(config, tmp_path, replay_setup):
    entries = validate_manifest(manifest())
    assert replay(config, entries, tmp_path, send=True) == 1
    assert replay(config, entries, tmp_path, send=True) == 0
    assert len(replay_setup) == 1
    html, headers = replay_setup[0]
    assert headers["subject"] == "Daily arXiv 2026/09/29 [revised]"
    assert html.count("A regenerated summary.") == 2
    assert "<details>" not in html
    assert html.index(entries[0]["urls"][0]) < html.index(entries[0]["urls"][1])


def test_replay_never_sends_partial_summary_batch(config, tmp_path, replay_setup, monkeypatch):
    def fail(paper, *args, **kwargs):
        raise RuntimeError("API balance insufficient")
    monkeypatch.setattr(Paper, "generate_tldr", fail)
    with pytest.raises(RuntimeError, match="balance insufficient"):
        replay(config, validate_manifest(manifest()), tmp_path, send=True)
    assert replay_setup == []


def test_replay_dry_run_then_send(config, tmp_path, replay_setup):
    entries = validate_manifest(manifest())
    assert replay(config, entries, tmp_path, send=False) == 0
    assert replay_setup == []
    assert replay(config, entries, tmp_path, send=True) == 1


def test_replay_stops_after_uncertain_delivery(config, tmp_path, replay_setup, monkeypatch):
    def fail_send(*args, **kwargs):
        raise OSError("Connection lost after DATA")
    monkeypatch.setattr(replay_module, "send_email", fail_send)
    entries = validate_manifest(manifest())
    with pytest.raises(OSError):
        replay(config, entries, tmp_path, send=True)
    assert json.loads((tmp_path / "state.json").read_text())["emails"]["123"]["status"] == "sending"
    with pytest.raises(RuntimeError, match="Uncertain delivery"):
        replay(config, entries, tmp_path, send=True)


def test_manifest_rejects_untrusted_links_and_duplicate_runs():
    payload = manifest()
    payload["emails"][0]["urls"][0] = "https://example.com/private"
    with pytest.raises(ValueError, match="exact arXiv"):
        validate_manifest(payload)
    payload = manifest()
    payload["emails"].append(payload["emails"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(payload)
