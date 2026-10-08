from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    client = MagicMock()
    monkeypatch.setattr(replay_module, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(replay_module.httpx, "Client", lambda **kwargs: nullcontext())
    monkeypatch.setattr(replay_module, "TopicClusterer", lambda *args: SimpleNamespace(
        cluster_papers=lambda papers: [PaperGroup("Topic", None, papers)],
    ))
    monkeypatch.setattr(replay_module, "fetch_api_balance", lambda config: None)
    monkeypatch.setattr(replay_module, "fetch_paper", lambda url, client: Paper(
        source="arxiv", title=url, authors=[], abstract="Original abstract", url=url,
    ))
    def score(config, papers):
        for paper in papers:
            paper.score = 7.5
    monkeypatch.setattr(replay_module, "score_papers", score)
    monkeypatch.setattr(replay_module, "fetch_affiliation_text", lambda paper, client: "Author, Test University")
    def affiliations(paper, client, params, *, strict=False):
        assert strict
        paper.affiliations = ["Test University"]
        return paper.affiliations
    monkeypatch.setattr(Paper, "generate_affiliations", affiliations)
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
    assert html.count("Test University") == 2
    assert html.count("7.5") == 2
    assert "Unknown Affiliation" not in html
    assert "未计算" not in html
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


def test_replay_bounds_clustering_retries(config, tmp_path, replay_setup, monkeypatch):
    client = replay_module.OpenAI()
    clusterer = MagicMock(wraps=replay_module.TopicClusterer)
    monkeypatch.setattr(replay_module, "TopicClusterer", clusterer)
    replay(config, validate_manifest(manifest()), tmp_path, send=False)
    client.with_options.assert_called_once_with(max_retries=0)
    client.with_options.return_value.with_options.assert_called_once_with(timeout=60)
    assert clusterer.call_args.args[0] is client.with_options.return_value.with_options.return_value


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
    with pytest.raises(RuntimeError, match="Uncertain delivery"):
        replay(config, entries, tmp_path, send=True, revision="metadata-v2")


def test_replay_upgrades_legacy_delivery_without_regenerating_summaries(config, tmp_path, replay_setup, monkeypatch):
    entries = validate_manifest(manifest())
    replay(config, entries, tmp_path, send=True)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    del state["emails"]["123"]["revision"]
    for paper in state["papers"].values():
        paper["score"] = None
        paper["affiliations"] = None
        paper["full_text"] = None
    state_path.write_text(json.dumps(state))
    def unexpected(*args, **kwargs):
        raise AssertionError("Existing summary must be preserved")
    monkeypatch.setattr(Paper, "generate_tldr", unexpected)
    assert replay(config, entries, tmp_path, send=True, revision="metadata-v2") == 1
    assert replay(config, entries, tmp_path, send=True, revision="metadata-v2") == 0
    assert replay(config, entries, tmp_path, send=True) == 0
    assert len(replay_setup) == 2
    assert replay_setup[-1][1]["subject"].endswith("[metadata-v2]")
    saved = json.loads(state_path.read_text())
    assert saved["delivery_history"][0]["status"] == "sent"
    assert all(paper["score"] == 7.5 and paper["affiliations"] for paper in saved["papers"].values())


@pytest.mark.parametrize("missing", ["affiliations", "score"])
def test_replay_blocks_missing_metadata(config, tmp_path, replay_setup, monkeypatch, missing):
    if missing == "affiliations":
        monkeypatch.setattr(Paper, "generate_affiliations", lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(replay_module, "score_papers", lambda *args: None)
    with pytest.raises(ValueError, match="Missing"):
        replay(config, validate_manifest(manifest()), tmp_path, send=True)
    assert replay_setup == []


def test_replay_blocks_pdf_failure(config, tmp_path, replay_setup, monkeypatch):
    def fail(*args):
        raise RuntimeError("PDF unavailable")
    monkeypatch.setattr(replay_module, "fetch_affiliation_text", fail)
    with pytest.raises(RuntimeError, match="PDF unavailable"):
        replay(config, validate_manifest(manifest()), tmp_path, send=True)
    assert replay_setup == []


def test_replay_retries_failed_paper_without_discarding_other_results(config, tmp_path, replay_setup, monkeypatch):
    original = Paper.generate_affiliations
    attempts = []
    def flaky(paper, *args, **kwargs):
        if paper.url.endswith("12346"):
            attempts.append(paper.url)
            if len(attempts) == 1:
                raise RuntimeError("Temporary model timeout")
        return original(paper, *args, **kwargs)
    monkeypatch.setattr(Paper, "generate_affiliations", flaky)
    assert replay(config, validate_manifest(manifest()), tmp_path, send=True) == 1
    assert len(attempts) == 2
    assert len(replay_setup) == 1


def test_replay_accepts_only_source_supported_affiliation_corrections(config, tmp_path, replay_setup):
    entries = validate_manifest(manifest())
    url = entries[0]["urls"][0]
    with pytest.raises(ValueError, match="not supported"):
        replay(config, entries, tmp_path, send=True, affiliation_overrides={url: ["Invented Institute"]})
    assert replay_setup == []
    assert replay(config, entries, tmp_path, send=True, affiliation_overrides={url: ["Test University"]}) == 1


def test_replay_rejects_corrections_for_unrequested_papers(config, tmp_path, replay_setup):
    with pytest.raises(ValueError, match="requested URLs"):
        replay(config, validate_manifest(manifest()), tmp_path, send=True,
               affiliation_overrides={"https://arxiv.org/abs/2609.99999": ["Test University"]})
    assert replay_setup == []


def test_manifest_rejects_untrusted_links_and_duplicate_runs():
    payload = manifest()
    payload["emails"][0]["urls"][0] = "https://example.com/private"
    with pytest.raises(ValueError, match="exact arXiv"):
        validate_manifest(payload)
    payload = manifest()
    payload["emails"].append(payload["emails"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(payload)
