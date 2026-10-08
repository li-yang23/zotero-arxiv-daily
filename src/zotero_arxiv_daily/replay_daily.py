"""Regenerate and resend the exact paper lists recovered from historical runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import httpx
from hydra import compose, initialize_config_dir
from loguru import logger
from openai import OpenAI

from .construct_email import render_email
from .paper_resolver import ScholarlyHTMLParser, extract_page_metadata
from .protocol import Paper
from .replay_metadata import fetch_affiliation_text, score_papers
from .topic_clusterer import PaperGroup, TopicClusterer
from .utils import fetch_api_balance, send_email


ARXIV_URL = re.compile(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}(?:v\d+)?\Z")


def validate_manifest(payload: dict) -> list[dict]:
    entries = payload.get("emails")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty emails list")
    seen = set()
    for entry in entries:
        run_id = str(entry["run_id"])
        date.fromisoformat(entry["date"])
        urls = entry["urls"]
        if not run_id.isdigit() or run_id in seen:
            raise ValueError("Each source run_id must be numeric and unique")
        seen.add(run_id)
        if not isinstance(urls, list) or not urls or len(urls) != len(set(urls)):
            raise ValueError("Each email must have a non-empty, unique paper list")
        if any(not isinstance(url, str) or not ARXIV_URL.fullmatch(url) for url in urls):
            raise ValueError("Only exact arXiv abstract URLs are supported")
    return sorted(entries, key=lambda entry: (entry["date"], str(entry["run_id"])))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_paper(url: str, client: httpx.Client) -> Paper:
    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            parser = ScholarlyHTMLParser()
            parser.feed(response.text)
            title = (parser.meta.get("citation_title") or [""])[0]
            if not title:
                raise ValueError(f"No scholarly title at {url}")
            metadata = extract_page_metadata(response.text, url, title)
            if len(metadata.abstract) < 100:
                raise ValueError(f"No usable abstract at {url}")
            return Paper(
                source="arxiv", title=metadata.title, authors=metadata.authors,
                abstract=metadata.abstract, url=url, pdf_url=metadata.pdf_url,
            )
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Cannot retrieve {url}")


def validate_complete_paper(paper: Paper) -> None:
    if not isinstance(paper.tldr, str) or not paper.tldr.strip():
        raise ValueError(f"Missing summary: {paper.url}")
    if paper.score is None or not math.isfinite(float(paper.score)):
        raise ValueError(f"Missing relevance score: {paper.url}")
    if not isinstance(paper.affiliations, list) or any(
        not isinstance(item, str) or not item.strip() for item in paper.affiliations
    ):
        raise ValueError(f"Missing affiliation extraction: {paper.url}")


def replay(config, entries, output_dir: Path, *, send: bool, workers: int = 8, revision: str = "revised", affiliation_overrides: dict | None = None) -> int:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", revision):
        raise ValueError("Invalid delivery revision")
    affiliation_overrides = affiliation_overrides or {}
    requested_urls = {url for entry in entries for url in entry["urls"]}
    if not isinstance(affiliation_overrides, dict) or any(
        url not in requested_urls or not isinstance(names, list) or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        for url, names in affiliation_overrides.items()
    ):
        raise ValueError("Affiliation corrections must contain requested URLs and non-empty institution names")
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"emails": {}, "papers": {}}
    summaries = state["papers"]
    pending = []
    for entry in entries:
        run_id = str(entry["run_id"])
        fingerprint = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        previous = state["emails"].get(run_id, {})
        if previous and previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Source manifest changed for run {run_id}")
        if previous.get("status") == "sending":
            raise RuntimeError(f"Uncertain delivery for run {run_id}; verify mailbox before retrying")
        deliveries = [previous] + [
            item for item in state.get("delivery_history", []) if item["run_id"] == run_id
        ]
        if any(item.get("status") == "sent" and item.get("revision", "revised") == revision for item in deliveries):
            logger.info(f"Already sent {revision} digest {entry['date']} (source run {run_id})")
            continue
        pending.append((entry, fingerprint))
    if not pending:
        return 0
    openai_client = OpenAI(
        api_key=config.llm.api.key, base_url=config.llm.api.base_url,
        timeout=float(config.llm.api.timeout), max_retries=int(config.llm.api.max_retries),
    )
    # Short extraction/classification calls should promptly use configured fallbacks.
    metadata_client = openai_client.with_options(max_retries=0)
    clusterer = TopicClusterer(
        metadata_client.with_options(timeout=min(60, float(config.llm.api.timeout))), config.llm,
    )
    sent = 0
    with httpx.Client(timeout=45, follow_redirects=True) as http_client, openai_client:
        unscored = []
        urls = dict.fromkeys(url for entry, _ in pending for url in entry["urls"])
        for url in urls:
            paper = Paper(**summaries[url]) if url in summaries else fetch_paper(url, http_client)
            summaries[url] = asdict(paper)
            if paper.score is None:
                unscored.append(paper)
        if unscored:
            score_papers(config, unscored)
            for paper in unscored:
                summaries[paper.url] = asdict(paper)
            atomic_json(state_path, state)
            logger.info(f"Computed Zotero relevance for {len(unscored)} papers")

        def summarize(url):
            paper = Paper(**summaries[url])
            if not paper.tldr:
                paper.generate_tldr(openai_client, config.llm, strict=True)
            if url in affiliation_overrides:
                if not paper.full_text:
                    paper.full_text = fetch_affiliation_text(paper, http_client)
                normalized_text = " ".join(paper.full_text.casefold().split())
                names = affiliation_overrides[url]
                if any(" ".join(name.casefold().split()) not in normalized_text for name in names):
                    raise ValueError(f"Affiliation correction not supported by extracted paper text: {url}")
                paper.affiliations = list(dict.fromkeys(name.strip() for name in names))
            elif paper.affiliations is None:
                if not paper.full_text:
                    paper.full_text = fetch_affiliation_text(paper, http_client)
                paper.generate_affiliations(metadata_client, config.llm, strict=True)
            validate_complete_paper(paper)
            return paper

        for entry, fingerprint in pending:
            run_id = str(entry["run_id"])
            previous = state["emails"].get(run_id, {})

            papers_by_url = {}
            # The first real summary checks API availability before starting the batch.
            first = summarize(entry["urls"][0])
            papers_by_url[first.url] = first
            summaries[first.url] = asdict(first)
            atomic_json(state_path, state)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(summarize, url): url for url in entry["urls"][1:]}
                failed_urls = []
                for future in as_completed(futures):
                    try:
                        paper = future.result()
                    except Exception as exc:
                        failed_urls.append(futures[future])
                        logger.warning(f"Will retry metadata after saving other completed papers: {exc}")
                        continue
                    papers_by_url[paper.url] = paper
                    summaries[paper.url] = asdict(paper)
                    atomic_json(state_path, state)
                    logger.info(f"Completed summary and metadata {entry['date']}: {len(papers_by_url)}/{len(entry['urls'])}")
            for url in failed_urls:
                paper = summarize(url)
                papers_by_url[paper.url] = paper
                summaries[paper.url] = asdict(paper)
                atomic_json(state_path, state)
            papers = [papers_by_url[url] for url in entry["urls"]]
            saved_groups = state.setdefault("groups", {}).get(run_id)
            if saved_groups:
                groups = [PaperGroup(group["label"], group["summary"], [papers_by_url[url] for url in group["urls"]]) for group in saved_groups]
            else:
                groups = clusterer.cluster_papers(papers)
                state["groups"][run_id] = [
                    {"label": group.label, "summary": group.summary, "urls": [paper.url for paper in group.papers]}
                    for group in groups
                ]
            html = render_email(groups, config.llm.language, api_balance=fetch_api_balance(config))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{entry['date']}-{run_id}-{revision}.html").write_text(html, encoding="utf-8")
            subject = f"Daily arXiv {entry['date'].replace('-', '/')} [{revision}]"
            if previous.get("status") == "sent":
                state.setdefault("delivery_history", []).append({"run_id": run_id, **previous})
            state["emails"][run_id] = {
                "date": entry["date"], "paper_count": len(papers), "fingerprint": fingerprint,
                "status": "ready", "subject": subject, "revision": revision,
            }
            atomic_json(state_path, state)
            if send:
                state["emails"][run_id]["status"] = "sending"
                atomic_json(state_path, state)
                send_email(config, html, subject=subject)
                state["emails"][run_id]["status"] = "sent"
                atomic_json(state_path, state)
                sent += 1
                logger.info(f"SMTP accepted revised digest {entry['date']}: {len(papers)} papers (source run {run_id})")
    return sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/digest-replay"))
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--revision", default="revised")
    args = parser.parse_args()
    raw = args.manifest.read_text() if args.manifest else os.environ["REPLAY_MANIFEST"]
    entries = validate_manifest(json.loads(raw))
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="default", overrides=["llm.language=Chinese"])
    corrections = json.loads(os.environ.get("REPLAY_AFFILIATION_OVERRIDES") or "{}")
    count = replay(config, entries, args.output_dir, send=args.send, revision=args.revision, affiliation_overrides=corrections)
    logger.info(f"Replay complete: {count} revised email(s) accepted by SMTP")


if __name__ == "__main__":
    main()
