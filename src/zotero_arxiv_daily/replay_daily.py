"""Regenerate and resend the exact paper lists recovered from historical runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
import hashlib
import json
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
from .topic_clusterer import TopicClusterer
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


def replay(config, entries, output_dir: Path, *, send: bool, workers: int = 4) -> int:
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"emails": {}, "papers": {}}
    summaries = state["papers"]
    openai_client = OpenAI(
        api_key=config.llm.api.key, base_url=config.llm.api.base_url,
        timeout=float(config.llm.api.timeout), max_retries=int(config.llm.api.max_retries),
    )
    clusterer = TopicClusterer(openai_client, config.llm)
    sent = 0
    with httpx.Client(timeout=45, follow_redirects=True) as http_client, openai_client:
        def summarize(url):
            if url in summaries:
                return Paper(**summaries[url])
            paper = fetch_paper(url, http_client)
            paper.generate_tldr(openai_client, config.llm, strict=True)
            return paper

        for entry in entries:
            run_id = str(entry["run_id"])
            fingerprint = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            previous = state["emails"].get(run_id, {})
            if previous and previous.get("fingerprint") != fingerprint:
                raise ValueError(f"Source manifest changed for run {run_id}")
            if previous.get("status") == "sent":
                logger.info(f"Already sent corrected digest {entry['date']} (source run {run_id})")
                continue
            if previous.get("status") == "sending":
                raise RuntimeError(f"Uncertain delivery for run {run_id}; verify mailbox before retrying")

            papers_by_url = {}
            # The first real summary checks API availability before starting the batch.
            first = summarize(entry["urls"][0])
            papers_by_url[first.url] = first
            summaries[first.url] = asdict(first)
            atomic_json(state_path, state)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(summarize, url): url for url in entry["urls"][1:]}
                for future in as_completed(futures):
                    paper = future.result()
                    papers_by_url[paper.url] = paper
                    summaries[paper.url] = asdict(paper)
                    atomic_json(state_path, state)
                    logger.info(f"Summarized {entry['date']}: {len(papers_by_url)}/{len(entry['urls'])}")
            papers = [papers_by_url[url] for url in entry["urls"]]
            groups = clusterer.cluster_papers(papers)
            html = render_email(groups, config.llm.language, api_balance=fetch_api_balance(config))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{entry['date']}-{run_id}.html").write_text(html, encoding="utf-8")
            subject = f"Daily arXiv {entry['date'].replace('-', '/')} [revised]"
            state["emails"][run_id] = {
                "date": entry["date"], "paper_count": len(papers), "fingerprint": fingerprint,
                "status": "ready", "subject": subject,
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
    args = parser.parse_args()
    raw = args.manifest.read_text() if args.manifest else os.environ["REPLAY_MANIFEST"]
    entries = validate_manifest(json.loads(raw))
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="default", overrides=["llm.language=Chinese"])
    count = replay(config, entries, args.output_dir, send=args.send)
    logger.info(f"Replay complete: {count} revised email(s) accepted by SMTP")


if __name__ == "__main__":
    main()
