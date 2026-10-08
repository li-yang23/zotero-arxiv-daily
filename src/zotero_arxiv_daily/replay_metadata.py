"""Recover source-backed affiliations and the normal Zotero relevance scores."""

import math
from threading import Lock
import time

import httpx

from .executor import Executor
from .protocol import Paper


PDF_LOCK = Lock()
MAX_PDF_BYTES = 50_000_000


def fetch_affiliation_text(paper: Paper, client: httpx.Client) -> str:
    import pymupdf

    # Derive the public PDF URL from the validated manifest, not scraped links.
    pdf_url = paper.url.replace("/abs/", "/pdf/")
    for attempt in range(3):
        try:
            with client.stream("GET", pdf_url) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_PDF_BYTES:
                        raise ValueError(f"PDF exceeds size limit: {paper.url}")
            if not content.startswith(b"%PDF"):
                raise ValueError(f"Not a PDF: {paper.url}")
            # PyMuPDF is not thread-safe; only downloads/LLM calls are concurrent.
            with PDF_LOCK, pymupdf.open(stream=bytes(content), filetype="pdf") as document:
                text = "\n".join(document[index].get_text(sort=True) for index in range(min(3, len(document))))
            if len(text.strip()) < 100:
                raise ValueError(f"No usable author/front-matter text: {paper.url}")
            return text
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Cannot retrieve author information: {paper.url}")


def score_papers(config, papers: list[Paper]) -> None:
    executor = Executor(config)
    try:
        corpus = executor.filter_corpus(executor.fetch_zotero_corpus())
        if not corpus:
            raise ValueError("No Zotero reference papers available for relevance scoring")
        # Rerank assigns scores in place. Keep the historical email order/list.
        executor.reranker.rerank(papers, corpus)
        for paper in papers:
            if paper.score is None or not math.isfinite(float(paper.score)):
                raise ValueError(f"Missing or invalid relevance score: {paper.url}")
            paper.score = float(paper.score)
    finally:
        executor.openai_client.close()
