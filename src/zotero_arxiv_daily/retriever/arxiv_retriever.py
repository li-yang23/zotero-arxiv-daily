from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlopen, urlretrieve
from tqdm import tqdm
import os
from collections import Counter
from omegaconf import OmegaConf
from loguru import logger


@dataclass
class RssArxivResult:
    paper_id: str
    title: str
    authors: list[str]
    summary: str
    entry_id: str
    pdf_url: str | None


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult | RssArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = self._parse_arxiv_rss_feed(f"https://rss.arxiv.org/atom/{query}")
        if feed.get("bozo") and len(feed.entries) == 0:
            raise RuntimeError(f"Failed to parse arXiv RSS feed for {query}: {feed.get('bozo_exception')}")
        feed_title = feed.feed.get("title", "")
        if 'Feed error for query' in feed_title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers: list[ArxivResult | RssArxivResult] = []
        announce_type_counts = Counter(i.get("arxiv_announce_type", "unknown") for i in feed.entries)
        logger.info(f"arXiv RSS returned {len(feed.entries)} entries for {query}: {dict(announce_type_counts)}")
        rss_entries = {
            i.id.removeprefix("oai:arXiv.org:"): i
            for i in feed.entries
            if self._is_daily_candidate(i)
        }
        all_paper_ids = list(rss_entries.keys())
        logger.info(f"Selected {len(all_paper_ids)} new or cross-listed arXiv entries for processing")
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        if not OmegaConf.select(self.config, "source.arxiv.api_enrich_metadata", default=False):
            logger.info("Using arXiv RSS metadata without API enrichment")
            return self._rss_entries_to_raw_papers(rss_entries, all_paper_ids)

        # Get full information of each paper from arxiv api
        client = arxiv.Client(
            num_retries=int(OmegaConf.select(self.config, "source.arxiv.api_num_retries", default=3)),
            delay_seconds=float(OmegaConf.select(self.config, "source.arxiv.api_delay_seconds", default=15)),
        )
        bar = tqdm(total=len(all_paper_ids))
        batch_size = max(1, int(OmegaConf.select(self.config, "source.arxiv.api_batch_size", default=5)))
        for i in range(0, len(all_paper_ids), batch_size):
            batch_ids = all_paper_ids[i:i + batch_size]
            search = arxiv.Search(id_list=batch_ids)
            try:
                batch = list(client.results(search))
                raw_papers.extend(batch)
            except arxiv.HTTPError as exc:
                logger.warning(f"arXiv API batch failed; falling back to RSS metadata for {len(batch_ids)} papers: {exc}")
                raw_papers.extend(self._rss_entries_to_raw_papers(rss_entries, batch_ids))
            bar.update(len(batch_ids))
        bar.close()

        return raw_papers

    def _parse_arxiv_rss_feed(self, url: str):
        feed = feedparser.parse(url)
        bozo_exception = str(feed.get("bozo_exception", ""))
        if (
            feed.get("bozo")
            and len(feed.entries) == 0
            and "XML or text declaration not at start of entity" in bozo_exception
        ):
            logger.warning("arXiv RSS has leading bytes before XML declaration; retrying with sanitized feed")
            with urlopen(url) as response:
                raw_feed = response.read()
            feed = feedparser.parse(raw_feed.lstrip(b"\xef\xbb\xbf \t\r\n"))
        return feed

    def _is_daily_candidate(self, entry) -> bool:
        announce_type = entry.get("arxiv_announce_type")
        return announce_type != "replace"

    def _rss_entries_to_raw_papers(self, rss_entries: dict[str, object], paper_ids: list[str]) -> list[RssArxivResult]:
        return [
            self._rss_entry_to_raw_paper(paper_id, rss_entries[paper_id])
            for paper_id in paper_ids
            if paper_id in rss_entries
        ]

    def _rss_entry_to_raw_paper(self, paper_id: str, entry) -> RssArxivResult:
        url = entry.link
        return RssArxivResult(
            paper_id=paper_id,
            title=entry.title,
            authors=self._parse_rss_authors(entry),
            summary=self._parse_rss_summary(entry.get("summary", "")),
            entry_id=url,
            pdf_url=url.replace("/abs/", "/pdf/") if "/abs/" in url else None,
        )

    def _parse_rss_authors(self, entry) -> list[str]:
        creators = entry.get("dc_creator") or entry.get("creator") or ""
        return [creator.strip() for creator in creators.split(",") if creator.strip()]

    def _parse_rss_summary(self, summary: str) -> str:
        marker = "Abstract:"
        if marker in summary:
            return summary.split(marker, 1)[1].strip()
        return summary.strip()

    def convert_to_paper(self, raw_paper: ArxivResult | RssArxivResult) -> Paper:
        title = raw_paper.title
        authors = raw_paper.authors if isinstance(raw_paper, RssArxivResult) else [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary or raw_paper.title
        pdf_url = raw_paper.pdf_url
        full_text = None
        if not isinstance(raw_paper, RssArxivResult):
            full_text = extract_text_from_pdf(raw_paper)
            if full_text is None:
                full_text = extract_text_from_tar(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )

def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None
        urlretrieve(paper.pdf_url, path)
        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None
        return full_text

def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None
        urlretrieve(source_url, path)
        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {paper.title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None
        return full_text
