from dataclasses import dataclass
from datetime import date, timedelta
import json
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from loguru import logger
from omegaconf import OmegaConf

from ..protocol import Paper
from .base import BaseRetriever, register_retriever


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"


DEFAULT_VENUES = [
    "International Conference on Learning Representations",
    "International Conference on Machine Learning",
    "Neural Information Processing Systems",
    "Network and Distributed System Security Symposium",
    "USENIX Security Symposium",
    "IEEE Symposium on Security and Privacy",
    "ACM Conference on Computer and Communications Security",
    "IEEE Transactions on Information Forensics and Security",
    "IEEE Transactions on Dependable and Secure Computing",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "Journal of Machine Learning Research",
    "Artificial Intelligence",
    "Nature",
    "Science",
]

DEFAULT_MODEL_KEYWORDS = [
    "large language model",
    "llm",
    "language model",
    "foundation model",
    "agent",
    "multi-agent",
    "diffusion model",
    "text-to-image",
]

DEFAULT_SECURITY_KEYWORDS = [
    "privacy",
    "private",
    "permission",
    "access control",
    "authorization",
    "authentication",
    "confidentiality",
    "data leakage",
    "information leakage",
    "security",
    "secure",
    "safety",
    "jailbreak",
]


@dataclass
class OpenAlexWork:
    venue: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None
    publication_date: str | None


@register_retriever("top_venue")
class TopVenueRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[OpenAlexWork]:
        venues = self._config_list("venues", DEFAULT_VENUES)
        model_keywords = self._config_list("model_keywords", DEFAULT_MODEL_KEYWORDS)
        security_keywords = self._config_list("security_keywords", DEFAULT_SECURITY_KEYWORDS)
        lookback_days = int(OmegaConf.select(self.config, "source.top_venue.lookback_days", default=21))
        per_page = int(OmegaConf.select(self.config, "source.top_venue.per_page", default=200))
        mailto = OmegaConf.select(self.config, "source.top_venue.mailto", default=None)
        delay_seconds = float(OmegaConf.select(self.config, "source.top_venue.request_delay_seconds", default=0.2))
        from_publication_date = (date.today() - timedelta(days=lookback_days)).isoformat()

        raw_papers: list[OpenAlexWork] = []
        seen_urls: set[str] = set()
        for venue in venues:
            source_ids = self._resolve_source_ids(venue, mailto)
            if not source_ids:
                logger.warning(f"Could not resolve OpenAlex source for venue {venue!r}")
                continue
            works = self._fetch_openalex_works(
                source_ids=source_ids,
                from_publication_date=from_publication_date,
                per_page=per_page,
                mailto=mailto,
            )
            for work in works:
                paper = self._parse_work(work, venue)
                if paper is None or paper.url in seen_urls:
                    continue
                if not self._matches_interest(paper, model_keywords, security_keywords):
                    continue
                seen_urls.add(paper.url)
                raw_papers.append(paper)
            if delay_seconds > 0:
                sleep(delay_seconds)

        logger.info(f"Top-venue monitor selected {len(raw_papers)} recent papers")
        return raw_papers

    def _config_list(self, key: str, default: list[str]) -> list[str]:
        value = OmegaConf.select(self.config, f"source.top_venue.{key}", default=default)
        return [str(item).strip() for item in value if str(item).strip()]

    def _resolve_source_ids(self, venue: str, mailto: str | None) -> list[str]:
        params = {"search": venue, "per-page": "3"}
        if mailto:
            params["mailto"] = mailto
        url = f"{OPENALEX_SOURCES_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "zotero-arxiv-daily/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to resolve OpenAlex source for {venue!r}: {exc}")
            return []

        source_ids = []
        for source in payload.get("results", []):
            source_id = source.get("id")
            if isinstance(source_id, str) and source_id:
                source_ids.append(source_id.removeprefix("https://openalex.org/"))
        return source_ids

    def _fetch_openalex_works(
        self,
        *,
        source_ids: list[str],
        from_publication_date: str,
        per_page: int,
        mailto: str | None,
    ) -> list[dict]:
        source_filter = "|".join(source_ids)
        params = {
            "filter": f"from_publication_date:{from_publication_date},primary_location.source.id:{source_filter}",
            "sort": "publication_date:desc",
            "per-page": str(per_page),
        }
        if mailto:
            params["mailto"] = mailto
        url = f"{OPENALEX_WORKS_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "zotero-arxiv-daily/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to fetch OpenAlex works for source IDs {source_ids}: {exc}")
            return []
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        return results

    def _parse_work(self, work: dict, configured_venue: str) -> OpenAlexWork | None:
        title = str(work.get("title") or "").strip()
        if not title:
            return None

        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))
        authors = self._parse_authors(work)
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = str(source.get("display_name") or configured_venue).strip()
        url = self._best_url(work, primary_location)
        if not url:
            return None
        pdf_url = self._best_pdf_url(primary_location)
        publication_date = work.get("publication_date")
        if isinstance(publication_date, str) and publication_date:
            abstract = f"Venue: {venue}. Publication date: {publication_date}. {abstract}".strip()
        else:
            abstract = f"Venue: {venue}. {abstract}".strip()

        return OpenAlexWork(
            venue=venue,
            title=title,
            authors=authors,
            abstract=abstract or title,
            url=url,
            pdf_url=pdf_url,
            publication_date=publication_date if isinstance(publication_date, str) else None,
        )

    def _reconstruct_abstract(self, inverted_index: object) -> str:
        if not isinstance(inverted_index, dict):
            return ""
        positioned_words: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int):
                    positioned_words.append((position, str(word)))
        positioned_words.sort(key=lambda item: item[0])
        return " ".join(word for _, word in positioned_words)

    def _parse_authors(self, work: dict) -> list[str]:
        authors = []
        for authorship in work.get("authorships") or []:
            author = authorship.get("author") or {}
            name = str(author.get("display_name") or "").strip()
            if name:
                authors.append(name)
        return authors

    def _best_url(self, work: dict, primary_location: dict) -> str | None:
        for key in ("landing_page_url",):
            value = primary_location.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("doi", "id"):
            value = work.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _best_pdf_url(self, primary_location: dict) -> str | None:
        value = primary_location.get("pdf_url")
        if isinstance(value, str) and value:
            return value
        return None

    def _matches_interest(
        self,
        paper: OpenAlexWork,
        model_keywords: list[str],
        security_keywords: list[str],
    ) -> bool:
        text = f"{paper.title}\n{paper.abstract}".lower()
        has_model_keyword = any(keyword.lower() in text for keyword in model_keywords)
        has_security_keyword = any(keyword.lower() in text for keyword in security_keywords)
        return has_model_keyword and has_security_keyword

    def convert_to_paper(self, raw_paper: OpenAlexWork) -> Paper:
        return Paper(
            source=f"top_venue:{raw_paper.venue}",
            title=raw_paper.title,
            authors=raw_paper.authors,
            abstract=raw_paper.abstract,
            url=raw_paper.url,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )
