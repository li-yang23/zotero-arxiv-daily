import html
import ipaddress
import json
import re
import socket
import tempfile
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import arxiv
import httpx
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .protocol import Paper
from .request_models import PaperRequest, PaperResolution
from .utils import extract_markdown_from_pdf


_ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|^)(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_SPACE_PATTERN = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(title)).casefold()
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def title_similarity(left: str, right: str) -> float:
    normalized_left = normalize_title(left)
    normalized_right = normalize_title(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    sequence_score = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(sequence_score, token_score)


def _plain_text(value: str | None) -> str:
    if not value:
        return ""
    without_tags = _TAG_PATTERN.sub(" ", html.unescape(value))
    return _SPACE_PATTERN.sub(" ", without_tags).strip()


def _abstract_from_inverted_index(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned_words.append((position, str(word)))
    positioned_words.sort(key=lambda item: item[0])
    return _plain_text(" ".join(word for _, word in positioned_words))


class ScholarlyHTMLParser(HTMLParser):
    BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "button", "li"}
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.links: list[tuple[str, str]] = []
        self.blocks: list[str] = []
        self.identified_blocks: dict[str, str] = {}
        self.json_ld: list[str] = []
        self._block_depth = 0
        self._block_parts: list[str] = []
        self._block_id: str | None = None
        self._anchor_depth = 0
        self._anchor_href: str | None = None
        self._anchor_parts: list[str] = []
        self._json_ld_depth = 0
        self._json_ld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs if value is not None}
        if tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or attributes.get("itemprop") or "").lower()
            content = attributes.get("content")
            if key and content:
                self.meta.setdefault(key, []).append(_plain_text(content))

        if self._block_depth:
            if tag == "br":
                self._block_parts.append(" ")
            elif tag not in self.VOID_TAGS:
                self._block_depth += 1
        elif tag in self.BLOCK_TAGS or (
            tag == "div" and "authorlist" in (attributes.get("class") or "").casefold().split()
        ):
            self._block_depth = 1
            self._block_parts = []
            self._block_id = attributes.get("id")

        if self._anchor_depth:
            if tag == "br":
                self._anchor_parts.append(" ")
            elif tag not in self.VOID_TAGS:
                self._anchor_depth += 1
        elif tag == "a" and attributes.get("href"):
            self._anchor_depth = 1
            self._anchor_href = attributes["href"]
            self._anchor_parts = []

        if tag == "script" and (attributes.get("type") or "").lower() == "application/ld+json":
            self._json_ld_depth = 1
            self._json_ld_parts = []
        elif self._json_ld_depth and tag not in self.VOID_TAGS:
            self._json_ld_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._block_depth:
            self._block_depth -= 1
            if self._block_depth == 0:
                block = _SPACE_PATTERN.sub(" ", "".join(self._block_parts)).strip()
                if block:
                    self.blocks.append(block)
                    if self._block_id:
                        self.identified_blocks[self._block_id] = block
                self._block_parts = []
                self._block_id = None

        if self._anchor_depth:
            self._anchor_depth -= 1
            if self._anchor_depth == 0:
                anchor_text = _SPACE_PATTERN.sub(" ", "".join(self._anchor_parts)).strip()
                if self._anchor_href:
                    self.links.append((anchor_text, self._anchor_href))
                self._anchor_href = None
                self._anchor_parts = []

        if self._json_ld_depth:
            self._json_ld_depth -= 1
            if self._json_ld_depth == 0:
                payload = "".join(self._json_ld_parts).strip()
                if payload:
                    self.json_ld.append(payload)
                self._json_ld_parts = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._block_depth:
            self._block_parts.append(data)
        if self._anchor_depth:
            self._anchor_parts.append(data)
        if self._json_ld_depth:
            self._json_ld_parts.append(data)


@dataclass
class PageMetadata:
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str | None


@dataclass
class _Candidate:
    title: str
    score: float
    source: str
    payload: Any


def _json_ld_articles(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _json_ld_articles(item)
        return
    if not isinstance(value, dict):
        return
    graph = value.get("@graph")
    if graph is not None:
        yield from _json_ld_articles(graph)
    value_type = value.get("@type")
    type_values = value_type if isinstance(value_type, list) else [value_type]
    if any(item in {"Article", "ScholarlyArticle", "TechArticle", "Report"} for item in type_values):
        yield value


def extract_page_metadata(html_text: str, base_url: str, requested_title: str) -> PageMetadata:
    parser = ScholarlyHTMLParser()
    parser.feed(html_text)

    json_articles: list[dict[str, Any]] = []
    for raw_payload in parser.json_ld:
        try:
            json_articles.extend(_json_ld_articles(json.loads(raw_payload)))
        except (json.JSONDecodeError, TypeError):
            continue

    def first_meta(*keys: str) -> str:
        for key in keys:
            values = parser.meta.get(key, [])
            if values and values[0]:
                return values[0]
        return ""

    title = first_meta("citation_title", "dc.title", "og:title", "twitter:title")
    authors = [value for value in parser.meta.get("citation_author", []) if value]
    abstract = first_meta("citation_abstract", "dc.description", "article:description")
    pdf_url = first_meta("citation_pdf_url", "eprints.document_url") or None

    for article in json_articles:
        title = title or _plain_text(str(article.get("headline") or article.get("name") or ""))
        abstract = abstract or _plain_text(str(article.get("abstract") or article.get("description") or ""))
        if not authors:
            raw_authors = article.get("author", [])
            if isinstance(raw_authors, (str, dict)):
                raw_authors = [raw_authors]
            for author in raw_authors:
                if isinstance(author, str):
                    name = author
                elif isinstance(author, dict):
                    name = author.get("name", "")
                else:
                    name = ""
                if name:
                    authors.append(_plain_text(str(name)))
        if not pdf_url:
            encoding = article.get("encoding") or article.get("associatedMedia")
            if isinstance(encoding, dict):
                pdf_url = encoding.get("contentUrl") or encoding.get("url")

    requested_block_index: int | None = None
    for index, block in enumerate(parser.blocks):
        if title_similarity(block, requested_title) >= 0.96:
            requested_block_index = index
            break

    if not authors:
        for anchor_text, href in parser.links:
            if title_similarity(anchor_text, requested_title) < 0.96 or not href.startswith("#"):
                continue
            author_block = parser.identified_blocks.get(href[1:])
            if not author_block:
                continue
            author_text = re.split(r"\s+\d+\s*:", author_block, maxsplit=1)[0]
            author_text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])\d+(?:,\d+)*", "", author_text)
            author_text = _SPACE_PATTERN.sub(" ", author_text).strip(" ,")
            if author_text:
                authors = [author_text]
            break

    if requested_block_index is not None:
        if not authors:
            for block in parser.blocks[requested_block_index + 1: requested_block_index + 4]:
                if block and len(block) <= 2000 and title_similarity(block, requested_title) < 0.8:
                    authors = [block]
                    break
        if not abstract or len(abstract) < 100:
            abstract_parts: list[str] = []
            stop_phrases = (
                "open access media",
                "view more papers",
                "usenix is committed",
                "bibtex",
                "support usenix",
            )
            for block in parser.blocks[requested_block_index + 1: requested_block_index + 12]:
                normalized_block = block.casefold()
                if any(phrase in normalized_block for phrase in stop_phrases):
                    break
                if len(block) >= 100:
                    abstract_parts.append(block)
                    if len(abstract_parts) >= 4:
                        break
                elif abstract_parts:
                    break
            if abstract_parts:
                abstract = "\n\n".join(abstract_parts)

    if not abstract or len(abstract) < 100:
        description = first_meta("description", "og:description", "twitter:description")
        if len(description) >= 100:
            abstract = description

    if not pdf_url:
        for anchor_text, href in parser.links:
            joined_url = urljoin(base_url, href)
            path = urlparse(joined_url).path.casefold()
            text = anchor_text.casefold()
            if path.endswith(".pdf") or " pdf" in f" {text}" or text in {"paper", "download"}:
                pdf_url = joined_url
                break

    if pdf_url:
        pdf_url = urljoin(base_url, str(pdf_url))
    if not title or title_similarity(title, requested_title) < 0.75:
        title = requested_title
    return PageMetadata(
        title=_plain_text(title) or requested_title,
        authors=authors,
        abstract=_plain_text(abstract),
        pdf_url=pdf_url,
    )


class PaperResolver:
    DEFAULT_ALLOWED_HOSTS = [
        "arxiv.org",
        "export.arxiv.org",
        "openreview.net",
        "ndss-symposium.org",
        "usenix.org",
        "ieee-security.org",
        "acm.org",
        "doi.org",
        "crossref.org",
    ]

    def __init__(
        self,
        config: DictConfig,
        *,
        http_client: httpx.Client | None = None,
        arxiv_client: arxiv.Client | None = None,
    ):
        self.config = config
        self.timeout = float(OmegaConf.select(config, "email_requests.http_timeout", default=30))
        self.match_threshold = float(OmegaConf.select(config, "email_requests.title_match_threshold", default=0.92))
        self.ambiguity_margin = float(OmegaConf.select(config, "email_requests.ambiguity_margin", default=0.02))
        self.max_html_bytes = int(OmegaConf.select(config, "email_requests.max_html_bytes", default=5_000_000))
        self.max_pdf_bytes = int(OmegaConf.select(config, "email_requests.max_pdf_bytes", default=50_000_000))
        configured_hosts = OmegaConf.select(config, "email_requests.allowed_link_hosts", default=None)
        self.allowed_hosts = [str(host).casefold() for host in (configured_hosts or self.DEFAULT_ALLOWED_HOSTS)]
        user_agent_email = str(OmegaConf.select(config, "email.receiver", default="") or "")
        self.user_agent = f"zotero-arxiv-daily/1.0 ({user_agent_email})"
        self.http_client = http_client or httpx.Client(
            follow_redirects=True,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
        )
        self._owns_http_client = http_client is None
        self.arxiv_client = arxiv_client or arxiv.Client(page_size=5, delay_seconds=3, num_retries=3)

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def resolve(self, request: PaperRequest) -> PaperResolution:
        page_paper: Paper | None = None
        try:
            if request.url:
                arxiv_id = self._extract_arxiv_id(request.url)
                if arxiv_id:
                    paper = self._lookup_arxiv_id(arxiv_id)
                    if paper is not None:
                        return PaperResolution(request=request, status="matched", paper=paper)
                page_paper = self._paper_from_web_page(request)
                if page_paper and (page_paper.abstract or page_paper.full_text):
                    return PaperResolution(request=request, status="matched", paper=page_paper)

            candidates = self._search_arxiv_candidates(request.title)
            exact_arxiv = next((candidate for candidate in candidates if candidate.score == 1.0), None)
            if exact_arxiv is None:
                candidates.extend(self._search_openalex_candidates(request.title))
            exact_scholarly_match = next(
                (candidate for candidate in candidates if candidate.score == 1.0),
                None,
            )
            if exact_scholarly_match is None:
                candidates.extend(self._search_crossref_candidates(request.title))

            selected, ambiguous_titles = self._select_candidate(candidates)
            if selected is not None:
                paper = self._materialize_candidate(selected)
                status = "matched" if paper.abstract or paper.full_text else "metadata_only"
                detail = None if status == "matched" else "找到了论文元数据，但没有找到公开摘要或 PDF"
                return PaperResolution(request=request, status=status, paper=paper, detail=detail)
            if ambiguous_titles:
                return PaperResolution(
                    request=request,
                    status="ambiguous",
                    detail="检索到多个相近候选：" + "；".join(ambiguous_titles),
                )
            if page_paper is not None:
                return PaperResolution(
                    request=request,
                    status="metadata_only",
                    paper=page_paper,
                    detail="找到了会议页面，但没有找到公开摘要或 PDF",
                )
            return PaperResolution(request=request, status="not_found", detail="没有找到可信的精确标题匹配")
        except Exception as exc:
            logger.warning(f"Failed to resolve requested paper {request.title}: {exc}")
            if page_paper is not None:
                return PaperResolution(
                    request=request,
                    status="metadata_only",
                    paper=page_paper,
                    detail=f"只提取到会议页面元数据；后续检索失败：{exc}",
                )
            return PaperResolution(request=request, status="failed", detail=str(exc))

    def _extract_arxiv_id(self, value: str) -> str | None:
        match = _ARXIV_ID_PATTERN.search(value)
        return match.group("id") if match else None

    def _lookup_arxiv_id(self, paper_id: str) -> Paper | None:
        try:
            results = list(self.arxiv_client.results(arxiv.Search(id_list=[paper_id])))
        except Exception as exc:
            logger.warning(f"arXiv ID lookup failed for {paper_id}: {exc}")
            return None
        return self._paper_from_arxiv(results[0]) if results else None

    def _search_arxiv_candidates(self, title: str) -> list[_Candidate]:
        escaped_title = title.replace('"', " ")
        search = arxiv.Search(
            query=f'ti:"{escaped_title}"',
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(self.arxiv_client.results(search))
        except Exception as exc:
            logger.warning(f"arXiv title lookup failed for {title}: {exc}")
            return []
        return [
            _Candidate(
                title=result.title,
                score=title_similarity(title, result.title),
                source="arxiv",
                payload=result,
            )
            for result in results
        ]

    def _search_openalex_candidates(self, title: str) -> list[_Candidate]:
        params: dict[str, str | int] = {
            "search": title,
            "per-page": 5,
            "select": (
                "id,doi,title,authorships,abstract_inverted_index,"
                "primary_location,best_oa_location"
            ),
        }
        receiver = str(OmegaConf.select(self.config, "email.receiver", default="") or "")
        if receiver:
            params["mailto"] = receiver
        try:
            response = self.http_client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
            items = response.json().get("results", [])
        except Exception as exc:
            logger.warning(f"OpenAlex title lookup failed for {title}: {exc}")
            return []

        candidates = []
        for item in items:
            candidate_title = str(item.get("title") or "")
            if candidate_title:
                candidates.append(
                    _Candidate(
                        title=candidate_title,
                        score=title_similarity(title, candidate_title),
                        source="openalex",
                        payload=item,
                    )
                )
        return candidates

    def _search_crossref_candidates(self, title: str) -> list[_Candidate]:
        params = {
            "query.title": title,
            "rows": 5,
            "select": "DOI,title,author,abstract,URL,link,container-title,published",
        }
        receiver = str(OmegaConf.select(self.config, "email.receiver", default="") or "")
        if receiver:
            params["mailto"] = receiver
        try:
            response = self.http_client.get("https://api.crossref.org/works", params=params)
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
        except Exception as exc:
            logger.warning(f"Crossref title lookup failed for {title}: {exc}")
            return []

        candidates = []
        for item in items:
            item_titles = item.get("title") or []
            candidate_title = str(item_titles[0]) if item_titles else ""
            if candidate_title:
                candidates.append(
                    _Candidate(
                        title=candidate_title,
                        score=title_similarity(title, candidate_title),
                        source="crossref",
                        payload=item,
                    )
                )
        return candidates

    def _select_candidate(self, candidates: list[_Candidate]) -> tuple[_Candidate | None, list[str]]:
        source_priority = {"arxiv": 2, "openalex": 1, "crossref": 0}
        viable = sorted(
            (candidate for candidate in candidates if candidate.score >= self.match_threshold),
            key=lambda candidate: (candidate.score, source_priority.get(candidate.source, -1)),
            reverse=True,
        )
        if not viable:
            return None, []
        top = viable[0]
        for alternative in viable[1:]:
            if normalize_title(alternative.title) == normalize_title(top.title):
                continue
            if top.score - alternative.score <= self.ambiguity_margin:
                return None, [top.title, alternative.title]
            break
        return top, []

    def _materialize_candidate(self, candidate: _Candidate) -> Paper:
        if candidate.source == "arxiv":
            return self._paper_from_arxiv(candidate.payload)
        if candidate.source == "openalex":
            return self._paper_from_openalex(candidate.payload)
        return self._paper_from_crossref(candidate.payload)

    def _paper_from_arxiv(self, result: arxiv.Result) -> Paper:
        full_text = None
        pdf_url = str(result.pdf_url) if result.pdf_url else None
        if pdf_url:
            full_text = self._download_pdf_text(pdf_url)
        return Paper(
            source="arxiv",
            title=_plain_text(result.title),
            authors=[author.name for author in result.authors],
            abstract=_plain_text(result.summary),
            url=str(result.entry_id),
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def _paper_from_openalex(self, item: dict[str, Any]) -> Paper:
        authors = []
        for authorship in item.get("authorships") or []:
            author = authorship.get("author") or {}
            name = str(author.get("display_name") or "").strip()
            if name:
                authors.append(name)

        location = item.get("best_oa_location") or item.get("primary_location") or {}
        url = str(location.get("landing_page_url") or item.get("doi") or item.get("id") or "")
        pdf_url = str(location.get("pdf_url") or "") or None
        if url.startswith("http://arxiv.org/"):
            url = "https://" + url.removeprefix("http://")
        if pdf_url and pdf_url.startswith("http://arxiv.org/"):
            pdf_url = "https://" + pdf_url.removeprefix("http://")
        full_text = self._download_pdf_text(pdf_url) if pdf_url else None
        return Paper(
            source="openalex",
            title=_plain_text(str(item.get("title") or "Untitled paper")),
            authors=authors,
            abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            url=url,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def _paper_from_crossref(self, item: dict[str, Any]) -> Paper:
        titles = item.get("title") or []
        title = _plain_text(str(titles[0])) if titles else "Untitled paper"
        authors = []
        for author in item.get("author") or []:
            name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
            if name:
                authors.append(name)
        doi = item.get("DOI")
        url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else ""))
        pdf_url = None
        for link in item.get("link") or []:
            content_type = str(link.get("content-type", "")).casefold()
            candidate_url = link.get("URL")
            if candidate_url and ("pdf" in content_type or str(candidate_url).casefold().endswith(".pdf")):
                pdf_url = str(candidate_url)
                break
        full_text = self._download_pdf_text(pdf_url) if pdf_url and self._url_is_allowed(pdf_url) else None
        return Paper(
            source="crossref",
            title=title,
            authors=authors,
            abstract=_plain_text(item.get("abstract")),
            url=url,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    def _paper_from_web_page(self, request: PaperRequest) -> Paper | None:
        if not request.url or not self._url_is_allowed(request.url):
            logger.warning(f"Skipping unapproved paper link host: {request.url}")
            return None
        response = self.http_client.get(request.url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").casefold()
        if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
            full_text = self._extract_pdf_bytes(response.content)
            return Paper(
                source=urlparse(request.url).hostname or "web",
                title=request.title,
                authors=[],
                abstract="",
                url=request.url,
                pdf_url=request.url,
                full_text=full_text,
            )
        if len(response.content) > self.max_html_bytes:
            raise ValueError(f"Paper page is larger than {self.max_html_bytes} bytes")
        metadata = extract_page_metadata(response.text, str(response.url), request.title)
        full_text = None
        if metadata.pdf_url and self._url_is_allowed(metadata.pdf_url, parent_url=str(response.url)):
            full_text = self._download_pdf_text(metadata.pdf_url)
        return Paper(
            source=urlparse(str(response.url)).hostname or "web",
            title=metadata.title,
            authors=metadata.authors,
            abstract=metadata.abstract,
            url=request.url,
            pdf_url=metadata.pdf_url,
            full_text=full_text,
        )

    def _download_pdf_text(self, url: str) -> str | None:
        if not self._url_is_allowed(url):
            return None
        try:
            response = self.http_client.get(url)
            response.raise_for_status()
            if len(response.content) > self.max_pdf_bytes:
                raise ValueError(f"PDF is larger than {self.max_pdf_bytes} bytes")
            if not response.content.startswith(b"%PDF") and "pdf" not in response.headers.get("content-type", "").casefold():
                raise ValueError("The paper download did not return a PDF")
            return self._extract_pdf_bytes(response.content)
        except Exception as exc:
            logger.warning(f"Failed to download or extract PDF {url}: {exc}")
            return None

    def _extract_pdf_bytes(self, content: bytes) -> str | None:
        if len(content) > self.max_pdf_bytes:
            raise ValueError(f"PDF is larger than {self.max_pdf_bytes} bytes")
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                temporary_file.write(content)
                path = Path(temporary_file.name)
            return extract_markdown_from_pdf(str(path))
        except Exception as exc:
            logger.warning(f"Failed to extract downloaded PDF: {exc}")
            return None
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _url_is_allowed(self, url: str, parent_url: str | None = None) -> bool:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        hostname = parsed.hostname.casefold().rstrip(".")
        parent_hostname = urlparse(parent_url).hostname.casefold().rstrip(".") if parent_url and urlparse(parent_url).hostname else None
        if parent_hostname and (hostname == parent_hostname or hostname.endswith("." + parent_hostname)):
            return self._host_is_public(hostname)
        if not any(hostname == allowed or hostname.endswith("." + allowed) for allowed in self.allowed_hosts):
            return False
        return self._host_is_public(hostname)

    def _host_is_public(self, hostname: str) -> bool:
        try:
            addresses = {entry[4][0] for entry in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
        except OSError as exc:
            logger.warning(f"Could not resolve link host {hostname}: {exc}")
            return False
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                return False
        return True
