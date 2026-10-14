"""Exact-venue, paginated metadata feeds for the independent daily monitor."""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
import re
from time import sleep
from urllib.parse import quote, urljoin, urlparse

import httpx


@dataclass(frozen=True)
class Venue:
    name: str
    provider: str
    identifier: str


VENUES = (
    Venue("ICLR", "openreview", "ICLR.cc/{year}/Conference"),
    Venue("ICML", "openreview", "ICML.cc/{year}/Conference"),
    Venue("NeurIPS", "openreview", "NeurIPS.cc/{year}/Conference"),
    Venue("NDSS", "dblp", "conf/ndss"),
    Venue("USENIX Security", "dblp", "conf/uss"),
    Venue("S&P", "dblp", "conf/sp"),
    Venue("CCS", "dblp", "conf/ccs"),
    Venue("TIFS", "crossref", "1556-6013"),
    Venue("TDSC", "crossref", "1545-5971"),
    Venue("PAMI", "crossref", "0162-8828"),
    Venue("JMLR", "jmlr", "https://www.jmlr.org/papers/v{volume}/"),
    Venue("Artificial Intelligence", "crossref", "0004-3702"),
    Venue("Nature", "crossref", "0028-0836"),
    Venue("Science", "crossref", "0036-8075"),
)


@dataclass
class VenuePaper:
    venue: str
    identifier: str
    title: str
    url: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    doi: str = ""
    affiliations: list[str] = field(default_factory=list)
    pdf_url: str = ""

    def to_dict(self):
        return asdict(self)


def plain(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(str(value or "")))).strip()


def crossref_paper(item, venue):
    dates = item.get("published", {}).get("date-parts", [[]])[0]
    authors = item.get("author", [])
    doi = item.get("DOI", "").lower()
    return VenuePaper(
        venue, doi, plain(next(iter(item.get("title", [])), "")),
        f"https://doi.org/{doi}",
        authors=[plain(f"{a.get('given', '')} {a.get('family', '')}") for a in authors],
        abstract=plain(item.get("abstract")),
        published="-".join(f"{part:02d}" for part in dates), doi=doi,
        affiliations=list(dict.fromkeys(plain(v.get("name")) for a in authors
                                      for v in a.get("affiliation", []) if v.get("name"))),
        pdf_url=next((v["URL"] for v in item.get("link", [])
                      if v.get("content-type") == "application/pdf" and v.get("URL")), ""),
    )


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.parts = []
        self.title = ""
        self.in_title = False
        self.title_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "p" and "title" in dict(attrs).get("class", "").split():
            self.in_title = True
            self.title_parts = []
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.parts = []

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.href:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.in_title:
            self.title = plain(" ".join(self.title_parts))
            self.in_title = False
        if tag == "a" and self.href:
            label = plain(" ".join(self.parts))
            self.links.append((self.href, self.title or label))
            self.href = None


class VenueSources:
    ALLOWED_HOSTS = ("openreview.net", "dblp.org", "dblp.uni-trier.de", "crossref.org", "doi.org",
                     "jmlr.org", "ndss-symposium.org", "usenix.org", "ieee.org", "ieee-security.org",
                     "acm.org", "nature.com", "science.org", "sciencedirect.com", "elsevier.com",
                     "proceedings.mlr.press", "papers.nips.cc", "papers.neurips.cc")

    def __init__(self, client, *, lookback_days=90, delay=0.3):
        self.client = client
        self.lookback_days = lookback_days
        self.delay = delay
        self.warnings = []

    def get(self, url, **kwargs):
        for attempt in range(3):
            try:
                current = str(url)
                for _ in range(8):
                    parsed = urlparse(current)
                    host = (parsed.hostname or "").lower()
                    if parsed.scheme != "https" or not any(
                        host == allowed or host.endswith("." + allowed) for allowed in self.ALLOWED_HOSTS
                    ):
                        raise ValueError("Unapproved metadata host")
                    response = self.client.get(current, follow_redirects=False, **kwargs)
                    if response.is_redirect:
                        current = urljoin(str(response.url), response.headers["location"])
                        continue
                    break
                else:
                    raise ValueError("Too many metadata redirects")
                response.raise_for_status()
                if self.delay:
                    sleep(self.delay)
                return response
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            sleep(2 ** attempt)
        raise RuntimeError("Metadata request exhausted retries")

    def fetch(self, venue, today, last_success=None):
        if venue.provider == "crossref":
            return self.crossref(venue, today, last_success)
        # Keep the preceding edition through the year boundary to catch late indexing.
        years = [today.year - 1, today.year] if today.month <= 3 else [today.year]
        papers = []
        for year in years:
            try:
                papers.extend(getattr(self, venue.provider)(venue, year))
            except httpx.HTTPError as exc:
                streams = {"ICLR": "conf/iclr", "ICML": "conf/icml", "NeurIPS": "conf/nips"}
                if venue.provider != "openreview":
                    raise
                fallback = self.dblp(Venue(venue.name, "dblp", streams[venue.name]), year)
                if not fallback:
                    raise ValueError(f"OpenReview unavailable and DBLP has no {venue.name} {year} papers") from exc
                self.warnings.append(f"{venue.name} {year}: OpenReview 不可用，已改查 DBLP；收录可能延迟。")
                papers.extend(fallback)
        return papers

    def openreview(self, venue, year):
        venue_id = venue.identifier.format(year=year)
        papers, offset, page_size = [], 0, 1000
        while True:
            payload = self.get("https://api2.openreview.net/notes", params={
                "content.venueid": venue_id, "limit": page_size, "offset": offset,
                "sort": "id:asc",
            }).json()
            notes = payload["notes"]
            for note in notes:
                def value(key, default=""):
                    v = note.get("content", {}).get(key, default)
                    return v.get("value", default) if isinstance(v, dict) else v

                # Submitted, rejected and withdrawn papers have different venue IDs.
                if value("venueid") != venue_id:
                    continue
                stamp = note.get("pdate")
                papers.append(VenuePaper(
                    venue.name, note["id"], plain(value("title")),
                    f"https://openreview.net/forum?id={note['id']}",
                    authors=value("authors", []), abstract=plain(value("abstract")),
                    published=datetime.fromtimestamp(stamp / 1000, timezone.utc).date().isoformat()
                    if stamp else str(year),
                    pdf_url=urljoin("https://openreview.net", value("pdf")) if value("pdf") else "",
                ))
            offset += len(notes)
            if not notes or offset >= payload.get("count", offset + 1):
                break
            if len(notes) < page_size and "count" not in payload:
                break
        return papers

    def dblp(self, venue, year):
        papers, offset, page_size = [], 0, 1000
        while True:
            payload = self.get("https://dblp.org/search/publ/api", params={
                "q": f"stream:{venue.identifier}: year:{year}:",
                "format": "json", "h": page_size, "f": offset,
            }).json()["result"]["hits"]
            hits = payload.get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]
            for hit in hits:
                info = hit["info"]
                key = info.get("key", "")
                if not key.startswith(venue.identifier + "/") or str(info.get("year")) != str(year):
                    continue
                if info.get("type") != "Conference and Workshop Papers":
                    continue
                authors = info.get("authors", {}).get("author", [])
                if not isinstance(authors, list):
                    authors = [authors]
                link = info.get("ee") or info.get("url")
                if isinstance(link, list):
                    link = link[0]
                if link and link.startswith("http://"):
                    link = "https://" + link.removeprefix("http://")
                doi = info.get("doi", "")
                if not doi and link and urlparse(link).hostname in {"doi.org", "dx.doi.org"}:
                    doi = urlparse(link).path.lstrip("/")
                papers.append(VenuePaper(
                    venue.name, key, plain(info.get("title")), link or f"https://dblp.org/rec/{key}",
                    authors=[a.get("text", "") if isinstance(a, dict) else a for a in authors],
                    published=str(year), doi=doi.lower(),
                ))
            offset += len(hits)
            if offset >= int(payload["@total"]):
                break
            if not hits:
                raise ValueError(f"Incomplete DBLP pagination for {venue.name}")
        return papers

    def crossref(self, venue, today, last_success):
        start = (date.fromisoformat(last_success) - timedelta(days=3)) if last_success else (
            today - timedelta(days=self.lookback_days))
        field = "index" if last_success else "pub"
        filters = f"type:journal-article,from-{field}-date:{start},until-{field}-date:{today}"
        cursor, papers, page_size, seen = "*", [], 200, set()
        while True:
            message = self.get(f"https://api.crossref.org/journals/{venue.identifier}/works", params={
                "filter": filters, "rows": page_size, "cursor": cursor,
            }).json()["message"]
            items = message["items"]
            for item in items:
                # The journals endpoint can include both print and electronic ISSNs.
                if not item.get("DOI") or not item.get("title"):
                    continue
                papers.append(crossref_paper(item, venue.name))
            if len(items) < page_size:
                break
            identifiers = {item["DOI"] for item in items}
            if identifiers <= seen:
                raise ValueError(f"Repeated Crossref page for {venue.name}")
            seen.update(identifiers)
            cursor = message["next-cursor"]
        return papers

    def jmlr(self, venue, year):
        url = venue.identifier.format(volume=year - 1999)
        parser = Links()
        parser.feed(self.get(url).text)
        papers = []
        for href, title in parser.links:
            link = urljoin(url, href)
            if re.search(rf"/papers/v{year - 1999}/[^/]+\.html$", link) and title:
                papers.append(VenuePaper(venue.name, link, title, link, published=str(year)))
        if not papers:
            raise ValueError(f"JMLR volume page contained no paper links: {url}")
        return papers

    def enrich(self, paper):
        if paper.doi and not paper.abstract:
            try:
                metadata = crossref_paper(self.get(
                    "https://api.crossref.org/works/" + quote(paper.doi, safe="")
                ).json()["message"], paper.venue)
                paper.abstract = metadata.abstract
                paper.affiliations = metadata.affiliations
                paper.pdf_url = paper.pdf_url or metadata.pdf_url
            except (httpx.HTTPError, ValueError, KeyError):
                pass  # Publisher landing pages are the next source for missing metadata.
        if not paper.abstract:
            from .paper_resolver import extract_page_metadata, ScholarlyHTMLParser, title_similarity

            response = self.get(paper.url)
            if len(response.content) > 5_000_000 or "pdf" in response.headers.get("content-type", ""):
                raise ValueError("No usable abstract page")
            parser = ScholarlyHTMLParser()
            parser.feed(response.text)
            page_title = next(iter(parser.meta.get("citation_title", parser.meta.get("dc.title", []))), "")
            if page_title and title_similarity(page_title, paper.title) < 0.9:
                raise ValueError("Publisher page title does not match the paper")
            meta = extract_page_metadata(response.text, str(response.url), paper.title)
            paper.abstract = meta.abstract
            paper.authors = paper.authors or meta.authors
            paper.pdf_url = paper.pdf_url or meta.pdf_url or ""
            paper.affiliations = list(dict.fromkeys(parser.meta.get("citation_author_institution", [])))
        return paper
