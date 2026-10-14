"""Exact-venue, paginated metadata feeds for the independent daily monitor."""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
import re
from time import sleep
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class Venue:
    name: str
    provider: str
    identifier: str


VENUES = (
    Venue("ICLR", "conference", "ICLR.cc/{year}/Conference"),
    Venue("ICML", "conference", "ICML.cc/{year}/Conference"),
    Venue("NeurIPS", "conference", "NeurIPS.cc/{year}/Conference"),
    Venue("NDSS", "conference", "conf/ndss"),
    Venue("USENIX Security", "conference", "conf/uss"),
    Venue("S&P", "conference", "conf/sp"),
    Venue("CCS", "conference", "conf/ccs"),
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
        if tag == "dt" or (tag == "p" and "title" in dict(attrs).get("class", "").split()):
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
        if tag in {"p", "dt"} and self.in_title:
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
                     "proceedings.mlr.press", "papers.nips.cc", "papers.neurips.cc", "iclr.cc",
                     "icml.cc", "neurips.cc", "sigsac.org", "semanticscholar.org")

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
                if exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("retry-after", "60")
                    if not retry_after.isdigit() or int(retry_after) > 60:
                        raise
                    sleep(int(retry_after))
            sleep(2 ** attempt)
        raise RuntimeError("Metadata request exhausted retries")

    def fetch(self, venue, today, last_success=None):
        if venue.provider == "crossref":
            return self.crossref(venue, today, last_success)
        # Keep the preceding edition through the year boundary to catch late indexing.
        years = [today.year - 1, today.year] if today.month <= 3 else [today.year]
        papers = []
        for year in years:
            papers.extend(getattr(self, venue.provider)(venue, year))
        return papers

    def conference(self, venue, year):
        if venue.name in {"ICLR", "ICML", "NeurIPS"}:
            try:
                return self.virtual_conference(venue, year)
            except (httpx.HTTPError, ValueError, KeyError):
                papers = self.openreview(venue, year)
                self.warnings.append(f"{venue.name}: 官方目录不可用，已改查 OpenReview 录用列表。")
                return papers
        urls = {
            "NDSS": f"https://www.ndss-symposium.org/ndss{year}/accepted-papers/",
            "USENIX Security": f"https://www.usenix.org/conference/usenixsecurity{year % 100:02d}/technical-sessions",
            "S&P": f"https://www.ieee-security.org/TC/SP{year}/accepted-papers.html",
            "CCS": f"https://www.sigsac.org/ccs/CCS{year}/program/accepted-papers.html",
        }
        url = urls[venue.name]
        try:
            soup = BeautifulSoup(self.get(url).text, "html.parser")
        except httpx.HTTPError:
            if venue.name != "USENIX Security":
                raise
            papers = self.semantic_scholar(venue, year)
            if not papers:
                raise ValueError("USENIX 官方目录不可用，备用索引尚无当年论文")
            self.warnings.append("USENIX Security: 官方站限制访问，已使用 Semantic Scholar；收录可能延迟。")
            return papers
        papers = []
        if venue.name == "NDSS":
            for entry in soup.select(".pt-cv-content-item"):
                title = entry.select_one(".pt-cv-title a")
                if title:
                    text = entry.select_one(".pt-cv-ctf-display_authors")
                    authors, institutions = split_authors(text.get_text(" ", strip=True) if text else "")
                    link = urljoin(url, title["href"])
                    papers.append(VenuePaper(venue.name, link, title.get_text(" ", strip=True), link,
                                              authors=authors, affiliations=institutions, published=str(year)))
        elif venue.name == "CCS":
            for index, row in enumerate(soup.select("table tr")):
                cells = row.find_all("td", recursive=False)
                if len(cells) != 2:
                    continue
                title = cells[0].get_text(" ", strip=True)
                authors, institutions = split_authors(cells[1].get_text("; ", strip=True))
                papers.append(VenuePaper(venue.name, f"{year}:{title}", title, url,
                                          authors=authors, affiliations=institutions, published=str(year)))
        elif venue.name == "S&P":
            for entry in soup.select(".list-group-item"):
                title = entry.select_one("b a[data-toggle='collapse']")
                if not title:
                    continue
                authors = entry.select_one(".authorlist")
                if authors:
                    before, after, in_affiliations = [], [], False
                    for node in authors.children:
                        if getattr(node, "name", None) == "br":
                            in_affiliations = True
                        elif getattr(node, "name", None) != "sup":
                            (after if in_affiliations else before).append(
                                node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node))
                    sections = [plain("".join(before)), plain("".join(after))]
                else:
                    sections = []
                link = urljoin(url, title["href"])
                papers.append(VenuePaper(venue.name, f"{year}:{title.get_text(strip=True)}",
                                          title.get_text(" ", strip=True), link,
                                          authors=[sections[0]] if sections else [],
                                          affiliations=[" ".join(sections[1:])] if len(sections) > 1 else [],
                                          published=str(year)))
        else:
            for title in soup.select("h2 a[href*='/presentation/'], h3 a[href*='/presentation/']"):
                link = urljoin(url, title["href"])
                papers.append(VenuePaper(venue.name, link, title.get_text(" ", strip=True), link, published=str(year)))
        if not papers:
            raise ValueError(f"官方目录没有解析出论文，需检查页面结构：{url}")
        return papers

    def virtual_conference(self, venue, year):
        name = venue.name.lower()
        base = f"https://{name}.cc"
        data_url = f"{base}/static/virtual/data/{name}-{year}-orals-posters.json"
        data = self.get(data_url).json()
        items = data["results"]
        if data.get("next") or int(data.get("count", len(items))) > len(items):
            raise ValueError("Official conference export is incomplete")
        try:
            abstracts = self.get(f"{base}/static/virtual/data/{name}-{year}-abstracts.json").json()
        except (httpx.HTTPError, ValueError):
            abstracts = {}
            self.warnings.append(f"{venue.name} {year}: 录用目录可用，摘要暂不可用；候选保留重试。")
        papers = []
        for item in items:
            if "workshop" in str(item.get("sourceurl", "")).lower():
                continue
            decision = str(item.get("decision", "")).lower()
            if "reject" in decision or "withdraw" in decision:
                continue
            authors = item.get("authors", [])
            papers.append(VenuePaper(
                venue.name, str(item["id"]), plain(item["name"]),
                urljoin(base, item.get("virtualsite_url") or f"/virtual/{year}/poster/{item['id']}"),
                authors=[a["fullname"] for a in authors], abstract=plain(abstracts.get(str(item["id"]), "")),
                published=str(year),
                affiliations=list(dict.fromkeys(a["institution"] for a in authors if a.get("institution"))),
                pdf_url=item.get("paper_pdf_url") or "",
            ))
        return papers

    def semantic_scholar(self, venue, year):
        token, papers, seen = None, [], set()
        while True:
            params = {"venue": "USENIX Security Symposium", "year": str(year),
                      "fields": "title,abstract,authors,venue,publicationDate,externalIds,url,openAccessPdf"}
            if token:
                params["token"] = token
            data = self.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=params).json()
            for item in data["data"]:
                if item.get("venue", "").casefold() not in {"usenix security symposium", "usenix security", "usenix security symp."}:
                    continue
                doi = item.get("externalIds", {}).get("DOI", "")
                papers.append(VenuePaper(venue.name, item["paperId"], item["title"], item["url"],
                                          authors=[a["name"] for a in item.get("authors", [])],
                                          abstract=item.get("abstract") or "", doi=doi.lower(),
                                          published=item.get("publicationDate") or str(year),
                                          pdf_url=(item.get("openAccessPdf") or {}).get("url", "")))
            token = data.get("token")
            if not token:
                break
            if token in seen:
                raise ValueError("Repeated Semantic Scholar page")
            seen.add(token)
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
        # Continuous-volume journals may give every paper only a year (January 1).
        # Index timestamps detect their actual arrivals; a publication-year floor excludes backfiles.
        year_floor = (today - timedelta(days=self.lookback_days)).year
        filters = (f"type:journal-article,from-index-date:{start},until-index-date:{today},"
                   f"from-pub-date:{year_floor}-01-01")
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
        if not paper.abstract and paper.venue in {"S&P", "CCS"} and not paper.doi:
            from .paper_resolver import title_similarity

            data = self.get("https://api.crossref.org/works", params={
                "query.bibliographic": paper.title, "rows": 3,
            }).json()["message"]["items"]
            matches = [item for item in data if item.get("title") and
                       title_similarity(item["title"][0], paper.title) >= 0.96]
            if matches:
                metadata = crossref_paper(matches[0], paper.venue)
                paper.doi = metadata.doi
                paper.abstract = metadata.abstract
                paper.url = metadata.url
                paper.affiliations = paper.affiliations or metadata.affiliations
                paper.pdf_url = metadata.pdf_url
            else:
                raise ValueError("录用目录尚无可核实的论文摘要，等待索引补齐")
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
            paper.affiliations = paper.affiliations or list(dict.fromkeys(parser.meta.get("citation_author_institution", [])))
        return paper


def split_authors(text):
    pairs = re.findall(r"([^();]+)\(([^()]+)\)", text)
    if pairs:
        return [name.strip(" ,;") for name, _ in pairs], list(dict.fromkeys(org.strip() for _, org in pairs))
    return ([text] if text else []), []
