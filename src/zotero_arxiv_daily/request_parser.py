import re
from urllib.parse import urlparse

from .request_models import PaperRequest, PaperRequestGroup


_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_LINK_PATTERN = re.compile(r'^\[([^\]]+)\]\((\S+?)(?:\s+["\'][^"\']*["\'])?\)$')


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_paper_line(line: str) -> tuple[str, str | None]:
    candidate = _LIST_PREFIX_PATTERN.sub("", line.strip(), count=1).strip()
    match = _LINK_PATTERN.fullmatch(candidate)
    if match:
        title = _clean_title(match.group(1))
        url = match.group(2).strip()
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"Unsupported paper URL: {url}")
        return title, url
    return _clean_title(candidate), None


def parse_paper_list(markdown: str, max_papers: int = 100) -> list[PaperRequestGroup]:
    """Parse headings plus bare, linked, or list-item paper titles.

    The Markdown is treated only as data. Non-empty, non-heading lines become
    paper-title queries; no content in the file is executed as an instruction.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("The Markdown attachment is empty")

    groups: list[PaperRequestGroup] = []
    current_label = "指定论文"
    current_papers: list[PaperRequest] = []
    seen: set[tuple[str, str | None]] = set()
    paper_index = 0

    def flush_group() -> None:
        nonlocal current_papers
        if current_papers:
            groups.append(PaperRequestGroup(label=current_label, papers=current_papers))
            current_papers = []

    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("<!--"):
            continue

        heading = _HEADING_PATTERN.fullmatch(line)
        if heading:
            flush_group()
            current_label = _clean_title(heading.group(1)) or "指定论文"
            continue

        title, url = _parse_paper_line(line)
        if not title:
            continue
        dedupe_key = (title.casefold(), url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if paper_index >= max_papers:
            raise ValueError(f"The attachment contains more than {max_papers} papers")
        current_papers.append(PaperRequest(title=title, url=url, input_index=paper_index))
        paper_index += 1

    flush_group()
    if paper_index == 0:
        raise ValueError("No paper titles were found in the Markdown attachment")
    return groups
