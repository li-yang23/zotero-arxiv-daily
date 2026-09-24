from dataclasses import dataclass, field
from typing import Literal

from .protocol import Paper
from .topic_clusterer import PaperGroup


@dataclass(frozen=True)
class PaperRequest:
    title: str
    url: str | None
    input_index: int


@dataclass(frozen=True)
class PaperRequestGroup:
    label: str
    papers: list[PaperRequest]


ResolutionStatus = Literal["matched", "metadata_only", "ambiguous", "not_found", "failed"]


@dataclass
class PaperResolution:
    request: PaperRequest
    status: ResolutionStatus
    paper: Paper | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RequestIssue:
    group: str
    title: str
    status: ResolutionStatus
    detail: str


@dataclass
class RequestReport:
    requested_count: int
    groups: list[PaperGroup] = field(default_factory=list)
    issues: list[RequestIssue] = field(default_factory=list)

    @property
    def resolved_count(self) -> int:
        return sum(len(group.papers) for group in self.groups)
