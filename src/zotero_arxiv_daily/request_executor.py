from loguru import logger
from omegaconf import DictConfig
from openai import OpenAI

from .paper_resolver import PaperResolver
from .quality_reviewer import QualityReviewer
from .request_models import PaperRequestGroup, RequestIssue, RequestReport
from .topic_clusterer import PaperGroup


class RequestExecutor:
    def __init__(
        self,
        config: DictConfig,
        *,
        resolver: PaperResolver | None = None,
        openai_client: OpenAI | None = None,
        quality_reviewer: QualityReviewer | None = None,
    ):
        self.config = config
        self.openai_client = openai_client or OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
            timeout=float(config.llm.api.timeout),
            max_retries=int(config.llm.api.max_retries),
        )
        self.resolver = resolver or PaperResolver(config)
        self._owns_resolver = resolver is None
        self.quality_reviewer = quality_reviewer or QualityReviewer(self.openai_client, config)

    def close(self) -> None:
        if self._owns_resolver:
            self.resolver.close()

    def process(self, request_groups: list[PaperRequestGroup]) -> RequestReport:
        requested_count = sum(len(group.papers) for group in request_groups)
        report = RequestReport(requested_count=requested_count)

        for request_group in request_groups:
            resolved_papers = []
            for request in request_group.papers:
                logger.info(f"Resolving requested paper {request.input_index + 1}/{requested_count}: {request.title}")
                resolution = self.resolver.resolve(request)
                if resolution.paper is None:
                    report.issues.append(
                        RequestIssue(
                            group=request_group.label,
                            title=request.title,
                            status=resolution.status,
                            detail=resolution.detail or "论文处理失败",
                        )
                    )
                    continue

                paper = resolution.paper
                if paper.abstract or paper.full_text:
                    review = self.quality_reviewer.review_paper(paper)
                    if review is not None:
                        paper.quality_review = review
                    paper.generate_tldr(self.openai_client, self.config.llm)
                    if paper.full_text:
                        paper.generate_affiliations(self.openai_client, self.config.llm)
                else:
                    paper.tldr = "未找到公开摘要或 PDF，暂时无法生成论文总结。"

                if resolution.status != "matched":
                    report.issues.append(
                        RequestIssue(
                            group=request_group.label,
                            title=request.title,
                            status=resolution.status,
                            detail=resolution.detail or "只提取到部分论文信息",
                        )
                    )
                resolved_papers.append(paper)

            if resolved_papers:
                report.groups.append(
                    PaperGroup(label=request_group.label, summary=None, papers=resolved_papers)
                )
        return report
