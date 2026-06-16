import json
from typing import Any

import httpx
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI, OpenAIError

from .protocol import Paper, QualityReview, truncate_text_by_tokens
from .llm_utils import iter_generation_kwargs


class QualityReviewer:
    def __init__(self, openai_client: OpenAI, config: DictConfig):
        self.openai_client = openai_client
        self.config = config
        self.llm_params = config.llm
        self.min_score = float(OmegaConf.select(config, "quality_filter.min_score", default=7.0))

    def filter_high_quality(self, papers: list[Paper]) -> list[Paper]:
        selected = []
        for paper in papers:
            review = self.review_paper(paper)
            if review is None:
                logger.info(f"Filtered out {paper.title}: quality review failed")
                continue
            paper.quality_review = review
            if review.overall_score >= self.min_score:
                selected.append(paper)
            else:
                logger.info(
                    f"Filtered out {paper.title}: quality score {review.overall_score:.1f} < {self.min_score:.1f}"
                )
        return selected

    def review_paper(self, paper: Paper) -> QualityReview | None:
        try:
            payload = self._build_payload(paper)
            for generation_kwargs in iter_generation_kwargs(self.llm_params):
                try:
                    response = self.openai_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        response_format={"type": "json_object"},
                        **generation_kwargs,
                    )
                    content = response.choices[0].message.content
                    return self._parse_review(content)
                except (OpenAIError, httpx.HTTPError) as exc:
                    logger.warning(f"Quality review request failed for {paper.url} with model {generation_kwargs.get('model')}: {exc}")
                except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    logger.warning(f"Invalid quality review output for {paper.url} with model {generation_kwargs.get('model')}: {exc}")
        except Exception as exc:
            logger.warning(f"Quality review failed for {paper.url}: {exc}")
        return None

    def _build_payload(self, paper: Paper) -> dict[str, str | None]:
        text = ""
        if paper.full_text:
            text = paper.full_text
        elif paper.abstract:
            text = paper.abstract

        if text:
            text = truncate_text_by_tokens(text, 8000)

        return {
            "title": paper.title,
            "abstract": paper.abstract,
            "paper_text": text,
        }

    def _system_prompt(self) -> str:
        lang = self.llm_params.get("language", "English")
        return (
            "You are a senior program committee reviewer for top AI conferences such as NeurIPS, ICML, "
            "ICLR, ACL, CVPR, and AAAI. Evaluate the paper from its abstract and available full-text preview. "
            "Extract the problem the paper tries to solve, the proposed method, and the core conclusion. "
            "Score the paper on a 0-10 scale for innovation, rigor, and significance. Then give an overall "
            "quality score on the same 0-10 scale. Be selective: a score of 7 means a clearly solid paper "
            "worthy of top-conference attention; 8+ means strong; 9+ means exceptional. Penalize incremental "
            "ideas, weak experimental evidence, missing baselines, unclear math, or limited impact. "
            f"Write all free-text fields in {lang}. Return JSON only with exactly these keys: "
            "problem, method, conclusion, innovation_score, rigor_score, significance_score, overall_score, rationale."
        )

    def _parse_review(self, content: str | None) -> QualityReview:
        if not content:
            raise ValueError("empty response")
        payload = json.loads(content)
        required = [
            "problem",
            "method",
            "conclusion",
            "innovation_score",
            "rigor_score",
            "significance_score",
            "overall_score",
            "rationale",
        ]
        for key in required:
            if key not in payload:
                raise ValueError(f"missing key: {key}")

        review = QualityReview(
            problem=str(payload["problem"]).strip(),
            method=str(payload["method"]).strip(),
            conclusion=str(payload["conclusion"]).strip(),
            innovation_score=self._score(payload["innovation_score"]),
            rigor_score=self._score(payload["rigor_score"]),
            significance_score=self._score(payload["significance_score"]),
            overall_score=self._score(payload["overall_score"]),
            rationale=str(payload["rationale"]).strip(),
        )
        self._validate_non_empty_text(review)
        return review

    def _score(self, value: Any) -> float:
        score = float(value)
        if score < 0 or score > 10:
            raise ValueError(f"score out of range: {score}")
        return score

    def _validate_non_empty_text(self, review: QualityReview) -> None:
        for key in ("problem", "method", "conclusion", "rationale"):
            if not getattr(review, key):
                raise ValueError(f"{key} must be non-empty")
