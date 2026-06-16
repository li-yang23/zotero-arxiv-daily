from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
from .llm_utils import iter_generation_kwargs
RawPaperItem = TypeVar('RawPaperItem')

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
        text_tokens = enc.encode(text)
        return enc.decode(text_tokens[:max_tokens])
    except Exception as e:
        logger.warning(f"Failed to load tokenizer, falling back to character truncation: {e}")
        return text[:max_tokens * 4]


@dataclass
class QualityReview:
    problem: str
    method: str
    conclusion: str
    innovation_score: float
    rigor_score: float
    significance_score: float
    overall_score: float
    rationale: str


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    detailed_summary: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    quality_review: Optional[QualityReview] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = (
            f"Given the following information of a paper, write a precise and concise research digest in {lang}. "
            "Do not copy the source abstract verbatim. Infer only what the paper text supports; if a point is unclear, say so briefly. "
            "Return JSON only with exactly these two string keys: concise_summary and detailed_summary.\n\n"
            "concise_summary should be one compact paragraph, preferably 4-6 sentences, answering these questions when possible: "
            "1) what problem the paper studies; 2) why the problem is worth studying; "
            "3) what existing methods roughly do and how far they get; "
            "4) why the problem still needs this paper; 5) how this paper solves it.\n\n"
            "detailed_summary should be a compact but complete analysis, preferably 8-12 short sentences or 4-6 dense bullet-like clauses, "
            "covering all of these questions when possible: "
            "1) what problem is studied; 2) why it matters; 3) what prior methods do and their current progress; "
            "4) why further work is needed; 5) how this paper solves it; "
            "6) what effect the paper claims, distinguishing solved, alleviated, improved, or analyzed; "
            "7) why the design should achieve that claimed effect; "
            "8) what experiments evaluate, and what dimensions should be evaluated for this problem; "
            "9) whether the results support the claimed effect; "
            "10) whether the problem still needs further research.\n\n"
        )
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        prompt = truncate_text_by_tokens(prompt, 4000)
        
        last_error = None
        for generation_kwargs in iter_generation_kwargs(llm_params):
            try:
                response = openai_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful research assistant writing high-density paper digests for a researcher. "
                                f"Write all free-text fields in {lang}. Return valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    **generation_kwargs
                )
                content = response.choices[0].message.content
                payload = json.loads(content or "{}")
                concise_summary = str(payload["concise_summary"]).strip()
                detailed_summary = str(payload["detailed_summary"]).strip()
                if not concise_summary or not detailed_summary:
                    raise ValueError("summary fields must be non-empty")
                self.detailed_summary = detailed_summary
                return concise_summary
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to generate summary of {self.url} with model {generation_kwargs.get('model')}: {e}")
        raise last_error or RuntimeError("No LLM model configured")
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self._fallback_tldr(llm_params)
            self.tldr = tldr
            self.detailed_summary = self.abstract
            return tldr

    def _fallback_tldr(self, llm_params: dict) -> str:
        lang = str(llm_params.get('language', 'English'))
        normalized_language = lang.strip().lower()
        if (
            "chinese" in normalized_language
            or "中文" in normalized_language
            or normalized_language.startswith("zh")
        ):
            return "中文总结生成失败，请打开论文链接查看原文。"
        return self.abstract

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            prompt = truncate_text_by_tokens(prompt, 2000)
            last_error = None
            for generation_kwargs in iter_generation_kwargs(llm_params):
                try:
                    affiliations = openai_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        **generation_kwargs
                    )
                    affiliations = affiliations.choices[0].message.content
                    affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                    affiliations = json.loads(affiliations)
                    affiliations = list(set(affiliations))
                    affiliations = [str(a) for a in affiliations]
                    return affiliations
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to generate affiliations of {self.url} with model {generation_kwargs.get('model')}: {e}")
            raise last_error or RuntimeError("No LLM model configured")
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
