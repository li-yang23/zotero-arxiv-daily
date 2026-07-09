import math
from html import escape

from .protocol import Paper
from .topic_clusterer import PaperGroup


GROUP_WRAPPER_STYLE = "margin: 0 0 32px 0;"
GROUP_HEADING_STYLE = "font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; color: #222; margin: 0 0 8px 0;"
GROUP_SUMMARY_STYLE = "font-family: Arial, sans-serif; font-size: 16px; color: #555; margin: 0 0 16px 0; line-height: 1.55;"
PAPER_SPACING = '<br></br><br>'


def _uses_chinese(language: str | None) -> bool:
    normalized_language = (language or "").strip().lower()
    return "chinese" in normalized_language or "中文" in normalized_language or normalized_language.startswith("zh")


def _email_labels(language: str | None) -> dict[str, str]:
    if _uses_chinese(language):
        return {
            "empty": "今天没有新论文，可以休息一下。",
            "unsubscribe": "如需退订，请从 GitHub Action 设置中移除你的邮箱。",
            "affiliation_unknown": "未知机构",
            "relevance": "相关度",
            "tldr": "论文总结",
            "full_summary": "完整分析",
            "show_full_summary": "展开完整分析",
            "quality": "质量评分",
            "innovation": "创新性",
            "rigor": "严谨性",
            "significance": "重要性",
            "problem": "问题",
            "method": "方法",
            "conclusion": "结论",
            "reviewer_note": "评审备注",
            "api_balance": "API 余额",
        }
    return {
        "empty": "No Papers Today. Take a Rest!",
        "unsubscribe": "To unsubscribe, remove your email in your Github Action setting.",
        "affiliation_unknown": "Unknown Affiliation",
        "relevance": "Relevance",
        "tldr": "TLDR",
        "full_summary": "Full Analysis",
        "show_full_summary": "Show full analysis",
        "quality": "Quality",
        "innovation": "Innovation",
        "rigor": "Rigor",
        "significance": "Significance",
        "problem": "Problem",
        "method": "Method",
        "conclusion": "Conclusion",
        "reviewer_note": "Reviewer Note",
        "api_balance": "API Balance",
    }




framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.5; color: #555;">
__API_BALANCE__
</div>
<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.5; color: #555;">
__UNSUBSCRIBE__
</div>

</body>
</html>
"""

def get_empty_html(labels: dict[str, str]):
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        {empty}
    </td>
  </tr>
  </table>
  """
  return block_template.format(empty=labels["empty"])

def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    labels: dict[str, str],
    affiliations: str = None,
    detailed_summary: str = "",
    quality_review: str = "",
):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 16px; color: #666; padding: 8px 0; line-height: 1.5;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 16px; color: #333; padding: 8px 0; line-height: 1.5;">
            <strong>{relevance_label}:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 16px; color: #333; padding: 8px 0; line-height: 1.5;">
            <strong>{tldr_label}:</strong> {tldr}
        </td>
    </tr>
    {detailed_summary}
    {quality_review}

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 16px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        relevance_label=labels["relevance"],
        tldr_label=labels["tldr"],
        affiliations=affiliations,
        detailed_summary=detailed_summary,
        quality_review=quality_review,
    )


def get_detailed_summary_html(paper: Paper, labels: dict[str, str]) -> str:
    if not paper.detailed_summary:
        return ""
    return f"""
    <tr>
        <td style="font-size: 16px; color: #333; padding: 8px 0; line-height: 1.5;">
            <details>
                <summary style="cursor: pointer; font-weight: bold; color: #2b5c8a;">{labels["show_full_summary"]}</summary>
                <div style="margin-top: 8px; line-height: 1.55; white-space: pre-wrap;">
                    <strong>{labels["full_summary"]}:</strong> {escape(paper.detailed_summary)}
                </div>
            </details>
        </td>
    </tr>
"""


def get_quality_review_html(paper: Paper, labels: dict[str, str]) -> str:
    review = paper.quality_review
    if review is None:
        return ""
    quality_score = round(review.overall_score, 1)
    innovation_score = round(review.innovation_score, 1)
    rigor_score = round(review.rigor_score, 1)
    significance_score = round(review.significance_score, 1)
    return f"""
    <tr>
        <td style="font-size: 16px; color: #333; padding: 8px 0; line-height: 1.5;">
            <strong>{labels["quality"]}:</strong> {quality_score}/10
            ({labels["innovation"]} {innovation_score}, {labels["rigor"]} {rigor_score}, {labels["significance"]} {significance_score})
            <br><strong>{labels["problem"]}:</strong> {escape(review.problem)}
            <br><strong>{labels["method"]}:</strong> {escape(review.method)}
            <br><strong>{labels["conclusion"]}:</strong> {escape(review.conclusion)}
            <br><strong>{labels["reviewer_note"]}:</strong> {escape(review.rationale)}
        </td>
    </tr>
"""

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def _render_paper_html(paper: Paper, labels: dict[str, str]) -> str:
    rate = round(paper.score, 1) if paper.score is not None else 'Unknown'
    author_list = [author for author in paper.authors]
    num_authors = len(author_list)
    if num_authors <= 5:
        authors = ', '.join(author_list)
    else:
        authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
    if paper.affiliations is not None:
        affiliations = paper.affiliations[:5]
        affiliations = ', '.join(affiliations)
        if len(paper.affiliations) > 5:
            affiliations += ', ...'
    else:
        affiliations = labels["affiliation_unknown"]
    return get_block_html(
        escape(paper.title),
        escape(authors),
        rate,
        escape(paper.tldr or ''),
        escape(paper.pdf_url or paper.url),
        labels,
        escape(affiliations),
        get_detailed_summary_html(paper, labels),
        get_quality_review_html(paper, labels),
    )


def get_group_html(label: str, summary: str | None, paper_html: str) -> str:
    escaped_label = escape(label)
    summary_html = f'<div style="{GROUP_SUMMARY_STYLE}">{escape(summary)}</div>' if summary else ''
    return (
        f'<div style="{GROUP_WRAPPER_STYLE}">'
        f'<h2 style="{GROUP_HEADING_STYLE}">{escaped_label}</h2>'
        f'{summary_html}'
        f'{paper_html}'
        '</div>'
    )


def _render_api_balance(api_balance: str | None, labels: dict[str, str]) -> str:
    if not api_balance:
        return ""
    return f'<div><strong>{labels["api_balance"]}:</strong> {escape(api_balance)}</div>'


def render_email(groups:list[PaperGroup], language: str | None = None, api_balance: str | None = None) -> str:
    labels = _email_labels(language)
    api_balance_html = _render_api_balance(api_balance, labels)
    if len(groups) == 0 :
        return (
            framework
            .replace('__CONTENT__', get_empty_html(labels))
            .replace('__API_BALANCE__', api_balance_html)
            .replace('__UNSUBSCRIBE__', labels["unsubscribe"])
        )

    rendered_groups = []
    for group in groups:
        paper_parts = [_render_paper_html(paper, labels) for paper in group.papers]
        paper_html = '<br>' + PAPER_SPACING.join(paper_parts) + '</br>'
        rendered_groups.append(get_group_html(group.label, group.summary, paper_html))

    content = '<br>' + PAPER_SPACING.join(rendered_groups) + '</br>'
    return (
        framework
        .replace('__CONTENT__', content)
        .replace('__API_BALANCE__', api_balance_html)
        .replace('__UNSUBSCRIBE__', labels["unsubscribe"])
    )
