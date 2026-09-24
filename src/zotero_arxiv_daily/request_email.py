from html import escape

from .construct_email import PAPER_SPACING, _email_labels, _render_paper_html, framework, get_group_html
from .request_models import RequestIssue, RequestReport


_STATUS_LABELS_ZH = {
    "metadata_only": "信息不完整",
    "ambiguous": "匹配有歧义",
    "not_found": "未找到",
    "failed": "处理失败",
}
_STATUS_LABELS_EN = {
    "metadata_only": "Incomplete metadata",
    "ambiguous": "Ambiguous match",
    "not_found": "Not found",
    "failed": "Failed",
}


def _uses_chinese(language: str | None) -> bool:
    normalized = (language or "").strip().casefold()
    return "chinese" in normalized or "中文" in normalized or normalized.startswith("zh")


def _summary_html(report: RequestReport, language: str | None) -> str:
    if _uses_chinese(language):
        text = f"本次请求共 {report.requested_count} 篇，已返回 {report.resolved_count} 篇；需要注意 {len(report.issues)} 项。"
    else:
        text = (
            f"Requested {report.requested_count} papers; returned {report.resolved_count}; "
            f"{len(report.issues)} item(s) need attention."
        )
    return (
        '<table border="0" cellpadding="0" cellspacing="0" width="100%" '
        'style="font-family: Arial, sans-serif; border: 1px solid #cfe2ff; border-radius: 8px; '
        'padding: 16px; background-color: #eef6ff; margin-bottom: 24px;">'
        f'<tr><td style="font-size: 16px; color: #222;">{escape(text)}</td></tr></table>'
    )


def _issues_html(issues: list[RequestIssue], language: str | None) -> str:
    if not issues:
        return ""
    chinese = _uses_chinese(language)
    heading = "未完整处理的论文" if chinese else "Papers needing attention"
    status_labels = _STATUS_LABELS_ZH if chinese else _STATUS_LABELS_EN
    rows = []
    for issue in issues:
        status = status_labels.get(issue.status, issue.status)
        rows.append(
            "<li style=\"margin-bottom: 10px;\">"
            f"<strong>{escape(issue.title)}</strong> "
            f"[{escape(issue.group)} · {escape(status)}]<br>"
            f"{escape(issue.detail)}"
            "</li>"
        )
    return (
        '<div style="font-family: Arial, sans-serif; border: 1px solid #f0c36d; border-radius: 8px; '
        'padding: 16px; background-color: #fff8e6; margin-top: 24px;">'
        f'<h2 style="font-size: 20px; color: #333; margin-top: 0;">{escape(heading)}</h2>'
        f'<ol style="padding-left: 24px;">{"".join(rows)}</ol></div>'
    )


def render_request_email(report: RequestReport, language: str | None = None) -> str:
    labels = _email_labels(language)
    rendered_groups = []
    for group in report.groups:
        paper_parts = [_render_paper_html(paper, labels) for paper in group.papers]
        paper_html = "<br>" + PAPER_SPACING.join(paper_parts) + "</br>"
        rendered_groups.append(get_group_html(group.label, group.summary, paper_html))

    content = _summary_html(report, language)
    if rendered_groups:
        content += "<br>" + PAPER_SPACING.join(rendered_groups) + "</br>"
    content += _issues_html(report.issues, language)
    footer = (
        "这是对你提交的 Markdown 论文清单的自动回复。"
        if _uses_chinese(language)
        else "This is an automated reply to your Markdown paper-list request."
    )
    return framework.replace("__CONTENT__", content).replace("__UNSUBSCRIBE__", footer)


def render_request_error(message: str, language: str | None = None) -> str:
    if _uses_chinese(language):
        title = "论文清单处理失败"
        lead = "没有处理这封邮件中的论文清单："
    else:
        title = "Paper-list request failed"
        lead = "The paper-list attachment was not processed:"
    content = (
        '<div style="font-family: Arial, sans-serif; border: 1px solid #d9534f; border-radius: 8px; '
        'padding: 16px; background-color: #fff5f5;">'
        f'<h2 style="margin-top: 0;">{escape(title)}</h2>'
        f'<p>{escape(lead)}</p><p>{escape(message)}</p></div>'
    )
    return framework.replace("__CONTENT__", content).replace("__UNSUBSCRIBE__", "")
