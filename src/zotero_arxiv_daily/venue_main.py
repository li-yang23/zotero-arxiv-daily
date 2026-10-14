"""Run with python -m zotero_arxiv_daily.venue_main; independent of executor.source."""

from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import dotenv
import hydra
import httpx
from openai import OpenAI

from .utils import fetch_api_balance, send_email
from .venue_monitor import VenueMonitor, VenueReviewer
from .venue_sources import VenueSources
from .venue_state import FileStateStore, GitHubStateStore


dotenv.load_dotenv()


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config):
    settings = config.venue_monitor
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=45, follow_redirects=True, headers={
        "User-Agent": "zotero-arxiv-daily/venue-monitor (https://github.com/li-yang23/zotero-arxiv-daily)",
    }) as client, httpx.Client(timeout=60, headers={
        "Authorization": "Bearer " + os.environ.get("GH_TOKEN", ""),
        "Accept": "application/vnd.github+json",
    }) as github:
        store = (GitHubStateStore(github, os.environ["GITHUB_REPOSITORY"])
                 if settings.state_backend == "github" else FileStateStore(settings.state_file))
        sources = VenueSources(client, lookback_days=int(settings.bootstrap_days))
        # Dry-run discovery deliberately requires neither mail nor LLM credentials.
        discovery = bool(settings.discover_only)
        llm = None if discovery else OpenAI(
            api_key=config.llm.api.key, base_url=config.llm.api.base_url,
            timeout=float(config.llm.api.timeout), max_retries=1,
        )

        def send(html, *, subject):
            (output / "latest-digest.html").write_text(html, encoding="utf-8")
            send_email(config, html, subject=subject)

        monitor = VenueMonitor(sources, VenueReviewer(llm, config.llm), store, send,
                               max_reviews=int(settings.max_reviews_per_run),
                               balance=partial(fetch_api_balance, config))
        try:
            report = monitor.run(today, discover_only=discovery)
        except Exception as exc:
            (output / "report.json").write_text(json.dumps({
                "date": today.isoformat(), "fatal_error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        finally:
            if llm:
                llm.close()
        if sources.warnings:
            report["source_warnings"] = sources.warnings
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"## 顶会顶刊每日检查 {today}", "", "| 来源 | 检查状态 | 获取 | 首次发现 |",
                 "| --- | --- | ---: | ---: |"]
        for name, result in report["venues"].items():
            lines.append(f"| {name} | {result['status']} | {result.get('fetched', '-')} | {result.get('new', '-')} |")
        lines.extend(["", f"已发送 {report['sent']} 篇；待处理 {report['pending']} 篇。"])
        lines.extend(report["errors"])
        lines.extend(sources.warnings)
        if report["deferred_errors"]:
            lines.append(f"{len(report['deferred_errors'])} 篇元数据/总结失败，已保留重试；详见 report.json。")
        summary = "\n".join(lines) + "\n"
        print(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
                handle.write(summary)
        if report["errors"]:
            raise RuntimeError("部分来源或邮件状态检查失败，见专题检查报告")
        if report["pending"] and report["deferred_errors"] and not report["sent"] and not discovery:
            raise RuntimeError("候选论文尚未成功总结，已保留重试；见专题检查报告")


if __name__ == "__main__":
    main()
