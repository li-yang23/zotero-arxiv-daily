"""Private reading queue bridge; credentials stay in Actions secrets."""
import hashlib
import json
import os
from html import escape
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from loguru import logger


class ReaderClient:
    def __init__(self, client=None):
        self.url = os.environ.get('PAPER_READER_URL', '').rstrip('/')
        self.token = os.environ.get('PAPER_READER_TOKEN', '')
        self.client = client or httpx.Client(timeout=45)

    @property
    def enabled(self):
        return self.url.startswith('https://') and bool(self.token)

    def post(self, path, payload):
        response = self.client.post(self.url + '/api/reader/worker/' + path,
            json=payload, headers={'OAI-Sites-Authorization': 'Bearer ' + self.token,
                                  'X-Reader-Token': self.token})
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()


def attach_reader(html, papers, title='论文日报'):
    """Publish candidates before mail, without making email depend on the reader."""
    if not papers or not os.environ.get('PAPER_READER_URL'):
        return html
    client = ReaderClient()
    try:
        if not client.enabled:
            return html
        date = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        digest_id = hashlib.sha256(json.dumps([date, title, papers], ensure_ascii=False,
            sort_keys=True).encode()).hexdigest()[:24]
        client.post('digests', {'id': digest_id, 'title': title + ' · ' + date, 'papers': papers})
        url = client.url + '/?digest=' + digest_id
        banner = ('<div style="padding:20px;margin:16px 0;background:#eef4e9;border-radius:8px;'
                  'font-family:Arial,sans-serif"><a style="color:#285b43;font-size:18px;font-weight:bold" '
                  f'href="{escape(url, quote=True)}">选择论文并精读 →</a>'
                  '<p>打开网页勾选感兴趣的论文，按你的预设 Prompt 阅读全文并保存结果。</p></div>')
        marker = html.find('>', html.lower().find('<body'))
        return html[:marker+1] + banner + html[marker+1:] if marker >= 0 else banner + html
    except Exception as exc:
        logger.warning(f'Reader upload unavailable ({type(exc).__name__}); delivering ordinary digest')
        return html
    finally:
        client.close()
