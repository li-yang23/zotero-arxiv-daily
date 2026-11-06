"""Import owner requests and claim one job before installing ML dependencies."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.github_reader import GitHubReaderClient

client = GitHubReaderClient()
if not client.enabled:
    raise RuntimeError('GitHub reader credentials are missing')
client.sync_requests()
job = client.post('claim', {}).get('job')
if job:
    path = Path(os.environ['PAPER_READER_JOB'])
    path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    path.chmod(0o600)
with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
    output.write('pending=' + ('true' if job else 'false') + '\n')
print('Claimed one reading job.' if job else 'No reading jobs queued.')
