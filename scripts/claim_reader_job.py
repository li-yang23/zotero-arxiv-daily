"""Cheap queue probe: skip installing the ML environment when there is no work."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

url = os.environ.get('PAPER_READER_URL', '').rstrip('/')
token = os.environ.get('PAPER_READER_TOKEN', '')
job = None
if url and token:
    request = Request(url + '/api/reader/worker/claim', data=b'{}', headers={
        'Content-Type': 'application/json',
        'OAI-Sites-Authorization': 'Bearer ' + token,
        'X-Reader-Token': token,
    })
    with urlopen(request, timeout=45) as response:
        job = json.load(response).get('job')
if job:
    path = Path(os.environ['PAPER_READER_JOB'])
    path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    path.chmod(0o600)
with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
    output.write('pending=' + ('true' if job else 'false') + '\n')
print('Claimed one reading job.' if job else 'No reading jobs queued.')
