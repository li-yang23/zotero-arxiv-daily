"""GitHub-backed reader queue. Standard library only, including the cheap probe.

Only the repository owner's Issues can create paid reading work. Every update
uses a Contents API compare-and-swap, so mail and reader runs cannot lose data.
"""
import base64
import copy
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler
import uuid

BRANCH = 'reader-data'
FILE = 'state.json'
PREFIX = '[Paper Reader]'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubReaderClient:
    def __init__(self):
        self.repository = os.environ.get('GITHUB_REPOSITORY', '')
        self.token = os.environ.get('GH_TOKEN', '')
        self.url = os.environ.get('PAPER_READER_URL', '').rstrip('/')
        self.opener = build_opener(NoRedirect())

    @property
    def enabled(self):
        return bool(re.fullmatch(r'[\w.-]+/[\w.-]+', self.repository) and self.token)

    def request(self, method, path, payload=None):
        req = Request('https://api.github.com/repos/' + self.repository + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method, headers={'Authorization': 'Bearer ' + self.token,
                'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json',
                'X-GitHub-Api-Version': '2022-11-28'})
        with self.opener.open(req, timeout=45) as response:
            return json.load(response)

    def load(self):
        item = self.request('GET', '/contents/' + FILE + '?ref=' + BRANCH)
        if item.get('encoding') != 'base64':
            blob = self.request('GET', '/git/blobs/' + item['sha'])
        else:
            blob = item
        state = json.loads(base64.b64decode(blob['content']))
        if state.get('version') != 1 or not isinstance(state.get('jobs'), list) or not isinstance(state.get('digests'), list):
            raise ValueError('Invalid reader state; refusing to replace it')
        return state, item['sha']

    def mutate(self, change):
        for attempt in range(5):
            state, sha = self.load()
            before = copy.deepcopy(state)
            result = change(state)
            if state == before:
                return result
            state['updated'] = int(time.time() * 1000)
            try:
                self.request('PUT', '/contents/' + FILE, {
                    'message': 'Update paper reader library', 'branch': BRANCH, 'sha': sha,
                    'content': base64.b64encode(json.dumps(state, ensure_ascii=False).encode()).decode()})
                return result
            except HTTPError as exc:
                if exc.code not in (409, 422) or attempt == 4:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError('Reader state update conflicted')

    def enqueue(self, request):
        def change(state):
            papers, prompt, request_id = resolve_request(request, state)
            ids = {j['id'] for j in state['jobs']}
            new = [(i, p) for i, p in enumerate(papers) if request_id + ':' + str(i) not in ids]
            if sum(j['status'] in ('queued', 'running') for j in state['jobs']) + len(new) > 50:
                raise ValueError('Queue full')
            now = int(time.time() * 1000)
            for i, paper in new:
                state['jobs'].append({'id': request_id + ':' + str(i), 'title': paper['title'],
                    'url': paper['url'], 'prompt': prompt, 'status': 'queued',
                    'created': now, 'updated': now, 'attempts': 0})
            return len(new)
        return self.mutate(change)

    def sync_requests(self):
        # A manual dispatch comes from a repository writer; never trust a
        # browser-supplied actor or token. The workflow also restricts dispatch.
        if os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch':
            event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
            batch = (event.get('inputs') or {}).get('request', '')
            if batch:
                owner = self.repository.split('/')[0]
                if event.get('sender', {}).get('login', '').lower() != owner.lower():
                    raise ValueError('Only the repository owner may submit reading work')
                self.enqueue(json.loads(batch))
        owner = self.repository.split('/')[0]
        for page in range(1, 11):
            issues = self.request('GET', f'/issues?state=open&creator={quote(owner)}&per_page=100&page={page}')
            for issue in issues:
                if not authorized_issue(issue, owner):
                    continue
                try:
                    self.enqueue(parse_issue(issue.get('body') or ''))
                except (ValueError, TypeError, KeyError):
                    print('Ignored invalid or already unavailable reading request in issue', issue['number'])
            if len(issues) < 100:
                break

    def post(self, path, payload):
        if path == 'draft':
            # Unreviewed model drafts stay off the public library and repository.
            directory = Path('outputs/reader-drafts')
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (payload['id'].replace(':', '-') + '.json')).write_text(json.dumps(payload, ensure_ascii=False))
            return {'ok': True}
        now = int(time.time() * 1000)
        lease = str(uuid.uuid4())
        def change(state):
            if path == 'digests':
                item = dict(payload, created=now)
                prior = next((d for d in state['digests'] if d['id'] == item['id']), None)
                if prior:
                    item['created'] = prior['created']
                state['digests'] = [item] + [d for d in state['digests'] if d['id'] != item['id']]
                return {'ok': True}
            if path == 'claim':
                for job in sorted(state['jobs'], key=lambda j: j['created']):
                    stale = job['status'] == 'running' and now - job['updated'] > 30 * 60 * 1000
                    if stale and job.get('attempts', 0) >= 3:
                        job.update(status='failed', error='任务多次中断，请重新提交', updated=now)
                        job.pop('lease', None)
                    elif job['status'] == 'queued' or stale:
                        job.update(status='running', lease=lease, updated=now, attempts=job.get('attempts', 0) + 1)
                        return {'job': copy.deepcopy(job)}
                return {'job': None}
            if path == 'complete':
                job = next((j for j in state['jobs'] if j['id'] == payload['id']), None)
                if not job or job['status'] != 'running' or job.get('lease') != payload['lease']:
                    raise ValueError('Expired reading job lease')
                if payload['status'] not in ('completed', 'failed'):
                    raise ValueError('Invalid completion status')
                if payload['status'] == 'completed':
                    json.loads(payload['result'])
                for key in ('status', 'result', 'model', 'error'):
                    if key in payload:
                        job[key] = payload[key]
                job['updated'] = now
                job.pop('lease', None)
                return {'ok': True}
            raise ValueError('Unknown reader operation')
        return self.mutate(change)

    def close(self):
        pass


def authorized_issue(issue, owner):
    return (not issue.get('pull_request') and issue.get('state') == 'open'
        and issue.get('user', {}).get('login', '').lower() == owner.lower()
        and issue.get('title', '').startswith(PREFIX))


def parse_issue(body):
    if len(body) > 60000:
        raise ValueError('Request too long')
    match = re.search(r'```json\s*\n(.*?)\n```', body, re.S)
    return json.loads(match.group(1) if match else body)


def resolve_request(request, state):
    if not isinstance(request, dict) or request.get('version') != 1:
        raise ValueError('Invalid request version')
    request_id = str(uuid.UUID(request['request_id']))
    digest = next((d for d in state['digests'] if d['id'] == request.get('digest')), None)
    indices = request.get('papers')
    if not digest or not isinstance(indices, list) or not 1 <= len(indices) <= 10:
        raise ValueError('Select 1–10 papers from an existing digest')
    if any(type(i) is not int or i < 0 or i >= len(digest['papers']) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Invalid paper selection')
    prompt = request.get('prompt', state['prompt'])
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError('Invalid reading prompt')
    return [digest['papers'][i] for i in indices], prompt.strip(), request_id
