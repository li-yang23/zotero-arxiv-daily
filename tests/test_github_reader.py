import copy
import json
import time
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from zotero_arxiv_daily.github_reader import GitHubReaderClient, authorized_issue, parse_issue


def library():
    return {'version': 1, 'prompt': 'Reviewer prompt', 'updated': 0, 'jobs': [],
        'digests': [{'id': 'digest', 'created': 1, 'title': 'Papers',
            'papers': [{'title': 'Paper', 'url': 'https://arxiv.org/abs/2610.07212'}]}]}


def request():
    return {'version': 1, 'request_id': '3a5bb2fc-81d1-48aa-bbf6-ea75bb3295df', 'digest': 'digest', 'papers': [0]}


def memory_client():
    client = GitHubReaderClient()
    state = library()
    client.mutate = lambda fn: fn(state)
    return client, state


def test_owner_only_and_no_pull_requests():
    issue = {'state': 'open', 'title': '[Paper Reader] 1 paper', 'user': {'login': 'owner'}}
    assert authorized_issue(issue, 'owner')
    assert not authorized_issue({**issue, 'user': {'login': 'stranger'}}, 'owner')
    assert not authorized_issue({**issue, 'pull_request': {'url': 'example'}}, 'owner')
    assert not authorized_issue({**issue, 'state': 'closed'}, 'owner')
    assert parse_issue('```json\n'+json.dumps(request())+'\n```') == request()


def test_request_is_idempotent_and_uses_saved_prompt():
    client, state = memory_client()
    assert client.enqueue(request()) == 1
    assert client.enqueue(request()) == 0
    assert state['jobs'][0]['prompt'] == state['prompt']
    job = client.post('claim', {})['job']
    assert job['status'] == 'running'
    assert client.post('claim', {}) == {'job': None}
    client.post('complete', dict(id=job['id'], lease=job['lease'], status='completed', result='{"sections":[]}'))
    assert client.enqueue(request()) == 0
    assert client.post('claim', {}) == {'job': None}


@pytest.mark.parametrize('patch', [{'papers': [0, 0]}, {'papers': [True]}, {'papers': [-1]}, {'papers': [10]},
    {'papers': []}, {'prompt': ''}, {'prompt': 'x'*12001}, {'digest': '../../secret'}, {'request_id': '../secret'}])
def test_invalid_requests_cannot_create_work(patch):
    client, state = memory_client()
    with pytest.raises((ValueError, TypeError)):
        client.enqueue({**request(), **patch})
    assert not state['jobs']


def test_stale_lease_cannot_overwrite_new_attempt_and_retries_are_bounded():
    client, state = memory_client()
    client.enqueue(request())
    first = client.post('claim', {})['job']
    state['jobs'][0]['updated'] = 0
    second = client.post('claim', {})['job']
    assert second['lease'] != first['lease'] and second['attempts'] == 2
    with pytest.raises(ValueError, match='lease'):
        client.post('complete', dict(id=first['id'], lease=first['lease'], status='failed'))
    state['jobs'][0].update(updated=0, attempts=3)
    assert client.post('claim', {}) == {'job': None}
    assert state['jobs'][0]['status'] == 'failed'


def test_conflicting_mail_write_is_preserved_when_retrying_queue_update(monkeypatch):
    client = GitHubReaderClient()
    first = library()
    second = copy.deepcopy(first)
    second['digests'].append({'id': 'concurrent-mail', 'papers': [], 'created': 2})
    client.load = Mock(side_effect=[(first, 'sha1'), (second, 'sha2')])
    client.request = Mock(side_effect=[HTTPError('api', 409, 'Conflict', {}, None), {}])
    monkeypatch.setattr(time, 'sleep', lambda n: None)
    assert client.enqueue(request()) == 1
    import base64
    written = json.loads(base64.b64decode(client.request.call_args.args[2]['content']))
    assert written['digests'][1]['id'] == 'concurrent-mail'
    assert len(written['jobs']) == 1


def test_forged_dispatch_sender_cannot_start_model_work(monkeypatch, tmp_path):
    client = GitHubReaderClient()
    client.repository = 'owner/repo'
    client.enqueue = Mock()
    event = tmp_path/'event.json'
    event.write_text(json.dumps({'inputs': {'request': json.dumps(request())}, 'sender': {'login': 'stranger'}}))
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event))
    with pytest.raises(ValueError, match='owner'):
        client.sync_requests()
    client.enqueue.assert_not_called()


def test_page_export_does_not_publish_drafts_or_leases():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('exporter', Path(__file__).parents[1]/'scripts/build_reader_library.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = library()
    state['jobs'] = [{'id': 'job', 'created': 0, 'status': 'running', 'lease': 'lease', 'draft': 'private draft'}]
    data = module.public_library(state, 'owner/repo')
    assert 'lease' not in data['jobs'][0] and 'draft' not in data['jobs'][0]
