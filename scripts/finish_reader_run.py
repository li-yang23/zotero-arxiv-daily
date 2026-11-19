"""Handle a reader completion event, then dispatch only actual queued work."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.github_reader import GitHubReaderClient


def finish(client, event):
    run = event.get('workflow_run') or {}
    repository = event.get('repository') or {}
    branch = repository.get('default_branch')
    if (repository.get('full_name') != client.repository or not branch or
            run.get('name') != 'Read selected papers' or
            (run.get('head_repository') or {}).get('full_name') != client.repository or
            run.get('head_branch') != branch or run.get('status') != 'completed' or
            type(run.get('id')) is not int or run['id'] <= 0 or
            run.get('conclusion') not in {'success', 'failure', 'timed_out', 'cancelled', 'skipped', 'neutral', 'action_required', 'startup_failure', 'stale'}):
        raise ValueError('Not a trusted default-branch reader completion event')
    if run['conclusion'] in {'cancelled', 'skipped', 'neutral'}:
        print('Reader did not execute or was cancelled; not starting further work.')
        return False
    if not client.finish_run(run['id'], run.get('run_attempt', 1), run['conclusion']):
        print('No continuation needed; queue empty, cancelled, or worker failed before claiming work.')
        return False
    client.request('POST', '/actions/workflows/paper-reader.yml/dispatches', {
        'ref': branch, 'inputs': {'continuation': 'true'},
    })
    print('Triggered the next queued paper; no scheduled poll is needed.')
    return True


if __name__ == '__main__':
    client = GitHubReaderClient()
    if not client.enabled:
        raise RuntimeError('GitHub reader credentials are missing')
    finish(client, json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text()))
