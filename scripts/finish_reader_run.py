"""Finish a reading job and explicitly dispatch publication and remaining work.

Run as a dependent job even after worker timeout. workflow_run is deliberately
not used: GITHUB_TOKEN-created runs do not reliably trigger that next event.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.github_reader import GitHubReaderClient


def finish(client, *, run_id, run_attempt, conclusion, branch):
    if (not str(run_id).isdigit() or int(run_id) <= 0 or
            not str(run_attempt).isdigit() or int(run_attempt) <= 0 or not branch or
            conclusion not in {'success', 'failure', 'cancelled', 'skipped'}):
        raise ValueError('Invalid reading job completion context')
    if conclusion in {'cancelled', 'skipped'}:
        print('Reader did not execute or was cancelled; not starting further work.')
        return False
    pending = client.finish_run(run_id, run_attempt, conclusion)
    errors = []
    # A failed publication request must not strand already-queued reading work.
    for workflow, inputs in [('reader-pages.yml', {'check_changes': 'true'})] + (
            [('paper-reader.yml', {'continuation': 'true'})] if pending else []):
        try:
            client.request('POST', f'/actions/workflows/{workflow}/dispatches', {
                'ref': branch, 'inputs': inputs,
            })
            print('Dispatched', workflow)
        except Exception as exc:
            errors.append(f'{workflow}: {type(exc).__name__}')
    if errors:
        raise RuntimeError('Follow-up dispatch failed: ' + ', '.join(errors))
    print('Triggered the next queued paper.' if pending else 'No queued papers; reading chain stopped.')
    return pending


if __name__ == '__main__':
    client = GitHubReaderClient()
    if not client.enabled:
        raise RuntimeError('GitHub reader credentials are missing')
    finish(client, run_id=os.environ['GITHUB_RUN_ID'],
           run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT', '1'),
           conclusion=os.environ['READER_CONCLUSION'],
           branch=os.environ['READER_DEFAULT_BRANCH'])
