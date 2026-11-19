import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from zotero_arxiv_daily.github_reader import GitHubReaderClient
from finish_reader_run import finish


class ReaderEventsTests(unittest.TestCase):
    def setUp(self):
        self.client = GitHubReaderClient()
        self.client.repository = 'owner/repo'
        self.state = {'jobs': [{'id': 'one', 'status': 'queued', 'created': 0, 'updated': 0, 'attempts': 0}]}
        self.client.mutate = lambda fn: fn(self.state)

    def claim(self, run_id, attempt='1'):
        with patch.dict('os.environ', {'GITHUB_RUN_ID': str(run_id), 'GITHUB_RUN_ATTEMPT': attempt}):
            return self.client.post('claim', {})['job']

    def event(self, **updates):
        run = {'id': 10, 'run_attempt': 1, 'name': 'Read selected papers', 'status': 'completed',
               'conclusion': 'success', 'head_branch': 'main', 'head_repository': {'full_name': 'owner/repo'}}
        run.update(updates)
        return {'repository': {'full_name': 'owner/repo', 'default_branch': 'main'}, 'workflow_run': run}

    def test_interrupted_jobs_retry_three_times_then_stop(self):
        for run_id in range(1, 4):
            job = self.claim(run_id)
            self.assertEqual(job['attempts'], run_id)
            self.assertEqual(self.client.finish_run(run_id, 1, 'timed_out'), run_id < 3)
        self.assertEqual(self.state['jobs'][0]['status'], 'failed')
        self.assertNotIn('lease', self.state['jobs'][0])
        self.assertIsNone(self.client.post('claim', {})['job'])

    def test_old_completion_cannot_requeue_a_new_workers_lease(self):
        self.claim(10)
        self.state['jobs'][0]['updated'] = 0
        newer = self.claim(11)
        self.assertFalse(self.client.finish_run(10, 1, 'failure'))
        self.assertEqual(self.state['jobs'][0]['lease'], newer['lease'])
        self.assertEqual(self.state['jobs'][0]['status'], 'running')

    def test_old_attempt_of_same_run_cannot_release_new_attempt(self):
        self.claim(10)
        self.state['jobs'][0]['updated'] = 0
        newer = self.claim(10, '2')
        self.assertFalse(self.client.finish_run(10, 1, 'failure'))
        self.assertEqual(self.state['jobs'][0]['lease'], newer['lease'])

    def test_manual_cancellation_does_not_restart_work(self):
        self.claim(10)
        before = copy.deepcopy(self.state)
        self.assertFalse(self.client.finish_run(10, 1, 'cancelled'))
        self.assertEqual(self.state, before)

    def test_failure_before_claiming_does_not_create_dispatch_loop(self):
        self.assertFalse(self.client.finish_run(10, 1, 'startup_failure'))
        self.assertEqual(self.state['jobs'][0]['status'], 'queued')

    def test_completed_paper_is_preserved_and_next_paper_is_dispatched(self):
        job = self.claim(10)
        self.client.post('complete', {'id': job['id'], 'lease': job['lease'],
                                     'status': 'completed', 'result': '{"sections":[]}'} )
        self.state['jobs'].append({'id': 'two', 'status': 'queued', 'created': 1})
        self.client.request = Mock(return_value={})
        self.assertTrue(finish(self.client, self.event()))
        self.client.request.assert_called_once_with('POST', '/actions/workflows/paper-reader.yml/dispatches',
                                                   {'ref': 'main', 'inputs': {'continuation': 'true'}})
        self.assertEqual(self.state['jobs'][0]['status'], 'completed')
        self.assertEqual(self.state['jobs'][0]['result'], '{"sections":[]}')

    def test_empty_queue_stops_without_dispatch(self):
        self.state['jobs'] = []
        self.client.request = Mock()
        self.assertFalse(finish(self.client, self.event()))
        self.client.request.assert_not_called()

    def test_untrusted_completion_events_are_rejected(self):
        for update in [{'head_branch': 'feature'}, {'head_repository': {'full_name': 'stranger/repo'}},
                       {'name': 'Other'}, {'status': 'in_progress'}, {'id': -1}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                finish(self.client, self.event(**update))

    def test_dispatch_accepts_the_github_no_content_response(self):
        response = io.BytesIO(b'')
        response.status = 204
        self.client.opener = Mock()
        self.client.opener.open.return_value = response
        self.assertEqual(self.client.request('POST', '/actions/workflows/paper-reader.yml/dispatches', {'ref': 'main'}), {})


if __name__ == '__main__':
    unittest.main()
