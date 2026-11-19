"""Operational checks run with stdlib unittest; no ML dependencies/services."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import check_reader_publish as publish


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.state = {'version': 1, 'prompt': 'read', 'updated': 1, 'digests': [],
                      'jobs': [{'id': 'j', 'created': 1, 'status': 'running', 'lease': 'old'}]}

    def run_probe(self, published, *, force='false', error=None):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            state.write_text(json.dumps(self.state))
            output = Path(directory) / 'output'
            response = io.BytesIO(json.dumps(published).encode())
            with patch.dict(os.environ, {'REPOSITORY': 'owner/repo', 'GITHUB_OUTPUT': str(output),
                                         'FORCE_PUBLISH': force, 'PAPER_READER_URL': 'https://owner.github.io/repo'}), \
                 patch.object(sys, 'argv', ['check', str(state)]), \
                 patch.object(publish, 'urlopen', return_value=response, side_effect=error) as fetch:
                publish.main()
                return output.read_text(), fetch.call_count

    def test_no_public_change_skips_build_even_if_internal_lease_changes(self):
        before = publish.public_library(self.state, 'owner/repo')
        self.state['jobs'][0]['lease'] = 'new'
        self.assertEqual(self.run_probe(before)[0], 'changed=false\n')

    def test_status_result_and_timestamp_changes_publish(self):
        before = publish.public_library(self.state, 'owner/repo')
        self.state['jobs'][0].update(status='completed', result='report')
        self.state['updated'] = 2
        self.assertEqual(self.run_probe(before)[0], 'changed=true\n')

    def test_unavailable_site_rebuilds_instead_of_skipping_recovery(self):
        self.assertEqual(self.run_probe({}, error=OSError('offline'))[0], 'changed=true\n')

    def test_frontend_push_and_manual_publish_force_a_build(self):
        before = publish.public_library(self.state, 'owner/repo')
        self.assertEqual(self.run_probe(before, force='true'), ('changed=true\n', 0))


if __name__ == '__main__':
    unittest.main()
