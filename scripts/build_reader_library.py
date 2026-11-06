"""Export only the public reader fields into the Pages artifact."""
import json
from pathlib import Path
import sys


def public_library(state, repository):
    fields = ('id', 'title', 'url', 'prompt', 'status', 'result', 'error', 'model', 'created', 'updated')
    return {
        'repository': repository, 'prompt': state['prompt'], 'updated': state['updated'],
        'digests': sorted(state['digests'], key=lambda d: d['created'], reverse=True),
        'jobs': [{k: j[k] for k in fields if k in j}
                 for j in sorted(state['jobs'], key=lambda j: j['created'], reverse=True)],
    }


if __name__ == '__main__':
    state_path, output, repository = sys.argv[1:]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(public_library(json.loads(Path(state_path).read_text()), repository), ensure_ascii=False))
