"""Skip expensive Pages builds after no-op reader polls; no third-party packages."""
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from build_reader_library import public_library


def needs_publish(current, published):
    # Compare every exported field, including updated and failed-job messages.
    return current != published


def main():
    repository = os.environ['REPOSITORY']
    state = json.loads(Path(sys.argv[1]).read_text())
    current = public_library(state, repository)
    changed = True
    if os.environ.get('FORCE_PUBLISH') != 'true':
        owner, repo = repository.split('/')
        site = os.environ.get('PAPER_READER_URL') or f'https://{owner}.github.io/{repo}'
        try:
            request = Request(site.rstrip('/') + '/data/library.json',
                              headers={'Cache-Control': 'no-cache'})
            with urlopen(request, timeout=20) as response:
                published = json.load(response)
            changed = needs_publish(current, published)
        except (OSError, ValueError):
            # Missing/unreachable snapshot must not prevent recovery publication.
            print('Published snapshot unavailable; rebuilding the reader.')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'changed={str(changed).lower()}\n')
    print('Public library changed; publishing.' if changed else 'No public changes; skipping build and deployment.')


if __name__ == '__main__':
    main()
