"""Download the three exact checkpoints; only publish SHA256-verified files."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for row in json.loads((ROOT/'models/manifest.json').read_text(encoding='utf-8')):
        target = ROOT/'models'/row['file']
        if target.is_file():
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                raise ValueError(f'Existing checkpoint failed verification: {target.name}; preserved for inspection')
            print('Verified:', target.name, flush=True)
            continue
        if args.verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.partial')
        offset = temporary.stat().st_size if temporary.exists() else 0
        # An interrupted verification can leave a complete download. Avoid an
        # unsatisfiable Range request starting at EOF; verify it locally first.
        if offset == row['bytes'] and digest(temporary) == row['sha256']:
            os.replace(temporary, target)
            print('Verified:', target.name, flush=True)
            continue
        if offset >= row['bytes']:
            offset = 0
        request = Request(row['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        with urlopen(request, timeout=120) as response:
            if offset and response.status == 206:
                if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected download range')
                mode = 'ab'
            else:
                mode = 'wb'
            print('Downloading:', target.name, flush=True)
            with temporary.open(mode) as stream:
                while chunk := response.read(4*1024*1024):
                    stream.write(chunk)
        if temporary.stat().st_size != row['bytes'] or digest(temporary) != row['sha256']:
            raise ValueError(f'Download integrity check failed: {target.name}')
        os.replace(temporary, target)
        print('Verified:', target.name, flush=True)


if __name__ == '__main__':
    main()
