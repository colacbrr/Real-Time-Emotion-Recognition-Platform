#!/usr/bin/env python3
"""Explicit public Kaggle reference download for provenance checks, not model fitting."""
import argparse
import json
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from evaluation import digest, write_json

BASE = 'https://www.kaggle.com/api/v1/datasets/'


def fetch(output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    urls = {
        'metadata.json': BASE + 'view/msambare/fer2013',
        'fer2013-msambare-v1.zip': BASE + 'download/msambare/fer2013?datasetVersionNumber=1',
    }
    evidence = {}
    for name, url in urls.items():
        with tempfile.NamedTemporaryFile(dir=output, suffix='.partial') as temporary:
            with urllib.request.urlopen(url, timeout=60) as response:
                n = 0
                while block := response.read(1024 * 1024):
                    n += len(block)
                    if n > 150 * 1024 * 1024:
                        raise ValueError('Unexpectedly large reference response')
                    temporary.write(block)
                content_type = response.headers.get('Content-Type', '')
                status = response.status
            temporary.flush()
            if name.endswith('.json'):
                metadata = json.loads(Path(temporary.name).read_text())
                if metadata.get('currentVersionNumber') != 1 or metadata.get('id') != 786787:
                    raise ValueError('Dataset identifier/version differs from requested reference')
            elif not zipfile.is_zipfile(temporary.name):
                raise ValueError('Public endpoint did not return a ZIP archive; no authentication bypass attempted')
            with (output / name).open('xb') as destination, open(temporary.name, 'rb') as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
        evidence[name] = {'url': url, 'http_status': status, 'content_type': content_type,
                          'bytes': n, 'sha256': digest(output / name)}
    record = {'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
              'distribution': 'Kaggle msambare/fer2013, version 1',
              'dataset_page': 'https://www.kaggle.com/datasets/msambare/fer2013',
              'files': evidence, 'implementation_sha256': digest(Path(__file__)),
              'scope': 'public reference only; a matching re-upload does not independently certify original image rights'}
    write_json(output / 'download_provenance.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.output), indent=2))
