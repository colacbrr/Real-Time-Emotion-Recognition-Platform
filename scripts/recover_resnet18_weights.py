#!/usr/bin/env python3
"""Explicitly recover official Torchvision ResNet18 weights; training never downloads."""
import argparse
import hashlib
import importlib.metadata
import inspect
import re
import tempfile
import urllib.request
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from evaluation import digest, write_json


def recover(output):
    if output.exists() or output.with_suffix('.provenance.json').exists():
        raise ValueError('Use a fresh weights path')
    weights = ResNet18_Weights.IMAGENET1K_V1
    url = weights.url
    if url != 'https://download.pytorch.org/models/resnet18-f37072fd.pth':
        raise ValueError(f'Official weights URL changed; review before downloading: {url}')
    expected_prefix = re.search(r'-([a-f0-9]+)\.pth$', url).group(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.partial') as temporary:
        sha = hashlib.sha256()
        n_bytes = 0
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.url != url:
                raise ValueError('Unexpected download redirect')
            while block := response.read(1024 * 1024):
                temporary.write(block)
                sha.update(block)
                n_bytes += len(block)
        temporary.flush()
        checksum = sha.hexdigest()
        if not checksum.startswith(expected_prefix):
            raise ValueError('Downloaded weights fail the publisher checksum-prefix check')
        state = torch.load(temporary.name, map_location='cpu', weights_only=True)
        model = resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        # Only publish the local artifact after checksum and architecture checks.
        with output.open('xb') as destination, open(temporary.name, 'rb') as source:
            while block := source.read(1024 * 1024):
                destination.write(block)
    metadata = {
        'architecture': 'resnet18', 'weights_enum': 'ResNet18_Weights.IMAGENET1K_V1',
        'url': url, 'bytes': n_bytes, 'sha256': checksum,
        'publisher_sha256_prefix': expected_prefix,
        'checksum_scope': 'official URL and published filename prefix; full SHA-256 computed locally, not a publisher signature',
        'torchvision_version': importlib.metadata.version('torchvision'),
        'torch_version': torch.__version__,
        'enum_source_sha256': digest(Path(inspect.getfile(ResNet18_Weights))),
        'versioned_source': 'https://github.com/pytorch/vision/blob/v0.28.0/torchvision/models/resnet.py',
        'documentation': 'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html',
        'pretraining_dataset': 'ImageNet-1K, as declared by Torchvision',
        'transforms': str(weights.transforms()),
        'conditions': 'Keep upstream model documentation and applicable pretraining-data terms; repository code license alone does not establish rights to all training images.',
        'architecture_check': 'strict state_dict load into unmodified ResNet18 with 1000-class ImageNet head',
        'implementation_sha256': digest(Path(__file__)),
    }
    write_json(output.with_suffix('.provenance.json'), metadata)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.output)
    print({k: result[k] for k in ('url', 'bytes', 'sha256', 'architecture_check')})
