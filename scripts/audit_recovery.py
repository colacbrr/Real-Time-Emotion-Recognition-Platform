#!/usr/bin/env python3
"""Read-only forensic inventory of the recovered ASPICV academic layout."""
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from evaluation import digest, read_csv, write_json


def labels_from(path):
    rows = read_csv(path)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs in {path}')
    return {r['id']: r['label'] for r in rows}


def inspect_image(path):
    with Image.open(path) as original:
        mode, size = original.mode, original.size
        im = ImageOps.exif_transpose(original).convert('RGB')
        pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        gray = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        difference = gray[:, 1:] > gray[:, :-1]
        dhash = int.from_bytes(np.packbits(difference).tobytes(), 'big')
    return mode, size, pixel, dhash


def audit_recovery(source, output, distance=4):
    if not 0 <= distance <= 64:
        raise ValueError('dHash distance must be between 0 and 64')
    if output.exists():
        raise ValueError('Use a fresh output directory')
    source = source.resolve()
    original = labels_from(source / 'train.csv')
    historical = {s: labels_from(source / 'dataset' / f'{s}.csv') for s in ('train', 'val')}
    if set(historical['train']) & set(historical['val']):
        raise ValueError('Historical train/val share IDs')
    merged = {**historical['train'], **historical['val']}
    if merged != original:
        raise ValueError('Historical labels do not reconstruct original train.csv')
    files = sorted([p for split in ('train', 'val', 'predict')
                    for p in (source / 'dataset' / split).glob('*.jpg')], key=lambda p: int(p.stem))
    ids = {p.stem for p in files}
    if not files or not set(original) <= ids:
        raise ValueError('Missing labeled images')
    if len(ids) != len(files):
        raise ValueError('Duplicate image ID')
    rows, modes, sizes, copy_errors, zip_errors, dhashes = [], Counter(), Counter(), [], [], {}
    with zipfile.ZipFile(source / 'dataset.zip') as archive:
        members = [x.filename for x in archive.infolist() if not x.is_dir()]
        if len(members) != len(files) or set(members) != {f'dataset/{p.name}' for p in files}:
            raise ValueError('ZIP member inventory differs from raw images')
        for path in files:
            ident = path.stem
            split = 'train' if ident in historical['train'] else 'val' if ident in historical['val'] else 'predict'
            sha = digest(path)
            if hashlib.sha256(archive.read(f'dataset/{path.name}')).hexdigest() != sha:
                zip_errors.append(ident)
            copy = source / 'dataset' / split / path.name
            if copy != path or not copy.is_file() or digest(copy) != sha:
                copy_errors.append(ident)
            mode, size, pixels, dhash = inspect_image(path)
            modes[mode] += 1
            sizes[str(size)] += 1
            dhashes[ident] = dhash
            rows.append({'id': ident, 'path': str(path.relative_to(source)), 'label': original.get(ident, ''),
                         'sha256': sha, 'pixel_sha256': pixels, 'split': split, 'subject_id': ''})
    for split in ('train', 'val', 'predict'):
        actual = {p.stem for p in (source / 'dataset' / split).glob('*.jpg')}
        expected = {r['id'] for r in rows if r['split'] == split}
        if actual != expected:
            raise ValueError(f'Unexpected split inventory in {split}')
    duplicates = {}
    for field in ('sha256', 'pixel_sha256'):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        duplicates[field] = [
            {'ids': [r['id'] for r in group], 'splits': sorted({r['split'] for r in group}),
             'labels': sorted({r['label'] for r in group if r['label']}),
             'cross_split': len({r['split'] for r in group}) > 1,
             'conflicting_labels': len({r['label'] for r in group if r['label']}) > 1}
            for group in grouped.values() if len(group) > 1]
    candidates = []
    for a, b in combinations(rows, 2):
        if a['pixel_sha256'] == b['pixel_sha256']:
            continue
        d = (dhashes[a['id']] ^ dhashes[b['id']]).bit_count()
        if d <= distance:
            candidates.append({'id_a': a['id'], 'id_b': b['id'], 'dhash_distance': d,
                               'split_a': a['split'], 'split_b': b['split'],
                               'label_a': a['label'], 'label_b': b['label']})
    artifacts = {}
    for pattern in ('models/*', 'reports/*.json', '*.pdf', 'dataset.zip', 'train.csv', 'dataset/*.csv'):
        for path in source.glob(pattern):
            if path.is_file():
                artifacts[str(path.relative_to(source))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    report = {'source_root': str(source), 'n_images': len(rows), 'n_labeled': len(original),
              'id_range_contiguous': ids == {str(i) for i in range(1, len(rows) + 1)},
              'split_counts': dict(Counter(r['split'] for r in rows)),
              'class_counts': {s: dict(Counter(r['label'] for r in rows if r['split'] == s and r['label'])) for s in ('train', 'val')},
              'image_modes': dict(modes), 'image_sizes': dict(sizes),
              'raw_directory_jpg_count': len(list((source / 'dataset/dataset').glob('*.jpg'))),
              'zip_mismatches': zip_errors, 'split_copy_mismatches': copy_errors,
              'exact_duplicate_groups': duplicates,
              'near_duplicate_candidates': {'method': '64-bit horizontal dHash; 9x8 grayscale bilinear resize',
                                            'max_hamming_distance': distance, 'n_pairs': len(candidates),
                                            'cross_split_pairs': sum(r['split_a'] != r['split_b'] for r in candidates),
                                            'interpretation': 'unreviewed visual-similarity candidates; not identity evidence or confirmed duplicates'},
              'identity_metadata': 'absent from recovered label CSVs; person independence not established',
              'official_test_labels_found': (source / 'test.csv').exists(),
              'artifacts': artifacts, 'implementation_sha256': digest(Path(__file__))}
    output.mkdir(parents=True)
    with (output / 'historical_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'similarity_candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id_a', 'id_b', 'dhash_distance', 'split_a', 'split_b', 'label_a', 'label_b'])
        writer.writeheader()
        writer.writerows(candidates)
    report['manifest_sha256'] = digest(output / 'historical_manifest.csv')
    write_json(output / 'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance', type=int, default=4)
    args = parser.parse_args()
    report = audit_recovery(args.source, args.output, args.distance)
    print(json.dumps({k: v for k, v in report.items() if k not in ('artifacts', 'exact_duplicate_groups')}, indent=2))
    print('Exact duplicate groups:', json.dumps(report['exact_duplicate_groups'], indent=2))


if __name__ == '__main__':
    main()
