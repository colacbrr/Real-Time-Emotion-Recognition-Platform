#!/usr/bin/env python3
"""Explicitly unofficial grouped split; never replaces an official benchmark split."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from evaluation import SPLITS, audit, digest, pixel_hash, read_csv, write_json


def assign(source, root, output, seed=42, test_folds=5, val_folds=4):
    """Approx. 60/20/20 at defaults; group sizes may prevent exact stratification."""
    if test_folds < 2 or val_folds < 2:
        raise ValueError('Fold counts must be at least 2')
    sidecar = output.with_suffix('.split.json')
    if output.exists() or sidecar.exists():
        raise ValueError('Output already exists')
    rows = read_csv(source)
    required = {'id', 'path', 'label'}
    if not rows or any(not required <= r.keys() or any(not r[k] for k in required) for r in rows):
        raise ValueError('Need nonempty id,path,label fields')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ID')
    if any(r.get('split') for r in rows):
        raise ValueError('Existing splits found; preserve official/historical splits and review them first')
    root = root.resolve()
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    # Stable ID ordering makes source row order irrelevant.
    rows.sort(key=lambda r: r['id'])
    for i, r in enumerate(rows):
        path = (root / r['path']).resolve()
        r['path'] = str(path.relative_to(root))
        r['sha256'], r['pixel_sha256'] = digest(path), pixel_hash(path)
        r['subject_id'] = r.get('subject_id') or ''
        r['duplicate_group'] = r.get('duplicate_group') or ''
        for key in ('sha256', 'pixel_sha256', 'subject_id', 'duplicate_group'):
            value = r[key]
            if value:
                token = (key, value)
                if token in seen:
                    parents[find(i)] = find(seen[token])
                else:
                    seen[token] = i
    # Union-find preserves transitive duplicate/person links, even across labels.
    groups = np.array([find(i) for i in range(len(rows))])
    labels = np.array([r['label'] for r in rows])
    for key in ('sha256', 'pixel_sha256'):
        label_by_hash = {}
        for r in rows:
            if r[key] in label_by_hash and label_by_hash[r[key]] != r['label']:
                raise ValueError('Duplicate image with conflicting labels')
            label_by_hash[r[key]] = r['label']
    if len(set(groups)) < test_folds:
        raise ValueError('Insufficient independent groups for test folds')
    splitter = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    rest, test = next(splitter.split(np.zeros(len(rows)), labels, groups))
    if len(set(groups[rest])) < val_folds:
        raise ValueError('Insufficient independent groups for validation folds')
    splitter = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    train_relative, val_relative = next(splitter.split(np.zeros(len(rest)), labels[rest], groups[rest]))
    indices = {'train': rest[train_relative], 'val': rest[val_relative], 'test': test}
    for split in SPLITS:
        for i in indices[split]:
            rows[i]['split'] = split
    report = audit(rows)  # Reject absent classes or any accidental leakage.
    report.update({'protocol': 'unofficial; source must have no applicable official split',
                   'algorithm': 'nested StratifiedGroupKFold; first fold at each stage',
                   'seed': seed, 'test_folds': test_folds, 'val_folds': val_folds,
                   'source_csv_sha256': digest(source), 'independent_groups': len(set(groups)),
                   'grouping': 'connected components of file hash, decoded pixels, supplied subject_id and reviewed duplicate_group',
                   'actual_fractions': {s: len(indices[s]) / len(rows) for s in SPLITS},
                   'warning': 'Approximate class balance; no identity inference; no near-duplicate detection'})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'path', 'label', 'split', 'subject_id', 'duplicate_group'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    report['split_csv_sha256'] = digest(output)
    write_json(sidecar, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-folds', type=int, default=5)
    parser.add_argument('--val-folds', type=int, default=4)
    args = parser.parse_args()
    report = assign(args.source, args.root, args.output, args.seed, args.test_folds, args.val_folds)
    print({key: report[key] for key in ('n', 'counts', 'actual_fractions', 'identity_control')})


if __name__ == '__main__':
    main()
