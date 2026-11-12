#!/usr/bin/env python3
"""Verify the public numerical study package without images, weights or fitting."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from evaluation import digest, metrics, read_csv
from scientific_study import bootstrap, LABELS, MODELS

ROOT = Path(__file__).resolve().parents[1]


def equal(actual, expected, where='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{where}: fields differ')
        for key in expected:
            equal(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{where}: length differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f'{where}[{i}]')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (float, int)) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{where}: numeric mismatch')
    elif actual != expected:
        raise ValueError(f'{where}: value differs')


def unique(rows, key='id'):
    if not rows or any(not r.get(key) for r in rows) or len({r[key] for r in rows}) != len(rows):
        raise ValueError(f'Empty or duplicate {key}')
    return {r[key]: r for r in rows}


def split_check(train, test, expected, rows):
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Repeated split ID')
    if set(train) & set(test) or set(train) | set(test) != set(expected):
        raise ValueError('Split overlap or incomplete coverage')
    for field in ['duplicate_group', 'sha256', 'pixel_sha256', 'subject_id']:
        a = {rows[i].get(field) for i in train} - {'', None}
        b = {rows[i].get(field) for i in test} - {'', None}
        if a & b:
            raise ValueError(f'Split overlap: {field}')
    for ids in [train, test]:
        if {rows[i]['label'] for i in ids} != set(LABELS):
            raise ValueError('Class missing from split')


def verify(plan, public):
    plan, public = Path(plan), Path(public)
    p = json.loads((plan / 'protocol.json').read_text())
    if digest(plan / 'cohort.csv') != p['cohort_sha256'] or digest(plan / 'folds.json') != p['folds_sha256']:
        raise ValueError('Cohort/fold integrity mismatch')
    for path, sha in p['inputs'].items():
        if Path(path).name in ['evaluation.py', 'scientific_study.py']:
            if digest(ROOT / 'scripts' / Path(path).name) != sha:
                raise ValueError('Frozen scientific implementation changed')
    for name, sha in json.loads((public / 'integrity.json').read_text()).items():
        target = (public / name).resolve()
        target.relative_to(public.resolve())
        if digest(target) != sha:
            raise ValueError(f'Public artifact integrity mismatch: {name}')
    result = json.loads((public / 'results.json').read_text())
    if result['protocol_sha256'] != digest(plan / 'protocol.json'):
        raise ValueError('Protocol/result mismatch')
    cohort = read_csv(plan / 'cohort.csv')
    rows = unique(cohort)
    if dict(Counter(r['role'] for r in cohort)) != p['counts']:
        raise ValueError('Cohort role counts differ')
    dev = [r['id'] for r in cohort if r['role'] == 'development']
    held = [r['id'] for r in cohort if r['role'] == 'holdout']
    split_check(dev, held, dev + held, rows)
    folds = json.loads((plan / 'folds.json').read_text())
    tested = []
    for outer in folds['outer']:
        split_check(outer['train_ids'], outer['test_ids'], dev, rows)
        tested.extend(outer['test_ids'])
        inner_test = []
        for inner in outer['inner']:
            split_check(inner['train_ids'], inner['val_ids'], outer['train_ids'], rows)
            inner_test.extend(inner['val_ids'])
        if Counter(inner_test) != Counter(outer['train_ids']):
            raise ValueError('Inner validation coverage differs')
    if Counter(tested) != Counter(dev):
        raise ValueError('OOF coverage differs')
    final_test = []
    for inner in folds['final_inner']:
        split_check(inner['train_ids'], inner['val_ids'], dev, rows)
        final_test.extend(inner['val_ids'])
    if Counter(final_test) != Counter(dev):
        raise ValueError('Final validation coverage differs')
    counts = {}
    for scope, ids in [('oof', dev), ('holdout', held)]:
        predictions = read_csv(public / f'{scope}_predictions.csv')
        indexed = unique(predictions)
        if set(indexed) != set(ids):
            raise ValueError('Prediction ID coverage differs')
        for r in predictions:
            if any(r[key] != rows[r['id']][key] for key in ['label', 'duplicate_group']):
                raise ValueError('Prediction label/group differs from cohort')
        predicted = {name: [r[name] for r in predictions] for name in MODELS}
        for name in MODELS:
            computed = metrics([r['label'] for r in predictions], predicted[name], LABELS)
            equal(computed, result['evaluations'][scope]['models'][name], f'{scope}.{name}')
        computed = bootstrap(predictions, predicted, p['bootstrap']['replicates'], p['bootstrap']['seed'])
        equal(computed, result['evaluations'][scope]['uncertainty'], f'{scope}.uncertainty')
        counts[scope] = len(predictions)
    return {'status': 'passed', 'predictions_per_model': counts, 'models': MODELS,
            'checks': ['public artifact hashes', 'portable frozen-code hashes', 'cohort/split consistency',
                       'known group/hash separation', 'outer and inner coverage', 'prediction alignment',
                       'accuracy, macro-F1, recall, confusion matrices', 'conditional paired bootstrap'],
            'images_read': False, 'weights_read': False, 'models_fitted': False, 'network_used': False,
            'scope': 'Internal consistency of saved evidence; not independent ground truth or model re-execution'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'docs/publicare/studiu_v2')
    parser.add_argument('--public', type=Path, default=ROOT / 'docs/publicare/studiu_v2/results')
    args = parser.parse_args()
    print(json.dumps(verify(args.plan, args.public), indent=2))
