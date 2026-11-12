#!/usr/bin/env python3
"""Descriptive evaluation of recovered HOG and majority baseline on historical val.

This does not certify a clean test or the checkpoint's original training membership.
Only load joblib artifacts from a trusted, user-identified recovery source.
"""
import argparse
import importlib.metadata
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib

from evaluation import digest, export, metrics, read_csv
from validate_model import build_features


def evaluate(source, recovery, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    audit = json.loads((recovery / 'audit.json').read_text())
    if audit['source_root'] != str(source.resolve()):
        raise ValueError('Recovery audit belongs to a different source')
    manifest = recovery / 'historical_manifest.csv'
    if digest(manifest) != audit['manifest_sha256']:
        raise ValueError('Recovery manifest changed')
    for name in ('train.csv', 'dataset/train.csv', 'dataset/val.csv', 'models/hog_svm.joblib', 'models/hog_svm.meta.json'):
        if digest(source / name) != audit['artifacts'][name]['sha256']:
            raise ValueError(f'Recovered artifact changed: {name}')
    rows = read_csv(manifest)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    for row in train + val:
        if digest(source / row['path']) != row['sha256']:
            raise ValueError(f'Recovered image changed: {row["id"]}')
    labels = sorted({r['label'] for r in train})
    common = {'evaluation_split': 'val', 'status': 'exploratory historical validation; known exact-duplicate leakage',
              'manifest_sha256': digest(manifest), 'audit_sha256': digest(recovery / 'audit.json'),
              'implementation_sha256': digest(Path(__file__)),
              'preprocessing_implementation_sha256': digest(Path(__file__).with_name('validate_model.py')),
              'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'scikit-image', 'joblib')},
              'limitations': ['No untouched test', 'Missing person IDs', 'Near-duplicate candidates not reviewed',
                              'Recovered checkpoint has no saved training-ID manifest; membership not independently certified']}
    counts = Counter(r['label'] for r in train)
    winner = sorted(counts, key=lambda k: (-counts[k], k))[0]
    output.mkdir(parents=True)
    (output / 'baseline').mkdir()
    export(output / 'baseline', val, [winner] * len(val), labels, {**common, 'model': 'baseline', 'majority_label': winner})
    meta = json.loads((source / 'models/hog_svm.meta.json').read_text())
    if meta['extractor'] != 'hog' or sorted(meta['labels']) != labels:
        raise ValueError('Recovered model metadata mismatch')
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = joblib.load(source / 'models/hog_svm.joblib')
        features = build_features([int(r['id']) for r in val], source / 'dataset/val', 'hog', meta['feature_params'])
        predictions = model.predict(features).tolist()
    (output / 'hog').mkdir()
    computed = metrics([r['label'] for r in val], predictions, labels)
    evidence = []
    for name, fingerprint in audit['artifacts'].items():
        if not name.startswith('reports/') or not name.endswith('.json'):
            continue
        if digest(source / name) != fingerprint['sha256']:
            raise ValueError(f'Historical report changed: {name}')
        historical_report = json.loads((source / name).read_text())
        if historical_report.get('confusion_matrix') == computed['confusion_matrix'] and historical_report.get('accuracy') == computed['accuracy']:
            evidence.append({'path': name, 'sha256': fingerprint['sha256'], 'match': 'accuracy and complete confusion matrix'})
    report = export(output / 'hog', val, predictions, labels,
                    {**common, 'model': 'recovered_hog', 'checkpoint_sha256': digest(source / 'models/hog_svm.joblib'),
                     'checkpoint_metadata': meta, 'warnings': list(dict.fromkeys(str(w.message) for w in caught)),
                     'matching_historical_reports': evidence,
                     'inference_seconds': time.perf_counter() - start,
                     'timing_scope': 'CPU; checkpoint load + original preprocessing + prediction; no training'})
    print(json.dumps({k: report[k] for k in ('accuracy', 'macro_f1', 'recall_per_class', 'warnings')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source, args.recovery, args.output)
