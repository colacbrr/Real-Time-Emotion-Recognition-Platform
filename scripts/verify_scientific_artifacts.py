#!/usr/bin/env python3
"""Additional saved-model and grouped-fold checks for scientific v2. Offline, no fit."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from evaluation import read_csv, digest, write_json
from scientific_study import MODELS, verified


def check(plan, run, root, public):
    protocol, rows = verified(plan, root)
    by_id = {r['id']: r for r in rows}
    development = {r['id'] for r in rows if r['role'] == 'development'}
    holdout = {r['id'] for r in rows if r['role'] == 'holdout'}
    if {by_id[i]['duplicate_group'] for i in development} & {by_id[i]['duplicate_group'] for i in holdout}:
        raise ValueError('Development/holdout group leakage')
    folds = json.loads((plan / 'folds.json').read_text())
    outer_test = []
    for outer in folds['outer']:
        tr, te = set(outer['train_ids']), set(outer['test_ids'])
        if tr & te or tr | te != development:
            raise ValueError('Outer ID leakage/coverage')
        outer_test.extend(te)
        for split in [outer] + outer['inner']:
            a, b = split['train_ids'], split.get('test_ids', split.get('val_ids'))
            if {by_id[i]['duplicate_group'] for i in a} & {by_id[i]['duplicate_group'] for i in b}:
                raise ValueError('Fold group leakage')
        for inner in outer['inner']:
            if set(inner['train_ids']) | set(inner['val_ids']) != tr or set(inner['train_ids']) & set(inner['val_ids']):
                raise ValueError('Inner ID leakage/coverage')
    if len(outer_test) != len(development) or set(outer_test) != development:
        raise ValueError('OOF coverage')
    index = {r['id']: i for i, r in enumerate(rows)}
    pred = read_csv(run / 'holdout_predictions.csv')
    for name in MODELS[1:]:
        x = np.load(run / (name.split('_')[0] + '_features.npy'), allow_pickle=False)
        model = joblib.load(run / (name + '.joblib'))
        if model.named_steps['standardscaler'].n_samples_seen_ != len(development):
            raise ValueError('Final scaler fitting sample count')
        recomputed = model.predict(x[[index[r['id']] for r in pred]]).tolist()
        if recomputed != [r[name] for r in pred]:
            raise ValueError('Saved model predictions do not match export')
    for name, sha in json.loads((public / 'integrity.json').read_text()).items():
        if digest(public / name) != sha:
            raise ValueError('Public evidence changed: ' + name)
    for name in ['results.json', 'tuning.json', 'oof_predictions.csv', 'holdout_predictions.csv']:
        if digest(public / name) != digest(run / name):
            raise ValueError('Public copy differs from original run: ' + name)
    return {'status': 'passed', 'development_n': len(development), 'holdout_n': len(holdout),
            'outer_folds': len(folds['outer']), 'group_crossings': 0,
            'final_saved_model_predictions_reproduced': 4, 'scalers_fit_n': len(development),
            'public_results_byte_identical': True, 'network_used': False, 'models_fitted': False,
            'protocol_sha256': digest(plan / 'protocol.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'run', 'root', 'public', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    report = check(a.plan, a.run, a.root, a.public)
    write_json(a.output, report)
    print(json.dumps(report, indent=2))
