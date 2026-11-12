#!/usr/bin/env python3
"""Source-aligned, retrospective representation/head controls; offline only.

Never changes v1 protocols or fills ASPICV predict labels. Commands must run in
order: plan, pilot, run, verify. New output directories are required.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import digest, read_csv, write_json, metrics, hog_features, cnn_features, pixel_hash

LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODELS = ['majority', 'hog_linear', 'hog_rbf', 'resnet_linear', 'resnet_rbf']
PAIRS = [('resnet_rbf', 'hog_rbf'), ('resnet_linear', 'hog_linear'),
         ('hog_rbf', 'hog_linear'), ('resnet_rbf', 'resnet_linear')]


def table(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cohort(eligible, matches):
    membership = defaultdict(set)
    for r in matches:
        if r['method'] == 'byte_exact':
            membership[r['id']].add(r['reference_split'])
    result = []
    train_groups = {r['duplicate_group'] for r in eligible if membership[r['id']] == {'train'}}
    for r in eligible:
        found = membership[r['id']]
        role = ('development' if found == {'train'} else
                'holdout' if found == {'test'} else 'excluded_ambiguous_source_split')
        if role == 'holdout' and r['duplicate_group'] in train_groups:
            role = 'excluded_known_group_crossing'
        result.append({**r, 'role': role, 'source_splits': '+'.join(sorted(found))})
    if any(not membership[r['id']] for r in eligible):
        raise ValueError('Unmatched eligible image')
    return result


def grouped_splits(rows, n, seed):
    y = np.array([r['label'] for r in rows])
    groups = np.array([r['duplicate_group'] for r in rows])
    result = list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Group leakage')
        if set(y[train]) != set(LABELS) or set(y[test]) != set(LABELS):
            raise ValueError('Missing class in fold')
    return result


def plan(a):
    if a.output.exists():
        raise ValueError('Use a fresh plan directory')
    rows = cohort(read_csv(a.inventory), read_csv(a.matches))
    development = [r for r in rows if r['role'] == 'development']
    holdout = [r for r in rows if r['role'] == 'holdout']
    folds = []
    for outer, (tr, te) in enumerate(grouped_splits(development, 5, 20261112)):
        train = [development[i] for i in tr]
        folds.append({'outer': outer, 'train_ids': [r['id'] for r in train],
                      'test_ids': [development[i]['id'] for i in te],
                      'inner': [{'train_ids': [train[i]['id'] for i in itr],
                                 'val_ids': [train[i]['id'] for i in ite]}
                                for itr, ite in grouped_splits(train, 3, 20261113 + outer)]})
    final_inner = [{'train_ids': [development[i]['id'] for i in tr],
                    'val_ids': [development[i]['id'] for i in te]}
                   for tr, te in grouped_splits(development, 3, 20261120)]
    a.output.mkdir(parents=True)
    table(a.output / 'cohort.csv', rows)
    write_json(a.output / 'folds.json', {'outer': folds, 'final_inner': final_inner})
    protocol = {'id': 'source-aligned-retrospective-v2', 'date': '2026-11-12',
                'scope': 'educational audit/reproducibility; no new algorithm; retrospective, not untouched benchmark',
                'cohort_sha256': digest(a.output / 'cohort.csv'), 'folds_sha256': digest(a.output / 'folds.json'),
                'inputs': {str(p): digest(p) for p in [a.inventory, a.matches, a.weights, Path(__file__), Path(__file__).with_name('evaluation.py')]},
                'counts': dict(Counter(r['role'] for r in rows)),
                'class_counts': {s: dict(Counter(r['label'] for r in rows if r['role'] == s)) for s in ['development', 'holdout']},
                'design': '5 outer grouped folds, 3 inner grouped folds; same folds for every model; final 3-fold selection on development for separate source-test subset',
                'models': MODELS, 'C_grid': [0.1, 1.0, 10.0], 'selection': 'mean inner macro-F1; smaller C at exact tie',
                'features': 'unchanged evaluation.py: HOG 900; frozen ImageNet V1 ResNet18 512, official transforms; no augmentation/fine-tuning',
                'heads': 'SVC(kernel=linear/rbf), gamma=scale for RBF; StandardScaler fit on each training fold only; unweighted classes',
                'baseline': 'train-fold majority; lexical tie break', 'labels': LABELS,
                'metrics': 'per-image accuracy, macro-F1 (primary), recall/class, confusion matrix; pooled OOF primary; fold scores descriptive',
                'bootstrap': {'replicates': 2000, 'seed': 20261121, 'unit': 'known image group; stratified by true class',
                              'interval': 'percentile 95%; pairs also Bonferroni 98.75% for four comparisons',
                              'pairs': PAIRS, 'caveat': 'conditional on realized fitted predictions; no refitting; excludes training/model-selection uncertainty and unidentified people'},
                'holdout': 'eligible source-test-only images; exclude known groups present in development; both-split exact matches excluded entirely; original ASPICV predict never labeled',
                'limitations': ['All local labeled data previously explored; this source-aligned holdout is retrospective',
                                'Subset sampling unknown; neither original FER2013 full benchmark nor cross-domain test',
                                'No person metadata, incomplete near-duplicate detection, single assistant review',
                                'No causal estimate of duplicate-induced score inflation; pretraining FER overlap not audited'],
                'expected_svm_fits': 240}
    write_json(a.output / 'protocol.json', protocol)
    print(json.dumps(protocol, indent=2))


def verified(plan_dir, root):
    p = json.loads((plan_dir / 'protocol.json').read_text())
    for path, sha in p['inputs'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'Changed protocol input: {path}')
    for filename, key in [('cohort.csv', 'cohort_sha256'), ('folds.json', 'folds_sha256')]:
        if digest(plan_dir / filename) != p[key]:
            raise ValueError(f'Changed plan: {filename}')
    rows = read_csv(plan_dir / 'cohort.csv')
    active = [r for r in rows if r['role'] in ('development', 'holdout')]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ID')
    for r in active:
        path = (root / r['path']).resolve()
        path.relative_to(root.resolve())
        if digest(path) != r['sha256'] or pixel_hash(path) != r['pixel_sha256']:
            raise ValueError(f'Image changed: {r["id"]}')
    return p, active


def classifier(model, c):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel=model.rsplit('_', 1)[1], gamma='scale'))


def pilot(a):
    p, rows = verified(a.plan, a.root)
    if p['inputs'].get(str(a.weights)) != digest(a.weights):
        raise ValueError('Backbone differs from protocol')
    if a.output.exists():
        raise ValueError('Pilot output exists')
    development = [r for r in rows if r['role'] == 'development']
    # Training-side extraction only; no performance metric and no tuning.
    sample = [r for label in LABELS for r in [s for s in development if s['label'] == label][:20]]
    times = {}
    for kind, feature_fn in [('hog', lambda: hog_features(sample, a.root)),
                             ('resnet', lambda: cnn_features(sample, a.root, a.weights, 16))]:
        start = time.perf_counter()
        x = feature_fn()
        extraction = time.perf_counter() - start
        times[kind] = {'sample_n': len(sample), 'extraction_seconds': extraction,
                       'linear_all_image_extraction_estimate_seconds': extraction / len(sample) * len(rows)}
        for kernel in ['linear', 'rbf']:
            start = time.perf_counter()
            classifier(f'{kind}_{kernel}', 10.).fit(x, [r['label'] for r in sample])
            seconds = time.perf_counter() - start
            times[kind][kernel + '_fit_seconds'] = seconds
    result = {'protocol_sha256': digest(a.plan / 'protocol.json'), 'purpose': 'cost only; not model-selection evidence',
              'pilot': times, 'svm_fits': p['expected_svm_fits'],
              'budget_seconds': 900, 'limit_policy': 'stop at completed model/fold boundary if >15min; report incomplete without fabrication',
              'estimate': 'CPU extraction roughly linear in images; 240 SVM fits may scale superlinearly; conservative 5–15 min budget, no fine-tuning',
              'hardware': p.get('hardware', 'CPU; torch maximum 4 threads; see v1 hardware record')}
    write_json(a.output, result)
    print(json.dumps(result, indent=2))


def cm_scores(cm):
    cm = np.asarray(cm, dtype=float)
    diagonal = np.diag(cm)
    denominator = cm.sum(0) + cm.sum(1)
    return np.array([diagonal.sum() / cm.sum(),
                     np.divide(2 * diagonal, denominator, out=np.zeros(len(cm)), where=denominator != 0).mean()])


def bootstrap(rows, predictions, replicates=2000, seed=20261121):
    """Paired class-stratified cluster bootstrap; per-image metric estimand."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['duplicate_group']].append(i)
    values = list(groups.values())
    strata = defaultdict(list)
    for g, indices in enumerate(values):
        labels = {rows[i]['label'] for i in indices}
        if len(labels) != 1:
            raise ValueError('Bootstrap groups must have consistent labels')
        strata[next(iter(labels))].append(g)
    group_index = np.empty(len(rows), dtype=int)
    for i, indices in enumerate(values):
        group_index[indices] = i
    y = np.array([LABELS.index(r['label']) for r in rows])
    encoded = {name: y * 7 + np.array([LABELS.index(v) for v in pred]) for name, pred in predictions.items()}
    if any(len(v) != len(rows) for v in encoded.values()):
        raise ValueError('Misaligned predictions')
    rng = np.random.default_rng(seed)
    sampled = {name: np.empty((replicates, 2)) for name in predictions}
    for b in range(replicates):
        chosen = np.concatenate([rng.choice(v, len(v), replace=True) for v in strata.values()])
        weight = np.bincount(chosen, minlength=len(values))[group_index]
        for name, codes in encoded.items():
            sampled[name][b] = cm_scores(np.bincount(codes, weights=weight, minlength=49).reshape(7, 7))
    return {'replicates': replicates, 'groups': len(values), 'images': len(rows),
            'model_intervals_95': {name: dict(zip(['accuracy', 'macro_f1'], np.quantile(v, [.025, .975], axis=0).T.tolist())) for name, v in sampled.items()},
            'paired_differences': [{'a': a, 'b': b, 'direction': 'a minus b',
                                    'interval_95': dict(zip(['accuracy', 'macro_f1'], np.quantile(sampled[a] - sampled[b], [.025, .975], axis=0).T.tolist())),
                                    'interval_familywise_95_four_pairs': dict(zip(['accuracy', 'macro_f1'], np.quantile(sampled[a] - sampled[b], [.00625, .99375], axis=0).T.tolist()))}
                                   for a, b in PAIRS if a in sampled and b in sampled],
            'scope': 'conditional prediction resampling; no model refits, no person-level guarantee; OOF training-fold dependence not accounted for'}


def select_fit(name, x, y, index, train_ids, inner, grid):
    history = []
    for c in grid:
        scores = []
        for split in inner:
            tr = [index[i] for i in split['train_ids']]
            va = [index[i] for i in split['val_ids']]
            if set(split['train_ids']) & set(split['val_ids']) or set(split['train_ids'] + split['val_ids']) != set(train_ids):
                raise ValueError('Invalid inner fold')
            model = classifier(name, c).fit(x[tr], y[tr])
            scores.append(metrics(y[va].tolist(), model.predict(x[va]).tolist(), LABELS)['macro_f1'])
        history.append({'C': c, 'fold_macro_f1': scores, 'mean_macro_f1': float(np.mean(scores))})
    best = max(history, key=lambda r: (r['mean_macro_f1'], -r['C']))['C']
    model = classifier(name, best).fit(x[[index[i] for i in train_ids]], y[[index[i] for i in train_ids]])
    return model, {'selected_C': best, 'inner_scores': history}


def run(a):
    import importlib.metadata
    import joblib
    p, rows = verified(a.plan, a.root)
    if p['inputs'].get(str(a.weights)) != digest(a.weights):
        raise ValueError('Backbone differs from protocol')
    cost = json.loads(a.pilot.read_text())
    if cost['protocol_sha256'] != digest(a.plan / 'protocol.json'):
        raise ValueError('Cost estimate does not match protocol')
    if a.output.exists():
        raise ValueError('Use a fresh output directory')
    a.output.mkdir(parents=True)
    start = time.perf_counter()
    index = {r['id']: i for i, r in enumerate(rows)}
    y = np.array([r['label'] for r in rows])
    timings = {}
    features = {}
    for kind, fn in [('hog', lambda: hog_features(rows, a.root)), ('resnet', lambda: cnn_features(rows, a.root, a.weights, 16))]:
        t = time.perf_counter()
        features[kind] = fn()
        np.save(a.output / f'{kind}_features.npy', features[kind])
        timings[kind + '_fixed_extraction_seconds'] = time.perf_counter() - t
    fold_plan = json.loads((a.plan / 'folds.json').read_text())
    development = [r for r in rows if r['role'] == 'development']
    holdout = [r for r in rows if r['role'] == 'holdout']
    predictions = {s: {} for s in ['oof', 'holdout']}
    tuning = []
    for name in MODELS:
        oof = {}
        for fold in fold_plan['outer']:
            train_ids, test_ids = fold['train_ids'], fold['test_ids']
            if name == 'majority':
                counts = Counter(y[[index[i] for i in train_ids]])
                pred = [sorted(counts, key=lambda k: (-counts[k], k))[0]] * len(test_ids)
                selected = {}
            else:
                x = features[name.split('_')[0]]
                model, selected = select_fit(name, x, y, index, train_ids, fold['inner'], p['C_grid'])
                pred = model.predict(x[[index[i] for i in test_ids]]).tolist()
            if set(oof) & set(test_ids):
                raise ValueError('Repeated outer test ID')
            oof.update(zip(test_ids, pred))
            tuning.append({'model': name, 'outer': fold['outer'], **selected,
                           'test': metrics([y[index[i]] for i in test_ids], pred, LABELS)})
            if time.perf_counter() - start > cost['budget_seconds']:
                raise RuntimeError('Cost budget exceeded; incomplete output, no final result')
        predictions['oof'][name] = [oof[r['id']] for r in development]
        tr_ids = [r['id'] for r in development]
        if name == 'majority':
            counts = Counter(r['label'] for r in development)
            winner = sorted(counts, key=lambda k: (-counts[k], k))[0]
            predictions['holdout'][name] = [winner] * len(holdout)
            selected = {'majority_label': winner}
        else:
            x = features[name.split('_')[0]]
            model, selected = select_fit(name, x, y, index, tr_ids, fold_plan['final_inner'], p['C_grid'])
            joblib.dump(model, a.output / f'{name}.joblib')
            predictions['holdout'][name] = model.predict(x[[index[r['id']] for r in holdout]]).tolist()
        tuning.append({'model': name, 'outer': 'final_development_fit', **selected})
        print(f'Completed {name}', flush=True)
    result = {'protocol_sha256': digest(a.plan / 'protocol.json'), 'timings': timings,
              'versions': {s: importlib.metadata.version(s) for s in ['numpy', 'scikit-learn', 'torch', 'torchvision']},
              'evaluations': {}}
    for scope, subset in [('oof', development), ('holdout', holdout)]:
        exported = [{**{k: r[k] for k in ['id', 'label', 'duplicate_group']},
                     **{name: predictions[scope][name][i] for name in MODELS}} for i, r in enumerate(subset)]
        table(a.output / f'{scope}_predictions.csv', exported)
        result['evaluations'][scope] = {'models': {name: metrics([r['label'] for r in subset], predictions[scope][name], LABELS) for name in MODELS},
                                        'uncertainty': bootstrap(subset, predictions[scope])}
    result['elapsed_seconds'] = time.perf_counter() - start
    write_json(a.output / 'tuning.json', tuning)
    write_json(a.output / 'results.json', result)
    write_json(a.output / 'integrity.json', {p.name: digest(p) for p in a.output.iterdir() if p.is_file()})
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'scores': {s: {k: [v['accuracy'], v['macro_f1']] for k, v in e['models'].items()} for s, e in result['evaluations'].items()}}, indent=2))


def verify(a):
    p, rows = verified(a.plan, a.root)
    for name, sha in json.loads((a.output / 'integrity.json').read_text()).items():
        if digest(a.output / name) != sha:
            raise ValueError(f'Artifact changed: {name}')
    result = json.loads((a.output / 'results.json').read_text())
    if result['protocol_sha256'] != digest(a.plan / 'protocol.json'):
        raise ValueError('Result protocol mismatch')
    for scope, role in [('oof', 'development'), ('holdout', 'holdout')]:
        expected = {r['id']: r for r in rows if r['role'] == role}
        pred = read_csv(a.output / f'{scope}_predictions.csv')
        if len(pred) != len(expected) or {r['id'] for r in pred} != set(expected):
            raise ValueError('Prediction ID mismatch')
        for r in pred:
            if any(r[k] != expected[r['id']][k] for k in ['label', 'duplicate_group']):
                raise ValueError('Prediction label/group changed')
        for name in MODELS:
            computed = metrics([r['label'] for r in pred], [r[name] for r in pred], LABELS)
            if computed != result['evaluations'][scope]['models'][name]:
                raise ValueError('Metric mismatch')
        computed = bootstrap(pred, {name: [r[name] for r in pred] for name in MODELS})
        if computed != result['evaluations'][scope]['uncertainty']:
            raise ValueError('Uncertainty mismatch')
    print('Verified image hashes, frozen inputs, saved artifacts, prediction IDs, metrics and bootstrap; no fitting/network')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    for key in ['inventory', 'matches', 'weights', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    for command in ['pilot', 'run', 'verify']:
        p = sub.add_parser(command)
        for key in ['plan', 'root', 'output']:
            p.add_argument('--' + key, type=Path, required=True)
        if command != 'verify':
            p.add_argument('--weights', type=Path, required=True)
        if command == 'run':
            p.add_argument('--pilot', type=Path, required=True)
    a = parser.parse_args()
    {'plan': plan, 'pilot': pilot, 'run': run, 'verify': verify}[a.command](a)


if __name__ == '__main__':
    main()
