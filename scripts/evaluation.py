#!/usr/bin/env python3
"""Manifest-first educational evaluation. No network access or implicit data discovery."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

SPLITS = ('train', 'val', 'test')
FIELDS = ['id', 'path', 'label', 'sha256', 'pixel_sha256', 'split', 'subject_id', 'duplicate_group']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def validate_rows(rows, required):
    if not rows or any(not required <= r.keys() or
                       any(not isinstance(r[k], str) or not r[k].strip() for k in required)
                       for r in rows):
        raise ValueError(f'Need nonempty CSV fields: {sorted(required)}')


def pixel_hash(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def audit(rows):
    validate_rows(rows, {'id', 'label', 'split'})
    ids = set()
    groups = {key: defaultdict(list) for key in ('sha256', 'pixel_sha256', 'subject_id', 'duplicate_group')}
    for r in rows:
        if not r['id'] or r['id'] in ids or not r['label'] or r['split'] not in SPLITS:
            raise ValueError('Empty/duplicate ID, empty label or invalid split')
        ids.add(r['id'])
        for key in groups:
            if r.get(key):
                groups[key][r[key]].append(r)
    duplicates = {}
    for key, values in groups.items():
        duplicates[key] = [[r['id'] for r in group] for group in values.values() if len(group) > 1]
        for group in values.values():
            if len({r['split'] for r in group}) > 1:
                raise ValueError(f'Cross-split leakage ({key}): {[r["id"] for r in group]}')
            if key != 'subject_id' and len({r['label'] for r in group}) > 1:
                raise ValueError('Duplicate image with conflicting labels')
    labels = sorted({r['label'] for r in rows})
    for split in SPLITS:
        if {r['label'] for r in rows if r['split'] == split} != set(labels):
            raise ValueError(f'Every class must be represented in {split}; recover more data or document a revised protocol')
    known = sum(bool(r.get('subject_id')) for r in rows)
    return {'n': len(rows), 'labels': labels,
            'counts': {s: dict(Counter(r['label'] for r in rows if r['split'] == s)) for s in SPLITS},
            'duplicate_groups': duplicates, 'subject_ids_known': known,
            'identity_control': 'complete metadata supplied; disjoint groups' if known == len(rows) else 'incomplete/absent; person independence NOT established',
            'near_duplicates': ('reviewed duplicate_group annotations supplied; candidate screening may be incomplete'
                                if any(r.get('duplicate_group') for r in rows) else
                                'not assessed; byte and decoded-pixel hashes detect exact duplicates only')}


def prepare(source, root, output):
    root = root.resolve()
    rows = read_csv(source)
    validate_rows(rows, {'id', 'path', 'label', 'split'})
    for r in rows:
        if not {'id', 'path', 'label', 'split'} <= r.keys():
            raise ValueError('Source CSV needs id,path,label,split and optional subject_id')
        path = (root / r['path']).resolve()
        r['path'] = str(path.relative_to(root))
        r['sha256'] = digest(path)
        r['pixel_sha256'] = pixel_hash(path)
        r['subject_id'] = r.get('subject_id') or ''
        r['duplicate_group'] = r.get('duplicate_group') or ''
    report = audit(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix('.audit.json').exists():
        raise ValueError('Output already exists; create a new version instead of replacing a split')
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['id']))
    report['manifest_sha256'] = digest(output)
    report['source_csv_sha256'] = digest(source)
    write_json(output.with_suffix('.audit.json'), report)
    return report


def load_verified(manifest, root):
    rows = read_csv(manifest)
    validate_rows(rows, set(FIELDS) - {'subject_id', 'duplicate_group'})
    report = audit(rows)
    root = root.resolve()
    for r in rows:
        path = (root / r['path']).resolve()
        path.relative_to(root)
        if digest(path) != r['sha256'] or pixel_hash(path) != r['pixel_sha256']:
            raise ValueError(f'Image changed: {r["id"]}')
    return rows, report


def metrics(truth, predictions, labels):
    if not truth or len(truth) != len(predictions):
        raise ValueError('Empty or misaligned predictions')
    if not (set(truth) | set(predictions)) <= set(labels):
        raise ValueError('Unknown label')
    return {'n': len(truth), 'labels': labels,
            'accuracy': float(accuracy_score(truth, predictions)),
            'macro_f1': float(f1_score(truth, predictions, labels=labels, average='macro', zero_division=0)),
            'recall_per_class': dict(zip(labels, recall_score(truth, predictions, labels=labels, average=None, zero_division=0).tolist())),
            'support_per_class': dict(Counter(truth)),
            'confusion_matrix': confusion_matrix(truth, predictions, labels=labels).tolist(),
            'confusion_matrix_axes': 'rows=true, columns=predicted; order=labels', 'zero_division': 0}


def export(output, rows, predictions, labels, metadata):
    report = metrics([r['label'] for r in rows], list(predictions), labels)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'true_label', 'predicted_label', 'split'])
        writer.writerows((r['id'], r['label'], p, r['split']) for r, p in zip(rows, predictions))
    report.update(metadata)
    write_json(output / 'metrics.json', report)
    return report


def hog_features(rows, root):
    from skimage.feature import hog
    features = []
    for r in rows:
        with Image.open(root / r['path']) as im:
            im = ImageOps.exif_transpose(im).convert('L').resize((48, 48), Image.Resampling.BILINEAR)
            features.append(hog(np.asarray(im, dtype=np.float32) / 255., orientations=9,
                                pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'))
    return np.asarray(features)


def cnn_features(rows, root, weights, batch_size):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = resnet18(weights=None)
    # Local, explicitly recovered ImageNet weights only; never silently download.
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    tfm = ResNet18_Weights.IMAGENET1K_V1.transforms()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = []
            for r in rows[start:start + batch_size]:
                with Image.open(root / r['path']) as im:
                    batch.append(tfm(ImageOps.exif_transpose(im).convert('RGB')))
            result.append(model(torch.stack(batch)).numpy())
    return np.concatenate(result)


def run(args):
    import importlib.metadata
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    rows, audit_report = load_verified(args.manifest, args.root)
    labels = audit_report['labels']
    splits = {s: [r for r in rows if r['split'] == s] for s in SPLITS}
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    if args.model == 'cnn' and not args.weights:
        raise ValueError('CNN requires --weights: verified local ResNet18 IMAGENET1K_V1 state_dict')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    meta = {'implementation_sha256': digest(Path(__file__)), 'model': args.model, 'manifest_sha256': digest(args.manifest), 'seed': args.seed,
            'evaluation_split': args.split, 'test_ids_sha256': hashlib.sha256(json.dumps(sorted(r['id'] for r in splits['test'])).encode()).hexdigest(),
            'audit': audit_report, 'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'scikit-image', 'Pillow')},
            'config': {'epochs': args.epochs, 'batch_size': args.batch_size},
            'timing_scope': 'CPU; fit includes train/validation preprocessing and extraction; inference includes evaluation preprocessing, excludes hash audit and serialization'}
    train = splits['train']
    evaluation = splits[args.split]
    y = [r['label'] for r in train]
    if args.model == 'baseline':
        winner = sorted(Counter(y), key=lambda k: (-Counter(y)[k], k))[0]
        joblib.dump({'majority_label': winner}, args.output / 'model.joblib')
        meta['fit_seconds'] = time.perf_counter() - start
        infer_start = time.perf_counter()
        predictions = [winner] * len(evaluation)
    elif args.model == 'hog':
        model = make_pipeline(StandardScaler(), SVC(C=10., kernel='rbf', gamma='scale'))
        model.fit(hog_features(train, args.root), y)
        val_pred = model.predict(hog_features(splits['val'], args.root))
        meta['validation'] = metrics([r['label'] for r in splits['val']], list(val_pred), labels)
        meta['config'].update({'C': 10., 'gamma': 'scale', 'kernel': 'rbf', 'resize': [48, 48], 'orientations': 9, 'pixels_per_cell': [8, 8], 'cells_per_block': [2, 2], 'block_norm': 'L2-Hys'})
        joblib.dump(model, args.output / 'model.joblib')
        meta['fit_seconds'] = time.perf_counter() - start
        infer_start = time.perf_counter()
        predictions = model.predict(hog_features(evaluation, args.root)).tolist()
    else:
        import torch
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        x = torch.tensor(cnn_features(train, args.root, args.weights, args.batch_size))
        xv = torch.tensor(cnn_features(splits['val'], args.root, args.weights, args.batch_size))
        yi = torch.tensor([labels.index(v) for v in y])
        head = torch.nn.Linear(x.shape[1], len(labels))
        optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
        best_score, best_state, history = -1., None, []
        for epoch in range(args.epochs):
            head.train()
            for idx in torch.randperm(len(x)).split(args.batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(x[idx]), yi[idx])
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                vp = [labels[i] for i in head(xv).argmax(1).tolist()]
            score = metrics([r['label'] for r in splits['val']], vp, labels)['macro_f1']
            history.append({'epoch': epoch + 1, 'val_macro_f1': score})
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        torch.save({'head_state': best_state, 'labels': labels, 'weights_sha256': digest(args.weights), 'history': history}, args.output / 'head.pt')
        meta.update({'head_optimizer': 'Adam lr=0.001; unweighted cross entropy', 'checkpoint_selection': 'max val macro-F1; first epoch at tie', 'architecture': 'resnet18', 'backbone': 'frozen; eval mode', 'pretraining': 'declared IMAGENET1K_V1; provenance must be verified externally',
                     'weights_sha256': digest(args.weights), 'history': history, 'best_val_macro_f1': best_score,
                     'torch': torch.__version__, 'torchvision': importlib.metadata.version('torchvision')})
        meta['fit_seconds'] = time.perf_counter() - start
        infer_start = time.perf_counter()
        xe = torch.tensor(cnn_features(evaluation, args.root, args.weights, args.batch_size))
        with torch.no_grad():
            predictions = [labels[i] for i in head(xe).argmax(1).tolist()]
    meta['inference_seconds'] = time.perf_counter() - infer_start
    meta['inference_batch_size'] = args.batch_size if args.model == 'cnn' else len(evaluation)
    meta['artifact_bytes'] = sum(p.stat().st_size for p in args.output.iterdir())
    artifact = args.output / ('head.pt' if args.model == 'cnn' else 'model.joblib')
    meta['artifact_sha256'] = digest(artifact)
    if args.model == 'cnn':
        meta['backbone_bytes_external'] = args.weights.stat().st_size
    report = export(args.output, evaluation, predictions, labels, meta)
    print(json.dumps({k: report[k] for k in ('model', 'accuracy', 'macro_f1', 'n')}))


def evaluate_saved(args):
    """Evaluate a fixed validation-selected artifact. Never fit on this path."""
    import joblib
    rows, audit_report = load_verified(args.manifest, args.root)
    source = json.loads((args.run / 'metrics.json').read_text())
    if source['evaluation_split'] != 'val' or source['manifest_sha256'] != digest(args.manifest):
        raise ValueError('Need a validation run using this exact manifest')
    if source['implementation_sha256'] != digest(Path(__file__)):
        raise ValueError('Evaluation code changed since fitting; review and create a new validation run')
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    labels = audit_report['labels']
    if source['labels'] != labels:
        raise ValueError('Label order changed')
    kind = source['model']
    artifact = args.run / ('head.pt' if kind == 'cnn' else 'model.joblib')
    if digest(artifact) != source['artifact_sha256']:
        raise ValueError('Saved artifact changed')
    if kind == 'cnn' and (not args.weights or digest(args.weights) != source['weights_sha256']):
        raise ValueError('Need the exact backbone weights used during fitting')
    evaluation = [r for r in rows if r['split'] == args.split]
    start = time.perf_counter()
    if kind == 'baseline':
        predictions = [joblib.load(artifact)['majority_label']] * len(evaluation)
    elif kind == 'hog':
        predictions = joblib.load(artifact).predict(hog_features(evaluation, args.root)).tolist()
    elif kind == 'cnn':
        import torch
        checkpoint = torch.load(artifact, map_location='cpu', weights_only=True)
        if checkpoint['labels'] != labels or checkpoint['weights_sha256'] != source['weights_sha256']:
            raise ValueError('Checkpoint metadata mismatch')
        head = torch.nn.Linear(512, len(labels))
        head.load_state_dict(checkpoint['head_state'])
        head.eval()
        x = torch.tensor(cnn_features(evaluation, args.root, args.weights, args.batch_size))
        with torch.inference_mode():
            predictions = [labels[i] for i in head(x).argmax(1).tolist()]
    else:
        raise ValueError('Unsupported saved model')
    metadata = {k: source[k] for k in ('model', 'manifest_sha256', 'seed', 'config',
                                      'implementation_sha256', 'artifact_sha256', 'versions', 'test_ids_sha256')}
    metadata.update({'evaluation_split': args.split, 'operation': 'evaluate saved artifact; no fitting',
                     'training_run_metrics_sha256': digest(args.run / 'metrics.json'),
                     'training_run': str(args.run.resolve()), 'inference_seconds': time.perf_counter() - start,
                     'inference_batch_size': args.batch_size if kind == 'cnn' else len(evaluation),
                     'timing_scope': 'CPU; artifact loading, preprocessing and inference; excludes manifest audit',
                     'audit': audit_report})
    args.output.mkdir(parents=True)
    report = export(args.output, evaluation, predictions, labels, metadata)
    print(json.dumps({k: report[k] for k in ('model', 'accuracy', 'macro_f1', 'n')}))


def estimate(args):
    rows, _ = load_verified(args.manifest, args.root)
    train = [r for r in rows if r['split'] == 'train']
    if not args.weights or args.sample < 1 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError('Need weights and positive sample/batch-size/epochs')
    start = time.perf_counter()
    sample = train[:args.sample]
    features = cnn_features(sample, args.root, args.weights, args.batch_size)
    seconds = time.perf_counter() - start
    import torch
    head = torch.nn.Linear(512, len({r['label'] for r in train}))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    x = torch.tensor(features)
    y = torch.tensor([sorted({r['label'] for r in train}).index(r['label']) for r in sample])
    pilot_start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(head(x), y).backward()
        optimizer.step()
    head_seconds = (time.perf_counter() - pilot_start) / 10
    print(json.dumps({'kind': 'rough CPU estimate, not an experiment result', 'sample_train_only': len(sample),
                      'sample_seconds_including_load': seconds,
                      'estimated_all_split_extraction_seconds': seconds / len(sample) * len(rows),
                      'head_pilot_steps': 10, 'head_pilot_step_seconds': head_seconds,
                      'estimated_head_seconds': head_seconds * len(train) / len(sample) * args.epochs,
                      'epochs': args.epochs,
                      'limitations': 'Includes model loading in each extraction; disk/cache variation; validation, hash audit and serialization excluded. Small-sample extrapolation; allow margin and review before a long run.'}, indent=2))



def compare(manifest, root, directories, output):
    rows, report = load_verified(manifest, root)
    expected = {r['id']: r['label'] for r in rows if r['split'] == 'test'}
    table = []
    seen = set()
    for directory in directories:
        m = json.loads((directory / 'metrics.json').read_text())
        if m['manifest_sha256'] != digest(manifest) or m['evaluation_split'] != 'test':
            raise ValueError('Reports must use this manifest and test split')
        predictions = read_csv(directory / 'predictions.csv')
        if len(predictions) != len(expected) or {r['id']: r['true_label'] for r in predictions} != expected:
            raise ValueError('Mismatched test IDs or labels')
        if any(r['split'] != 'test' for r in predictions):
            raise ValueError('Non-test predictions')
        computed = metrics([r['true_label'] for r in predictions], [r['predicted_label'] for r in predictions], report['labels'])
        for key in ('n', 'labels', 'support_per_class', 'accuracy', 'macro_f1', 'recall_per_class', 'confusion_matrix'):
            if computed[key] != m[key]:
                raise ValueError('Metrics do not match saved predictions')
        if m['model'] in seen:
            raise ValueError('Duplicate model in comparison')
        seen.add(m['model'])
        table.append({'model': m['model'], **computed})
    if seen != {'baseline', 'hog', 'cnn'}:
        raise ValueError('Initial comparison requires baseline, hog, cnn')
    if output.exists():
        raise ValueError('Comparison output already exists')
    write_json(output, {'manifest_sha256': digest(manifest), 'evaluation_split': 'test', 'models': table})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('evaluate-saved', help='Evaluate an existing validation-selected artifact, without fitting')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--weights', type=Path)
    p.add_argument('--batch-size', type=int, default=16)
    for name in ('run', 'estimate-cnn'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--root', type=Path, required=True)
        p.add_argument('--weights', type=Path)
        p.add_argument('--batch-size', type=int, default=16)
        p.add_argument('--epochs', type=int, default=10)
        if name == 'run':
            p.add_argument('--model', choices=['baseline', 'hog', 'cnn'], required=True)
            p.add_argument('--split', choices=['val', 'test'], default='val')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--output', type=Path, required=True)
        else:
            p.add_argument('--sample', type=int, default=16)
    p = sub.add_parser('compare')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.source, args.root, args.output), indent=2))
    elif args.command == 'run':
        run(args)
    elif args.command == 'estimate-cnn':
        estimate(args)
    elif args.command == 'evaluate-saved':
        if args.batch_size < 1:
            raise ValueError('batch-size must be positive')
        evaluate_saved(args)
    else:
        compare(args.manifest, args.root, args.runs, args.output)


if __name__ == '__main__':
    main()
