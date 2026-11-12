#!/usr/bin/env python3
"""Verify and export one frozen retrospective baseline/HOG/CNN comparison."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

from evaluation import compare, digest, load_verified, write_json

MODELS = ('baseline', 'hog', 'cnn')


def export_results(protocol_dir, inventory_dir, runs_dir, source, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    protocol = json.loads((protocol_dir / 'protocol.json').read_text())
    manifest = protocol_dir / 'manifest.csv'
    if digest(manifest) != protocol['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    if digest(inventory_dir / 'summary.json') != protocol['inventory_summary_sha256']:
        raise ValueError('Derived inventory summary changed')
    if str(source.resolve()) != protocol['source_root']:
        raise ValueError('Unexpected source directory')
    for name, checksum in protocol['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f'Frozen implementation changed: {name}')
    _, audit = load_verified(manifest, source)
    with tempfile.TemporaryDirectory(prefix='aspicv-results-verification-') as temporary:
        regenerated = Path(temporary) / 'comparison.json'
        compare(manifest, source, [runs_dir / f'{m}-test' for m in MODELS], regenerated)
        if digest(regenerated) != digest(runs_dir / 'comparison.json'):
            raise ValueError('Stored comparison differs from recalculated predictions')
    records = []
    for name in MODELS:
        fit_dir, test_dir = runs_dir / f'{name}-val', runs_dir / f'{name}-test'
        fit = json.loads((fit_dir / 'metrics.json').read_text())
        test = json.loads((test_dir / 'metrics.json').read_text())
        artifact = fit_dir / ('head.pt' if name == 'cnn' else 'model.joblib')
        if (fit['evaluation_split'] != 'val' or test['evaluation_split'] != 'test' or
                test.get('operation') != 'evaluate saved artifact; no fitting' or
                fit['model'] != name or test['model'] != name or
                fit['manifest_sha256'] != protocol['manifest_sha256'] or
                fit['implementation_sha256'] != protocol['code_sha256']['evaluation.py'] or
                test['implementation_sha256'] != fit['implementation_sha256'] or
                test['training_run_metrics_sha256'] != digest(fit_dir / 'metrics.json') or
                digest(artifact) != fit['artifact_sha256'] or test['artifact_sha256'] != fit['artifact_sha256']):
            raise ValueError(f'Training/test provenance mismatch: {name}')
        if name == 'cnn':
            planned = protocol['models']['cnn']
            if (fit['weights_sha256'] != planned['weights_sha256'] or
                    fit['seed'] != planned['seed'] or
                    any(fit['config'][k] != planned[k] for k in ('epochs', 'batch_size'))):
                raise ValueError('CNN differs from frozen configuration')
        records.append({
            'model': name,
            **{k: test[k] for k in ('n', 'labels', 'accuracy', 'macro_f1', 'recall_per_class',
                                    'support_per_class', 'confusion_matrix', 'confusion_matrix_axes')},
            'validation_accuracy': fit['accuracy'], 'validation_macro_f1': fit['macro_f1'],
            'fit_seconds': fit['fit_seconds'], 'validation_inference_seconds': fit['inference_seconds'],
            'test_inference_seconds': test['inference_seconds'],
            'test_inference_batch_size': test['inference_batch_size'],
            'fitted_artifact_bytes': fit['artifact_bytes'],
            'external_backbone_bytes': fit.get('backbone_bytes_external', 0),
            'selected_epoch': (max(fit['history'], key=lambda r: r['val_macro_f1'])['epoch'] if name == 'cnn' else None),
            'fit_metrics_sha256': digest(fit_dir / 'metrics.json'),
            'test_metrics_sha256': digest(test_dir / 'metrics.json'),
            'predictions_sha256': digest(test_dir / 'predictions.csv'),
            'predictions_path': str((test_dir / 'predictions.csv').resolve()),
            'artifact_sha256': digest(artifact),
        })
    result = {
        'study': 'internal-retrospective-v1', 'status': 'exploratory; not official benchmark results',
        'protocol_sha256': digest(protocol_dir / 'protocol.json'),
        'manifest_sha256': digest(manifest), 'split_sizes': protocol['split_sizes'],
        'labels': audit['labels'], 'hardware': protocol['hardware'], 'versions': protocol['versions'],
        'models': records, 'limitations': protocol['limitations'],
        'cost_estimate': json.loads((protocol_dir / 'cnn-cost-estimate.json').read_text()),
        'timing_scope': 'fit includes train/validation feature extraction, fitting, checkpoint selection and serialization; '
                        'test inference includes artifact load and preprocessing, excludes hash audit; CPU batch timings, not service latency',
        'implementation_sha256': digest(Path(__file__)),
    }
    output.mkdir(parents=True)
    write_json(output / 'results.json', result)
    with (output / 'recall_per_class.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'test_support', *MODELS])
        for label in audit['labels']:
            writer.writerow([label, records[0]['support_per_class'][label], *[r['recall_per_class'][label] for r in records]])
    for record in records:
        with (output / f'{record["model"]}_confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true_label / predicted_label', *audit['labels']])
            writer.writerows([label, *row] for label, row in zip(audit['labels'], record['confusion_matrix']))
    lines = ['# Rezultate interne v1 — exploratorii', '',
             'Aceleași 536 imagini de test intern; 1605 train și 536 validare. '
             'Nu sunt rezultatele testului oficial și nu se afirmă că testul intern este istoric neatins.', '',
             '| Model | Accuracy | Macro-F1 | Fit CPU (s) | Test batch CPU (s) |',
             '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["model"]} | {r["accuracy"]:.6f} | {r["macro_f1"]:.6f} | {r["fit_seconds"]:.3f} | {r["test_inference_seconds"]:.3f} |')
    lines += ['', '## Recall pe clase', '', '| Clasă | Suport test | baseline | hog | cnn |', '|---|---:|---:|---:|---:|']
    for label in audit['labels']:
        scores = ' | '.join(f'{r["recall_per_class"][label]:.6f}' for r in records)
        lines.append(f'| {label} | {records[0]["support_per_class"][label]} | {scores} |')
    lines += ['', '## Limite', '', *['- ' + text for text in protocol['limitations']], '',
              'Timpii includ costurile precizate în results.json; nu sunt măsurători de latență a serviciului sau dovadă real-time.', '',
              'Predicțiile sunt păstrate local în outputs/internal-runs-v1; hash-urile și traseul lor sunt în results.json. '
              'Matricile CSV au rânduri=adevăr și coloane=predicție. Nu s-au modificat hiperparametrii după consultarea testului.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol-dir', 'inventory-dir', 'runs-dir', 'source', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    report = export_results(args.protocol_dir, args.inventory_dir, args.runs_dir, args.source, args.output)
    print({r['model']: {'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']} for r in report['models']})
