#!/usr/bin/env python3
"""Verify the recorded internal v1 study without fitting models or downloading data."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from build_internal_inventory import build
from evaluation import digest, load_verified, read_csv, write_json
from export_internal_results import export_results

ROOT = Path(__file__).resolve().parents[1]


def verify_study(source, inventory, protocol, runs, results, recovery, evidence, review, record, policy):
    with tempfile.TemporaryDirectory(prefix='aspicv-study-verification-') as temporary:
        derived = Path(temporary) / 'inventory'
        built = build(recovery, evidence, review, record, policy, source, derived)
        for name in (*built['files'], 'summary.json'):
            if digest(derived / name) != digest(inventory / name):
                raise ValueError(f'Derived inventory differs from reconstruction: {name}')
        regenerated = Path(temporary) / 'results'
        exported = export_results(protocol, inventory, runs, source, regenerated)
        for path in regenerated.iterdir():
            if digest(path) != digest(results / path.name):
                raise ValueError(f'Results differ from saved predictions: {path.name}')
    original = {r['id']: r for r in read_csv(recovery / 'historical_manifest.csv')}
    ledger = read_csv(inventory / 'selection_ledger.csv')
    if len(ledger) != len(original) or {r['id'] for r in ledger} != set(original):
        raise ValueError('Selection ledger does not preserve all original IDs')
    for row in ledger:
        if any(row[k] != value for k, value in original[row['id']].items()):
            raise ValueError(f'Original provenance/label/split changed: {row["id"]}')
    eligible = {r['id'] for r in read_csv(inventory / 'eligible_inventory.csv')}
    excluded = {r['id'] for r in read_csv(inventory / 'excluded_inventory.csv')}
    official = {r['id'] for r in read_csv(inventory / 'official_unlabeled_inventory.csv')}
    if eligible & excluded or eligible & official or excluded & official:
        raise ValueError('Inventory ID partitions overlap')
    if eligible | excluded | official != set(original):
        raise ValueError('Inventory partitions do not reconstruct source IDs')
    manifest, audit = load_verified(protocol / 'manifest.csv', source)
    if {r['id'] for r in manifest} != eligible:
        raise ValueError('Model manifest must contain exactly the eligible inventory')
    if audit['counts'] != json.loads((protocol / 'protocol.json').read_text())['class_counts']:
        raise ValueError('Split class counts differ from frozen protocol')
    verification = built['source_verification']
    return {
        'status': 'passed',
        'source_images_verified': verification['source_images_verified'],
        'source_artifacts_verified': verification['source_artifacts_verified'],
        'original_ids_labels_hashes_splits_preserved': True,
        'inventory_reconstruction': 'byte-identical CSVs and JSON',
        'results_reconstruction': 'byte-identical generated JSON, Markdown and CSVs',
        'counts': {'eligible': len(eligible), 'excluded': len(excluded), 'official_unlabeled': len(official)},
        'split_counts': dict(Counter(r['split'] for r in manifest)),
        'test_predictions_per_model': {m['model']: m['n'] for m in exported['models']},
        'official_test_used': False,
        'models_fitted_during_verification': False, 'network_used': False,
        'protocol_sha256': digest(protocol / 'protocol.json'),
        'results_sha256': digest(results / 'results.json'),
        'implementation_sha256': digest(Path(__file__)),
        'scope': 'artifact integrity and protocol consistency, not proof of person independence or an untouched test',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    defaults = {
        'inventory': 'outputs/internal-inventory-v1',
        'protocol': 'outputs/internal-protocol-v1',
        'runs': 'outputs/internal-runs-v1',
        'results': 'docs/publicare/rezultate_interne_v1',
        'recovery': 'outputs/recovery-v1',
        'evidence': 'outputs/duplicate-review-v2',
        'review': 'docs/publicare/duplicate_review.csv',
        'record': 'docs/publicare/duplicate_review.provenance.json',
        'policy': 'outputs/duplicate-policy-v1',
    }
    for name, relative in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=ROOT / relative)
    parser.add_argument('--output', type=Path, help='Optional fresh JSON output; otherwise print the verification.')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Use a fresh output file')
    report = verify_study(args.source, **{name: getattr(args, name) for name in defaults})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
