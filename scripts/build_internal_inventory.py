#!/usr/bin/env python3
"""Apply the reviewed exclusion policy to derived CSVs, without modifying source data."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import digest, read_csv, write_json
from verify_duplicate_review import verify

ACTIONS = {
    'eligible_subject_to_source_and_protocol_review': 'eligible_internal',
    'keep_image_group_together_if_resplitting': 'eligible_internal',
    'quarantine_label_disagreement_pending_source_review': 'excluded_internal',
    'quarantine_uncertain_image_match_pending_second_review': 'excluded_internal',
    'exclude_from_internal_development_due_official_overlap': 'excluded_internal',
    'preserve_official_unlabeled; do_not_impute_label': 'official_unlabeled',
}


def partition(proposal):
    """Validate the complete ledger before writing any derived inventory."""
    if not proposal or len({r['id'] for r in proposal}) != len(proposal):
        raise ValueError('Empty proposal or duplicate ID')
    partitions = {role: [] for role in set(ACTIONS.values())}
    groups = defaultdict(list)
    for row in proposal:
        action = row['proposed_action']
        if action not in ACTIONS:
            raise ValueError(f'Unknown policy action: {action}')
        role = ACTIONS[action]
        if not all(row.get(k) for k in ('id', 'path', 'sha256', 'pixel_sha256', 'duplicate_group')):
            raise ValueError('Incomplete provenance/group fields')
        if role == 'official_unlabeled':
            if row['split'] != 'predict' or row['label']:
                raise ValueError('Official unlabeled rows must stay predict with empty labels')
        elif row['split'] not in ('train', 'val') or not row['label']:
            raise ValueError('Internal rows must come from labeled historical train/val')
        if role == 'eligible_internal' and (row['label_disagreement'] != 'false' or row['uncertain_image_match'] != 'false'):
            raise ValueError('Risk-flagged image cannot enter eligible inventory')
        partitions[role].append({**row, 'inventory_role': role})
        groups[row['duplicate_group']].append(row)
    for members in groups.values():
        labels = {r['label'] for r in members if r['label']}
        if len(labels) > 1 and any(ACTIONS[r['proposed_action']] == 'eligible_internal' for r in members):
            raise ValueError('Label-disagreement group cannot enter eligible inventory')
    eligible = partitions['eligible_internal']
    held_out = partitions['excluded_internal'] + partitions['official_unlabeled']
    for key in ('duplicate_group', 'sha256', 'pixel_sha256'):
        if {r[key] for r in eligible} & {r[key] for r in held_out}:
            raise ValueError(f'Eligible and held-out inventories overlap by {key}')
    return partitions


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def build(recovery, evidence, review, record, policy, source, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Derived output must be outside the recovered source directory')
    verification = verify(recovery, evidence, review, record, policy, source)
    proposal = read_csv(policy / 'group_proposal.csv')
    partitions = partition(proposal)
    fields = list(proposal[0]) + ['inventory_role']
    ledger = sorted([r for rows in partitions.values() for r in rows], key=lambda r: int(r['id']))
    eligible = []
    for row in partitions['eligible_internal']:
        item = dict(row)
        # Retain the historical assignment as provenance; do not silently reuse it.
        item['historical_split'] = item.pop('split')
        eligible.append(item)
    eligible.sort(key=lambda r: int(r['id']))
    output.mkdir(parents=True)
    write_csv(output / 'eligible_inventory.csv', eligible,
              [k for k in fields if k != 'split'] + ['historical_split'])
    write_csv(output / 'excluded_inventory.csv', partitions['excluded_internal'], fields)
    write_csv(output / 'official_unlabeled_inventory.csv', partitions['official_unlabeled'], fields)
    write_csv(output / 'selection_ledger.csv', ledger, fields)
    summary = {
        'status': 'reviewed policy applied to derived CSV inventories only',
        'source_root': str(source.resolve()), 'source_verification': verification,
        'counts': {role: len(rows) for role, rows in sorted(partitions.items())},
        'exclusion_reasons': dict(Counter(r['proposed_action'] for r in partitions['excluded_internal'])),
        'class_counts': {role: dict(sorted(Counter(r['label'] for r in rows if r['label']).items()))
                         for role, rows in sorted(partitions.items())},
        'eligible_historical_split_counts': dict(Counter(r['historical_split'] for r in eligible)),
        'eligible_image_groups': len({r['duplicate_group'] for r in eligible}),
        'source_ids_preserved': len(ledger), 'partition_ids_disjoint': True,
        'eligible_overlap_with_excluded_or_official_by_hash_or_reviewed_group': False,
        'files': {p.name: {'sha256': digest(p), 'rows': len(read_csv(p))} for p in sorted(output.glob('*.csv'))},
        'policy_summary_sha256': digest(policy / 'summary.json'),
        'review_record_sha256': digest(record), 'implementation_sha256': digest(Path(__file__)),
        'protocol': 'retrospective internal development inventory from original labeled pool; no new split assigned',
        'limitations': ['Not a clean-data certification or proof of person independence',
                        'Single assistant visual review; two uncertain pairs excluded conservatively',
                        'Official test labels and original dataset source/license remain unavailable',
                        'No images moved/copied, no relabeling, no model fitting',
                        'No new untouched test; original historical memberships retained as provenance'],
    }
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovery', 'evidence', 'review', 'record', 'policy', 'source', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.recovery, args.evidence, args.review, args.record, args.policy, args.source, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'exclusion_reasons', 'class_counts', 'eligible_image_groups')}, indent=2))
