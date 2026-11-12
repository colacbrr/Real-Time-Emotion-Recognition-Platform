#!/usr/bin/env python3
"""Verify recorded duplicate-review evidence and reproduce the group proposal."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from duplicate_policy import propose
from evaluation import digest, read_csv, write_json


def verify(recovery, evidence, review_path, record_path, policy, source):
    record = json.loads(record_path.read_text())
    targets = {
        'review_sha256': review_path,
        'historical_manifest_sha256': recovery / 'historical_manifest.csv',
        'recovery_audit_sha256': recovery / 'audit.json',
        'candidates_sha256': recovery / 'similarity_candidates.csv',
        'evidence_provenance_sha256': evidence / 'provenance.json',
        'evidence_template_sha256': evidence / 'review_template.csv',
        'review_script_sha256': Path(__file__).with_name('review_candidates.py'),
        'policy_script_sha256': Path(__file__).with_name('duplicate_policy.py'),
        'proposal_sha256': policy / 'group_proposal.csv',
        'proposal_summary_sha256': policy / 'summary.json',
    }
    for key, path in targets.items():
        if digest(path) != record[key]:
            raise ValueError(f'Hash mismatch: {path}')
    for name, checksum in record['sheets'].items():
        if digest(evidence / name) != checksum:
            raise ValueError(f'Review sheet changed: {name}')
    audit = json.loads((recovery / 'audit.json').read_text())
    if str(source.resolve()) != audit['source_root']:
        raise ValueError('Source does not match recovery audit')
    for name, fingerprint in audit['artifacts'].items():
        if digest(source / name) != fingerprint['sha256']:
            raise ValueError(f'Recovered source artifact changed: {name}')
    rows = read_csv(recovery / 'historical_manifest.csv')
    for row in rows:
        if digest(source / row['path']) != row['sha256']:
            raise ValueError(f'Recovered image changed: {row["id"]}')
    reviews = read_csv(review_path)
    if dict(Counter(r['decision'] for r in reviews)) != record['decisions']:
        raise ValueError('Review decision counts differ from recorded provenance')
    # Reconstruct in temporary storage; do not overwrite any recorded output.
    with tempfile.TemporaryDirectory(prefix='aspicv-review-verification-') as temporary:
        regenerated = Path(temporary) / 'policy'
        propose(recovery, evidence, review_path, regenerated)
        for name in ('group_proposal.csv', 'summary.json'):
            if digest(regenerated / name) != digest(policy / name):
                raise ValueError(f'Group policy is not reproducible: {name}')
    proposal = read_csv(policy / 'group_proposal.csv')
    original_by_id = {row['id']: row for row in rows}
    if len(proposal) != len(rows) or {r['id'] for r in proposal} != set(original_by_id):
        raise ValueError('Proposed inventory does not preserve all original IDs')
    for row in proposal:
        if any(row[k] != value for k, value in original_by_id[row['id']].items()):
            raise ValueError(f'Original manifest fields changed in proposal: {row["id"]}')
    return {
        'status': 'verified', 'source_images_verified': len(rows),
        'source_artifacts_verified': len(audit['artifacts']),
        'reviewed_pairs': len(reviews), 'decision_counts': record['decisions'],
        'review_sheets_verified': len(record['sheets']),
        'policy_reproduction': 'byte-identical CSV and JSON in temporary directory',
        'original_manifest_fields': 'preserved for every proposed row, including empty test labels',
        'proposed_actions': dict(Counter(r['proposed_action'] for r in proposal)),
        'review_record_sha256': digest(record_path),
        'implementation_sha256': digest(Path(__file__)),
        'scope': 'integrity and reproducibility; not an independent visual review or proof of person independence',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    result = verify(args.recovery, args.evidence, args.review, args.record, args.policy, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
