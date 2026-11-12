#!/usr/bin/env python3
"""Propose image-duplication groups and exclusions; never modify data or split assignments."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import digest, read_csv, write_json

DECISIONS = {'near_duplicate', 'different_image', 'uncertain'}


def propose(recovery, evidence, review_path, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    audit = json.loads((recovery / 'audit.json').read_text())
    provenance = json.loads((evidence / 'provenance.json').read_text())
    manifest = recovery / 'historical_manifest.csv'
    candidates = recovery / 'similarity_candidates.csv'
    if (digest(manifest) != audit['manifest_sha256'] or
            provenance['manifest_sha256'] != digest(manifest) or
            provenance['candidates_sha256'] != digest(candidates) or
            provenance['audit_sha256'] != digest(recovery / 'audit.json')):
        raise ValueError('Recovery/evidence hashes do not match')
    rows = read_csv(manifest)
    by_id = {r['id']: r for r in rows}
    expected = read_csv(evidence / 'review_template.csv')
    original_candidates = read_csv(candidates)
    if len(expected) != len(original_candidates):
        raise ValueError('Evidence candidate count changed')
    for candidate, item in zip(original_candidates, expected):
        if any(candidate[k] != item[k] for k in candidate):
            raise ValueError('Evidence differs from original candidates')
    reviews = read_csv(review_path)
    if len(reviews) != len(expected) or len({r['pair_index'] for r in reviews}) != len(expected):
        raise ValueError('Review must cover every candidate exactly once')
    reviews_by_index = {r['pair_index']: r for r in reviews}
    for item in expected:
        r = reviews_by_index.get(item['pair_index'])
        keys = ('id_a', 'id_b', 'dhash_distance', 'ssim', 'rmse', 'aligned_ncc', 'scale', 'shift_y', 'shift_x')
        if r is None or any(r[k] != item[k] for k in keys):
            raise ValueError('Review pair or numerical evidence changed')
        if r['decision'] not in DECISIONS or not r['reason'].strip() or not r['reviewer'].strip():
            raise ValueError('Every review requires a valid decision, reason and reviewer')
    parents = {ident: ident for ident in by_id}

    def find(ident):
        if parents[ident] != ident:
            parents[ident] = find(parents[ident])
        return parents[ident]

    def union(a, b):
        parents[find(b)] = find(a)

    for group in audit['exact_duplicate_groups']['pixel_sha256']:
        for ident in group['ids'][1:]:
            union(group['ids'][0], ident)
    for r in reviews:
        if r['decision'] in ('near_duplicate', 'uncertain'):
            union(r['id_a'], r['id_b'])
    components = defaultdict(list)
    for ident in by_id:
        components[find(ident)].append(ident)
    uncertain_roots = {find(r['id_a']) for r in reviews if r['decision'] == 'uncertain'}
    output_rows, grouped = [], []
    for root, ids in components.items():
        ids.sort(key=int)
        members = [by_id[ident] for ident in ids]
        labels = sorted({r['label'] for r in members if r['label']})
        splits = sorted({r['split'] for r in members})
        conflict = len(labels) > 1
        uncertain = root in uncertain_roots
        official_overlap = 'predict' in splits and len(splits) > 1
        group_id = f'image-{ids[0]}'
        if len(ids) > 1:
            grouped.append({'group_id': group_id, 'ids': ids, 'labels': labels, 'splits': splits,
                            'label_disagreement': conflict, 'uncertain_image_match': uncertain,
                            'official_unlabeled_overlap': official_overlap})
        for row in members:
            if row['split'] == 'predict':
                action = 'preserve_official_unlabeled; do_not_impute_label'
            elif conflict:
                action = 'quarantine_label_disagreement_pending_source_review'
            elif uncertain:
                action = 'quarantine_uncertain_image_match_pending_second_review'
            elif official_overlap:
                action = 'exclude_from_internal_development_due_official_overlap'
            elif len(ids) > 1:
                action = 'keep_image_group_together_if_resplitting'
            else:
                action = 'eligible_subject_to_source_and_protocol_review'
            output_rows.append({**row, 'duplicate_group': group_id,
                                'label_disagreement': str(conflict).lower(),
                                'uncertain_image_match': str(uncertain).lower(), 'proposed_action': action})
    output_rows.sort(key=lambda r: int(r['id']))
    summary = {'kind': 'proposal only; original data, labels and official split unchanged',
               'review_decisions': dict(Counter(r['decision'] for r in reviews)),
               'reviewers': sorted({r['reviewer'] for r in reviews}),
               'n_images': len(rows), 'n_nontrivial_groups': len(grouped),
               'n_images_in_nontrivial_groups': sum(len(g['ids']) for g in grouped),
               'cross_train_val_groups': sum({'train', 'val'} <= set(g['splits']) for g in grouped),
               'label_disagreement_groups': sum(g['label_disagreement'] for g in grouped),
               'uncertain_groups': sum(g['uncertain_image_match'] for g in grouped),
               'official_overlap_groups': sum(g['official_unlabeled_overlap'] for g in grouped),
               'proposed_actions': dict(Counter(r['proposed_action'] for r in output_rows)),
               'components': grouped,
               'manifest_sha256': digest(manifest), 'review_sha256': digest(review_path),
               'evidence_provenance_sha256': digest(evidence / 'provenance.json'),
               'implementation_sha256': digest(Path(__file__)),
               'limits': ['Single assistant visual review, no independent second reviewer',
                          'Image duplication only; no person identity inferred',
                          'Only dHash candidates at distance <=4 were reviewed; screening is not exhaustive',
                          'No conflict resolved by majority vote or model prediction',
                          'Groups include uncertain edges conservatively; not all edges are confirmed duplicates']}
    output.mkdir(parents=True)
    with (output / 'group_proposal.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary['group_proposal_sha256'] = digest(output / 'group_proposal.csv')
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = propose(args.recovery, args.evidence, args.review, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('components', 'limits')}, indent=2))
