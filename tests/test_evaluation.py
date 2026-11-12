"""Synthetic protocol checks; these are not facial-expression performance results."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from argparse import Namespace
from collections import Counter
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluation.py'
spec = importlib.util.spec_from_file_location('evaluation', SCRIPT)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
sys.path.insert(0, str(SCRIPT.parent))
from split_protocol import assign
from audit_recovery import audit_recovery
from review_candidates import aligned_evidence, render_review
from duplicate_policy import propose
from verify_duplicate_review import verify
from build_internal_inventory import build, partition
from match_fer2013_reference import match


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.manifest = self.root / 'manifest.csv'
        self.rows = []
        rng = np.random.default_rng(7)
        for split in e.SPLITS:
            for label in ('a', 'b'):
                for j in range(2):
                    ident = f'{split}-{label}-{j}'
                    Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(self.root / f'{ident}.png')
                    self.rows.append(dict(id=ident, path=f'{ident}.png', label=label, split=split, subject_id=ident))
        self.write_source()

    def write_source(self):
        with self.source.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)

    def prepare(self):
        return e.prepare(self.source, self.root, self.manifest)

    def test_exact_metrics(self):
        m = e.metrics(['a', 'a', 'b', 'b'], ['a', 'a', 'a', 'a'], ['a', 'b'])
        self.assertEqual(m['accuracy'], 0.5)
        self.assertAlmostEqual(m['macro_f1'], 1 / 3)
        self.assertEqual(m['recall_per_class'], {'a': 1., 'b': 0.})
        self.assertEqual(m['confusion_matrix'], [[2, 0], [2, 0]])

    def test_manifest_stable_and_images_verified(self):
        self.prepare()
        second = self.root / 'v2.csv'
        e.prepare(self.source, self.root, second)
        self.assertEqual(e.digest(second), e.digest(self.manifest))
        e.load_verified(self.manifest, self.root)
        Image.new('RGB', (48, 48)).save(self.root / self.rows[0]['path'])
        with self.assertRaisesRegex(ValueError, 'changed'):
            e.load_verified(self.manifest, self.root)

    def test_cross_split_pixels(self):
        # PNG metadata differs but decoded image is identical.
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text('note', 'different encoding metadata')
        with Image.open(self.root / self.rows[0]['path']) as im:
            im.save(self.root / self.rows[4]['path'], pnginfo=info)
        with self.assertRaisesRegex(ValueError, 'pixel_sha256'):
            self.prepare()
        self.assertFalse(self.manifest.exists())

    def test_subject_leakage(self):
        self.rows[4]['subject_id'] = self.rows[0]['subject_id']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'subject_id'):
            self.prepare()

    def test_missing_identity_and_duplicate_id(self):
        for r in self.rows:
            r['subject_id'] = ''
        self.write_source()
        self.assertIn('NOT established', self.prepare()['identity_control'])
        self.rows[1]['id'] = self.rows[0]['id']
        with self.assertRaises(ValueError):
            e.audit(self.rows)

    def test_path_escape_and_missing_class(self):
        self.rows[0]['path'] = '../outside.png'
        self.write_source()
        with self.assertRaises(ValueError):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'Every class'):
            e.audit([r for r in self.rows if not (r['split'] == 'test' and r['label'] == 'b')])

    def test_baseline_hog_common_test_exports(self):
        self.prepare()
        reports = []
        for model in ('baseline', 'hog'):
            out = self.root / model
            subprocess.run([sys.executable, str(SCRIPT), 'run', '--manifest', str(self.manifest), '--root', str(self.root), '--model', model, '--split', 'val', '--output', str(out)], check=True, capture_output=True)
            test_out = self.root / f'{model}-test'
            subprocess.run([sys.executable, str(SCRIPT), 'evaluate-saved', '--manifest', str(self.manifest), '--root', str(self.root), '--run', str(out), '--output', str(test_out)], check=True, capture_output=True)
            out = test_out
            reports.append(json.loads((out / 'metrics.json').read_text()))
            preds = e.read_csv(out / 'predictions.csv')
            self.assertEqual([r['id'] for r in preds], [r['id'] for r in e.read_csv(self.manifest) if r['split'] == 'test'])
            self.assertEqual(reports[-1]['n'], 4)
        self.assertEqual(reports[0]['manifest_sha256'], reports[1]['manifest_sha256'])
        self.assertEqual(reports[0]['test_ids_sha256'], reports[1]['test_ids_sha256'])

    def test_cnn_local_state_smoke(self):
        # Random weights ONLY to verify plumbing. Not ImageNet, not transfer-learning evidence.
        import torch
        from torchvision.models import resnet18
        weights = self.root / 'synthetic_random_weights.pt'
        torch.manual_seed(1)
        torch.save(resnet18(weights=None).state_dict(), weights)
        self.prepare()
        out = self.root / 'cnn'
        subprocess.run([sys.executable, str(SCRIPT), 'run', '--manifest', str(self.manifest), '--root', str(self.root), '--model', 'cnn', '--weights', str(weights), '--epochs', '1', '--batch-size', '4', '--split', 'test', '--output', str(out)], check=True, capture_output=True)
        report = json.loads((out / 'metrics.json').read_text())
        self.assertEqual(report['n'], 4)
        self.assertEqual(report['weights_sha256'], e.digest(weights))
        self.assertTrue((out / 'head.pt').exists())
        val_out = self.root / 'cnn-val'
        subprocess.run([sys.executable, str(SCRIPT), 'run', '--manifest', str(self.manifest), '--root', str(self.root), '--model', 'cnn', '--weights', str(weights), '--epochs', '1', '--batch-size', '4', '--output', str(val_out)], check=True, capture_output=True)
        saved_out = self.root / 'cnn-saved'
        subprocess.run([sys.executable, str(SCRIPT), 'evaluate-saved', '--manifest', str(self.manifest), '--root', str(self.root), '--run', str(val_out), '--weights', str(weights), '--output', str(saved_out), '--batch-size', '4'], check=True, capture_output=True)
        self.assertEqual(e.read_csv(out / 'predictions.csv'), e.read_csv(saved_out / 'predictions.csv'))
        self.assertEqual(json.loads((saved_out / 'metrics.json').read_text())['operation'], 'evaluate saved artifact; no fitting')

        for model in ('baseline', 'hog'):
            subprocess.run([sys.executable, str(SCRIPT), 'run', '--manifest', str(self.manifest), '--root', str(self.root), '--model', model, '--split', 'test', '--output', str(self.root / model)], check=True, capture_output=True)
        directories = [self.root / name for name in ('baseline', 'hog', 'cnn')]
        e.compare(self.manifest, self.root, directories, self.root / 'comparison.json')
        self.assertEqual(len(json.loads((self.root / 'comparison.json').read_text())['models']), 3)
        report['accuracy'] = -1
        e.write_json(out / 'metrics.json', report)
        with self.assertRaisesRegex(ValueError, 'Metrics do not match'):
            e.compare(self.manifest, self.root, directories, self.root / 'tampered.json')
        pilot = subprocess.run([sys.executable, str(SCRIPT), 'estimate-cnn', '--manifest', str(self.manifest), '--root', str(self.root), '--weights', str(weights), '--sample', '2', '--epochs', '1'], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(pilot.stdout)['sample_train_only'], 2)



    def test_saved_artifact_rejects_mutation(self):
        self.prepare()
        out = self.root / 'fit'
        subprocess.run([sys.executable, str(SCRIPT), 'run', '--manifest', str(self.manifest), '--root', str(self.root), '--model', 'baseline', '--output', str(out)], check=True, capture_output=True)
        args = Namespace(manifest=self.manifest, root=self.root, run=out, output=self.root / 'evaluated', split='test', weights=None, batch_size=4)
        with (out / 'model.joblib').open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            e.evaluate_saved(args)
        self.assertFalse(args.output.exists())

    def test_missing_csv_field(self):
        self.rows[0]['path'] = ''
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'nonempty CSV fields'):
            self.prepare()

    def unsplit_source(self):
        rng = np.random.default_rng(11)
        rows = []
        for subject in range(30):
            for label in ('a', 'b'):
                ident = f's{subject:02}-{label}'
                Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(self.root / f'{ident}.png')
                rows.append(dict(id=ident, path=f'{ident}.png', label=label, subject_id=f's{subject:02}'))
        self.rows = rows
        self.write_source()

    def test_group_split_stable_and_disjoint(self):
        self.unsplit_source()
        # Transitive link: a duplicate connects two otherwise distinct subjects.
        self.rows.append(dict(id='duplicate-link', path=self.rows[0]['path'], label='a', subject_id='s01'))
        self.write_source()
        output = self.root / 'split.csv'
        report = assign(self.source, self.root, output)
        self.assertEqual(report['independent_groups'], 29)
        self.assertEqual(set(report['counts']), set(e.SPLITS))
        self.rows.reverse()
        self.write_source()
        second = self.root / 'split2.csv'
        assign(self.source, self.root, second)
        self.assertEqual(e.digest(output), e.digest(second))
        e.prepare(output, self.root, self.manifest)
        verified, _ = e.load_verified(self.manifest, self.root)
        linked = [r for r in verified if r['subject_id'] in ('s00', 's01')]
        self.assertEqual(len({r['split'] for r in linked}), 1)

    def test_group_split_preserves_existing_splits(self):
        with self.assertRaisesRegex(ValueError, 'Existing splits'):
            assign(self.source, self.root, self.root / 'split.csv')

    def test_group_split_rejects_conflicting_duplicates(self):
        self.unsplit_source()
        self.rows[1]['path'] = self.rows[0]['path']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            assign(self.source, self.root, self.root / 'split.csv')


    def recovered_fixture(self):
        source = self.root / 'recovered'
        for split in ('train', 'val', 'predict'):
            (source / 'dataset' / split).mkdir(parents=True)
        rng = np.random.default_rng(9)
        mapping = [{'id': str(i), 'label': 'a' if i % 2 else 'b'} for i in range(1, 5)]
        for path, rows in ((source / 'train.csv', mapping), (source / 'dataset/train.csv', mapping[:3]), (source / 'dataset/val.csv', mapping[3:])):
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['id', 'label'])
                w.writeheader()
                w.writerows(rows)
        for i in range(1, 7):
            split = 'train' if i <= 3 else 'val' if i == 4 else 'predict'
            image = Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8))
            image.save(source / 'dataset' / split / f'{i}.jpg')
        # Deliberately conflicting duplicate across historical train and val.
        (source / 'dataset/val/4.jpg').write_bytes((source / 'dataset/train/1.jpg').read_bytes())
        with zipfile.ZipFile(source / 'dataset.zip', 'w') as archive:
            for path in (source / 'dataset').glob('*/*.jpg'):
                archive.writestr(f'dataset/{path.name}', path.read_bytes())
        return source

    def test_recovery_inventory_reports_leakage(self):
        source = self.recovered_fixture()
        out = self.root / 'audit'
        report = audit_recovery(source, out)
        self.assertEqual(report['n_images'], 6)
        self.assertEqual(report['n_labeled'], 4)
        self.assertEqual(report['zip_mismatches'], [])
        self.assertEqual(report['split_counts'], {'train': 3, 'val': 1, 'predict': 2})
        group = report['exact_duplicate_groups']['pixel_sha256'][0]
        self.assertTrue(group['conflicting_labels'])
        self.assertTrue(group['cross_split'])
        self.assertEqual(len(e.read_csv(out / 'historical_manifest.csv')), 6)

    def test_recovery_detects_archive_mismatch(self):
        source = self.recovered_fixture()
        Image.new('L', (48, 48)).save(source / 'dataset/train/2.jpg')
        report = audit_recovery(source, self.root / 'audit')
        self.assertEqual(report['zip_mismatches'], ['2'])


    def test_review_alignment_and_constant_images(self):
        from scipy.ndimage import shift
        a = np.random.default_rng(19).normal(100, 20, (48, 48))
        b = shift(a, (1, -1), order=1, mode='nearest') + 7
        result = aligned_evidence(a, b)
        self.assertGreater(float(result['aligned_ncc']), .999)
        self.assertEqual(result['shift_y'], '1')
        self.assertEqual(result['shift_x'], '-1')
        self.assertEqual(float(aligned_evidence(np.ones((48, 48)), np.ones((48, 48)))['aligned_ncc']), 0.)

    def test_review_policy_preserves_unknown_labels_and_quarantines(self):
        source = self.recovered_fixture()
        recovery, evidence = self.root / 'audit', self.root / 'evidence'
        audit_recovery(source, recovery, distance=64)
        template = render_review(recovery, source, evidence)
        for r in template:
            r['reviewer'], r['reason'], r['decision'] = 'synthetic-test', 'Test fixture annotation, no real visual review.', 'different_image'
            if (r['id_a'], r['id_b']) == ('2', '5'):
                r['decision'] = 'uncertain'
            elif (r['id_a'], r['id_b']) == ('3', '6'):
                r['decision'] = 'near_duplicate'
        review = self.root / 'review.csv'

        def save(items):
            with review.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(template[0]))
                w.writeheader()
                w.writerows(items)

        save(template)
        out = self.root / 'proposal'
        result = propose(recovery, evidence, review, out)
        self.assertEqual(result['uncertain_groups'], 1)
        self.assertEqual(result['label_disagreement_groups'], 1)
        rows = {r['id']: r for r in e.read_csv(out / 'group_proposal.csv')}
        self.assertIn('quarantine_uncertain', rows['2']['proposed_action'])
        self.assertIn('exclude_from_internal', rows['3']['proposed_action'])
        self.assertIn('quarantine_label_disagreement', rows['1']['proposed_action'])
        self.assertEqual(rows['5']['label'], '')
        self.assertIn('do_not_impute_label', rows['5']['proposed_action'])
        # Check the verifier end-to-end on synthetic recovery artifacts, then
        # prove a recorded annotation change is rejected before any regeneration.
        record = {
            'review_sha256': e.digest(review),
            'historical_manifest_sha256': e.digest(recovery / 'historical_manifest.csv'),
            'recovery_audit_sha256': e.digest(recovery / 'audit.json'),
            'candidates_sha256': e.digest(recovery / 'similarity_candidates.csv'),
            'evidence_provenance_sha256': e.digest(evidence / 'provenance.json'),
            'evidence_template_sha256': e.digest(evidence / 'review_template.csv'),
            'review_script_sha256': e.digest(SCRIPT.with_name('review_candidates.py')),
            'policy_script_sha256': e.digest(SCRIPT.with_name('duplicate_policy.py')),
            'proposal_sha256': e.digest(out / 'group_proposal.csv'),
            'proposal_summary_sha256': e.digest(out / 'summary.json'),
            'sheets': {p.name: e.digest(p) for p in evidence.glob('candidates_*.png')},
            'decisions': dict(Counter(r['decision'] for r in template)),
        }
        record_path = self.root / 'record.json'
        e.write_json(record_path, record)
        verified = verify(recovery, evidence, review, record_path, out, source)
        self.assertEqual(verified['source_images_verified'], 6)
        self.assertEqual(verified['status'], 'verified')
        derived = self.root / 'derived'
        result = build(recovery, evidence, review, record_path, out, source, derived)
        self.assertEqual(sum(result['counts'].values()), 6)
        ledger = e.read_csv(derived / 'selection_ledger.csv')
        original = {r['id']: r for r in e.read_csv(recovery / 'historical_manifest.csv')}
        for row in ledger:
            for key, value in original[row['id']].items():
                self.assertEqual(row[key], value)
        self.assertEqual(result['counts']['excluded_internal'], 4)
        self.assertEqual(result['counts']['official_unlabeled'], 2)
        self.assertEqual(len(e.read_csv(derived / 'eligible_inventory.csv')), 0)
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            build(recovery, evidence, review, record_path, out, source, derived)

        save(template[:-1])
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            verify(recovery, evidence, review, record_path, out, source)
        with self.assertRaisesRegex(ValueError, 'every candidate exactly once'):
            propose(recovery, evidence, review, self.root / 'invalid')
        template[0]['ssim'] = 'changed'
        save(template)
        with self.assertRaisesRegex(ValueError, 'numerical evidence changed'):
            propose(recovery, evidence, review, self.root / 'invalid')

    def test_reviewed_duplicate_group_leakage_is_blocked(self):
        for r in self.rows:
            r['duplicate_group'] = ''
        self.rows[0]['duplicate_group'] = self.rows[4]['duplicate_group'] = 'reviewed-image-group'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'duplicate_group'):
            self.prepare()

    def test_group_split_preserves_reviewed_duplicate_annotations(self):
        self.unsplit_source()
        for r in self.rows:
            r['duplicate_group'] = ''
        self.rows[0]['duplicate_group'] = self.rows[2]['duplicate_group'] = 'reviewed-image-group'
        self.write_source()
        output = self.root / 'split.csv'
        report = assign(self.source, self.root, output)
        self.assertEqual(report['independent_groups'], 29)
        split_rows = e.read_csv(output)
        joined = [r for r in split_rows if r['subject_id'] in ('s00', 's01')]
        self.assertEqual(len({r['split'] for r in joined}), 1)
        e.prepare(output, self.root, self.manifest)
        verified, report = e.load_verified(self.manifest, self.root)
        self.assertEqual(sum(r['duplicate_group'] == 'reviewed-image-group' for r in verified), 2)
        self.assertIn('annotations supplied', report['near_duplicates'])


    def policy_row(self, ident='1', action='eligible_subject_to_source_and_protocol_review', split='train', label='a', group=None):
        return {'id': ident, 'path': f'{ident}.jpg', 'label': label, 'sha256': f'file-{ident}',
                'pixel_sha256': f'pixels-{ident}', 'split': split, 'subject_id': '',
                'duplicate_group': group or f'image-{ident}', 'label_disagreement': 'false',
                'uncertain_image_match': 'false', 'proposed_action': action}

    def test_inventory_partition_does_not_infer_official_labels(self):
        eligible = self.policy_row()
        official = self.policy_row('2', 'preserve_official_unlabeled; do_not_impute_label', 'predict', '')
        parts = partition([eligible, official])
        self.assertEqual(len(parts['eligible_internal']), 1)
        self.assertEqual(parts['official_unlabeled'][0]['label'], '')
        official['label'] = 'a'
        with self.assertRaisesRegex(ValueError, 'empty labels'):
            partition([eligible, official])

    def test_inventory_partition_rejects_leakage_or_unknown_actions(self):
        eligible = self.policy_row()
        excluded = self.policy_row('2', 'quarantine_uncertain_image_match_pending_second_review')
        for key in ('duplicate_group', 'sha256', 'pixel_sha256'):
            modified = dict(excluded)
            modified[key] = eligible[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overlap'):
                partition([eligible, modified])
        excluded['proposed_action'] = 'unreviewed-action'
        with self.assertRaisesRegex(ValueError, 'Unknown policy action'):
            partition([eligible, excluded])

    def test_inventory_partition_rejects_risk_flagged_eligible(self):
        for key in ('label_disagreement', 'uncertain_image_match'):
            row = self.policy_row()
            row[key] = 'true'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Risk-flagged'):
                partition([row])
        with self.assertRaisesRegex(ValueError, 'duplicate ID'):
            partition([self.policy_row(), self.policy_row()])


    def reference_fixture(self):
        from PIL.PngImagePlugin import PngInfo
        local = self.root / 'local-reference-fixture'
        reference = self.root / 'public-reference-fixture'
        local.mkdir()
        reference.mkdir()
        rng = np.random.default_rng(32)
        rows = []
        for i, label, split, suffix in ((1, 'happy', 'train', 'jpg'), (2, '', 'predict', 'jpg'), (3, 'neutral', 'val', 'png'), (4, 'angry', 'train', 'jpg')):
            path = local / f'{i}.{suffix}'
            Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(path)
            rows.append({'id': str(i), 'path': path.name, 'label': label, 'split': split,
                         'sha256': e.digest(path), 'pixel_sha256': e.pixel_hash(path), 'subject_id': ''})
        manifest = local / 'manifest.csv'
        with manifest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        archive = reference / 'fer2013-msambare-v1.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('train/happy/example.jpg', (local / '1.jpg').read_bytes())
            z.writestr('test/sad/hidden.jpg', (local / '2.jpg').read_bytes())
            import io
            extra = PngInfo()
            extra.add_text('note', 'same pixels, different file bytes')
            data = io.BytesIO()
            with Image.open(local / '3.png') as im:
                im.save(data, format='PNG', pnginfo=extra)
            z.writestr('test/neutral/example.png', data.getvalue())
        e.write_json(reference / 'metadata.json', {'id': 123, 'currentVersionNumber': 1, 'licenseName': 'synthetic test declaration only'})
        e.write_json(reference / 'download_provenance.json', {
            'dataset_page': 'https://example.invalid/synthetic-reference',
            'files': {name: {'sha256': e.digest(reference / name)} for name in ('metadata.json', archive.name)}})
        return local, reference, manifest

    def test_reference_matching_exact_pixels_and_official_redaction(self):
        source, reference, manifest = self.reference_fixture()
        out = self.root / 'matched'
        result = match(manifest, source, reference, out)
        self.assertEqual(result['content_match_counts'], {'byte_exact': 2, 'pixel_exact': 1, 'unmatched': 1})
        self.assertEqual(result['unmatched_ids'], ['4'])
        self.assertEqual(result['labeled_agreement_counts']['all_candidates_agree'], 2)
        rows = e.read_csv(out / 'content_matches.csv')
        hidden = [r for r in rows if r['id'] == '2'][0]
        self.assertEqual(hidden['local_label'], '')
        self.assertEqual(hidden['reference_class'], '')
        self.assertEqual(hidden['reference_path'], '')
        self.assertTrue(hidden['reference_path_sha256'])
        self.assertEqual(result['reference_license_declaration'], 'synthetic test declaration only')

    def test_reference_matching_rejects_changed_source_or_metadata(self):
        source, reference, manifest = self.reference_fixture()
        Image.new('L', (48, 48)).save(source / '1.jpg')
        with self.assertRaisesRegex(ValueError, 'Local image changed'):
            match(manifest, source, reference, self.root / 'invalid-match')
        (reference / 'metadata.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Reference artifact changed'):
            match(manifest, source, reference, self.root / 'invalid-match')


if __name__ == '__main__':
    unittest.main()
