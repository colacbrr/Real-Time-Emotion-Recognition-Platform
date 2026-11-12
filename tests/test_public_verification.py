import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluation import digest
from scientific_study import LABELS
from verify_published_results import equal, split_check, unique, verify


class PublicVerificationTests(unittest.TestCase):
    def test_changed_metric_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary) / 'public'
            shutil.copytree(ROOT / 'docs/publicare/studiu_v2/results', public)
            path = public / 'results.json'
            report = json.loads(path.read_text())
            report['evaluations']['oof']['models']['majority']['accuracy'] += .01
            path.write_text(json.dumps(report))
            integrity = json.loads((public / 'integrity.json').read_text())
            integrity['results.json'] = digest(path)
            (public / 'integrity.json').write_text(json.dumps(integrity))
            with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
                verify(ROOT / 'docs/publicare/studiu_v2', public)

    def test_duplicate_prediction_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate id'):
            unique([{'id': '1'}, {'id': '1'}])

    def test_exact_hash_crossing_rejected_even_when_group_ids_differ(self):
        rows = {f'{side}-{label}': {'label': label, 'duplicate_group': f'{side}-{label}',
                                   'sha256': 'same' if label == 'happy' else f'{side}-{label}'}
                for side in ['a', 'b'] for label in LABELS}
        a, b = [[i for i in rows if i.startswith(side)] for side in ['a', 'b']]
        with self.assertRaisesRegex(ValueError, 'sha256'):
            split_check(a, b, list(rows), rows)

    def test_small_rounding_difference_allowed_but_invalid_number_rejected(self):
        equal({'score': .3 + .6}, {'score': .9})
        with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
            equal({'score': float('nan')}, {'score': .9})


if __name__ == '__main__':
    unittest.main()
