import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluation import metrics
from scientific_study import LABELS, cohort, grouped_splits, cm_scores, bootstrap


class ScientificTests(unittest.TestCase):
    def test_source_partition_is_conservative(self):
        rows = [{'id': str(i), 'duplicate_group': g} for i, g in [(1, 'a'), (2, 'a'), (3, 'b'), (4, 'c')]]
        matches = [{'id': str(i), 'method': 'byte_exact', 'reference_split': split}
                   for i, split in [(1, 'train'), (2, 'test'), (3, 'train'), (3, 'test'), (4, 'test')]]
        actual = cohort(rows, matches)
        self.assertEqual([r['role'] for r in actual], ['development', 'excluded_known_group_crossing',
                         'excluded_ambiguous_source_split', 'holdout'])

    def test_metrics_formula_including_absent_prediction_classes(self):
        truth = LABELS * 4
        pred = ['happy'] * len(truth)
        actual = cm_scores(confusion_matrix(truth, pred, labels=LABELS))
        expected = metrics(truth, pred, LABELS)
        np.testing.assert_allclose(actual, [expected['accuracy'], expected['macro_f1']])

    def test_identical_models_have_exact_zero_paired_intervals(self):
        rows = [{'label': label, 'duplicate_group': f'{label}-{j // 2}'} for label in LABELS for j in range(8)]
        pred = [r['label'] for r in rows]
        actual = bootstrap(rows, {'resnet_rbf': pred, 'hog_rbf': pred}, replicates=50)
        self.assertEqual(actual['groups'], 28)
        self.assertEqual(actual['paired_differences'][0]['interval_95']['macro_f1'], [0., 0.])
        self.assertEqual(actual['model_intervals_95']['resnet_rbf']['macro_f1'], [1., 1.])

    def test_mixed_group_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'consistent labels'):
            bootstrap([{'label': 'happy', 'duplicate_group': 'a'}, {'label': 'sad', 'duplicate_group': 'a'}],
                      {'majority': ['happy', 'happy']}, replicates=5)

    def test_outer_groups_disjoint_and_each_image_tested_once(self):
        rows = [{'label': label, 'duplicate_group': f'{label}-{j // 2}'} for label in LABELS for j in range(30)]
        splits = grouped_splits(rows, 5, 42)
        tested = []
        for tr, te in splits:
            self.assertFalse({rows[i]['duplicate_group'] for i in tr} & {rows[i]['duplicate_group'] for i in te})
            tested.extend(te)
        self.assertEqual(sorted(tested), list(range(len(rows))))


if __name__ == '__main__':
    unittest.main()
