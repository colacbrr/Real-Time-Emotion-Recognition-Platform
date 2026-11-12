"""Synthetic classification regression tests; no real data, weights or network."""
import base64
import io
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import server


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('_ready', True), ('_init_error', None), ('_models', {}), ('_cnn_models', {})]:
            self.stack.enter_context(patch.object(server, name, value))
        self.stack.enter_context(patch.object(server, '_append_metrics_event'))
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        buffer = io.BytesIO()
        Image.new('L', (48, 48), color=128).save(buffer, format='PNG')
        self.payload = {'image_base64': base64.b64encode(buffer.getvalue()).decode()}

    def test_routes_expose_classification_and_status(self):
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertTrue({'/predict', '/compare', '/models', '/status', '/reports', '/events', '/logs'} <= set(paths))
        self.assertFalse({'/search', '/rag', '/explain', '/ollama/models'} & set(paths))
        status = self.client.get('/status').json()
        self.assertNotIn('mir', status)
        self.assertEqual(status['models'], [])

    def test_no_model_returns_error_without_fabricated_label(self):
        result = self.client.post('/predict', json=self.payload).json()
        self.assertIn('error', result)
        self.assertNotIn('label', result)
        self.assertIn('error', self.client.post('/compare', json=self.payload).json())

    def test_classical_prediction_and_comparison(self):
        params = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        model = DummyClassifier(strategy='constant', constant='happy').fit(np.zeros((2, 900)), ['happy', 'sad'])
        server._models['synthetic_hog'] = {'pipeline': model, 'meta': {'extractor': 'hog', 'feature_params': params}}
        result = self.client.post('/predict', json=self.payload).json()
        self.assertEqual(result['label'], 'happy')
        self.assertEqual(result['model'], 'synthetic_hog')
        comparison = self.client.post('/compare', json=self.payload).json()
        self.assertEqual(comparison['results'][0]['label'], 'happy')

    def test_model_failure_is_not_replaced_by_neutral(self):
        server._models['broken'] = {'pipeline': None, 'meta': {}}
        result = self.client.post('/predict', json=self.payload).json()
        self.assertIn('Prediction failed', result['error'])
        self.assertNotIn('label', result)

    def test_cnn_works_without_retrieval_imports(self):
        self.assertIsNotNone(server.torch)
        self.assertIsNotNone(server.tv_models)
        server.torch.set_num_threads(min(4, server.torch.get_num_threads()))
        model = server._build_cnn_model('small_cnn', 2).eval()
        server._cnn_models['synthetic_cnn'] = {
            'checkpoint': {'labels': ['happy', 'sad'], 'image_size': 48, 'use_rgb': False},
            'model': model, 'device': server.torch.device('cpu')}
        result = self.client.post('/predict', json=self.payload).json()
        self.assertIn(result['label'], ['happy', 'sad'])
        self.assertEqual(result['model'], 'synthetic_cnn')
        self.assertAlmostEqual(sum(result['normalized_scores'].values()), 1., places=6)


if __name__ == '__main__':
    unittest.main()
