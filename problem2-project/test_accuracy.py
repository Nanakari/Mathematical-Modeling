import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from accuracy_model import TinyDual, class_weights, predict


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def batch(self):
        return {'ids': torch.tensor([[101, 2054, 0, 102], [0, 0, 0, 0]]),
                'attention': torch.tensor([[True, True, False, True], [False] * 4]),
                'segments': torch.zeros(2, 4, dtype=torch.long),
                'pool': torch.tensor([[False, True, False, False], [False] * 4]),
                'av': torch.zeros(2, 657)}

    def test_class_weights_preserve_population_scale(self):
        y = np.array([0, 1, 1, 2, 2, 2])
        for gamma in [0., .5, 1.]:
            w = class_weights(y, gamma)
            self.assertAlmostEqual(float(w[y].mean()), 1.)
        np.testing.assert_allclose(class_weights(y, 0), [1, 1, 1])
        np.testing.assert_allclose(class_weights(y, 1), len(y) / (3 * np.bincount(y)))

    def test_missing_tokens_invariant_empty_input_finite_reload(self):
        model = TinyDual('models/bert_tiny')
        b = self.batch(); expected = predict(model, b)
        b['ids'][~b['attention']] = 999999
        b['segments'][~b['attention']] = 999999
        for a, c in zip(expected, predict(model, b)):
            np.testing.assert_array_equal(a, c)
            self.assertTrue(np.isfinite(a).all())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'; torch.save(model.state_dict(), path)
            reloaded = TinyDual('models/bert_tiny')
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            for a, c in zip(expected, predict(reloaded, b)):
                np.testing.assert_array_equal(a, c)

    def test_encoder_changes_only_when_trainable(self):
        for frozen in [False, True]:
            model = TinyDual('models/bert_tiny', frozen=frozen, dropout=0.)
            model.train()
            before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            logits, reg = model(**self.batch())
            (torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])) + reg.square().mean()).backward()
            optimizer.step()
            changed = not torch.equal(before, model.encoder.embeddings.word_embeddings.weight)
            self.assertEqual(changed, not frozen)
            self.assertEqual(model.encoder.training, not frozen)


if __name__ == '__main__':
    unittest.main()
