import unittest
import numpy as np
from p2 import MODALITIES, TinyMaskedModel, corrupt, metrics, pooled_features


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = {m: rng.normal(size=(3, 10, d)) for m, d in zip(MODALITIES, [5, 3, 2])}
        self.v = {m: np.tile(np.arange(10) < 8, (3, 1)) for m in MODALITIES}
        self.a = {m: v.copy() for m, v in self.v.items()}
        self.ids = ["one", "two", "three"]

    def test_reproducible_local_masks_and_no_source_mutation(self):
        before = {m: x.copy() for m, x in self.x.items()}
        args = (self.x, self.v, self.a, self.ids)
        one = corrupt(*args, modalities=MODALITIES, rate=.5, seed=3)
        two = corrupt(*args, modalities=MODALITIES, rate=.5, seed=3)
        for m in MODALITIES:
            np.testing.assert_array_equal(one[0][m], two[0][m])
            np.testing.assert_array_equal(self.x[m], before[m])
            self.assertFalse(np.any(one[2][m] & ~self.v[m]))
            np.testing.assert_array_equal(one[2][m].sum(1), [4, 4, 4])
        self.assertEqual(one[3], two[3])

    def test_position_multiple_sync_and_short_sequence(self):
        for pos, interval in [("start", [0, 4]), ("middle", [2, 6]), ("end", [4, 8])]:
            out = corrupt(self.x, self.v, self.a, self.ids, rate=.5, position=pos)
            self.assertEqual(out[3][0]["intervals"], [interval])
        out = corrupt(self.x, self.v, self.a, self.ids, modalities=MODALITIES,
                      rate=.75, segments=2, synchronized=True)
        for m in MODALITIES:
            np.testing.assert_array_equal(out[2][m], out[2]["text"])
            self.assertTrue(np.all(out[2][m].sum(1) == 6))
        short = {m: np.zeros_like(v) for m, v in self.v.items()}
        for m in MODALITIES:
            short[m][:, 0] = True
        out = corrupt(self.x, short, short, self.ids, rate=.9)
        self.assertEqual(int(out[2]["text"].sum()), 0)

    def test_padding_invariance_and_empty_modality(self):
        net = TinyMaskedModel({m: x.shape[-1] for m, x in self.x.items()})
        p, r, _ = net.forward(self.x, self.a)
        extended = {m: np.pad(x, ((0, 0), (0, 5), (0, 0)), constant_values=999) for m, x in self.x.items()}
        masks = {m: np.pad(a, ((0, 0), (0, 5))) for m, a in self.a.items()}
        p2, r2, _ = net.forward(extended, masks)
        np.testing.assert_allclose(p, p2)
        np.testing.assert_allclose(r, r2)
        masks = {m: np.zeros_like(a) for m, a in self.a.items()}
        self.assertTrue(np.isfinite(net.forward(self.x, masks)[0]).all())
        self.assertTrue(np.isfinite(pooled_features(self.x, masks)).all())

    def test_gradients_against_finite_differences_and_training(self):
        net = TinyMaskedModel({m: x.shape[-1] for m, x in self.x.items()}, hidden=4)
        y, r = np.array([0, 1, 2]), np.array([-1., 0., 1.])
        loss, grads = net.loss_grad(self.x, self.a, y, r)
        for key, value in net.parameters.items():
            idx = tuple(0 for _ in value.shape)
            old, eps = value[idx], 1e-5
            value[idx] = old + eps
            high = net.loss_grad(self.x, self.a, y, r)[0]
            value[idx] = old - eps
            low = net.loss_grad(self.x, self.a, y, r)[0]
            value[idx] = old
            self.assertAlmostEqual((high-low)/(2*eps), grads[key][idx], places=6)
        for _ in range(40):
            _, g = net.loss_grad(self.x, self.a, y, r)
            for k in g:
                net.parameters[k] -= .02 * g[k]
        self.assertLess(net.loss_grad(self.x, self.a, y, r)[0], loss)

    def test_metrics_constant_pearson(self):
        result = metrics([0, 1, 2], [-1, 0, 1], [0, 1, 2], [0, 0, 0])
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["f1_macro"], 1.)
        self.assertIsNone(result["pearson"])

    def test_preserve_available_information_and_stable_ids(self):
        sparse = {m: np.zeros_like(a) for m, a in self.a.items()}
        for m in MODALITIES:
            sparse[m][:, 3] = True
        result = corrupt(self.x, self.v, sparse, self.ids, modalities=MODALITIES,
                         rate=.9, segments=3, synchronized=True)
        for m in MODALITIES:
            self.assertTrue(np.all(result[1][m].sum(1) == 1))
            np.testing.assert_array_equal(result[2][m], result[2]["text"])
        first = corrupt(self.x, self.v, self.a, self.ids, rate=.5, seed=77)
        reverse = corrupt({m:x[::-1] for m,x in self.x.items()},
                          {m:x[::-1] for m,x in self.v.items()},
                          {m:x[::-1] for m,x in self.a.items()}, self.ids[::-1], rate=.5, seed=77)
        np.testing.assert_array_equal(first[2]["text"], reverse[2]["text"][::-1])


if __name__ == "__main__":
    unittest.main()
