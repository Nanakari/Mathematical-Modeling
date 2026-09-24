import unittest
import numpy as np
from aligned_dataset import AlignedDataset


class DatasetTest(unittest.TestCase):
    def part(self):
        b = np.zeros((1, 3, 50), dtype=np.float32)
        b[0, 0, :5] = [101, 1200, 100, 2200, 102]
        b[0, 1, :5] = 1
        a, v = np.zeros((1, 50, 74)), np.zeros((1, 50, 35))
        a[:, 1:4] = 1
        v[:, 1] = 1
        return {"text_bert": b, "audio": a, "vision": v,
                "classification_labels": np.array([1.]), "regression_labels": np.array([0.])}

    def test_types_masks_label_and_no_mutation(self):
        p = self.part()
        item = AlignedDataset(p, labeled=True, special_token_ids=(101, 102))[0]
        self.assertEqual(item["input_ids"].dtype, np.int64)
        self.assertEqual(item["audio"].dtype, np.float32)
        self.assertEqual(int(item["text_pool_mask"].sum()), 3)
        self.assertTrue(item["text_pool_mask"][2])  # ID 100 is not assumed missing.
        self.assertEqual(int(item["audio_observed"].sum()), 3)
        self.assertEqual(int(item["vision_observed"].sum()), 1)
        item["audio"][:] = 99
        self.assertEqual(p["audio"][0, 0, 0], 0)

    def test_unlabeled_has_no_fake_targets(self):
        item = AlignedDataset(self.part(), labeled=False)[0]
        self.assertFalse(item["has_label"])
        self.assertNotIn("class_label", item)

    def test_reject_invalid_inputs(self):
        for mutation in (lambda p: p["text_bert"].__setitem__((0, 0, 1), 2.5),
                         lambda p: p["text_bert"].__setitem__((0, 1, 2), 0),
                         lambda p: p["audio"].__setitem__((0, 49, 0), 1),
                         lambda p: p.pop("text_bert")):
            p = self.part()
            mutation(p)
            with self.assertRaises(ValueError):
                AlignedDataset(p, labeled=True)


if __name__ == "__main__":
    unittest.main()
