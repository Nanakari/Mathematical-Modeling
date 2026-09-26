"""Focused CPU checks for optional post-encoder training losses and schedules."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_train


def _batch(labels=(0, 1, 2)):
    return {
        "class_label": torch.tensor(labels, dtype=torch.long),
        "regression_label": torch.tensor([-1.0, 0.0, 1.0][:len(labels)]),
    }


def _output(*, intensity_raw=None, intensity_direct=None):
    values = {
        "logits": torch.tensor([[1.2, -0.4, 0.1], [-0.2, 0.7, 0.3], [0.0, 0.2, 0.9]],
                                dtype=torch.float32),
        "intensity_raw": (torch.tensor([-0.6, 0.2, 0.8]) if intensity_raw is None
                          else intensity_raw),
        "neutral_logit": None,
    }
    if intensity_direct is not None:
        values["intensity_direct"] = intensity_direct
    return values


class PostEncoderLossTests(unittest.TestCase):
    def test_default_classification_loss_matches_existing_ce_and_weighted_ce(self):
        batch = _batch()
        output = _output()
        expected = F.cross_entropy(output["logits"], batch["class_label"])
        losses = v5_train.training_objective(output, batch, training={"neutral_aux_weight": 0})
        torch.testing.assert_close(losses["classification"], expected)
        self.assertEqual(losses["consistency"].item(), 0.0)

        weights = [1.0, 2.5, 0.75]
        expected_weighted = F.cross_entropy(output["logits"], batch["class_label"],
                                            weight=torch.tensor(weights))
        weighted = v5_train.training_objective(
            output, batch,
            training={"neutral_aux_weight": 0, "class_weights": weights})
        torch.testing.assert_close(weighted["classification"], expected_weighted)

    def test_focal_gamma_zero_matches_ce_and_focal_backward_is_finite(self):
        batch = _batch()
        logits = _output()["logits"].detach().requires_grad_(True)
        ce = v5_train._classification_loss(logits, batch["class_label"])
        focal_zero = v5_train._classification_loss(
            logits, batch["class_label"], loss_type="focal", focal_gamma=0)
        torch.testing.assert_close(focal_zero, ce)

        weights = torch.tensor([1.0, 2.5, 0.75])
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, batch["class_label"].view(-1, 1)).squeeze(1)
        expected = (((1.0 - target_log_probs.exp()) ** 1.5)
                    * -target_log_probs * weights[batch["class_label"]]).mean()
        focal = v5_train._classification_loss(
            logits, batch["class_label"], class_weights=weights,
            loss_type="focal", focal_gamma=1.5)
        torch.testing.assert_close(focal, expected)
        focal.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_consistency_loss_backpropagates_to_direct_and_raw_and_requires_direct(self):
        batch = _batch(labels=(0, 1))
        direct = torch.tensor([1.0, -0.7], requires_grad=True)
        raw = torch.tensor([-0.5, 0.8], requires_grad=True)
        output = {
            "logits": torch.tensor([[0.8, 0.1, -0.2], [0.1, 0.7, -0.3]], requires_grad=True),
            "intensity_raw": raw,
            "intensity_direct": direct,
            "neutral_logit": None,
        }
        losses = v5_train.training_objective(
            output, batch,
            training={"regression_loss_weight": 0, "neutral_aux_weight": 0,
                      "consistency_weight": 1.0})
        expected = F.smooth_l1_loss(direct, raw)
        torch.testing.assert_close(losses["consistency"], expected)
        losses["total"].backward()
        self.assertIsNotNone(direct.grad)
        self.assertGreater(torch.count_nonzero(direct.grad).item(), 0)
        self.assertIsNotNone(raw.grad)
        self.assertGreater(torch.count_nonzero(raw.grad).item(), 0)

        with self.assertRaisesRegex(ValueError, "intensity_direct"):
            v5_train.training_objective(
                _output(), _batch(),
                training={"neutral_aux_weight": 0, "consistency_weight": 0.1})

    def test_severity_ramp_rates_at_start_end_and_after_ramp(self):
        training = {"missing_curriculum": "severity_ramp", "mask_rates": [0.1, 0.3, 0.5]}
        first = v5_train._missing_curriculum_state(training, epoch_index=0)
        fifth = v5_train._missing_curriculum_state(training, epoch_index=4)
        later = v5_train._missing_curriculum_state(training, epoch_index=12)
        self.assertEqual(first["factor"], 0.5)
        torch.testing.assert_close(torch.tensor(first["mask_rates"]),
                                   torch.tensor((0.05, 0.15, 0.25)))
        self.assertEqual(fifth["factor"], 1.0)
        torch.testing.assert_close(torch.tensor(fifth["mask_rates"]),
                                   torch.tensor((0.1, 0.3, 0.5)))
        self.assertEqual(later, fifth)

    def test_default_missing_view_sequence_matches_original_sampler(self):
        samples = [{"sample_id": str(i)} for i in range(20)]
        indices = [0, 3, 5, 7, 9, 11, 13, 17]

        def fake_make_missing_view(sample, modalities, rate, position, seed, *, nested_rates):
            return {"sample_id": sample["sample_id"], "plan":
                    (tuple(modalities), rate, position, seed, tuple(nested_rates))}

        def sample_using_default_helper():
            with patch.object(v5_train, "make_missing_view", side_effect=fake_make_missing_view):
                return v5_train._make_missing_batch_views(
                    samples, indices, 1729, 2,
                    {"training": {"view_policy": "mixed", "mask_rates": [0.1, 0.3, 0.5],
                                  "clean_probability": 0.6}})

        def sample_using_previous_path():
            with patch.object(v5_train, "make_missing_view", side_effect=fake_make_missing_view):
                return v5_train._sample_views(
                    samples, indices, policy="mixed", seed=1729, epoch=2,
                    rates=(0.1, 0.3, 0.5), clean_probability=0.6)

        self.assertEqual(sample_using_default_helper(), sample_using_previous_path())
