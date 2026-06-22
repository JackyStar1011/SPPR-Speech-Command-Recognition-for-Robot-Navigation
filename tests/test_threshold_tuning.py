import unittest

import torch

from src.training.tune_threshold import (
    apply_confidence_threshold,
    make_thresholds,
    select_safety_threshold,
    tune_threshold,
)


class ThresholdTuningTests(unittest.TestCase):
    def test_low_confidence_predictions_become_unknown(self) -> None:
        probabilities = torch.tensor(
            [
                [0.80, 0.10, 0.10],
                [0.40, 0.35, 0.25],
                [0.10, 0.10, 0.80],
            ]
        )

        predictions = apply_confidence_threshold(probabilities, unknown_index=2, threshold=0.70)

        self.assertEqual(predictions, [0, 2, 2])

    def test_tuning_selects_best_validation_macro_f1(self) -> None:
        y_true = [0, 2, 2]
        probabilities = torch.tensor(
            [
                [0.80, 0.10, 0.10],
                [0.45, 0.35, 0.20],
                [0.10, 0.10, 0.80],
            ]
        )

        threshold, curve = tune_threshold(
            y_true,
            probabilities,
            unknown_index=2,
            thresholds=[0.40, 0.50, 0.90],
        )

        self.assertEqual(threshold, 0.50)
        self.assertEqual(len(curve), 3)

    def test_threshold_range_includes_both_ends(self) -> None:
        self.assertEqual(make_thresholds(0.50, 0.52, 0.01), [0.50, 0.51, 0.52])

    def test_safety_threshold_meets_unknown_recall_target(self) -> None:
        curve = [
            {"threshold": 0.50, "accuracy": 0.90, "macro_f1": 0.88, "unknown_recall": 0.70},
            {"threshold": 0.60, "accuracy": 0.86, "macro_f1": 0.85, "unknown_recall": 0.82},
            {"threshold": 0.70, "accuracy": 0.80, "macro_f1": 0.78, "unknown_recall": 0.90},
        ]

        threshold = select_safety_threshold(curve, minimum_unknown_recall=0.80)

        self.assertEqual(threshold, 0.60)


if __name__ == "__main__":
    unittest.main()
