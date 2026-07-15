import unittest

import torch

from src.features.bc_resnet_recipe import (
    BCResNetRecipePreprocessor,
    NaturalLogMelExtractor,
    SpecAugment,
)
from src.features.logmel import build_logmel_extractor
from src.training.train_bc_resnet_recipe import paper_learning_rate


class BCResNetRecipeTests(unittest.TestCase):
    def test_natural_logmel_shape(self) -> None:
        extractor = NaturalLogMelExtractor()

        features = extractor(torch.zeros(3, 1, 16000))

        self.assertEqual(features.shape, (3, 1, 40, 101))
        self.assertTrue(torch.isfinite(features).all())

    def test_shared_feature_factory_selects_recipe_frontend(self) -> None:
        config = {
            "data": {"sample_rate": 16000},
            "features": {
                "type": "natural_log_mel",
                "n_fft": 512,
                "win_length": 480,
                "hop_length": 160,
                "n_mels": 40,
                "log_epsilon": 1e-6,
            },
        }

        extractor = build_logmel_extractor(config)

        self.assertIsInstance(extractor, NaturalLogMelExtractor)

    def test_frequency_mask_param_one_matches_reference_noop(self) -> None:
        augmenter = SpecAugment(
            frequency_mask_param=1,
            time_mask_param=0,
            frequency_mask_count=2,
            time_mask_count=0,
        )
        features = torch.randn(2, 1, 40, 101)

        augmented = augmenter(features)

        self.assertTrue(torch.equal(augmented, features))

    def test_waveform_recipe_adds_registered_background_noise(self) -> None:
        preprocessor = BCResNetRecipePreprocessor(
            feature_extractor=NaturalLogMelExtractor(),
            spec_augment=SpecAugment(0, 0, 0, 0),
            sample_rate=16000,
            sample_length=16000,
            augmentation_config={
                "enabled": True,
                "probability": 1.0,
                "time_shift_ms": 0.0,
                "noise_amplitude_min": 0.1,
                "noise_amplitude_max": 0.1,
            },
        )
        preprocessor.register_buffer("_background_noise_0", torch.ones(1, 16000))
        preprocessor.background_noise_count = 1

        augmented = preprocessor.augment_waveforms(torch.zeros(2, 1, 16000))

        self.assertTrue(torch.allclose(augmented, torch.full_like(augmented, 0.1)))

    def test_learning_rate_warmup_and_cosine_endpoints(self) -> None:
        self.assertAlmostEqual(paper_learning_rate(1, 100, 10, 0.1), 0.01)
        self.assertAlmostEqual(paper_learning_rate(10, 100, 10, 0.1), 0.1)
        self.assertAlmostEqual(paper_learning_rate(100, 100, 10, 0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
