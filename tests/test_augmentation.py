import unittest

import torch

from src.data.augmentation import add_noise_at_snr, apply_waveform_augmentation, shift_waveform


class WaveformAugmentationTests(unittest.TestCase):
    def test_shift_waveform_preserves_length_with_zero_padding(self) -> None:
        waveform = torch.arange(1, 6, dtype=torch.float32).unsqueeze(0)

        shifted = shift_waveform(waveform, shift_samples=2)

        self.assertEqual(shifted.shape, waveform.shape)
        self.assertTrue(torch.equal(shifted, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]])))

    def test_add_noise_at_snr_matches_target_ratio(self) -> None:
        waveform = torch.ones(1, 16000)
        noise = torch.full_like(waveform, 0.5)

        augmented = add_noise_at_snr(waveform, snr_db=10.0, noise=noise)
        added_noise = augmented - waveform
        actual_snr = 20.0 * torch.log10(
            waveform.square().mean().sqrt() / added_noise.square().mean().sqrt()
        )

        self.assertAlmostEqual(actual_snr.item(), 10.0, places=4)

    def test_disabled_config_returns_original_waveform(self) -> None:
        waveform = torch.randn(1, 16000)

        augmented = apply_waveform_augmentation(
            waveform,
            sample_rate=16000,
            config={"enabled": False},
        )

        self.assertTrue(torch.equal(augmented, waveform))

    def test_enabled_config_keeps_fixed_shape_and_audio_range(self) -> None:
        generator = torch.Generator().manual_seed(7)
        waveform = torch.zeros(1, 16000)
        waveform[:, 4000:8000] = 0.5
        config = {
            "enabled": True,
            "normalize_after": True,
            "gain": {"enabled": True, "probability": 1.0, "min_db": 3.0, "max_db": 3.0},
            "time_shift": {"enabled": True, "probability": 1.0, "max_ms": 10.0},
            "add_noise": {"enabled": True, "probability": 1.0, "min_snr_db": 20.0, "max_snr_db": 20.0},
        }

        augmented = apply_waveform_augmentation(
            waveform,
            sample_rate=16000,
            config=config,
            generator=generator,
        )

        self.assertEqual(augmented.shape, waveform.shape)
        self.assertLessEqual(augmented.abs().max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
