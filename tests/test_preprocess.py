import unittest

import math

import torch

from src.data.preprocess import align_active_speech, preprocess_waveform, reduce_stationary_noise


class SpeechAlignmentTests(unittest.TestCase):
    def test_aligns_late_speech_without_cutting_it(self) -> None:
        sample_rate = 1000
        waveform = torch.zeros(1, 2000)
        waveform[:, 750:1350] = 0.8

        aligned = preprocess_waveform(
            waveform,
            sample_rate=sample_rate,
            target_sample_rate=sample_rate,
            target_num_samples=1000,
            normalize=False,
            align_speech=True,
        )

        active = (aligned.abs() > 0.1).nonzero(as_tuple=False)[:, -1]
        self.assertEqual(aligned.shape, (1, 1000))
        self.assertEqual(active.numel(), 600)
        self.assertGreater(active.min().item(), 0)
        self.assertLess(active.max().item(), 999)

    def test_disabled_alignment_preserves_original_front_crop(self) -> None:
        waveform = torch.zeros(1, 2000)
        waveform[:, 1200:1500] = 1.0

        processed = preprocess_waveform(
            waveform,
            sample_rate=1000,
            target_sample_rate=1000,
            target_num_samples=1000,
            normalize=False,
            align_speech=False,
        )

        self.assertEqual(processed.count_nonzero().item(), 0)

    def test_silence_returns_fixed_length_zeros(self) -> None:
        aligned = align_active_speech(
            torch.zeros(1, 400),
            sample_rate=1000,
            target_num_samples=1000,
        )

        self.assertEqual(aligned.shape, (1, 1000))
        self.assertEqual(aligned.count_nonzero().item(), 0)

    def test_noise_reduction_lowers_stationary_noise_region(self) -> None:
        sample_rate = 16000
        time_axis = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        speech = torch.zeros(1, sample_rate)
        speech[:, 6000:10000] = 0.6 * torch.sin(2 * math.pi * 440 * time_axis[6000:10000])
        noise = 0.05 * torch.sin(2 * math.pi * 120 * time_axis).unsqueeze(0)
        noisy = speech + noise

        reduced = reduce_stationary_noise(
            noisy,
            n_fft=400,
            win_length=400,
            hop_length=160,
            reduction_strength=0.9,
        )

        before_noise_rms = noisy[:, :3000].square().mean().sqrt()
        after_noise_rms = reduced[:, :3000].square().mean().sqrt()
        self.assertEqual(reduced.shape, noisy.shape)
        self.assertLess(after_noise_rms.item(), before_noise_rms.item())


if __name__ == "__main__":
    unittest.main()
