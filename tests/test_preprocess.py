import unittest

import torch

from src.data.preprocess import align_active_speech, preprocess_waveform


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


if __name__ == "__main__":
    unittest.main()
