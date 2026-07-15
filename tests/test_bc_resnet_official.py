import unittest

import torch

from src.models.bc_resnet_official import SpeechCommandBCResNet15, SubSpectralNorm, build_model


class OfficialBCResNetTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SpeechCommandBCResNet15(num_classes=6)

        outputs = model(torch.randn(4, 1, 40, 101))

        self.assertEqual(outputs.shape, (4, 6))

    def test_build_model_from_config(self) -> None:
        config = {
            "model": {"type": "bc_resnet_1_5", "tau": 1.5, "dropout": 0.1},
        }

        model = build_model(config, num_classes=6)

        self.assertEqual(model(torch.randn(2, 1, 40, 101)).shape, (2, 6))

    def test_rejects_invalid_input_rank(self) -> None:
        model = SpeechCommandBCResNet15(num_classes=6)

        with self.assertRaises(ValueError):
            model(torch.randn(1, 40, 101))

    def test_subspectral_norm_requires_divisible_frequency(self) -> None:
        norm = SubSpectralNorm(8, spec_groups=5)

        with self.assertRaises(ValueError):
            norm(torch.randn(2, 8, 12, 20))

    def test_backward_pass_produces_gradients(self) -> None:
        model = SpeechCommandBCResNet15(num_classes=6, tau=1.0)
        logits = model(torch.randn(2, 1, 40, 50))

        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5])).backward()

        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
