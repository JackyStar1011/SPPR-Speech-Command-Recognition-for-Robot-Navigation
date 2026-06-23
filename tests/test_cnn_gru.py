import unittest

import torch

from src.models.cnn_gru import SpeechCommandCNNGRU, build_model


class CNNGRUTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SpeechCommandCNNGRU(num_classes=6, n_mels=64)
        inputs = torch.randn(4, 1, 64, 101)

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (4, 6))

    def test_supports_variable_time_axis(self) -> None:
        model = SpeechCommandCNNGRU(num_classes=6, n_mels=64)

        outputs = model(torch.randn(2, 1, 64, 80))

        self.assertEqual(outputs.shape, (2, 6))

    def test_build_model_from_config(self) -> None:
        config = {
            "features": {"n_mels": 64},
            "model": {
                "type": "cnn_gru",
                "conv_channels": [8, 16, 32],
                "gru_hidden_size": 48,
                "gru_layers": 1,
                "bidirectional": False,
                "dropout": 0.2,
            },
        }

        model = build_model(config, num_classes=6)

        self.assertEqual(model(torch.randn(2, 1, 64, 101)).shape, (2, 6))

    def test_rejects_invalid_input_rank(self) -> None:
        model = SpeechCommandCNNGRU(num_classes=6)

        with self.assertRaises(ValueError):
            model(torch.randn(1, 64, 101))

    def test_backward_pass_produces_gradients(self) -> None:
        model = SpeechCommandCNNGRU(
            num_classes=6,
            conv_channels=(4, 8, 16),
            gru_hidden_size=16,
            gru_layers=1,
        )
        logits = model(torch.randn(2, 1, 64, 40))

        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5])).backward()

        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
