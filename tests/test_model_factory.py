import unittest

import torch

from src.models import (
    SpeechCommandBCResNet,
    SpeechCommandCNN,
    SpeechCommandCNNGRU,
    SpeechCommandTCResNet,
    build_model,
)


class ModelFactoryTests(unittest.TestCase):
    def test_builds_baseline_cnn(self) -> None:
        config = {
            "features": {"n_mels": 64},
            "model": {"type": "cnn", "dropout": 0.2},
        }

        model = build_model(config, num_classes=6)

        self.assertIsInstance(model, SpeechCommandCNN)
        self.assertEqual(model(torch.randn(2, 1, 64, 101)).shape, (2, 6))

    def test_builds_cnn_gru(self) -> None:
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

        self.assertIsInstance(model, SpeechCommandCNNGRU)
        self.assertEqual(model(torch.randn(2, 1, 64, 101)).shape, (2, 6))

    def test_builds_tc_resnet(self) -> None:
        config = {
            "features": {"n_mels": 64},
            "model": {
                "type": "tc_resnet",
                "channels": [8, 12, 16],
                "blocks_per_stage": [1, 1, 1],
                "kernel_size": 5,
                "dropout": 0.1,
            },
        }

        model = build_model(config, num_classes=6)

        self.assertIsInstance(model, SpeechCommandTCResNet)
        self.assertEqual(model(torch.randn(2, 1, 64, 101)).shape, (2, 6))

    def test_builds_bc_resnet(self) -> None:
        config = {
            "features": {"n_mels": 64},
            "model": {
                "type": "bc_resnet",
                "channels": [8, 12, 16],
                "blocks_per_stage": [1, 1, 1],
                "kernel_size": 3,
                "dropout": 0.1,
            },
        }

        model = build_model(config, num_classes=6)

        self.assertIsInstance(model, SpeechCommandBCResNet)
        self.assertEqual(model(torch.randn(2, 1, 64, 101)).shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
