from __future__ import annotations

import torch
from torch import nn


class SpeechCommandCNNGRU(nn.Module):
    """CRNN classifier for fixed-length Log-Mel spectrograms.

    The CNN reduces the frequency axis and downsamples time moderately. The GRU
    then models the remaining frame sequence before temporal mean pooling.
    """

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 64,
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden_size: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError("conv_channels must contain exactly three values")
        if n_mels < 8:
            raise ValueError("n_mels must be at least 8 for three frequency pooling stages")

        c1, c2, c3 = conv_channels
        self.cnn = nn.Sequential(
            self._conv_block(1, c1, pool_size=(2, 2)),
            self._conv_block(c1, c2, pool_size=(2, 2)),
            self._conv_block(c2, c3, pool_size=(2, 1)),
        )

        frequency_bins = n_mels // 8
        gru_input_size = c3 * frequency_bins
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_output_size = gru_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, num_classes),
        )

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        pool_size: tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        if logmel.ndim != 4:
            raise ValueError("Expected Log-Mel input with shape [batch, 1, mel, time]")

        features = self.cnn(logmel)
        # [B, C, F, T] -> [B, T, C * F]
        sequence = features.permute(0, 3, 1, 2).contiguous().flatten(start_dim=2)
        sequence, _ = self.gru(sequence)
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)


def build_model(config: dict, num_classes: int) -> SpeechCommandCNNGRU:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "cnn_gru")
    if model_type != "cnn_gru":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SpeechCommandCNNGRU(
        num_classes=num_classes,
        n_mels=config["features"]["n_mels"],
        conv_channels=tuple(model_cfg.get("conv_channels", [16, 32, 64])),
        gru_hidden_size=model_cfg.get("gru_hidden_size", 128),
        gru_layers=model_cfg.get("gru_layers", 2),
        bidirectional=model_cfg.get("bidirectional", False),
        dropout=model_cfg.get("dropout", 0.3),
    )
