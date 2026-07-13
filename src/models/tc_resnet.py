from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    """Residual block that applies convolution along the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal alignment")

        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs) + self.shortcut(inputs))


class SpeechCommandTCResNet(nn.Module):
    """TC-ResNet style classifier for Log-Mel speech commands.

    The original TC-ResNet paper uses MFCC features. This project keeps the
    existing Log-Mel pipeline and applies the same core idea: temporal
    convolutions with residual connections over the frame sequence.
    """

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 64,
        channels: tuple[int, ...] = (16, 24, 32, 48),
        blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1),
        kernel_size: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")
        if not channels:
            raise ValueError("channels must contain at least one stage")
        if n_mels <= 0:
            raise ValueError("n_mels must be positive")

        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_channels = channels[0]
        for stage_index, (out_channels, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            if num_blocks <= 0:
                raise ValueError("each stage must contain at least one residual block")
            for block_index in range(num_blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(
                    TemporalResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        if logmel.ndim != 4:
            raise ValueError("Expected Log-Mel input with shape [batch, 1, mel, time]")
        if logmel.size(1) != 1:
            raise ValueError("Expected a single-channel Log-Mel input")

        # [B, 1, Mel, T] -> [B, Mel, T], treating Mel bins as per-frame features.
        sequence = logmel.squeeze(1)
        features = self.stem(sequence)
        features = self.stages(features)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)


def build_model(config: dict, num_classes: int) -> SpeechCommandTCResNet:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "tc_resnet")
    if model_type != "tc_resnet":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SpeechCommandTCResNet(
        num_classes=num_classes,
        n_mels=config["features"]["n_mels"],
        channels=tuple(model_cfg.get("channels", [16, 24, 32, 48])),
        blocks_per_stage=tuple(model_cfg.get("blocks_per_stage", [1, 1, 1, 1])),
        kernel_size=model_cfg.get("kernel_size", 9),
        dropout=model_cfg.get("dropout", 0.2),
    )
