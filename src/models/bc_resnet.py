from __future__ import annotations

import torch
from torch import nn


class BroadcastedResidualBlock(nn.Module):
    """BC-ResNet style block with local 2D features and broadcasted temporal context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal alignment")

        padding = kernel_size // 2
        self.local = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local_features = self.local(inputs)
        temporal_context = local_features.mean(dim=2)
        temporal_context = self.temporal(temporal_context).unsqueeze(2)
        fused = self.fuse(local_features + temporal_context)
        return self.activation(fused + self.shortcut(inputs))


class SpeechCommandBCResNet(nn.Module):
    """BC-ResNet style classifier for Log-Mel speech commands.

    The model keeps two complementary views of the Log-Mel input: local
    time-frequency patterns from 2D convolutions and temporal context from a
    frequency-pooled 1D branch that is broadcast back to the spectrogram map.
    """

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 64,
        channels: tuple[int, ...] = (16, 24, 32, 48),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 3,
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
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_channels = channels[0]
        for stage_index, (out_channels, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            if num_blocks <= 0:
                raise ValueError("each stage must contain at least one residual block")
            for block_index in range(num_blocks):
                stride = (2, 2) if stage_index > 0 and block_index == 0 else (1, 1)
                stages.append(
                    BroadcastedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        if logmel.ndim != 4:
            raise ValueError("Expected Log-Mel input with shape [batch, 1, mel, time]")
        if logmel.size(1) != 1:
            raise ValueError("Expected a single-channel Log-Mel input")

        features = self.stem(logmel)
        features = self.stages(features)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(pooled)


def build_model(config: dict, num_classes: int) -> SpeechCommandBCResNet:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "bc_resnet")
    if model_type != "bc_resnet":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SpeechCommandBCResNet(
        num_classes=num_classes,
        n_mels=config["features"]["n_mels"],
        channels=tuple(model_cfg.get("channels", [16, 24, 32, 48])),
        blocks_per_stage=tuple(model_cfg.get("blocks_per_stage", [2, 2, 2, 2])),
        kernel_size=model_cfg.get("kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.2),
    )
