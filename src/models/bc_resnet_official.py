from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SubSpectralNorm(nn.Module):
    """Sub-spectral normalization used by the official BC-ResNet blocks."""

    def __init__(self, num_features: int, spec_groups: int = 5) -> None:
        super().__init__()
        if spec_groups <= 0:
            raise ValueError("spec_groups must be positive")
        self.spec_groups = spec_groups
        self.norm = nn.BatchNorm2d(num_features * spec_groups, affine=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, time = inputs.shape
        if freq % self.spec_groups != 0:
            raise ValueError("frequency bins must be divisible by spec_groups")
        features = inputs.contiguous().view(
            batch,
            channels * self.spec_groups,
            freq // self.spec_groups,
            time,
        )
        features = self.norm(features)
        return features.view(batch, channels, freq, time)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_index: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        use_dilation: bool = False,
        activation: str | None = "relu",
        subspectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = tuple(self._padding(k, stage_index, use_dilation)[0] for k in kernel_size)
            dilation = tuple(self._padding(k, stage_index, use_dilation)[1] for k in kernel_size)
        else:
            padding, dilation = self._padding(kernel_size, stage_index, use_dilation)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        ]
        layers.append(SubSpectralNorm(out_channels, spec_groups=5) if subspectral_norm else nn.BatchNorm2d(out_channels))
        if activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    @staticmethod
    def _padding(kernel_size: int, stage_index: int, use_dilation: bool) -> tuple[int, int]:
        dilation = 2**stage_index if use_dilation and kernel_size > 1 else 1
        return dilation * ((kernel_size - 1) // 2), dilation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class OfficialBCResBlock(nn.Module):
    """Broadcasted residual block following the official BC-ResNet design."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_index: int,
        stride: tuple[int, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.transition = in_channels != out_channels

        two_dimensional: list[nn.Module] = []
        if self.transition:
            two_dimensional.append(ConvNormAct(in_channels, out_channels, stage_index, kernel_size=1))
            in_channels = out_channels
        two_dimensional.append(
            ConvNormAct(
                in_channels,
                out_channels,
                stage_index,
                kernel_size=(3, 1),
                stride=(stride[0], 1),
                groups=in_channels,
                activation=None,
                subspectral_norm=True,
            )
        )
        self.local_branch = nn.Sequential(*two_dimensional)
        self.frequency_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_branch = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                stage_index,
                kernel_size=(1, 3),
                stride=(1, stride[1]),
                groups=out_channels,
                use_dilation=True,
                activation="silu",
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs
        local_features = self.local_branch(inputs)
        temporal_context = self.frequency_pool(local_features)
        temporal_context = self.temporal_branch(temporal_context)
        outputs = temporal_context + local_features
        if not self.transition:
            outputs = outputs + shortcut
        return F.relu(outputs, inplace=True)


class SpeechCommandBCResNet15(nn.Module):
    """Official-style BC-ResNet-1.5 classifier adapted to the project labels."""

    def __init__(
        self,
        num_classes: int,
        tau: float = 1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")

        base_channels = int(tau * 8)
        self.repeats = (2, 2, 4, 4)
        self.channels = (
            base_channels * 2,
            base_channels,
            int(base_channels * 1.5),
            base_channels * 2,
            int(base_channels * 2.5),
            base_channels * 4,
        )
        self.stride_stages = {1, 2}

        self.stem = nn.Sequential(
            nn.Conv2d(1, self.channels[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        for stage_index, repeat in enumerate(self.repeats):
            in_channels = self.channels[stage_index]
            out_channels = self.channels[stage_index + 1]
            blocks: list[nn.Module] = []
            for block_index in range(repeat):
                block_in = in_channels if block_index == 0 else out_channels
                stride = (2, 1) if stage_index in self.stride_stages and block_index == 0 else (1, 1)
                blocks.append(
                    OfficialBCResBlock(
                        block_in,
                        out_channels,
                        stage_index=stage_index,
                        stride=stride,
                        dropout=dropout,
                    )
                )
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        classifier_channels = self.channels[-2]
        self.classifier = nn.Sequential(
            nn.Conv2d(
                classifier_channels,
                classifier_channels,
                kernel_size=(5, 5),
                padding=(0, 2),
                groups=classifier_channels,
                bias=False,
            ),
            nn.Conv2d(classifier_channels, self.channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channels[-1], num_classes, kernel_size=1),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        if logmel.ndim != 4:
            raise ValueError("Expected Log-Mel input with shape [batch, 1, mel, time]")
        if logmel.size(1) != 1:
            raise ValueError("Expected a single-channel Log-Mel input")

        features = self.stem(logmel)
        features = self.stages(features)
        logits = self.classifier(features)
        return logits.flatten(start_dim=1)


def build_model(config: dict, num_classes: int) -> SpeechCommandBCResNet15:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "bc_resnet_1_5")
    if model_type != "bc_resnet_1_5":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SpeechCommandBCResNet15(
        num_classes=num_classes,
        tau=float(model_cfg.get("tau", 1.5)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
