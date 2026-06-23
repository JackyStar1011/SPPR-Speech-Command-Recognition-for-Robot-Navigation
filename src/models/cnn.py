from __future__ import annotations

from torch import nn


class SpeechCommandCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(1, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(config: dict, num_classes: int) -> SpeechCommandCNN:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "cnn")
    if model_type != "cnn":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SpeechCommandCNN(
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.3),
    )
