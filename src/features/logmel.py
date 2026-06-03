from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        logmel = self.to_db(self.mel(waveform))
        if logmel.dim() == 3:
            logmel = logmel.unsqueeze(1)
        return logmel


def build_logmel_extractor(config: dict) -> LogMelExtractor:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    return LogMelExtractor(
        sample_rate=data_cfg["sample_rate"],
        n_fft=feature_cfg["n_fft"],
        win_length=feature_cfg["win_length"],
        hop_length=feature_cfg["hop_length"],
        n_mels=feature_cfg["n_mels"],
    )
