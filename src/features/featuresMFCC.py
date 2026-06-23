from __future__ import annotations

import torch
from torch import nn
import torchaudio


class MFCCExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
    ) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "power": 2.0,
            },
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        mfcc = self.mfcc(waveform)
        if mfcc.dim() == 3:
            mfcc = mfcc.unsqueeze(1)
        return mfcc


def build_mfcc_extractor(config: dict) -> MFCCExtractor:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    return MFCCExtractor(
        sample_rate=data_cfg["sample_rate"],
        n_mfcc=feature_cfg["n_mfcc"],
        n_fft=feature_cfg["n_fft"],
        win_length=feature_cfg["win_length"],
        hop_length=feature_cfg["hop_length"],
        n_mels=feature_cfg["n_mels"],
    )
