from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def fix_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current = waveform.size(-1)
    if current > target_num_samples:
        return waveform[..., :target_num_samples]
    if current < target_num_samples:
        return F.pad(waveform, (0, target_num_samples - current))
    return waveform


def normalize_amplitude(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
    target_num_samples: int = 16000,
    normalize: bool = True,
) -> torch.Tensor:
    waveform = to_mono(waveform.float())
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    waveform = fix_length(waveform, target_num_samples)
    if normalize:
        waveform = normalize_amplitude(waveform)
    return waveform


def load_audio_file(
    path: str,
    target_sample_rate: int = 16000,
    target_num_samples: int = 16000,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    return preprocess_waveform(
        waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
    )
