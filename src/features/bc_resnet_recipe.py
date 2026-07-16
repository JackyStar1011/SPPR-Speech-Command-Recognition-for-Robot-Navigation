from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torchaudio
from torch import nn

from src.data.augmentation import shift_waveform
from src.data.preprocess import load_waveform, to_mono


class NaturalLogMelExtractor(nn.Module):
    """Log-Mel frontend used by the official BC-ResNet implementation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 480,
        hop_length: int = 160,
        n_mels: int = 40,
        log_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if log_epsilon <= 0:
            raise ValueError("log_epsilon must be positive")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.log_epsilon = log_epsilon

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0).unsqueeze(0)
        elif waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        if waveforms.ndim != 3 or waveforms.size(1) != 1:
            raise ValueError("Expected waveforms with shape [batch, 1, samples]")
        return (self.mel(waveforms) + self.log_epsilon).log()


class SpecAugment(nn.Module):
    """Mask Log-Mel features with the same sampling convention as the official code."""

    def __init__(
        self,
        frequency_mask_param: int = 1,
        time_mask_param: int = 20,
        frequency_mask_count: int = 2,
        time_mask_count: int = 2,
    ) -> None:
        super().__init__()
        values = (
            frequency_mask_param,
            time_mask_param,
            frequency_mask_count,
            time_mask_count,
        )
        if any(value < 0 for value in values):
            raise ValueError("SpecAugment parameters must be non-negative")
        self.frequency_mask_param = frequency_mask_param
        self.time_mask_param = time_mask_param
        self.frequency_mask_count = frequency_mask_count
        self.time_mask_count = time_mask_count

    @staticmethod
    def _mask_axis(features: torch.Tensor, axis: int, mask_param: int, count: int) -> None:
        axis_length = features.size(axis)
        if mask_param <= 0 or axis_length == 0:
            return
        for batch_index in range(features.size(0)):
            for _ in range(count):
                # The reference implementation samples uniform [0, mask_param)
                # and truncates to int, so param=1 intentionally produces width 0.
                width = int(random.random() * mask_param)
                if width == 0:
                    continue
                width = min(width, axis_length)
                start = random.randint(0, axis_length - width)
                slices = [batch_index, slice(None), slice(None), slice(None)]
                slices[axis] = slice(start, start + width)
                features[tuple(slices)] = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("Expected Log-Mel features with shape [batch, 1, mel, time]")
        augmented = features.clone()
        self._mask_axis(
            augmented,
            axis=2,
            mask_param=self.frequency_mask_param,
            count=self.frequency_mask_count,
        )
        self._mask_axis(
            augmented,
            axis=3,
            mask_param=self.time_mask_param,
            count=self.time_mask_count,
        )
        return augmented


class BCResNetRecipePreprocessor(nn.Module):
    """Official-style waveform augmentation followed by Log-Mel and SpecAugment."""

    def __init__(
        self,
        feature_extractor: NaturalLogMelExtractor,
        spec_augment: SpecAugment,
        sample_rate: int,
        sample_length: int,
        augmentation_config: dict,
        background_noise_paths: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.spec_augment = spec_augment
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.augmentation_config = augmentation_config

        for index, path in enumerate(background_noise_paths or []):
            waveform, source_rate = load_waveform(str(path))
            waveform = to_mono(waveform.float())
            if source_rate != sample_rate:
                waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
            self.register_buffer(f"_background_noise_{index}", waveform, persistent=False)
        self.background_noise_count = len(background_noise_paths or [])

    def _random_background_segment(self) -> torch.Tensor:
        if self.background_noise_count == 0:
            raise RuntimeError(
                "Background-noise augmentation is enabled but no WAV files were found"
            )
        noise_index = random.randrange(self.background_noise_count)
        noise = getattr(self, f"_background_noise_{noise_index}")
        if noise.size(-1) < self.sample_length:
            noise = noise.repeat(1, math.ceil(self.sample_length / noise.size(-1)))
        max_start = noise.size(-1) - self.sample_length
        start = random.randint(0, max_start)
        return noise[..., start : start + self.sample_length]

    def augment_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        config = self.augmentation_config
        if not config.get("enabled", False):
            return waveforms

        probability = float(config.get("probability", 0.8))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be between 0 and 1")
        maximum_shift = int(
            round(self.sample_rate * float(config.get("time_shift_ms", 100.0)) / 1000.0)
        )
        minimum_amplitude = float(config.get("noise_amplitude_min", 0.0))
        maximum_amplitude = float(config.get("noise_amplitude_max", 0.1))
        if minimum_amplitude > maximum_amplitude:
            raise ValueError("noise_amplitude_min must be <= noise_amplitude_max")

        augmented = waveforms.clone()
        for index in range(augmented.size(0)):
            if random.random() > probability:
                continue
            shift = int(random.uniform(-maximum_shift, maximum_shift)) if maximum_shift else 0
            shifted = shift_waveform(augmented[index], shift)
            noise = self._random_background_segment()
            amplitude = random.uniform(minimum_amplitude, maximum_amplitude)
            augmented[index] = shifted + amplitude * noise
        return augmented.clamp(-1.0, 1.0)

    def forward(self, waveforms: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            waveforms = self.augment_waveforms(waveforms)
        features = self.feature_extractor(waveforms)
        if augment and self.augmentation_config.get("spec_augment", {}).get("enabled", True):
            features = self.spec_augment(features)
        return features


def build_recipe_preprocessor(
    config: dict,
    background_noise_paths: list[Path] | None = None,
) -> BCResNetRecipePreprocessor:
    data = config["data"]
    features = config["features"]
    augmentation = config.get("augmentation", {})
    spec = augmentation.get("spec_augment", {})
    extractor = NaturalLogMelExtractor(
        sample_rate=int(data["sample_rate"]),
        n_fft=int(features["n_fft"]),
        win_length=int(features["win_length"]),
        hop_length=int(features["hop_length"]),
        n_mels=int(features["n_mels"]),
        log_epsilon=float(features.get("log_epsilon", 1e-6)),
    )
    spec_augment = SpecAugment(
        frequency_mask_param=int(spec.get("frequency_mask_param", 1)),
        time_mask_param=int(spec.get("time_mask_param", 20)),
        frequency_mask_count=int(spec.get("frequency_mask_count", 2)),
        time_mask_count=int(spec.get("time_mask_count", 2)),
    )
    return BCResNetRecipePreprocessor(
        extractor,
        spec_augment,
        sample_rate=int(data["sample_rate"]),
        sample_length=int(data["sample_rate"] * data["duration_seconds"]),
        augmentation_config=augmentation,
        background_noise_paths=background_noise_paths,
    )
