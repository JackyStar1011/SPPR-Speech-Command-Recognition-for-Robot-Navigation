from __future__ import annotations

import math

import torch

from src.data.preprocess import normalize_amplitude


def _uniform(
    minimum: float,
    maximum: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> float:
    if minimum > maximum:
        raise ValueError("minimum must be <= maximum")
    value = torch.rand((), generator=generator, device=device)
    return float(minimum + (maximum - minimum) * value.item())


def _should_apply(
    probability: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    return bool(torch.rand((), generator=generator, device=device).item() < probability)


def apply_gain_db(
    waveform: torch.Tensor,
    gain_db: float,
) -> torch.Tensor:
    return waveform * (10.0 ** (gain_db / 20.0))


def random_gain(
    waveform: torch.Tensor,
    minimum_gain_db: float = -6.0,
    maximum_gain_db: float = 6.0,
    probability: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not _should_apply(probability, generator, waveform.device):
        return waveform
    gain_db = _uniform(minimum_gain_db, maximum_gain_db, generator, waveform.device)
    return apply_gain_db(waveform, gain_db)


def shift_waveform(
    waveform: torch.Tensor,
    shift_samples: int,
) -> torch.Tensor:
    if shift_samples == 0:
        return waveform
    output = torch.zeros_like(waveform)
    if shift_samples > 0:
        output[..., shift_samples:] = waveform[..., :-shift_samples]
    else:
        shift = abs(shift_samples)
        output[..., :-shift] = waveform[..., shift:]
    return output


def random_time_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    maximum_shift_ms: float = 100.0,
    probability: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if maximum_shift_ms < 0:
        raise ValueError("maximum_shift_ms must be non-negative")
    if not _should_apply(probability, generator, waveform.device):
        return waveform

    maximum_shift = int(round(sample_rate * maximum_shift_ms / 1000.0))
    if maximum_shift == 0:
        return waveform
    shift = int(round(_uniform(-maximum_shift, maximum_shift, generator, waveform.device)))
    return shift_waveform(waveform, shift)


def add_noise_at_snr(
    waveform: torch.Tensor,
    snr_db: float,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn(
            waveform.shape,
            generator=generator,
            device=waveform.device,
            dtype=waveform.dtype,
        )
    else:
        noise = noise.to(device=waveform.device, dtype=waveform.dtype)
        if noise.shape[-1] < waveform.shape[-1]:
            repeats = math.ceil(waveform.shape[-1] / noise.shape[-1])
            repeat_shape = [1] * noise.ndim
            repeat_shape[-1] = repeats
            noise = noise.repeat(*repeat_shape)
        noise = noise[..., : waveform.shape[-1]]
        if noise.shape[:-1] != waveform.shape[:-1]:
            noise = noise.expand_as(waveform)

    signal_rms = waveform.square().mean().sqrt()
    noise_rms = noise.square().mean().sqrt().clamp_min(eps)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return waveform + noise * (target_noise_rms / noise_rms)


def random_add_noise(
    waveform: torch.Tensor,
    minimum_snr_db: float = 5.0,
    maximum_snr_db: float = 20.0,
    probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not _should_apply(probability, generator, waveform.device):
        return waveform
    snr_db = _uniform(minimum_snr_db, maximum_snr_db, generator, waveform.device)
    return add_noise_at_snr(waveform, snr_db=snr_db, generator=generator)


def apply_waveform_augmentation(
    waveform: torch.Tensor,
    sample_rate: int,
    config: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not config.get("enabled", False):
        return waveform

    augmented = waveform
    gain_cfg = config.get("gain", {})
    if gain_cfg.get("enabled", False):
        augmented = random_gain(
            augmented,
            minimum_gain_db=float(gain_cfg.get("min_db", -6.0)),
            maximum_gain_db=float(gain_cfg.get("max_db", 6.0)),
            probability=float(gain_cfg.get("probability", 1.0)),
            generator=generator,
        )

    shift_cfg = config.get("time_shift", {})
    if shift_cfg.get("enabled", False):
        augmented = random_time_shift(
            augmented,
            sample_rate=sample_rate,
            maximum_shift_ms=float(shift_cfg.get("max_ms", 100.0)),
            probability=float(shift_cfg.get("probability", 1.0)),
            generator=generator,
        )

    noise_cfg = config.get("add_noise", {})
    if noise_cfg.get("enabled", False):
        augmented = random_add_noise(
            augmented,
            minimum_snr_db=float(noise_cfg.get("min_snr_db", 5.0)),
            maximum_snr_db=float(noise_cfg.get("max_snr_db", 20.0)),
            probability=float(noise_cfg.get("probability", 0.5)),
            generator=generator,
        )

    if config.get("normalize_after", True):
        augmented = normalize_amplitude(augmented)
    return augmented.clamp(-1.0, 1.0)
