from __future__ import annotations

import wave

import numpy as np
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


def align_active_speech(
    waveform: torch.Tensor,
    sample_rate: int,
    target_num_samples: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    padding_ms: float = 120.0,
    max_gap_ms: float = 150.0,
    noise_multiplier: float = 3.0,
    minimum_peak_ratio: float = 0.08,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Center the strongest speech region in a fixed-length waveform window.

    Speech activity is estimated from short-frame RMS energy. Nearby active
    regions are merged so short pauses inside a command do not split the word.
    A margin is retained to protect low-energy consonants such as /f/ and /s/.
    """
    if target_num_samples <= 0:
        raise ValueError("target_num_samples must be positive")
    if waveform.size(-1) == 0:
        return waveform.new_zeros((*waveform.shape[:-1], target_num_samples))

    frame_length = max(1, int(round(sample_rate * frame_length_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_length_ms / 1000.0)))
    analysis = waveform.mean(dim=0)
    if analysis.numel() < frame_length:
        analysis = F.pad(analysis, (0, frame_length - analysis.numel()))

    frames = analysis.unfold(0, frame_length, hop_length)
    frame_rms = frames.square().mean(dim=1).add(eps).sqrt()
    peak_rms = frame_rms.max()
    if peak_rms.item() <= eps**0.5:
        return _center_pad_or_crop(waveform, target_num_samples)

    noise_floor = torch.quantile(frame_rms, 0.2)
    threshold = torch.maximum(
        noise_floor * noise_multiplier,
        peak_rms * minimum_peak_ratio,
    )
    # A high background estimate must not suppress every frame.
    threshold = torch.minimum(threshold, peak_rms * 0.5)
    active_indices = (frame_rms >= threshold).nonzero(as_tuple=False).flatten().tolist()
    if not active_indices:
        return _center_pad_or_crop(waveform, target_num_samples)

    max_gap_frames = max(0, int(round(max_gap_ms / hop_length_ms)))
    regions: list[tuple[int, int]] = []
    region_start = previous = active_indices[0]
    for index in active_indices[1:]:
        if index - previous - 1 > max_gap_frames:
            regions.append((region_start, previous))
            region_start = index
        previous = index
    regions.append((region_start, previous))

    def region_score(region: tuple[int, int]) -> float:
        start, end = region
        excess_energy = (frame_rms[start : end + 1] - threshold).clamp_min(0.0)
        return float(excess_energy.sum().item())

    start_frame, end_frame = max(regions, key=region_score)
    padding = int(round(sample_rate * padding_ms / 1000.0))
    start_sample = max(0, start_frame * hop_length - padding)
    end_sample = min(
        waveform.size(-1),
        end_frame * hop_length + frame_length + padding,
    )
    speech = waveform[..., start_sample:end_sample]
    return _center_pad_or_crop(speech, target_num_samples)


def _center_pad_or_crop(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current = waveform.size(-1)
    if current > target_num_samples:
        start = (current - target_num_samples) // 2
        return waveform[..., start : start + target_num_samples]
    if current < target_num_samples:
        total_padding = target_num_samples - current
        left_padding = total_padding // 2
        return F.pad(waveform, (left_padding, total_padding - left_padding))
    return waveform


def normalize_amplitude(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def reduce_stationary_noise(
    waveform: torch.Tensor,
    n_fft: int = 400,
    win_length: int = 400,
    hop_length: int = 160,
    noise_quantile: float = 0.2,
    reduction_strength: float = 0.8,
    residual_floor: float = 0.05,
) -> torch.Tensor:
    """Reduce near-stationary background noise with simple spectral subtraction."""
    if waveform.size(-1) == 0:
        return waveform
    if not 0.0 <= noise_quantile <= 1.0:
        raise ValueError("noise_quantile must be between 0 and 1")
    if not 0.0 <= reduction_strength <= 1.0:
        raise ValueError("reduction_strength must be between 0 and 1")
    if not 0.0 <= residual_floor <= 1.0:
        raise ValueError("residual_floor must be between 0 and 1")

    length = waveform.size(-1)
    n_fft = min(n_fft, max(2, length))
    win_length = min(win_length, n_fft)
    hop_length = min(hop_length, win_length)
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)

    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    phase = torch.angle(spectrum)
    noise_profile = torch.quantile(magnitude, noise_quantile, dim=-1, keepdim=True)
    reduced_magnitude = (magnitude - reduction_strength * noise_profile).clamp_min(
        magnitude * residual_floor
    )
    reduced_spectrum = torch.polar(reduced_magnitude, phase)
    return torch.istft(
        reduced_spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
    target_num_samples: int = 16000,
    normalize: bool = True,
    align_speech: bool = False,
    speech_alignment: dict | None = None,
    apply_noise_reduction: bool = False,
    noise_reduction: dict | None = None,
) -> torch.Tensor:
    waveform = to_mono(waveform.float())
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    if apply_noise_reduction:
        waveform = reduce_stationary_noise(waveform, **(noise_reduction or {}))
    if align_speech:
        waveform = align_active_speech(
            waveform,
            sample_rate=target_sample_rate,
            target_num_samples=target_num_samples,
            **(speech_alignment or {}),
        )
    else:
        waveform = fix_length(waveform, target_num_samples)
    if normalize:
        waveform = normalize_amplitude(waveform)
    return waveform


def load_audio_file(
    path: str,
    target_sample_rate: int = 16000,
    target_num_samples: int = 16000,
    align_speech: bool = False,
    speech_alignment: dict | None = None,
    apply_noise_reduction: bool = False,
    noise_reduction: dict | None = None,
) -> torch.Tensor:
    waveform, sample_rate = load_waveform(path)
    return preprocess_waveform(
        waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
        align_speech=align_speech,
        speech_alignment=speech_alignment,
        apply_noise_reduction=apply_noise_reduction,
        noise_reduction=noise_reduction,
    )


def get_speech_alignment_config(config: dict, inference: bool = False) -> tuple[bool, dict]:
    alignment = dict(config.get("preprocessing", {}).get("speech_alignment", {}))
    enabled = bool(alignment.pop("enabled", False))
    if inference:
        enabled = bool(config.get("inference", {}).get("align_speech", enabled))
    return enabled, alignment


def get_noise_reduction_config(config: dict, inference: bool = False) -> tuple[bool, dict]:
    noise_reduction = dict(config.get("preprocessing", {}).get("noise_reduction", {}))
    enabled = bool(noise_reduction.pop("enabled", False))
    if inference:
        enabled = bool(config.get("inference", {}).get("noise_reduction", enabled))
    return enabled, noise_reduction


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except (ImportError, RuntimeError, OSError) as error:
        try:
            return _load_pcm_wav(path)
        except Exception:
            raise error


def _load_pcm_wav(path: str) -> tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    waveform = torch.from_numpy(audio.reshape(-1, channels).T.copy())
    return waveform, sample_rate
