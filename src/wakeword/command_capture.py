from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


def waveform_rms(waveform: torch.Tensor) -> float:
    audio = waveform.detach().float().cpu().reshape(-1)
    if audio.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(audio * audio)).item())


@dataclass(frozen=True)
class CommandCaptureResult:
    waveform: torch.Tensor | None
    reason: str
    speech_detected: bool
    duration_seconds: float
    peak_rms: float


class WakeTriggeredCommandCapture:
    """Capture one spoken command from microphone frames after a wake word."""

    def __init__(
        self,
        sample_rate: int,
        *,
        speech_rms_threshold: float = 0.01,
        speech_start_timeout_seconds: float = 3.0,
        silence_duration_seconds: float = 0.5,
        min_command_seconds: float = 0.25,
        max_command_seconds: float = 2.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if speech_rms_threshold < 0.0:
            raise ValueError("speech_rms_threshold must be non-negative")
        if speech_start_timeout_seconds <= 0.0:
            raise ValueError("speech_start_timeout_seconds must be positive")
        if silence_duration_seconds <= 0.0:
            raise ValueError("silence_duration_seconds must be positive")
        if min_command_seconds <= 0.0:
            raise ValueError("min_command_seconds must be positive")
        if max_command_seconds < min_command_seconds:
            raise ValueError("max_command_seconds must be at least min_command_seconds")

        self.sample_rate = sample_rate
        self.speech_rms_threshold = speech_rms_threshold
        self.speech_start_samples = int(sample_rate * speech_start_timeout_seconds)
        self.silence_samples = int(sample_rate * silence_duration_seconds)
        self.min_command_samples = int(sample_rate * min_command_seconds)
        self.max_command_samples = int(sample_rate * max_command_seconds)

    def capture(
        self,
        frames: Iterable[torch.Tensor],
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> CommandCaptureResult:
        command_frames: list[torch.Tensor] = []
        waited_samples = 0
        captured_samples = 0
        trailing_silence_samples = 0
        peak_rms = 0.0
        speech_detected = False
        reason = "stream_ended"

        for frame in frames:
            if stop_requested is not None and stop_requested():
                reason = "stop_requested"
                break
            normalized_frame = self._normalize_frame(frame)
            frame_samples = normalized_frame.size(-1)
            level = waveform_rms(normalized_frame)
            peak_rms = max(peak_rms, level)

            if not speech_detected:
                waited_samples += frame_samples
                if level < self.speech_rms_threshold:
                    if waited_samples >= self.speech_start_samples:
                        reason = "speech_start_timeout"
                        break
                    continue
                speech_detected = True

            command_frames.append(normalized_frame)
            captured_samples += frame_samples

            if level < self.speech_rms_threshold:
                trailing_silence_samples += frame_samples
            else:
                trailing_silence_samples = 0

            if captured_samples >= self.max_command_samples:
                reason = "max_duration"
                break
            if (
                captured_samples >= self.min_command_samples
                and trailing_silence_samples >= self.silence_samples
            ):
                reason = "silence"
                break

        if not command_frames:
            return CommandCaptureResult(
                waveform=None,
                reason=reason,
                speech_detected=speech_detected,
                duration_seconds=0.0,
                peak_rms=peak_rms,
            )

        waveform = torch.cat(command_frames, dim=-1)
        waveform = waveform[:, : self.max_command_samples]
        duration_seconds = waveform.size(-1) / self.sample_rate
        if waveform.size(-1) < self.min_command_samples:
            return CommandCaptureResult(
                waveform=None,
                reason="command_too_short",
                speech_detected=True,
                duration_seconds=duration_seconds,
                peak_rms=peak_rms,
            )

        return CommandCaptureResult(
            waveform=waveform,
            reason=reason,
            speech_detected=True,
            duration_seconds=duration_seconds,
            peak_rms=peak_rms,
        )

    @staticmethod
    def _normalize_frame(frame: torch.Tensor) -> torch.Tensor:
        normalized = frame.detach().float().cpu()
        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0)
        if normalized.ndim != 2 or normalized.size(0) != 1:
            raise ValueError("microphone frame must have shape [samples] or [1, samples]")
        return normalized
