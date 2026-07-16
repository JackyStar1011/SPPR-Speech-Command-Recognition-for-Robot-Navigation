from __future__ import annotations

import queue
from collections.abc import Iterator

import numpy as np
import sounddevice as sd
import torch


class MicrophoneFrameStream:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: float,
        *,
        queue_capacity: int = 8,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.queue_capacity = queue_capacity
        self.frame_samples = int(sample_rate * frame_ms / 1000.0)
        if self.frame_samples <= 0:
            raise ValueError("frame size must be positive")

    @staticmethod
    def _to_waveform(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame.T.copy())

    def frames(self, timeout_seconds: float | None = None) -> Iterator[torch.Tensor]:
        audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.queue_capacity)

        def callback(indata, frames, time, status) -> None:
            if status:
                print(status)
            frame = indata.copy()
            try:
                audio_queue.put_nowait(frame)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(frame)
                except queue.Full:
                    pass

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        ):
            while True:
                try:
                    frame = audio_queue.get(timeout=timeout_seconds)
                except queue.Empty:
                    break
                yield self._to_waveform(frame)
