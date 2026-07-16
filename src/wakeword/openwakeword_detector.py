from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class WakeWordResult:
    detected: bool
    score: float
    threshold: float
    model_name: str


def is_detected(score: float, threshold: float) -> bool:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return score >= threshold


def waveform_to_pcm16(waveform: torch.Tensor) -> np.ndarray:
    audio = waveform.detach().float().cpu().squeeze().clamp(-1.0, 1.0)
    return (audio.numpy() * 32767.0).astype(np.int16)


class OpenWakeWordDetector:
    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        inference_framework: str = "onnx",
        wakeword_models: list[str] | None = None,
        auto_download: bool = True,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model_name = model_name
        self.prediction_key = model_name
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.auto_download = auto_download
        self._model = model or self._load_model(wakeword_models)

    @staticmethod
    def is_available() -> bool:
        try:
            import openwakeword  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_model_path(self, openwakeword_module: Any) -> str:
        model_dir = Path(openwakeword_module.__file__).resolve().parent / "resources" / "models"
        extension = "onnx" if self.inference_framework == "onnx" else "tflite"
        for candidate in self._model_candidates(model_dir, extension):
            if candidate.exists():
                self.prediction_key = candidate.stem
                return str(candidate)

        if self.auto_download:
            try:
                from openwakeword.utils import download_models

                download_models([self.model_name])
            except Exception as error:
                raise RuntimeError(
                    f"Failed to download openWakeWord model {self.model_name!r}: {error}"
                ) from error
            for candidate in self._model_candidates(model_dir, extension):
                if candidate.exists():
                    self.prediction_key = candidate.stem
                    return str(candidate)

        raise FileNotFoundError(
            f"Could not find openWakeWord model '{self.model_name}' in {model_dir}. "
            "Enable wakeword.auto_download, provide wakeword.model_path, or download the model."
        )

    def _model_candidates(self, model_dir: Path, extension: str) -> list[Path]:
        return [
            model_dir / f"{self.model_name}.{extension}",
            model_dir / f"{self.model_name}_v0.1.{extension}",
            model_dir / f"{self.model_name}_v0.2.{extension}",
        ]

    def _load_model(self, wakeword_models: list[str] | None) -> Any:
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError("openWakeWord is not installed. Install it with `pip install openwakeword`.") from error

        kwargs: dict[str, Any] = {"inference_framework": self.inference_framework}
        if wakeword_models:
            kwargs["wakeword_models"] = wakeword_models
            self.prediction_key = Path(wakeword_models[0]).stem
        else:
            kwargs["wakeword_models"] = [self._resolve_model_path(openwakeword)]
        return Model(**kwargs)

    def predict_frame(self, frame: torch.Tensor, sample_rate: int) -> WakeWordResult:
        if sample_rate != 16000:
            raise ValueError("openWakeWord expects 16 kHz audio")
        predictions = self._model.predict(waveform_to_pcm16(frame))
        score = float(predictions.get(self.prediction_key, 0.0))
        return WakeWordResult(
            detected=is_detected(score, self.threshold),
            score=score,
            threshold=self.threshold,
            model_name=self.model_name,
        )
