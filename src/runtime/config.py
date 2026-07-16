from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_config


@dataclass(frozen=True)
class ModelRuntimeConfig:
    config_path: str = "configs/models/cnn_gru.yaml"
    checkpoint_path: str | None = None
    device: str = "auto"
    prediction_threshold: float = 0.0


@dataclass(frozen=True)
class WakeWordRuntimeConfig:
    enabled: bool = True
    model_name: str = "hey_jarvis"
    model_path: str | None = None
    threshold: float = 0.5
    inference_framework: str = "onnx"
    frame_ms: float = 80.0


@dataclass(frozen=True)
class CommandCaptureRuntimeConfig:
    speech_rms_threshold: float = 0.01
    speech_start_timeout_seconds: float = 3.0
    silence_duration_seconds: float = 0.5
    min_command_seconds: float = 0.25
    max_command_seconds: float = 2.0


@dataclass(frozen=True)
class WebotsRuntimeConfig:
    command_host: str = "127.0.0.1"
    command_port: int = 5005
    telemetry_host: str = "127.0.0.1"
    telemetry_port: int = 5006
    stop_on_exit: bool = True


@dataclass(frozen=True)
class RuntimeBehaviorConfig:
    frame_timeout_seconds: float = 2.0
    cooldown_seconds: float = 0.5
    snapshot_path: str = "outputs/runtime/live_state.json"
    event_log_path: str = "outputs/runtime/events.jsonl"
    trajectory_limit: int = 500


@dataclass(frozen=True)
class VoiceWebotsRuntimeConfig:
    model: ModelRuntimeConfig
    wakeword: WakeWordRuntimeConfig
    capture: CommandCaptureRuntimeConfig
    webots: WebotsRuntimeConfig
    runtime: RuntimeBehaviorConfig


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"runtime config section {name!r} must be a mapping")
    return value


def load_runtime_config(path: str | Path) -> VoiceWebotsRuntimeConfig:
    raw = load_config(path)
    if not isinstance(raw, dict):
        raise ValueError("runtime config root must be a mapping")

    model = _section(raw, "model")
    wakeword = _section(raw, "wakeword")
    capture = _section(raw, "capture")
    webots = _section(raw, "webots")
    runtime = _section(raw, "runtime")

    config = VoiceWebotsRuntimeConfig(
        model=ModelRuntimeConfig(
            config_path=str(model.get("config_path", ModelRuntimeConfig.config_path)),
            checkpoint_path=_optional_string(model.get("checkpoint_path")),
            device=str(model.get("device", ModelRuntimeConfig.device)),
            prediction_threshold=float(
                model.get("prediction_threshold", ModelRuntimeConfig.prediction_threshold)
            ),
        ),
        wakeword=WakeWordRuntimeConfig(
            enabled=bool(wakeword.get("enabled", WakeWordRuntimeConfig.enabled)),
            model_name=str(wakeword.get("model_name", WakeWordRuntimeConfig.model_name)),
            model_path=_optional_string(wakeword.get("model_path")),
            threshold=float(wakeword.get("threshold", WakeWordRuntimeConfig.threshold)),
            inference_framework=str(
                wakeword.get("inference_framework", WakeWordRuntimeConfig.inference_framework)
            ),
            frame_ms=float(wakeword.get("frame_ms", WakeWordRuntimeConfig.frame_ms)),
        ),
        capture=CommandCaptureRuntimeConfig(
            speech_rms_threshold=float(
                capture.get(
                    "speech_rms_threshold",
                    CommandCaptureRuntimeConfig.speech_rms_threshold,
                )
            ),
            speech_start_timeout_seconds=float(
                capture.get(
                    "speech_start_timeout_seconds",
                    CommandCaptureRuntimeConfig.speech_start_timeout_seconds,
                )
            ),
            silence_duration_seconds=float(
                capture.get(
                    "silence_duration_seconds",
                    CommandCaptureRuntimeConfig.silence_duration_seconds,
                )
            ),
            min_command_seconds=float(
                capture.get(
                    "min_command_seconds",
                    CommandCaptureRuntimeConfig.min_command_seconds,
                )
            ),
            max_command_seconds=float(
                capture.get(
                    "max_command_seconds",
                    CommandCaptureRuntimeConfig.max_command_seconds,
                )
            ),
        ),
        webots=WebotsRuntimeConfig(
            command_host=str(webots.get("command_host", WebotsRuntimeConfig.command_host)),
            command_port=int(webots.get("command_port", WebotsRuntimeConfig.command_port)),
            telemetry_host=str(
                webots.get("telemetry_host", WebotsRuntimeConfig.telemetry_host)
            ),
            telemetry_port=int(webots.get("telemetry_port", WebotsRuntimeConfig.telemetry_port)),
            stop_on_exit=bool(webots.get("stop_on_exit", WebotsRuntimeConfig.stop_on_exit)),
        ),
        runtime=RuntimeBehaviorConfig(
            frame_timeout_seconds=float(
                runtime.get("frame_timeout_seconds", RuntimeBehaviorConfig.frame_timeout_seconds)
            ),
            cooldown_seconds=float(
                runtime.get("cooldown_seconds", RuntimeBehaviorConfig.cooldown_seconds)
            ),
            snapshot_path=str(
                runtime.get("snapshot_path", RuntimeBehaviorConfig.snapshot_path)
            ),
            event_log_path=str(
                runtime.get("event_log_path", RuntimeBehaviorConfig.event_log_path)
            ),
            trajectory_limit=int(
                runtime.get("trajectory_limit", RuntimeBehaviorConfig.trajectory_limit)
            ),
        ),
    )
    _validate_runtime_config(config)
    return config


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _validate_runtime_config(config: VoiceWebotsRuntimeConfig) -> None:
    if not 0.0 <= config.model.prediction_threshold <= 1.0:
        raise ValueError("model.prediction_threshold must be between 0 and 1")
    if not 0.0 <= config.wakeword.threshold <= 1.0:
        raise ValueError("wakeword.threshold must be between 0 and 1")
    if config.wakeword.frame_ms <= 0.0:
        raise ValueError("wakeword.frame_ms must be positive")
    if config.capture.speech_rms_threshold < 0.0:
        raise ValueError("capture.speech_rms_threshold must be non-negative")
    if config.capture.speech_start_timeout_seconds <= 0.0:
        raise ValueError("capture.speech_start_timeout_seconds must be positive")
    if config.capture.silence_duration_seconds <= 0.0:
        raise ValueError("capture.silence_duration_seconds must be positive")
    if config.capture.min_command_seconds <= 0.0:
        raise ValueError("capture.min_command_seconds must be positive")
    if config.capture.max_command_seconds < config.capture.min_command_seconds:
        raise ValueError(
            "capture.max_command_seconds must be at least capture.min_command_seconds"
        )
    if not 1 <= config.webots.command_port <= 65535:
        raise ValueError("webots.command_port must be between 1 and 65535")
    if not 1 <= config.webots.telemetry_port <= 65535:
        raise ValueError("webots.telemetry_port must be between 1 and 65535")
    if config.runtime.frame_timeout_seconds <= 0.0:
        raise ValueError("runtime.frame_timeout_seconds must be positive")
    if config.runtime.cooldown_seconds < 0.0:
        raise ValueError("runtime.cooldown_seconds must be non-negative")
    if config.runtime.trajectory_limit <= 0:
        raise ValueError("runtime.trajectory_limit must be positive")
