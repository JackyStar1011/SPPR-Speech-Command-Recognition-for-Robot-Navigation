from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from typing import Any

import torch

from src.robot.actions import VALID_WEBOTS_ACTIONS
from src.robot.safety import SafetyDecision, SafetyDecisionLayer
from src.robot.webots_udp import WebotsUDPClient
from src.runtime.state import RuntimeEvent, RuntimeState
from src.wakeword.command_capture import (
    CommandCaptureResult,
    WakeTriggeredCommandCapture,
)
from src.wakeword.openwakeword_detector import OpenWakeWordDetector


EventHandler = Callable[[RuntimeEvent], None]
StopPredicate = Callable[[], bool]


class VoiceWebotsPipeline:
    """Orchestrate wake word, command inference, safety, and Webots dispatch."""

    def __init__(
        self,
        *,
        detector: OpenWakeWordDetector | None,
        predictor: Any,
        safety_layer: SafetyDecisionLayer,
        command_capture: WakeTriggeredCommandCapture,
        webots_client: WebotsUDPClient,
        sample_rate: int,
        prediction_threshold: float = 0.0,
        cooldown_seconds: float = 0.5,
        wakeword_enabled: bool = True,
        event_handler: EventHandler | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must be between 0 and 1")
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be non-negative")
        if wakeword_enabled and detector is None:
            raise ValueError("detector is required when wakeword is enabled")

        self.detector = detector
        self.predictor = predictor
        self.safety_layer = safety_layer
        self.command_capture = command_capture
        self.webots_client = webots_client
        self.sample_rate = sample_rate
        self.prediction_threshold = prediction_threshold
        self.cooldown_samples = int(sample_rate * cooldown_seconds)
        self.wakeword_enabled = wakeword_enabled
        self.event_handler = event_handler
        self.state = RuntimeState.IDLE

    def run_frames(
        self,
        frames: Iterable[torch.Tensor],
        *,
        stop_requested: StopPredicate | None = None,
    ) -> int:
        iterator = iter(frames)
        commands_processed = 0
        self._transition(RuntimeState.IDLE, "runtime_started")

        while not self._should_stop(stop_requested):
            try:
                frame = next(iterator)
            except StopIteration:
                break

            if self.wakeword_enabled:
                assert self.detector is not None
                wake_result = self.detector.predict_frame(frame, sample_rate=self.sample_rate)
                self._emit(
                    "wake_score",
                    score=wake_result.score,
                    threshold=wake_result.threshold,
                    model_name=wake_result.model_name,
                    detected=wake_result.detected,
                )
                if not wake_result.detected:
                    continue
                command_frames: Iterator[torch.Tensor] = iterator
            else:
                command_frames = chain([frame], iterator)

            self._transition(RuntimeState.LISTENING, "wake_word_detected")
            capture = self.command_capture.capture(command_frames)
            self._emit_capture(capture)

            if capture.waveform is None:
                self._transition(RuntimeState.COOLDOWN, "command_not_captured")
                self._consume_cooldown(iterator)
                self._transition(RuntimeState.IDLE, "waiting_for_wake_word")
                continue

            self._transition(RuntimeState.PROCESSING, "command_captured")
            prediction = self.predictor.predict_waveform(
                capture.waveform,
                sample_rate=self.sample_rate,
                threshold=self.prediction_threshold,
            )
            decision = self.safety_layer.decide(
                raw_label=str(prediction["raw_label"]),
                confidence=float(prediction["confidence"]),
                wake_word_detected=True,
                listening=True,
                elapsed_since_wake_seconds=capture.duration_seconds,
            )
            self._emit_prediction(prediction, decision)

            if decision.accepted and decision.action in VALID_WEBOTS_ACTIONS:
                self._transition(RuntimeState.DISPATCHING, "command_accepted")
                self.webots_client.send_action(decision.action)
                self._emit("action_dispatched", action=decision.action)
            else:
                self._emit(
                    "action_not_dispatched",
                    action=decision.action,
                    status=decision.status,
                    reason=decision.reason,
                )

            commands_processed += 1
            self._transition(RuntimeState.COOLDOWN, "command_complete")
            self._consume_cooldown(iterator)
            self._transition(RuntimeState.IDLE, "waiting_for_wake_word")

        self._transition(RuntimeState.STOPPED, "runtime_stopped")
        return commands_processed

    def fail(self, error: Exception) -> None:
        self._transition(
            RuntimeState.ERROR,
            "runtime_error",
            error_type=type(error).__name__,
            message=str(error),
        )

    def _consume_cooldown(self, frames: Iterator[torch.Tensor]) -> None:
        remaining_samples = self.cooldown_samples
        while remaining_samples > 0:
            try:
                frame = next(frames)
            except StopIteration:
                return
            remaining_samples -= int(frame.numel())

    def _emit_capture(self, capture: CommandCaptureResult) -> None:
        self._emit(
            "command_capture",
            reason=capture.reason,
            speech_detected=capture.speech_detected,
            duration_seconds=capture.duration_seconds,
            peak_rms=capture.peak_rms,
            captured=capture.waveform is not None,
        )

    def _emit_prediction(
        self,
        prediction: dict[str, Any],
        decision: SafetyDecision,
    ) -> None:
        self._emit(
            "prediction",
            raw_label=str(prediction["raw_label"]),
            predicted_label=str(prediction["label"]),
            confidence=float(prediction["confidence"]),
            action=decision.action,
            status=decision.status,
            reason=decision.reason,
            accepted=decision.accepted,
        )

    def _transition(self, state: RuntimeState, kind: str, **data: Any) -> None:
        self.state = state
        self._emit(kind, **data)

    def _emit(self, kind: str, **data: Any) -> None:
        if self.event_handler is not None:
            self.event_handler(RuntimeEvent(kind=kind, state=self.state, data=data))

    @staticmethod
    def _should_stop(stop_requested: StopPredicate | None) -> bool:
        return bool(stop_requested and stop_requested())
