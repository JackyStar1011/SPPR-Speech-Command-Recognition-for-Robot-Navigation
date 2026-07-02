from __future__ import annotations

from dataclasses import dataclass

from src.robot.actions import label_to_action


@dataclass(frozen=True)
class SafetyDecision:
    label: str
    raw_label: str
    confidence: float | None
    action: str
    status: str
    reason: str
    accepted: bool


@dataclass(frozen=True)
class SafetyConfig:
    confidence_threshold: float = 0.70
    unknown_label: str = "unknown"
    allowed_commands: tuple[str, ...] = ("forward", "backward", "left", "right", "stop")
    require_wake_word: bool = False
    command_timeout_seconds: float | None = None
    stop_confidence_threshold: float = 0.0


class SafetyDecisionLayer:
    """Rule-based decision layer between model prediction and robot control."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        unknown_label: str = "unknown",
        allowed_commands: tuple[str, ...] | list[str] | None = None,
        require_wake_word: bool = False,
        command_timeout_seconds: float | None = None,
        stop_confidence_threshold: float = 0.0,
    ) -> None:
        self.config = SafetyConfig(
            confidence_threshold=confidence_threshold,
            unknown_label=unknown_label,
            allowed_commands=tuple(allowed_commands or ("forward", "backward", "left", "right", "stop")),
            require_wake_word=require_wake_word,
            command_timeout_seconds=command_timeout_seconds,
            stop_confidence_threshold=stop_confidence_threshold,
        )
        self._validate_config()

    @classmethod
    def from_config(cls, config: dict) -> "SafetyDecisionLayer":
        safety_cfg = config.get("safety", {})
        data_cfg = config.get("data", {})
        return cls(
            confidence_threshold=float(
                safety_cfg.get(
                    "confidence_threshold",
                    config.get("inference", {}).get("threshold", 0.70),
                )
            ),
            unknown_label=str(safety_cfg.get("unknown_label", data_cfg.get("unknown_label", "unknown"))),
            allowed_commands=tuple(safety_cfg.get("allowed_commands", data_cfg.get("commands", []))),
            require_wake_word=bool(safety_cfg.get("require_wake_word", False)),
            command_timeout_seconds=safety_cfg.get("command_timeout_seconds"),
            stop_confidence_threshold=float(safety_cfg.get("stop_confidence_threshold", 0.0)),
        )

    @property
    def confidence_threshold(self) -> float:
        return self.config.confidence_threshold

    @property
    def unknown_label(self) -> str:
        return self.config.unknown_label

    def _validate_config(self) -> None:
        if not 0.0 <= self.config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.config.stop_confidence_threshold <= 1.0:
            raise ValueError("stop_confidence_threshold must be between 0 and 1")
        if self.config.command_timeout_seconds is not None and self.config.command_timeout_seconds <= 0.0:
            raise ValueError("command_timeout_seconds must be positive when provided")

    def decide(
        self,
        raw_label: str,
        confidence: float | None,
        wake_word_detected: bool = True,
        listening: bool = True,
        elapsed_since_wake_seconds: float | None = None,
    ) -> SafetyDecision:
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if self.config.require_wake_word and not wake_word_detected:
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="ignored",
                reason="wake word not detected",
                accepted=False,
            )

        if not listening:
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="ignored",
                reason="system is not listening",
                accepted=False,
            )

        if (
            self.config.command_timeout_seconds is not None
            and elapsed_since_wake_seconds is not None
            and elapsed_since_wake_seconds > self.config.command_timeout_seconds
        ):
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="rejected",
                reason="command window timeout",
                accepted=False,
            )

        allowed_labels = set(self.config.allowed_commands) | {self.config.unknown_label}
        if raw_label not in allowed_labels:
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="rejected",
                reason="command is not allowed",
                accepted=False,
            )

        if raw_label == self.config.unknown_label:
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="ignored",
                reason="unknown command",
                accepted=False,
            )

        if raw_label == "stop" and (
            confidence is None or confidence >= self.config.stop_confidence_threshold
        ):
            return SafetyDecision(
                label=raw_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(raw_label),
                status="accepted",
                reason="stop command accepted",
                accepted=True,
            )

        if confidence is not None and confidence < self.config.confidence_threshold:
            return SafetyDecision(
                label=self.config.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.config.unknown_label),
                status="rejected",
                reason="confidence below threshold",
                accepted=False,
            )

        return SafetyDecision(
            label=raw_label,
            raw_label=raw_label,
            confidence=confidence,
            action=label_to_action(raw_label),
            status="accepted",
            reason="command accepted",
            accepted=True,
        )
