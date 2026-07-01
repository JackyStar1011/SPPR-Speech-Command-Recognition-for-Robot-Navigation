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


class SafetyDecisionLayer:
    """Rule-based decision layer between model prediction and robot control."""

    def __init__(self, confidence_threshold: float, unknown_label: str = "unknown") -> None:
        self.confidence_threshold = confidence_threshold
        self.unknown_label = unknown_label

    def decide(
        self,
        raw_label: str,
        confidence: float | None,
        wake_word_detected: bool = True,
    ) -> SafetyDecision:
        if not wake_word_detected:
            return SafetyDecision(
                label=self.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.unknown_label),
                status="ignored",
                reason="wake word not detected",
                accepted=False,
            )

        if raw_label == self.unknown_label:
            return SafetyDecision(
                label=self.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.unknown_label),
                status="ignored",
                reason="unknown command",
                accepted=False,
            )

        if confidence is not None and confidence < self.confidence_threshold:
            return SafetyDecision(
                label=self.unknown_label,
                raw_label=raw_label,
                confidence=confidence,
                action=label_to_action(self.unknown_label),
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
