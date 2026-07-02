import unittest

from src.robot.safety import SafetyDecisionLayer
from src.robot.simulator import RobotSimulator


class RobotSafetyTests(unittest.TestCase):
    def test_requires_wake_word_when_configured(self) -> None:
        decision = SafetyDecisionLayer(
            confidence_threshold=0.70,
            require_wake_word=True,
        ).decide(
            raw_label="forward",
            confidence=0.95,
            wake_word_detected=False,
        )

        self.assertEqual(decision.label, "unknown")
        self.assertEqual(decision.action, "IGNORE")
        self.assertEqual(decision.status, "ignored")
        self.assertEqual(decision.reason, "wake word not detected")
        self.assertFalse(decision.accepted)

    def test_not_listening_state_ignores_command(self) -> None:
        decision = SafetyDecisionLayer(confidence_threshold=0.70).decide(
            raw_label="forward",
            confidence=0.95,
            listening=False,
        )

        self.assertEqual(decision.status, "ignored")
        self.assertEqual(decision.reason, "system is not listening")
        self.assertFalse(decision.accepted)

    def test_command_window_timeout_rejects_command(self) -> None:
        decision = SafetyDecisionLayer(
            confidence_threshold=0.70,
            command_timeout_seconds=2.0,
        ).decide(
            raw_label="forward",
            confidence=0.95,
            elapsed_since_wake_seconds=2.5,
        )

        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.reason, "command window timeout")
        self.assertFalse(decision.accepted)

    def test_disallowed_label_is_rejected(self) -> None:
        decision = SafetyDecisionLayer(confidence_threshold=0.70).decide(
            raw_label="up",
            confidence=0.99,
        )

        self.assertEqual(decision.label, "unknown")
        self.assertEqual(decision.action, "IGNORE")
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.reason, "command is not allowed")
        self.assertFalse(decision.accepted)

    def test_unknown_command_is_ignored(self) -> None:
        decision = SafetyDecisionLayer(confidence_threshold=0.70).decide(
            raw_label="unknown",
            confidence=0.99,
        )

        self.assertEqual(decision.label, "unknown")
        self.assertEqual(decision.action, "IGNORE")
        self.assertEqual(decision.status, "ignored")
        self.assertFalse(decision.accepted)

    def test_low_confidence_command_is_rejected(self) -> None:
        decision = SafetyDecisionLayer(confidence_threshold=0.70).decide(
            raw_label="right",
            confidence=0.42,
        )

        self.assertEqual(decision.label, "unknown")
        self.assertEqual(decision.raw_label, "right")
        self.assertEqual(decision.action, "IGNORE")
        self.assertEqual(decision.status, "rejected")
        self.assertFalse(decision.accepted)

    def test_stop_command_uses_priority_threshold(self) -> None:
        decision = SafetyDecisionLayer(
            confidence_threshold=0.90,
            stop_confidence_threshold=0.20,
        ).decide(
            raw_label="stop",
            confidence=0.40,
        )

        self.assertEqual(decision.label, "stop")
        self.assertEqual(decision.action, "STOP")
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.reason, "stop command accepted")
        self.assertTrue(decision.accepted)

    def test_builds_safety_layer_from_project_config(self) -> None:
        layer = SafetyDecisionLayer.from_config(
            {
                "data": {
                    "commands": ["forward", "stop"],
                    "unknown_label": "unknown",
                },
                "safety": {
                    "confidence_threshold": 0.80,
                    "require_wake_word": True,
                    "command_timeout_seconds": 3.0,
                    "stop_confidence_threshold": 0.10,
                },
            }
        )

        self.assertEqual(layer.confidence_threshold, 0.80)
        self.assertEqual(layer.unknown_label, "unknown")
        self.assertTrue(layer.config.require_wake_word)

    def test_left_turns_relative_to_current_direction_then_moves(self) -> None:
        simulator = RobotSimulator(width=12, height=12)
        event = simulator.apply_command("left", confidence=0.91)

        self.assertEqual(event["from_position"], (6, 6))
        self.assertEqual(event["position"], (5, 6))
        self.assertEqual(event["direction"], "WEST")
        self.assertTrue(event["moved"])
        self.assertFalse(event["blocked"])

        simulator.state.direction = "EAST"
        event = simulator.apply_command("left", confidence=0.91)

        self.assertEqual(event["from_position"], (5, 6))
        self.assertEqual(event["position"], (5, 7))
        self.assertEqual(event["direction"], "NORTH")
        self.assertTrue(event["moved"])
        self.assertFalse(event["blocked"])

    def test_right_turns_relative_to_current_direction_then_moves(self) -> None:
        simulator = RobotSimulator(width=12, height=12)
        event = simulator.apply_command("right", confidence=0.91)

        self.assertEqual(event["from_position"], (6, 6))
        self.assertEqual(event["position"], (7, 6))
        self.assertEqual(event["direction"], "EAST")
        self.assertTrue(event["moved"])
        self.assertFalse(event["blocked"])

        simulator.state.direction = "SOUTH"
        event = simulator.apply_command("right", confidence=0.91)

        self.assertEqual(event["from_position"], (7, 6))
        self.assertEqual(event["position"], (6, 6))
        self.assertEqual(event["direction"], "WEST")
        self.assertTrue(event["moved"])
        self.assertFalse(event["blocked"])

    def test_boundary_move_is_blocked(self) -> None:
        simulator = RobotSimulator(width=3, height=3)
        simulator.state.x = 0
        simulator.state.y = 1
        simulator.state.direction = "NORTH"

        event = simulator.apply_command("left", confidence=0.91)

        self.assertEqual(event["position"], (0, 1))
        self.assertEqual(event["direction"], "WEST")
        self.assertEqual(event["status"], "blocked")
        self.assertTrue(event["blocked"])


if __name__ == "__main__":
    unittest.main()
