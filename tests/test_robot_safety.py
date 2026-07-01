import unittest

from src.robot.safety import SafetyDecisionLayer
from src.robot.simulator import RobotSimulator


class RobotSafetyTests(unittest.TestCase):
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
