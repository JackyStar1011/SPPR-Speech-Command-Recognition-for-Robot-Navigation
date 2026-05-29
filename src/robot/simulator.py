from __future__ import annotations

from src.robot.actions import label_to_action


class RobotSimulator:
    def apply_command(self, label: str) -> str:
        return label_to_action(label)
