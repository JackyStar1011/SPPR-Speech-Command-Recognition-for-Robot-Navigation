from __future__ import annotations


ACTION_MAP = {
    "forward": "MOVE_FORWARD",
    "backward": "MOVE_BACKWARD",
    "left": "TURN_LEFT",
    "right": "TURN_RIGHT",
    "stop": "STOP",
    "unknown": "IGNORE",
}


def label_to_action(label: str) -> str:
    return ACTION_MAP.get(label, ACTION_MAP["unknown"])
