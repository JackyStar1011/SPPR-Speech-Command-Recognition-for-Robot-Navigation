from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from src.robot.actions import label_to_action


DIRECTIONS = ("NORTH", "EAST", "SOUTH", "WEST")
DIRECTION_DELTAS = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0),
}
DIRECTION_MARKERS = {
    "NORTH": "^",
    "EAST": ">",
    "SOUTH": "v",
    "WEST": "<",
}


@dataclass
class RobotState:
    width: int = 8
    height: int = 8
    x: int = 3
    y: int = 3
    direction: str = "NORTH"
    path: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.path:
            self.path.append((self.x, self.y))

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


class RobotSimulator:
    def __init__(self, width: int = 8, height: int = 8) -> None:
        self.width = width
        self.height = height
        self.state = self._initial_state()
        self.history: list[dict[str, Any]] = []

    def _initial_state(self) -> RobotState:
        return RobotState(
            width=self.width,
            height=self.height,
            x=self.width // 2,
            y=self.height // 2,
            direction="NORTH",
        )

    def reset(self) -> None:
        self.state = self._initial_state()
        self.history = []

    def apply_command(self, label: str, confidence: float | None = None) -> dict[str, Any]:
        action = label_to_action(label)
        previous_position = self.state.position
        previous_direction = self.state.direction

        moved = False
        blocked = False
        if label == "forward":
            moved, blocked = self._move(multiplier=1)
        elif label == "backward":
            moved, blocked = self._move(multiplier=-1)
        elif label == "left":
            self._turn(step=-1)
        elif label == "right":
            self._turn(step=1)

        event = {
            "step": len(self.history) + 1,
            "command": label,
            "confidence": confidence,
            "action": action,
            "from_position": previous_position,
            "position": self.state.position,
            "from_direction": previous_direction,
            "direction": self.state.direction,
            "moved": moved,
            "blocked": blocked,
        }
        self.history.append(event)
        return event

    def _move(self, multiplier: int) -> tuple[bool, bool]:
        dx, dy = DIRECTION_DELTAS[self.state.direction]
        next_x = self.state.x + dx * multiplier
        next_y = self.state.y + dy * multiplier

        if not (0 <= next_x < self.width and 0 <= next_y < self.height):
            return False, True

        self.state.x = next_x
        self.state.y = next_y
        self.state.path.append(self.state.position)
        return True, False

    def _turn(self, step: int) -> None:
        current_index = DIRECTIONS.index(self.state.direction)
        self.state.direction = DIRECTIONS[(current_index + step) % len(DIRECTIONS)]

    def render(self):
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(color="#d1d5db", linewidth=1.0)
        ax.set_facecolor("#f8fafc")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if self.state.path:
            path_x = [point[0] for point in self.state.path]
            path_y = [point[1] for point in self.state.path]
            ax.plot(
                path_x,
                path_y,
                color="#2563eb",
                linewidth=2.8,
                marker="o",
                markersize=6,
                markerfacecolor="#93c5fd",
                markeredgecolor="#1d4ed8",
                zorder=2,
            )
            ax.scatter(
                path_x[0],
                path_y[0],
                s=170,
                marker="s",
                color="#10b981",
                edgecolor="white",
                linewidth=1.8,
                zorder=3,
            )

        marker = DIRECTION_MARKERS[self.state.direction]
        ax.scatter(
            self.state.x,
            self.state.y,
            s=850,
            marker=marker,
            color="#ef4444",
            edgecolor="white",
            linewidth=2.2,
            zorder=4,
        )
        ax.text(
            self.state.x,
            self.state.y - 0.33,
            "Robot",
            ha="center",
            va="center",
            fontsize=9,
            color="#111827",
            fontweight="bold",
            zorder=5,
        )

        ax.set_title("Robot Navigation Map", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    def history_rows(self) -> list[dict[str, Any]]:
        rows = []
        for event in self.history:
            confidence = event["confidence"]
            rows.append(
                {
                    "Step": event["step"],
                    "Command": event["command"],
                    "Confidence": "" if confidence is None else f"{confidence:.2%}",
                    "Action": event["action"],
                    "Position": str(event["position"]),
                    "Direction": event["direction"],
                    "Status": "blocked" if event["blocked"] else "applied",
                }
            )
        return rows
