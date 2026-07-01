from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.robot.actions import label_to_action
from src.robot.safety import SafetyDecision


DIRECTIONS = ("NORTH", "EAST", "SOUTH", "WEST")
DIRECTION_DELTAS = {
    "NORTH": (0, 1),
    "EAST": (1, 0),
    "SOUTH": (0, -1),
    "WEST": (-1, 0),
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
    def __init__(self, width: int = 12, height: int = 12) -> None:
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

    def apply_command(
        self,
        label: str,
        confidence: float | None = None,
        action: str | None = None,
        status: str = "accepted",
        reason: str = "command accepted",
        raw_label: str | None = None,
    ) -> dict[str, Any]:
        action = action or label_to_action(label)
        previous_position = self.state.position
        previous_direction = self.state.direction

        moved = False
        blocked = False
        if status != "accepted":
            pass
        elif label == "forward":
            moved, blocked = self._move(multiplier=1)
        elif label == "backward":
            moved, blocked = self._move(multiplier=-1)
        elif label == "left":
            self._turn(step=-1)
            moved, blocked = self._move(multiplier=1)
        elif label == "right":
            self._turn(step=1)
            moved, blocked = self._move(multiplier=1)
        elif label == "stop":
            pass

        event_status = "blocked" if blocked else ("applied" if status == "accepted" else status)
        event_reason = "map boundary reached" if blocked else reason

        event = {
            "step": len(self.history) + 1,
            "command": label,
            "raw_command": raw_label or label,
            "confidence": confidence,
            "action": action,
            "from_position": previous_position,
            "position": self.state.position,
            "from_direction": previous_direction,
            "direction": self.state.direction,
            "moved": moved,
            "blocked": blocked,
            "status": event_status,
            "reason": event_reason,
        }
        self.history.append(event)
        return event

    def apply_decision(self, decision: SafetyDecision) -> dict[str, Any]:
        return self.apply_command(
            decision.label,
            confidence=decision.confidence,
            action=decision.action,
            status=decision.status,
            reason=decision.reason,
            raw_label=decision.raw_label,
        )

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
        fig, ax = plt.subplots(figsize=(7.8, 7.8))
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(color="#cbd5e1", linewidth=1.0)
        ax.set_facecolor("#eef2f7")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.add_patch(
            patches.Rectangle(
                (-0.5, -0.5),
                self.width,
                self.height,
                fill=False,
                edgecolor="#334155",
                linewidth=2.2,
                zorder=1,
            )
        )

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

        ax.add_patch(
            patches.Rectangle(
                (self.state.x - 0.48, self.state.y - 0.48),
                0.96,
                0.96,
                facecolor="#dbeafe",
                edgecolor="#2563eb",
                linewidth=1.8,
                alpha=0.55,
                zorder=3,
            )
        )
        self._draw_wheelchair(ax)
        ax.text(
            self.state.x,
            self.state.y - 0.58,
            "Wheelchair",
            ha="center",
            va="center",
            fontsize=8,
            color="#111827",
            fontweight="bold",
            zorder=7,
        )

        ax.set_title("Wheelchair Navigation Map", fontsize=16, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    def _draw_wheelchair(self, ax) -> None:
        x, y = self.state.position
        dx, dy = DIRECTION_DELTAS[self.state.direction]

        ax.add_patch(
            patches.Circle(
                (x - 0.27, y - 0.08),
                0.23,
                facecolor="#bfdbfe",
                edgecolor="#1e40af",
                linewidth=2.2,
                zorder=5,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x + 0.27, y - 0.08),
                0.23,
                facecolor="#bfdbfe",
                edgecolor="#1e40af",
                linewidth=2.2,
                zorder=5,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x - 0.27, y - 0.08),
                0.08,
                facecolor="#1e3a8a",
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x + 0.27, y - 0.08),
                0.08,
                facecolor="#1e3a8a",
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.25, y - 0.2),
                0.5,
                0.42,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0f766e",
                edgecolor="white",
                linewidth=1.8,
                zorder=6,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.18, y + 0.13),
                0.36,
                0.2,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#14b8a6",
                edgecolor="white",
                linewidth=1.4,
                zorder=6,
            )
        )
        ax.add_patch(
            patches.FancyArrowPatch(
                (x - dx * 0.08, y - dy * 0.08),
                (x + dx * 0.58, y + dy * 0.58),
                arrowstyle="-|>",
                mutation_scale=24,
                linewidth=3.0,
                color="#ef4444",
                zorder=8,
            )
        )

    def history_rows(self) -> list[dict[str, Any]]:
        rows = []
        for event in self.history:
            confidence = event["confidence"]
            rows.append(
                {
                    "Step": event["step"],
                    "Command": event["command"],
                    "Raw command": event.get("raw_command", event["command"]),
                    "Confidence": "" if confidence is None else f"{confidence:.2%}",
                    "Action": event["action"],
                    "Position": str(event["position"]),
                    "Direction": event["direction"],
                    "Status": event.get("status", "blocked" if event["blocked"] else "applied"),
                    "Reason": event.get("reason", ""),
                }
            )
        return rows
