from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from src.robot.webots_telemetry import WebotsTelemetryReceiver
from src.runtime.state import RuntimeEvent, RuntimeState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CompositeEventHandler:
    def __init__(self, *handlers: Callable[[RuntimeEvent], None]) -> None:
        self.handlers = handlers

    def __call__(self, event: RuntimeEvent) -> None:
        for handler in self.handlers:
            handler(event)


class RuntimeStateStore:
    """Persist the latest runtime state and significant events for dashboards."""

    def __init__(
        self,
        snapshot_path: str | Path,
        event_log_path: str | Path,
        *,
        trajectory_limit: int = 500,
    ) -> None:
        if trajectory_limit <= 0:
            raise ValueError("trajectory_limit must be positive")

        self.snapshot_path = Path(snapshot_path)
        self.event_log_path = Path(event_log_path)
        self.trajectory_limit = trajectory_limit
        self._lock = RLock()
        self._state: dict[str, Any] = {
            "updated_at": utc_now(),
            "runtime_state": RuntimeState.STOPPED.value,
            "last_event": None,
            "wakeword": {},
            "capture": {},
            "prediction": {},
            "dispatch": {},
            "webots": {"telemetry_available": False},
            "trajectory": [],
            "error": None,
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_snapshot()

    def handle_event(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._state["updated_at"] = event.timestamp
            self._state["runtime_state"] = event.state.value
            self._state["last_event"] = event.kind

            if event.kind == "wake_score":
                self._state["wakeword"] = dict(event.data)
            elif event.kind == "command_capture":
                self._state["capture"] = dict(event.data)
            elif event.kind == "prediction":
                self._state["prediction"] = dict(event.data)
            elif event.kind in {"action_dispatched", "action_not_dispatched"}:
                self._state["dispatch"] = dict(event.data)
            elif event.kind == "runtime_error":
                self._state["error"] = dict(event.data)

            self._write_snapshot()
            if event.kind != "wake_score" or bool(event.data.get("detected")):
                self._append_event(
                    {
                        "timestamp": event.timestamp,
                        "kind": event.kind,
                        "state": event.state.value,
                        "data": event.data,
                    }
                )

    def update_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self._lock:
            received_at = utc_now()
            self._state["updated_at"] = received_at
            self._state["webots"] = {
                **telemetry,
                "telemetry_available": True,
                "received_at": received_at,
            }
            self._append_trajectory_point(telemetry)
            self._write_snapshot()

    def mark_telemetry_error(self, error: Exception) -> None:
        with self._lock:
            self._state["webots"] = {
                **self._state.get("webots", {}),
                "telemetry_error": str(error),
                "telemetry_error_at": utc_now(),
            }
            self._write_snapshot()

    def _append_trajectory_point(self, telemetry: dict[str, Any]) -> None:
        point = {
            "x": float(telemetry["x"]),
            "y": float(telemetry["y"]),
            "yaw": float(telemetry["yaw"]),
            "timestamp": float(telemetry["timestamp"]),
        }
        trajectory: list[dict[str, float]] = self._state["trajectory"]
        if trajectory:
            previous = trajectory[-1]
            distance = math.hypot(point["x"] - previous["x"], point["y"] - previous["y"])
            if distance < 0.005 and abs(point["yaw"] - previous["yaw"]) < 0.01:
                return
        trajectory.append(point)
        if len(trajectory) > self.trajectory_limit:
            del trajectory[: len(trajectory) - self.trajectory_limit]

    def _write_snapshot(self) -> None:
        temporary_path = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.snapshot_path)

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.event_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


class TelemetryMonitor:
    def __init__(
        self,
        receiver: WebotsTelemetryReceiver,
        state_store: RuntimeStateStore,
        *,
        poll_timeout_seconds: float = 0.25,
    ) -> None:
        self.receiver = receiver
        self.state_store = state_store
        self.poll_timeout_seconds = poll_timeout_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="webots-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_timeout_seconds + 0.5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                telemetry = self.receiver.receive(self.poll_timeout_seconds)
            except Exception as error:
                self.state_store.mark_telemetry_error(error)
                continue
            if telemetry is not None:
                self.state_store.update_telemetry(telemetry)

    def __enter__(self) -> TelemetryMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
