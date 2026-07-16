from __future__ import annotations

import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.robot.webots_udp import WebotsUDPClient  # noqa: E402
from src.runtime import load_runtime_config  # noqa: E402


def load_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def load_recent_events(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            lines = deque(file, maxlen=limit)
    except OSError:
        return []

    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return list(reversed(events))


def is_telemetry_fresh(webots: dict[str, Any], max_age_seconds: float = 1.0) -> bool:
    received_at = webots.get("received_at")
    if not received_at:
        return False
    try:
        received_time = datetime.fromisoformat(str(received_at))
    except ValueError:
        return False
    if received_time.tzinfo is None:
        received_time = received_time.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - received_time).total_seconds() <= max_age_seconds


def build_webots_map(state: dict[str, Any], arena_size: float):
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    half_size = arena_size / 2.0
    axis.set_xlim(-half_size, half_size)
    axis.set_ylim(-half_size, half_size)
    axis.set_aspect("equal")
    axis.set_xlabel("Webots X (m)")
    axis.set_ylabel("Webots Y (m)")
    axis.set_title(f"Wheelchair trajectory — {arena_size:.1f} m × {arena_size:.1f} m")
    axis.grid(color="#cbd5e1", linewidth=0.8, alpha=0.8)
    axis.add_patch(
        patches.Rectangle(
            (-half_size, -half_size),
            arena_size,
            arena_size,
            fill=False,
            edgecolor="#334155",
            linewidth=2.0,
        )
    )

    trajectory = state.get("trajectory", [])
    valid_points = [
        point
        for point in trajectory
        if isinstance(point, dict) and "x" in point and "y" in point
    ]
    if valid_points:
        axis.plot(
            [float(point["x"]) for point in valid_points],
            [float(point["y"]) for point in valid_points],
            color="#2563eb",
            linewidth=2.2,
            alpha=0.85,
            label="Trajectory",
        )

    webots = state.get("webots", {})
    if webots.get("telemetry_available"):
        x = float(webots.get("x", 0.0))
        y = float(webots.get("y", 0.0))
        yaw = float(webots.get("yaw", 0.0))
        axis.scatter([x], [y], s=260, color="#0f172a", edgecolor="white", zorder=5)
        arrow_length = 0.45
        axis.arrow(
            x,
            y,
            math.cos(yaw) * arrow_length,
            math.sin(yaw) * arrow_length,
            width=0.035,
            head_width=0.16,
            head_length=0.16,
            length_includes_head=True,
            color="#ef4444",
            zorder=6,
        )
        axis.text(x, y - 0.2, "Wheelchair", ha="center", va="top", fontsize=9)

    figure.tight_layout()
    return figure


def display_value(value: Any, fallback: str = "—") -> str:
    return fallback if value is None or value == "" else str(value)


def main() -> None:
    st.set_page_config(page_title="Voice Wheelchair Dashboard", layout="wide")
    st.title("Wake-word Voice Wheelchair — Webots Dashboard")
    st.caption("Monitoring only: microphone and wake-word run in the standalone runtime process.")

    with st.sidebar:
        st.subheader("Runtime")
        config_path = st.text_input(
            "Runtime config",
            value="configs/runtime/voice_webots.yaml",
        )
        arena_size = st.number_input(
            "Webots arena size (m)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )

    try:
        runtime_config = load_runtime_config(config_path)
    except Exception as error:
        st.error(f"Failed to load runtime config: {error}")
        return

    snapshot_path = Path(runtime_config.runtime.snapshot_path)
    event_log_path = Path(runtime_config.runtime.event_log_path)

    if st.button("EMERGENCY STOP", type="primary", use_container_width=True):
        try:
            with WebotsUDPClient(
                runtime_config.webots.command_host,
                runtime_config.webots.command_port,
            ) as client:
                client.stop()
        except Exception as error:
            st.error(f"Failed to send STOP: {error}")
        else:
            st.success("STOP sent to Webots.")

    @st.fragment(run_every=0.5)
    def render_live_dashboard() -> None:
        state = load_snapshot(snapshot_path)
        if state is None:
            st.warning(
                f"Waiting for runtime state at `{snapshot_path}`. "
                "Start `python scripts/run_voice_webots.py` first."
            )
            return

        webots = state.get("webots", {})
        wakeword = state.get("wakeword", {})
        capture = state.get("capture", {})
        prediction = state.get("prediction", {})
        dispatch = state.get("dispatch", {})

        status_columns = st.columns(4)
        status_columns[0].metric("Runtime", display_value(state.get("runtime_state")))
        status_columns[1].metric(
            "Webots telemetry",
            "ONLINE" if is_telemetry_fresh(webots) else "OFFLINE",
        )
        status_columns[2].metric("Motion", display_value(webots.get("motion_state")))
        status_columns[3].metric("Last event", display_value(state.get("last_event")))

        map_column, detail_column = st.columns([1.35, 1.0])
        with map_column:
            figure = build_webots_map(state, float(arena_size))
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)

        with detail_column:
            st.subheader("Wake word")
            wake_columns = st.columns(3)
            wake_columns[0].metric("Model", display_value(wakeword.get("model_name")))
            wake_columns[1].metric(
                "Score",
                f"{float(wakeword.get('score', 0.0)):.3f}",
            )
            wake_columns[2].metric(
                "Threshold",
                f"{float(wakeword.get('threshold', 0.0)):.3f}",
            )

            st.subheader("Speech command")
            prediction_columns = st.columns(2)
            prediction_columns[0].metric(
                "Raw command",
                display_value(prediction.get("raw_label")),
            )
            prediction_columns[1].metric(
                "Confidence",
                f"{float(prediction.get('confidence', 0.0)):.2%}",
            )
            st.metric("Action", display_value(prediction.get("action")))
            st.metric("Safety status", display_value(prediction.get("status")))
            if prediction.get("reason"):
                st.caption(str(prediction["reason"]))

            st.subheader("Audio capture")
            st.write(
                {
                    "captured": capture.get("captured"),
                    "duration_seconds": capture.get("duration_seconds"),
                    "peak_rms": capture.get("peak_rms"),
                    "reason": capture.get("reason"),
                }
            )

            st.subheader("Webots")
            st.write(
                {
                    "x": webots.get("x"),
                    "y": webots.get("y"),
                    "yaw": webots.get("yaw"),
                    "left_velocity": webots.get("left_velocity"),
                    "right_velocity": webots.get("right_velocity"),
                    "last_dispatch": dispatch,
                }
            )

        st.subheader("Recent events")
        events = load_recent_events(event_log_path)
        if events:
            rows = [
                {
                    "Time": event.get("timestamp"),
                    "State": event.get("state"),
                    "Event": event.get("kind"),
                    "Details": json.dumps(event.get("data", {}), ensure_ascii=False),
                }
                for event in events
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No runtime events yet.")

    render_live_dashboard()


if __name__ == "__main__":
    main()
