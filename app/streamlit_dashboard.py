from __future__ import annotations

import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
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


def file_version(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(ttl=1.0, show_spinner=False)
def load_snapshot(path_value: str, version: int) -> dict[str, Any] | None:
    del version
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


@st.cache_data(ttl=5.0, show_spinner=False)
def load_recent_events(
    path_value: str,
    version: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    del version
    path = Path(path_value)
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
    axis.set_title(f"Wheelchair trajectory - {arena_size:.1f} m x {arena_size:.1f} m")
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


def render_map_png(state: dict[str, Any], arena_size: float) -> bytes:
    figure = build_webots_map(state, arena_size)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def display_value(value: Any, fallback: str = "-") -> str:
    return fallback if value is None or value == "" else str(value)


def rounded_number(value: Any, digits: int = 3) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def main() -> None:
    st.set_page_config(page_title="Voice Wheelchair Dashboard", layout="wide")
    st.title("Wake-word Voice Wheelchair - Webots Dashboard")
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
        st.subheader("Refresh")
        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_interval = st.slider(
            "Status interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            disabled=not auto_refresh,
        )
        event_limit = st.number_input(
            "Recent events",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

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

    status_run_every = float(refresh_interval) if auto_refresh else None
    map_run_every = max(2.0, float(refresh_interval) * 2.0) if auto_refresh else None
    events_run_every = max(5.0, float(refresh_interval) * 5.0) if auto_refresh else None

    @st.fragment(run_every=status_run_every)
    def render_status() -> None:
        state = load_snapshot(str(snapshot_path), file_version(snapshot_path))
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

        voice_column, robot_column = st.columns(2)
        with voice_column:
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

        with robot_column:
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

    @st.fragment(run_every=map_run_every)
    def render_map() -> None:
        state = load_snapshot(str(snapshot_path), file_version(snapshot_path))
        st.subheader("Webots map")
        if state is None:
            st.info("Map will appear after the runtime creates its first state snapshot.")
            return

        webots = state.get("webots", {})
        trajectory = state.get("trajectory", [])
        last_point = trajectory[-1] if trajectory else {}
        signature = (
            str(snapshot_path),
            float(arena_size),
            rounded_number(webots.get("x")),
            rounded_number(webots.get("y")),
            rounded_number(webots.get("yaw")),
            len(trajectory),
            last_point.get("timestamp") if isinstance(last_point, dict) else None,
        )
        if st.session_state.get("dashboard_map_signature") != signature:
            st.session_state.dashboard_map_png = render_map_png(state, float(arena_size))
            st.session_state.dashboard_map_signature = signature

        map_png = st.session_state.get("dashboard_map_png")
        if map_png:
            st.image(map_png, use_container_width=True)
        else:
            st.info("Waiting for map data.")

    @st.fragment(run_every=events_run_every)
    def render_events() -> None:
        st.subheader("Recent events")
        events = load_recent_events(
            str(event_log_path),
            file_version(event_log_path),
            int(event_limit),
        )
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

    render_status()
    render_map()
    render_events()


if __name__ == "__main__":
    main()
