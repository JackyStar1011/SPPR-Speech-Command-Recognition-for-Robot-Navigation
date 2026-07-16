from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path
import queue
import sys
import time
import wave

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import streamlit as st
import torch

# Streamlit's file watcher can accidentally inspect torch.classes as a Python
# package and emit noisy warnings on some Torch/Windows combinations.
try:
    torch.classes.__path__ = type("_TorchClassesPath", (), {"_path": []})()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.logmel import build_logmel_extractor
from src.inference.predictor import SpeechCommandPredictor
from src.robot.safety import SafetyDecisionLayer
from src.robot.simulator import RobotSimulator
from src.utils.config import load_config
from src.wakeword import OpenWakeWordDetector
from src.wakeword.audio_stream import MicrophoneFrameStream


@st.cache_resource
def load_predictor(checkpoint_path: str, config_path: str, device: str) -> SpeechCommandPredictor:
    return SpeechCommandPredictor(checkpoint_path, config_path=config_path, device_name=device)


def get_robot_simulator() -> RobotSimulator:
    if "robot_simulator" not in st.session_state:
        st.session_state.robot_simulator = RobotSimulator(width=12, height=12)
    return st.session_state.robot_simulator


def get_prediction_results() -> list[dict[str, object]]:
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = []
    return st.session_state.prediction_results


def record_microphone(sample_rate: int, seconds: float) -> torch.Tensor:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    frames_needed = int(seconds * sample_rate)
    collected_frames = 0
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time, status) -> None:
        if status:
            print(status)
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while collected_frames < frames_needed:
                chunk = audio_queue.get(timeout=seconds + 1.0)
                chunks.append(chunk)
                collected_frames += chunk.shape[0]
    finally:
        sd.stop()

    if not chunks:
        raise RuntimeError("No audio was recorded from the microphone.")

    audio = np.concatenate(chunks, axis=0)[:frames_needed]
    return torch.from_numpy(audio.T.copy())


def waveform_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    buffer = BytesIO()
    audio = waveform.squeeze().detach().cpu().clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).short().numpy().tobytes()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def plot_waveform(waveform, sample_rate: int):
    fig, ax = plt.subplots(figsize=(10, 3))
    time_axis = [index / sample_rate for index in range(waveform.size(-1))]
    ax.plot(time_axis, waveform.squeeze().cpu().numpy(), linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_logmel(logmel):
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(logmel.squeeze().cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def waveform_rms(waveform: torch.Tensor) -> float:
    audio = waveform.detach().float().cpu().squeeze()
    if audio.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(audio * audio)).item())


def build_export_html(rows: list[dict[str, object]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Step']))}</td>"
            f"<td>{html.escape(str(row['Command']))}</td>"
            f"<td>{html.escape(str(row['Raw command']))}</td>"
            f"<td>{html.escape(str(row['Confidence']))}</td>"
            f"<td>{html.escape(str(row['Action']))}</td>"
            f"<td>{html.escape(str(row['Position']))}</td>"
            f"<td>{html.escape(str(row['Direction']))}</td>"
            f"<td>{html.escape(str(row.get('Status', '')))}</td>"
            f"<td>{html.escape(str(row.get('Reason', '')))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Speech Command Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    img {{ width: 280px; max-width: 100%; }}
  </style>
</head>
<body>
  <h1>Speech Command Results</h1>
  <table>
    <thead>
      <tr>
        <th>Step</th>
        <th>Command</th>
        <th>Raw command</th>
        <th>Confidence</th>
        <th>Action</th>
        <th>Position</th>
        <th>Direction</th>
        <th>Status</th>
        <th>Reason</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def safety_status_label(status: str) -> str:
    labels = {
        "accepted": "ACCEPT",
        "applied": "EXECUTED",
        "blocked": "BLOCKED",
        "rejected": "REJECT",
        "ignored": "IGNORE",
    }
    return labels.get(status, status.upper())


def get_system_state(require_wake_word: bool) -> dict[str, object]:
    if "system_state" not in st.session_state:
        st.session_state.system_state = "IDLE"
    if "wake_word_detected" not in st.session_state:
        st.session_state.wake_word_detected = False

    state = str(st.session_state.system_state)
    wake_word_detected = bool(st.session_state.wake_word_detected)
    listening = state == "LISTENING" if require_wake_word else True
    return {
        "state": state,
        "wake_word_detected": wake_word_detected,
        "listening": listening,
    }


def reset_system_state() -> None:
    st.session_state.system_state = "IDLE"
    st.session_state.wake_word_detected = False


def apply_waveform_command(
    waveform: torch.Tensor,
    sample_rate: int,
    config: dict,
    config_path: str,
    checkpoint_path: str,
    device: str,
    threshold: float,
    require_wake_word: bool,
    wake_word_detected: bool,
    listening: bool,
    elapsed_since_wake_seconds: float,
    command_timeout: float,
    stop_threshold: float,
    simulator: RobotSimulator,
    prediction_results: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], object, torch.Tensor, bytes]:
    audio_bytes = waveform_to_wav_bytes(waveform, sample_rate)
    feature_extractor = build_logmel_extractor(config)
    logmel = feature_extractor(waveform)

    predictor = load_predictor(checkpoint_path, config_path, device)
    result = predictor.predict_waveform(
        waveform,
        sample_rate=sample_rate,
        threshold=0.0,
    )
    safety_config = dict(config)
    safety_settings = dict(config.get("safety", {}))
    safety_settings["confidence_threshold"] = threshold
    safety_settings["unknown_label"] = str(predictor.unknown_label)
    safety_settings["require_wake_word"] = require_wake_word
    safety_settings["command_timeout_seconds"] = command_timeout
    safety_settings["stop_confidence_threshold"] = stop_threshold
    safety_config["safety"] = safety_settings
    safety_decision = SafetyDecisionLayer.from_config(safety_config).decide(
        raw_label=str(result["raw_label"]),
        confidence=float(result["confidence"]),
        wake_word_detected=wake_word_detected,
        listening=listening,
        elapsed_since_wake_seconds=elapsed_since_wake_seconds,
    )
    applied_event = simulator.apply_decision(safety_decision)
    prediction_results.append(
        {
            "Step": applied_event["step"],
            "Command": safety_decision.label,
            "Raw command": result["raw_label"],
            "Confidence": f"{float(result['confidence']):.2%}",
            "Action": safety_decision.action,
            "Position": str(applied_event["position"]),
            "Direction": applied_event["direction"],
            "Status": applied_event["status"],
            "Reason": applied_event["reason"],
        }
    )
    return result, applied_event, safety_decision, logmel, audio_bytes


def render_navigation(
    simulator: RobotSimulator,
    result,
    applied_event,
    safety_decision,
    require_wake_word: bool,
    wake_word_display: str,
    listening: bool,
    command_timeout: float,
    threshold: float,
    stop_threshold: float,
    map_placeholder,
    state_placeholder,
) -> None:
    with map_placeholder.container():
        map_figure = simulator.render()
        st.pyplot(map_figure, clear_figure=True)
        plt.close(map_figure)

    with state_placeholder.container():
        state = simulator.state
        state_metric_1, state_metric_2 = st.columns(2)
        state_metric_1.metric("Position", str(state.position))
        state_metric_2.metric("Direction", state.direction)

        if result is None:
            st.metric("Wheelchair action", "WAITING")
        else:
            display_label = safety_decision.label if safety_decision is not None else str(result["label"])
            display_action = safety_decision.action if safety_decision is not None else str(result["action"])
            st.metric("Predicted command", str(display_label))
            st.metric("Raw command", str(result["raw_label"]))
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Wheelchair action", str(display_action))

            if safety_decision is not None and applied_event is not None:
                decision_status = safety_status_label(str(applied_event["status"]))
                if safety_decision.accepted and not applied_event["blocked"]:
                    st.success(f"Safety decision: {decision_status}")
                elif applied_event["blocked"]:
                    st.warning(f"Safety decision: {decision_status}")
                elif applied_event["status"] == "rejected":
                    st.warning(f"Safety decision: {decision_status}")
                else:
                    st.info(f"Safety decision: {decision_status}")

                st.caption(str(applied_event["reason"]))
                st.dataframe(
                    [
                        {"Rule": "Wake word required", "Value": str(require_wake_word)},
                        {"Rule": "Wake word detected", "Value": wake_word_display},
                        {"Rule": "Listening state", "Value": str(listening)},
                        {"Rule": "Command timeout", "Value": f"{command_timeout:.1f}s"},
                        {"Rule": "Confidence threshold", "Value": f"{threshold:.2f}"},
                        {"Rule": "Stop threshold", "Value": f"{stop_threshold:.2f}"},
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            if applied_event is not None and applied_event["blocked"]:
                st.warning("Movement blocked by map boundary.")
            elif applied_event is not None and applied_event["status"] == "rejected":
                st.warning("Command rejected because confidence is below the safety threshold.")
            elif applied_event is not None and applied_event["status"] == "ignored":
                st.info("Command ignored by the safety layer.")

        history_rows = simulator.history_rows()
        if history_rows:
            st.dataframe(history_rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Speech Command Wheelchair Demo", layout="wide")
    st.title("Speech Command Wheelchair Demo")

    with st.sidebar:
        st.subheader("Model")
        config_path = st.text_input("Config", value="configs/models/cnn_gru.yaml")
        config = load_config(config_path)
        safety_cfg = config.get("safety", {})
        checkpoint_path = st.text_input("Checkpoint", value=config["training"]["checkpoint_path"])
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

        st.subheader("Recording")
        record_seconds = st.number_input(
            "Command record seconds",
            min_value=0.25,
            max_value=3.0,
            value=2.0,
            step=0.25,
            key="command_record_seconds_v2",
        )

        st.subheader("Safety")
        configured_threshold = float(
            safety_cfg.get(
                "confidence_threshold",
                config.get("inference", {}).get("threshold", 0.70),
            )
        )
        default_threshold = configured_threshold if configured_threshold > 0.0 else 0.70
        threshold = st.slider("Confidence threshold", 0.0, 1.0, default_threshold, 0.01)
        require_wake_word = st.checkbox(
            "Require wake word",
            value=bool(safety_cfg.get("require_wake_word", False)),
        )
        command_timeout = st.number_input(
            "Command timeout (s)",
            min_value=0.5,
            max_value=10.0,
            value=float(safety_cfg.get("command_timeout_seconds") or 2.0),
            step=0.5,
            key="command_timeout_seconds_v2",
        )
        stop_threshold = st.slider(
            "Stop confidence threshold",
            0.0,
            1.0,
            float(safety_cfg.get("stop_confidence_threshold", 0.0)),
            0.01,
        )

        st.subheader("Wake word")
        wakeword_cfg = config.get("wakeword", {})
        auto_wake_listener = st.checkbox(
            "Auto wake listener",
            value=bool(wakeword_cfg.get("enabled", False)),
        )
        wakeword_model_name = st.text_input(
            "Wake word model",
            value=str(wakeword_cfg.get("model_name", "hey_jarvis")),
        )
        wakeword_threshold = st.slider(
            "Wake word threshold",
            0.0,
            1.0,
            float(wakeword_cfg.get("threshold", 0.5)),
            0.01,
        )
        wake_frame_ms = st.number_input(
            "Wake frame (ms)",
            min_value=80.0,
            max_value=800.0,
            value=float(wakeword_cfg.get("frame_ms", 320.0)),
            step=80.0,
        )
        wake_session_seconds = st.number_input(
            "Auto listener session (s)",
            min_value=5.0,
            max_value=300.0,
            value=float(wakeword_cfg.get("session_seconds", 60.0)),
            step=5.0,
        )
        speech_rms_threshold = st.slider(
            "Speech activity threshold",
            0.001,
            0.100,
            float(wakeword_cfg.get("speech_rms_threshold", 0.010)),
            0.001,
            format="%.3f",
        )

        system_state = get_system_state(require_wake_word)
        wake_word_detected = bool(system_state["wake_word_detected"])
        listening = bool(system_state["listening"])

        st.subheader("System state")
        st.metric("State", str(system_state["state"]) if require_wake_word else "COMMAND_ONLY")
        st.metric("Wake listener", "ON" if auto_wake_listener else "OFF")
        st.caption(f"Wake model: {wakeword_model_name}")
        wake_word_display = "DETECTED" if wake_word_detected else "WAITING"
        reset_clicked = st.button("Reset simulator")

    simulator = get_robot_simulator()
    prediction_results = get_prediction_results()
    record_clicked = st.button("Record command", type="primary")

    if reset_clicked:
        simulator.reset()
        reset_system_state()
        prediction_results.clear()

    result = None
    applied_event = None
    safety_decision = None
    waveform = None
    logmel = None
    audio_bytes = None
    data_cfg = config["data"]

    map_col, state_col = st.columns([1.35, 1.0])
    map_placeholder = map_col.empty()
    state_placeholder = state_col.empty()
    render_navigation(
        simulator,
        result,
        applied_event,
        safety_decision,
        require_wake_word,
        wake_word_display,
        listening,
        command_timeout,
        threshold,
        stop_threshold,
        map_placeholder,
        state_placeholder,
    )

    if record_clicked:
        if Path(checkpoint_path).exists():
            sample_rate = data_cfg["sample_rate"]
            try:
                with st.spinner("Recording..."):
                    waveform = record_microphone(sample_rate, float(record_seconds))
                result, applied_event, safety_decision, logmel, audio_bytes = apply_waveform_command(
                    waveform,
                    sample_rate,
                    config,
                    config_path,
                    checkpoint_path,
                    device,
                    threshold,
                    require_wake_word,
                    wake_word_detected,
                    listening,
                    float(record_seconds),
                    command_timeout,
                    stop_threshold,
                    simulator,
                    prediction_results,
                )
                render_navigation(
                    simulator,
                    result,
                    applied_event,
                    safety_decision,
                    require_wake_word,
                    wake_word_display,
                    listening,
                    command_timeout,
                    threshold,
                    stop_threshold,
                    map_placeholder,
                    state_placeholder,
                )
            except Exception as error:
                st.error(f"Microphone recording failed: {error}")
        else:
            st.warning(f"Checkpoint not found: {checkpoint_path}")

    if auto_wake_listener:
        if not Path(checkpoint_path).exists():
            st.warning(f"Checkpoint not found: {checkpoint_path}")
        elif not OpenWakeWordDetector.is_available():
            st.error("openWakeWord is not installed. Run `pip install openwakeword` in the active environment.")
        else:
            sample_rate = data_cfg["sample_rate"]
            status_placeholder = st.empty()
            wake_score_placeholder = st.empty()
            audio_level_placeholder = st.empty()
            try:
                detector = OpenWakeWordDetector(
                    model_name=wakeword_model_name,
                    threshold=wakeword_threshold,
                )
                deadline = time.monotonic() + float(wake_session_seconds)
                wake_active = False
                last_command_activity = 0.0
                command_frames: list[torch.Tensor] = []
                command_samples = int(sample_rate * float(record_seconds))
                frame_stream = MicrophoneFrameStream(sample_rate, float(wake_frame_ms))
                for frame in frame_stream.frames(timeout_seconds=1.0):
                    if time.monotonic() >= deadline:
                        break
                    now = time.monotonic()
                    frame_level = waveform_rms(frame)
                    audio_level_placeholder.metric("Audio level", f"{frame_level:.3f}")

                    if not wake_active:
                        wake_result = detector.predict_frame(frame, sample_rate=sample_rate)
                        wake_score_placeholder.metric("Wake score", f"{wake_result.score:.2f}")
                        status_placeholder.info(f"Listening for wake word: {wakeword_model_name}")
                        if wake_result.detected:
                            wake_active = True
                            last_command_activity = now
                            command_frames = []
                            st.session_state.system_state = "LISTENING"
                            st.session_state.wake_word_detected = True
                            status_placeholder.success("Wake word detected. Say a command.")
                        continue

                    if frame_level >= float(speech_rms_threshold):
                        last_command_activity = now
                        command_frames.append(frame)
                    elif command_frames:
                        command_frames.append(frame)

                    idle_seconds = now - last_command_activity
                    if idle_seconds >= float(command_timeout):
                        wake_active = False
                        command_frames = []
                        st.session_state.system_state = "IDLE"
                        st.session_state.wake_word_detected = False
                        status_placeholder.info(
                            f"Command session timed out after {command_timeout:.1f}s of silence. "
                            "Listening for wake word again."
                        )
                        continue

                    status_placeholder.info(
                        f"Command session active. Silence timeout in "
                        f"{max(0.0, float(command_timeout) - idle_seconds):.1f}s."
                    )
                    if not command_frames:
                        continue

                    captured = torch.cat(command_frames, dim=-1)
                    if captured.size(-1) < command_samples:
                        continue

                    command_audio = captured[:, :command_samples]
                    remaining = captured[:, command_samples:]
                    command_frames = [remaining] if remaining.numel() else []
                    if waveform_rms(command_audio) < float(speech_rms_threshold):
                        continue

                    result, applied_event, safety_decision, logmel, audio_bytes = apply_waveform_command(
                        command_audio,
                        sample_rate,
                        config,
                        config_path,
                        checkpoint_path,
                        device,
                        threshold,
                        True,
                        True,
                        True,
                        0.0,
                        command_timeout,
                        stop_threshold,
                        simulator,
                        prediction_results,
                    )
                    wake_word_display = "DETECTED"
                    render_navigation(
                        simulator,
                        result,
                        applied_event,
                        safety_decision,
                        True,
                        wake_word_display,
                        True,
                        command_timeout,
                        threshold,
                        stop_threshold,
                        map_placeholder,
                        state_placeholder,
                    )
                    status_placeholder.success(
                        f"Executed: raw={result['raw_label']} confidence={float(result['confidence']):.2%} "
                        f"status={safety_decision.status}"
                    )
            except Exception as error:
                st.error(f"Auto wake listener failed: {error}")

    if waveform is not None and logmel is not None:
        if audio_bytes is not None:
            st.subheader("Recorded audio")
            st.audio(audio_bytes, format="audio/wav")
        left_col, right_col = st.columns(2)
        with left_col:
            waveform_figure = plot_waveform(waveform, data_cfg["sample_rate"])
            st.pyplot(waveform_figure, clear_figure=True)
            plt.close(waveform_figure)
        with right_col:
            logmel_figure = plot_logmel(logmel)
            st.pyplot(logmel_figure, clear_figure=True)
            plt.close(logmel_figure)

    if prediction_results:
        st.subheader("Export results")
        st.dataframe(prediction_results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download HTML report",
            data=build_export_html(prediction_results),
            file_name="speech_command_results.html",
            mime="text/html",
        )


if __name__ == "__main__":
    main()
