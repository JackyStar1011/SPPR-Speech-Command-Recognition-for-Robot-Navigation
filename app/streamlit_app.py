from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import sounddevice as sd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.logmel import build_logmel_extractor
from src.inference.predictor import SpeechCommandPredictor
from src.robot.simulator import RobotSimulator
from src.utils.config import load_config


@st.cache_resource
def load_predictor(checkpoint_path: str, config_path: str, device: str) -> SpeechCommandPredictor:
    return SpeechCommandPredictor(checkpoint_path, config_path=config_path, device_name=device)


def get_robot_simulator() -> RobotSimulator:
    if "robot_simulator" not in st.session_state:
        st.session_state.robot_simulator = RobotSimulator(width=8, height=8)
    return st.session_state.robot_simulator


def record_microphone(sample_rate: int, seconds: float) -> torch.Tensor:
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return torch.from_numpy(audio.T.copy())


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


def main() -> None:
    st.set_page_config(page_title="Speech Command Robot Demo", layout="wide")
    st.title("Speech Command Robot Demo")

    with st.sidebar:
        config_path = st.text_input("Config", value="configs/baseline.yaml")
        config = load_config(config_path)
        checkpoint_path = st.text_input("Checkpoint", value=config["training"]["checkpoint_path"])
        threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.70, 0.01)
        record_seconds = st.number_input("Record seconds", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        reset_clicked = st.button("Reset simulator")

    simulator = get_robot_simulator()
    record_clicked = st.button("Record command", type="primary")

    if reset_clicked:
        simulator.reset()

    result = None
    applied_event = None
    waveform = None
    logmel = None
    data_cfg = config["data"]

    if record_clicked:
        if Path(checkpoint_path).exists():
            sample_rate = data_cfg["sample_rate"]
            try:
                with st.spinner("Recording..."):
                    waveform = record_microphone(sample_rate, float(record_seconds))
                feature_extractor = build_logmel_extractor(config)
                logmel = feature_extractor(waveform)

                predictor = load_predictor(checkpoint_path, config_path, device)
                result = predictor.predict_waveform(
                    waveform,
                    sample_rate=sample_rate,
                    threshold=threshold,
                )
                applied_event = simulator.apply_command(
                    str(result["label"]),
                    confidence=float(result["confidence"]),
                )
            except Exception as error:
                st.error(f"Microphone recording failed: {error}")
        else:
            st.warning(f"Checkpoint not found: {checkpoint_path}")

    map_col, state_col = st.columns([1.35, 1.0])
    with map_col:
        st.pyplot(simulator.render(), clear_figure=True)
    with state_col:
        state = simulator.state
        state_metric_1, state_metric_2 = st.columns(2)
        state_metric_1.metric("Position", str(state.position))
        state_metric_2.metric("Direction", state.direction)

        if result is None:
            st.metric("Robot action", "WAITING")
        else:
            st.metric("Predicted command", str(result["label"]))
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Robot action", str(result["action"]))
            if applied_event is not None and applied_event["blocked"]:
                st.warning("Movement blocked by map boundary.")

        history_rows = simulator.history_rows()
        if history_rows:
            st.dataframe(history_rows, use_container_width=True, hide_index=True)

    if waveform is not None and logmel is not None:
        left_col, right_col = st.columns(2)
        with left_col:
            st.pyplot(plot_waveform(waveform, data_cfg["sample_rate"]), clear_figure=True)
        with right_col:
            st.pyplot(plot_logmel(logmel), clear_figure=True)


if __name__ == "__main__":
    main()
