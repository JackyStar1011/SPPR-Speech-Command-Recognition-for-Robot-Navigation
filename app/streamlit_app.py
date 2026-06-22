from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_waveform, preprocess_waveform
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


def make_upload_key(uploaded_file, audio_bytes: bytes) -> str:
    digest = hashlib.sha256(audio_bytes).hexdigest()
    return f"{uploaded_file.name}:{len(audio_bytes)}:{digest}"


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
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        reset_clicked = st.button("Reset simulator")

    simulator = get_robot_simulator()
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    audio_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    upload_key = make_upload_key(uploaded_file, audio_bytes) if uploaded_file is not None and audio_bytes is not None else None

    if reset_clicked:
        simulator.reset()
        if upload_key is not None:
            st.session_state.last_robot_audio_key = upload_key

    result = None
    applied_event = None
    temp_path = None
    waveform = None
    logmel = None
    data_cfg = config["data"]

    if uploaded_file is not None and audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        waveform, sample_rate = load_waveform(str(temp_path))
        waveform = preprocess_waveform(
            waveform,
            sample_rate=sample_rate,
            target_sample_rate=data_cfg["sample_rate"],
            target_num_samples=int(data_cfg["sample_rate"] * data_cfg["duration_seconds"]),
        )
        feature_extractor = build_logmel_extractor(config)
        logmel = feature_extractor(waveform)

        if Path(checkpoint_path).exists():
            predictor = load_predictor(checkpoint_path, config_path, device)
            result = predictor.predict_file(temp_path, threshold=threshold)
            if upload_key != st.session_state.get("last_robot_audio_key"):
                applied_event = simulator.apply_command(
                    str(result["label"]),
                    confidence=float(result["confidence"]),
                )
                st.session_state.last_robot_audio_key = upload_key
        else:
            st.warning(f"Checkpoint not found: {checkpoint_path}")

        temp_path.unlink(missing_ok=True)

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
