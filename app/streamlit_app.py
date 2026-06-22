from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import get_speech_alignment_config, load_waveform, preprocess_waveform
from src.features.logmel import build_logmel_extractor
from src.inference.predictor import SpeechCommandPredictor
from src.utils.config import load_config


@st.cache_resource
def load_predictor(checkpoint_path: str, config_path: str, device: str) -> SpeechCommandPredictor:
    return SpeechCommandPredictor(checkpoint_path, config_path=config_path, device_name=device)


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
        config_path = st.text_input("Config", value="configs/cnn_gru.yaml")
        config = load_config(config_path)
        checkpoint_path = st.text_input("Checkpoint", value=config["training"]["checkpoint_path"])
        default_threshold = float(config.get("inference", {}).get("threshold", 0.70))
        threshold = st.slider("Confidence threshold", 0.0, 1.0, default_threshold, 0.01)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_file is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    waveform, sample_rate = load_waveform(str(temp_path))
    data_cfg = config["data"]
    align_speech, speech_alignment = get_speech_alignment_config(config, inference=True)
    waveform = preprocess_waveform(
        waveform,
        sample_rate=sample_rate,
        target_sample_rate=data_cfg["sample_rate"],
        target_num_samples=int(data_cfg["sample_rate"] * data_cfg["duration_seconds"]),
        align_speech=align_speech,
        speech_alignment=speech_alignment,
    )
    feature_extractor = build_logmel_extractor(config)
    logmel = feature_extractor(waveform)

    left_col, right_col = st.columns(2)
    with left_col:
        st.pyplot(plot_waveform(waveform, data_cfg["sample_rate"]), clear_figure=True)
    with right_col:
        st.pyplot(plot_logmel(logmel), clear_figure=True)

    if not Path(checkpoint_path).exists():
        st.warning(f"Checkpoint not found: {checkpoint_path}")
        temp_path.unlink(missing_ok=True)
        return

    predictor = load_predictor(checkpoint_path, config_path, device)
    result = predictor.predict_file(temp_path, threshold=threshold)
    temp_path.unlink(missing_ok=True)

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Predicted command", str(result["label"]))
    metric_col_2.metric("Confidence", f"{result['confidence']:.2%}")
    metric_col_3.metric("Robot action", str(result["action"]))


if __name__ == "__main__":
    main()
