from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
import sys
import wave

import matplotlib.pyplot as plt
import sounddevice as sd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.logmel import build_logmel_extractor
from src.inference.predictor import SpeechCommandPredictor
from src.robot.safety import SafetyDecisionLayer
from src.robot.simulator import RobotSimulator
from src.utils.config import load_config


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
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
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


def audio_to_data_uri(audio_bytes: bytes) -> str:
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


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


def figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
            f"<td><audio controls src=\"{row.get('AudioData', '')}\"></audio></td>"
            f"<td><img src=\"{row['Waveform']}\" alt=\"Waveform\"/></td>"
            f"<td><img src=\"{row['Log-Mel']}\" alt=\"Log-Mel spectrogram\"/></td>"
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
        <th>Audio</th>
        <th>Waveform</th>
        <th>Log-Mel</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    st.set_page_config(page_title="Speech Command Wheelchair Demo", layout="wide")
    st.title("Speech Command Wheelchair Demo")

    with st.sidebar:
        config_path = st.text_input("Config", value="configs/cnn_gru.yaml")
        config = load_config(config_path)
        checkpoint_path = st.text_input("Checkpoint", value=config["training"]["checkpoint_path"])
        configured_threshold = float(config.get("inference", {}).get("threshold", 0.70))
        default_threshold = configured_threshold if configured_threshold > 0.0 else 0.70
        threshold = st.slider("Confidence threshold", 0.0, 1.0, default_threshold, 0.01)
        record_seconds = st.number_input("Record seconds", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        reset_clicked = st.button("Reset simulator")

    simulator = get_robot_simulator()
    prediction_results = get_prediction_results()
    record_clicked = st.button("Record command", type="primary")

    if reset_clicked:
        simulator.reset()
        prediction_results.clear()

    result = None
    applied_event = None
    safety_decision = None
    waveform = None
    logmel = None
    audio_bytes = None
    data_cfg = config["data"]

    if record_clicked:
        if Path(checkpoint_path).exists():
            sample_rate = data_cfg["sample_rate"]
            try:
                with st.spinner("Recording..."):
                    waveform = record_microphone(sample_rate, float(record_seconds))
                audio_bytes = waveform_to_wav_bytes(waveform, sample_rate)
                feature_extractor = build_logmel_extractor(config)
                logmel = feature_extractor(waveform)

                predictor = load_predictor(checkpoint_path, config_path, device)
                result = predictor.predict_waveform(
                    waveform,
                    sample_rate=sample_rate,
                    threshold=0.0,
                )
                safety_decision = SafetyDecisionLayer(
                    confidence_threshold=threshold,
                    unknown_label=str(predictor.unknown_label),
                ).decide(
                    raw_label=str(result["raw_label"]),
                    confidence=float(result["confidence"]),
                )
                applied_event = simulator.apply_decision(safety_decision)
                waveform_image = figure_to_data_uri(plot_waveform(waveform, sample_rate))
                logmel_image = figure_to_data_uri(plot_logmel(logmel))
                audio_label = f"Recording {applied_event['step']}"
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
                        "Audio": audio_label,
                        "AudioData": audio_to_data_uri(audio_bytes),
                        "Waveform": waveform_image,
                        "Log-Mel": logmel_image,
                    }
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
            st.metric("Wheelchair action", "WAITING")
        else:
            display_label = safety_decision.label if safety_decision is not None else str(result["label"])
            display_action = safety_decision.action if safety_decision is not None else str(result["action"])
            st.metric("Predicted command", str(display_label))
            st.metric("Raw command", str(result["raw_label"]))
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Wheelchair action", str(display_action))
            if applied_event is not None and applied_event["blocked"]:
                st.warning("Movement blocked by map boundary.")
            elif applied_event is not None and applied_event["status"] == "rejected":
                st.warning("Command rejected because confidence is below the safety threshold.")
            elif applied_event is not None and applied_event["status"] == "ignored":
                st.info("Command ignored by the safety layer.")

        history_rows = simulator.history_rows()
        if history_rows:
            st.dataframe(history_rows, use_container_width=True, hide_index=True)

    if waveform is not None and logmel is not None:
        if audio_bytes is not None:
            st.subheader("Recorded audio")
            st.audio(audio_bytes, format="audio/wav")
        left_col, right_col = st.columns(2)
        with left_col:
            st.pyplot(plot_waveform(waveform, data_cfg["sample_rate"]), clear_figure=True)
        with right_col:
            st.pyplot(plot_logmel(logmel), clear_figure=True)

    if prediction_results:
        st.subheader("Export results")
        st.dataframe(
            [{key: value for key, value in row.items() if key != "AudioData"} for row in prediction_results],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Waveform": st.column_config.ImageColumn("Waveform"),
                "Log-Mel": st.column_config.ImageColumn("Log-Mel"),
            },
        )
        with st.expander("Recorded audio history"):
            for row in prediction_results:
                st.caption(
                    f"Step {row['Step']} - {row['Command']} "
                    f"({row['Confidence']}) - {row['Status']}"
                )
                audio_data = str(row.get("AudioData", ""))
                if audio_data.startswith("data:audio/wav;base64,"):
                    st.audio(base64.b64decode(audio_data.split(",", 1)[1]), format="audio/wav")
        st.download_button(
            "Download HTML report",
            data=build_export_html(prediction_results),
            file_name="speech_command_results.html",
            mime="text/html",
        )


if __name__ == "__main__":
    main()
