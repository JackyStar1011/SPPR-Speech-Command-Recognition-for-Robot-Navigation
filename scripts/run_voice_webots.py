from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from pathlib import Path
from threading import Event


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import SpeechCommandPredictor  # noqa: E402
from src.robot.safety import SafetyDecisionLayer  # noqa: E402
from src.robot.webots_telemetry import WebotsTelemetryReceiver  # noqa: E402
from src.robot.webots_udp import WebotsUDPClient  # noqa: E402
from src.runtime import (  # noqa: E402
    CompositeEventHandler,
    RuntimeEvent,
    RuntimeStateStore,
    TelemetryMonitor,
    VoiceWebotsPipeline,
    load_runtime_config,
)
from src.utils.config import load_config  # noqa: E402
from src.wakeword import OpenWakeWordDetector, WakeTriggeredCommandCapture  # noqa: E402
from src.wakeword.audio_stream import MicrophoneFrameStream  # noqa: E402


class ConsoleEventPrinter:
    def __init__(self) -> None:
        self._last_wake_score_print = 0.0

    def __call__(self, event: RuntimeEvent) -> None:
        if event.kind == "wake_score":
            now = time.monotonic()
            detected = bool(event.data.get("detected"))
            if not detected and now - self._last_wake_score_print < 1.0:
                return
            self._last_wake_score_print = now
            print(
                "[WAKE] "
                f"score={float(event.data['score']):.3f} "
                f"threshold={float(event.data['threshold']):.3f} "
                f"detected={detected}",
                flush=True,
            )
            return

        if event.kind == "prediction":
            print(
                "[PREDICTION] "
                f"raw={event.data['raw_label']} "
                f"confidence={float(event.data['confidence']):.2%} "
                f"action={event.data['action']} "
                f"status={event.data['status']} "
                f"reason={event.data['reason']}",
                flush=True,
            )
            return

        if event.kind == "command_capture":
            print(
                "[CAPTURE] "
                f"captured={event.data['captured']} "
                f"duration={float(event.data['duration_seconds']):.2f}s "
                f"peak_rms={float(event.data['peak_rms']):.4f} "
                f"reason={event.data['reason']}",
                flush=True,
            )
            return

        if event.kind == "action_dispatched":
            print(f"[WEBOTS] sent={event.data['action']}", flush=True)
            return

        if event.kind in {
            "runtime_started",
            "wake_word_detected",
            "command_captured",
            "command_complete",
            "waiting_for_wake_word",
            "runtime_stopped",
            "runtime_error",
        }:
            message = event.data.get("message", event.kind)
            print(f"[STATE] {event.state.value}: {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the always-on wake-word speech pipeline for Webots wheelchair control."
    )
    parser.add_argument(
        "--config",
        default="configs/runtime/voice_webots.yaml",
        help="Path to the voice-Webots runtime YAML file.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override model checkpoint path.")
    parser.add_argument("--device", default=None, help="Override inference device.")
    parser.add_argument(
        "--wake-model",
        default=None,
        help="Override custom openWakeWord ONNX/TFLite model path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = load_runtime_config(args.config)
    model_config_path = Path(runtime_config.model.config_path)
    if not model_config_path.exists():
        raise FileNotFoundError(f"model config not found: {model_config_path}")

    model_config = load_config(model_config_path)
    sample_rate = int(model_config["data"]["sample_rate"])
    checkpoint_path = Path(
        args.checkpoint
        or runtime_config.model.checkpoint_path
        or model_config["training"]["checkpoint_path"]
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {checkpoint_path}")

    wake_model_path = args.wake_model or runtime_config.wakeword.model_path
    wakeword_models = None
    if wake_model_path:
        custom_model = Path(wake_model_path)
        if not custom_model.exists():
            raise FileNotFoundError(f"wake-word model not found: {custom_model}")
        wakeword_models = [str(custom_model)]

    detector = None
    if runtime_config.wakeword.enabled:
        detector = OpenWakeWordDetector(
            model_name=runtime_config.wakeword.model_name,
            threshold=runtime_config.wakeword.threshold,
            inference_framework=runtime_config.wakeword.inference_framework,
            wakeword_models=wakeword_models,
            auto_download=runtime_config.wakeword.auto_download,
        )

    predictor = SpeechCommandPredictor(
        checkpoint_path=str(checkpoint_path),
        config_path=str(model_config_path),
        device_name=args.device or runtime_config.model.device,
    )
    safety_config = copy.deepcopy(model_config)
    safety_settings = dict(safety_config.get("safety", {}))
    safety_settings["require_wake_word"] = runtime_config.wakeword.enabled
    safety_config["safety"] = safety_settings
    safety_layer = SafetyDecisionLayer.from_config(safety_config)
    command_capture = WakeTriggeredCommandCapture(
        sample_rate,
        speech_rms_threshold=runtime_config.capture.speech_rms_threshold,
        speech_start_timeout_seconds=runtime_config.capture.speech_start_timeout_seconds,
        silence_duration_seconds=runtime_config.capture.silence_duration_seconds,
        min_command_seconds=runtime_config.capture.min_command_seconds,
        max_command_seconds=runtime_config.capture.max_command_seconds,
    )
    microphone = MicrophoneFrameStream(sample_rate, runtime_config.wakeword.frame_ms)
    stop_event = Event()
    state_store = RuntimeStateStore(
        runtime_config.runtime.snapshot_path,
        runtime_config.runtime.event_log_path,
        trajectory_limit=runtime_config.runtime.trajectory_limit,
    )

    def request_stop(signum, frame) -> None:
        print("\n[SHUTDOWN] stop requested", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    event_handler = CompositeEventHandler(ConsoleEventPrinter(), state_store.handle_event)
    with WebotsTelemetryReceiver(
        host=runtime_config.webots.telemetry_host,
        port=runtime_config.webots.telemetry_port,
    ) as telemetry_receiver:
        with TelemetryMonitor(telemetry_receiver, state_store):
            with WebotsUDPClient(
                host=runtime_config.webots.command_host,
                port=runtime_config.webots.command_port,
                stop_on_close=runtime_config.webots.stop_on_exit,
            ) as webots_client:
                pipeline = VoiceWebotsPipeline(
                    detector=detector,
                    predictor=predictor,
                    safety_layer=safety_layer,
                    command_capture=command_capture,
                    webots_client=webots_client,
                    sample_rate=sample_rate,
                    prediction_threshold=runtime_config.model.prediction_threshold,
                    cooldown_seconds=runtime_config.runtime.cooldown_seconds,
                    wakeword_enabled=runtime_config.wakeword.enabled,
                    event_handler=event_handler,
                )
                try:
                    pipeline.run_frames(
                        microphone.frames(
                            timeout_seconds=runtime_config.runtime.frame_timeout_seconds
                        ),
                        stop_requested=stop_event.is_set,
                    )
                    if not stop_event.is_set():
                        raise RuntimeError(
                            "microphone stream ended unexpectedly; check the selected input device"
                        )
                except Exception as error:
                    pipeline.fail(error)
                    raise

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[FATAL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
