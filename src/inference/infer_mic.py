from __future__ import annotations

import argparse

import sounddevice as sd
import torch

from src.inference.predictor import SpeechCommandPredictor
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record microphone audio and classify a robot command.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sample_rate = config["data"]["sample_rate"]
    checkpoint = args.checkpoint or config["training"]["checkpoint_path"]

    predictor = SpeechCommandPredictor(
        checkpoint_path=checkpoint,
        config_path=args.config,
        device_name=args.device,
    )

    print(f"Recording {args.seconds:.2f}s at {sample_rate} Hz...")
    audio = sd.rec(int(args.seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    waveform = torch.from_numpy(audio.T)
    result = predictor.predict_waveform(waveform, sample_rate=sample_rate, threshold=args.threshold)

    print(f"predicted_command={result['label']}")
    print(f"raw_command={result['raw_label']}")
    print(f"confidence={result['confidence']:.4f}")
    print(f"robot_action={result['action']}")


if __name__ == "__main__":
    main()
