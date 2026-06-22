from __future__ import annotations

import argparse

from src.inference.predictor import SpeechCommandPredictor
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a robot navigation command from a WAV file.")
    parser.add_argument("--file", required=True, help="Path to a WAV file.")
    parser.add_argument("--config", default="configs/cnn_gru.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override the threshold from the config.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or config["training"]["checkpoint_path"]
    predictor = SpeechCommandPredictor(
        checkpoint_path=checkpoint,
        config_path=args.config,
        device_name=args.device,
    )
    result = predictor.predict_file(args.file, threshold=args.threshold)

    print(f"predicted_command={result['label']}")
    print(f"raw_command={result['raw_label']}")
    print(f"confidence={result['confidence']:.4f}")
    print(f"robot_action={result['action']}")


if __name__ == "__main__":
    main()
