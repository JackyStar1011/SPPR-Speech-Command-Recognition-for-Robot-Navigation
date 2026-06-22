from __future__ import annotations

from pathlib import Path

import torch

from src.data.preprocess import get_speech_alignment_config, load_waveform, preprocess_waveform
from src.features.logmel import build_logmel_extractor
from src.models.cnn_gru import build_model
from src.robot.actions import label_to_action
from src.utils.config import load_config
from src.utils.seed import resolve_device


class SpeechCommandPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/cnn_gru.yaml",
        device_name: str = "auto",
    ) -> None:
        self.config = load_config(config_path)
        self.device = resolve_device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_names = checkpoint.get(
            "classes",
            self.config["data"]["commands"] + [self.config["data"]["unknown_label"]],
        )
        self.unknown_label = self.config["data"]["unknown_label"]
        self.align_speech, self.speech_alignment = get_speech_alignment_config(
            self.config,
            inference=True,
        )
        self.model = build_model(self.config, num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feature_extractor = build_logmel_extractor(self.config).to(self.device)
        self.feature_extractor.eval()

    @torch.no_grad()
    def predict_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        threshold: float | None = None,
    ) -> dict[str, float | str]:
        data_cfg = self.config["data"]
        threshold = (
            threshold
            if threshold is not None
            else self.config.get("inference", {}).get("threshold", 0.70)
        )
        waveform = preprocess_waveform(
            waveform,
            sample_rate=sample_rate,
            target_sample_rate=data_cfg["sample_rate"],
            target_num_samples=int(data_cfg["sample_rate"] * data_cfg["duration_seconds"]),
            align_speech=self.align_speech,
            speech_alignment=self.speech_alignment,
        ).to(self.device)

        features = self.feature_extractor(waveform)
        logits = self.model(features)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, index = probabilities.max(dim=0)

        raw_label = self.class_names[index.item()]
        output_label = raw_label if confidence.item() >= threshold else self.unknown_label
        return {
            "label": output_label,
            "raw_label": raw_label,
            "confidence": confidence.item(),
            "action": label_to_action(output_label),
        }

    def predict_file(self, file_path: str | Path, threshold: float | None = None) -> dict[str, float | str]:
        data_cfg = self.config["data"]
        waveform, sample_rate = load_waveform(str(file_path))
        return self.predict_waveform(waveform, sample_rate, threshold=threshold)
