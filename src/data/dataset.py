from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.preprocess import get_speech_alignment_config, load_waveform, preprocess_waveform

Split = Literal["training", "validation", "testing"]


class SpeechCommandsRobotDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split,
        commands: list[str],
        unknown_label: str = "unknown",
        unknown_ratio: float = 1.0,
        download: bool = True,
        sample_rate: int = 16000,
        duration_seconds: float = 1.0,
        seed: int = 42,
        align_speech: bool = False,
        speech_alignment: dict | None = None,
    ) -> None:
        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=download,
            subset=split,
        )
        self.commands = list(commands)
        self.unknown_label = unknown_label
        self.classes = self.commands + [self.unknown_label]
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_seconds)
        self.align_speech = align_speech
        self.speech_alignment = speech_alignment or {}
        self.samples = self._build_balanced_index(unknown_ratio, seed)

    def _label_for_index(self, index: int) -> str:
        return Path(self.base._walker[index]).parent.name

    def _path_for_index(self, index: int) -> Path:
        path = Path(self.base._walker[index])
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _build_balanced_index(self, unknown_ratio: float, seed: int) -> list[tuple[int, str]]:
        known_by_label = {label: [] for label in self.commands}
        unknown_indices: list[int] = []

        for index in range(len(self.base)):
            label = self._label_for_index(index)
            if label in known_by_label:
                known_by_label[label].append(index)
            else:
                unknown_indices.append(index)

        known_counts = [len(indices) for indices in known_by_label.values() if indices]
        if not known_counts:
            raise RuntimeError("No target command samples were found in Speech Commands.")

        rng = random.Random(seed)
        unknown_target = int(round((sum(known_counts) / len(known_counts)) * unknown_ratio))
        unknown_target = max(1, min(unknown_target, len(unknown_indices)))
        sampled_unknown = rng.sample(unknown_indices, unknown_target)

        samples: list[tuple[int, str]] = []
        for label, indices in known_by_label.items():
            samples.extend((index, label) for index in indices)
        samples.extend((index, self.unknown_label) for index in sampled_unknown)
        rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        base_index, label = self.samples[index]
        waveform, sample_rate = load_waveform(str(self._path_for_index(base_index)))
        waveform = preprocess_waveform(
            waveform,
            sample_rate=sample_rate,
            target_sample_rate=self.sample_rate,
            target_num_samples=self.num_samples,
            align_speech=self.align_speech,
            speech_alignment=self.speech_alignment,
        )
        return waveform, self.class_to_idx[label]


def create_dataset(config: dict, split: Split) -> SpeechCommandsRobotDataset:
    data_cfg = config["data"]
    align_speech, speech_alignment = get_speech_alignment_config(config)
    return SpeechCommandsRobotDataset(
        root=data_cfg["root"],
        split=split,
        commands=data_cfg["commands"],
        unknown_label=data_cfg["unknown_label"],
        unknown_ratio=data_cfg["unknown_ratio"],
        download=data_cfg["download"],
        sample_rate=data_cfg["sample_rate"],
        duration_seconds=data_cfg["duration_seconds"],
        seed=config["seed"],
        align_speech=align_speech,
        speech_alignment=speech_alignment,
    )


def create_datasets(config: dict) -> tuple[SpeechCommandsRobotDataset, SpeechCommandsRobotDataset, SpeechCommandsRobotDataset]:
    return (
        create_dataset(config, "training"),
        create_dataset(config, "validation"),
        create_dataset(config, "testing"),
    )
