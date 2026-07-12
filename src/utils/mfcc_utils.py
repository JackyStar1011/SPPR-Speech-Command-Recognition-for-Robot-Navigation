from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ACTION_MAP = {
    "forward": "MOVE_FORWARD",
    "backward": "MOVE_BACKWARD",
    "left": "TURN_LEFT",
    "right": "TURN_RIGHT",
    "stop": "STOP",
    "unknown": "IGNORE",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def label_to_action(label: str) -> str:
    return ACTION_MAP.get(label, ACTION_MAP["unknown"])


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> tuple[dict[str, Any], str]:
    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"],
            }
            for name in class_names
        },
    }
    return metrics, report_text


def save_metrics(
    metrics_dir: str | Path,
    metrics: dict[str, Any],
    report_text: str,
    prefix: str = "test",
) -> None:
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    with (metrics_path / f"{prefix}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_name = "classification_report.txt" if prefix == "test" else f"{prefix}_classification_report.txt"
    with (metrics_path / report_name).open("w", encoding="utf-8") as file:
        file.write(report_text)


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="MFCC-CNN Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
