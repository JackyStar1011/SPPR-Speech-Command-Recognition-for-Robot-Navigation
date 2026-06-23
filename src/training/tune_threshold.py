from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataset
from src.features.logmel import build_logmel_extractor
from src.models import build_model
from src.training.train import build_loader
from src.utils.config import load_config
from src.utils.metrics import (
    compute_classification_metrics,
    save_confusion_matrix,
    save_metrics,
)
from src.utils.seed import resolve_device, set_seed


@torch.no_grad()
def collect_probabilities(
    model: nn.Module,
    feature_extractor: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[int], torch.Tensor]:
    model.eval()
    feature_extractor.eval()
    y_true: list[int] = []
    probability_batches: list[torch.Tensor] = []

    for waveforms, labels in tqdm(data_loader, desc=description, leave=False):
        waveforms = waveforms.to(device)
        features = feature_extractor(waveforms)
        probabilities = torch.softmax(model(features), dim=1)

        y_true.extend(labels.tolist())
        probability_batches.append(probabilities.cpu())

    if not probability_batches:
        raise RuntimeError(f"No samples were available while {description.lower()}.")
    return y_true, torch.cat(probability_batches, dim=0)


def apply_confidence_threshold(
    probabilities: torch.Tensor,
    unknown_index: int,
    threshold: float,
) -> list[int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [num_samples, num_classes]")
    if not 0 <= unknown_index < probabilities.shape[1]:
        raise ValueError("unknown_index is outside the class dimension")

    confidence, predictions = probabilities.max(dim=1)
    predictions = predictions.clone()
    predictions[confidence < threshold] = unknown_index
    return predictions.tolist()


def make_thresholds(minimum: float, maximum: float, step: float) -> list[float]:
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= minimum <= maximum <= 1")
    if step <= 0.0:
        raise ValueError("threshold step must be positive")
    return [float(value) for value in np.arange(minimum, maximum + step / 2.0, step)]


def tune_threshold(
    y_true: list[int],
    probabilities: torch.Tensor,
    unknown_index: int,
    thresholds: list[float],
) -> tuple[float, list[dict[str, float]]]:
    if not thresholds:
        raise ValueError("at least one threshold is required")

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = apply_confidence_threshold(probabilities, unknown_index, threshold)
        curve.append(
            {
                "threshold": threshold,
                "accuracy": float(accuracy_score(y_true, predictions)),
                "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
                "unknown_recall": float(
                    recall_score(
                        y_true,
                        predictions,
                        labels=[unknown_index],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    best = max(curve, key=lambda row: (row["macro_f1"], row["unknown_recall"], row["threshold"]))
    return best["threshold"], curve


def select_safety_threshold(
    curve: list[dict[str, float]],
    minimum_unknown_recall: float,
) -> float:
    if not 0.0 <= minimum_unknown_recall <= 1.0:
        raise ValueError("minimum_unknown_recall must be between 0 and 1")
    eligible = [row for row in curve if row["unknown_recall"] >= minimum_unknown_recall]
    if not eligible:
        raise ValueError(
            f"No threshold reached unknown recall >= {minimum_unknown_recall:.2f} "
            "on the validation split."
        )
    best = max(eligible, key=lambda row: (row["macro_f1"], row["accuracy"], -row["threshold"]))
    return best["threshold"]


def load_model_and_features(config: dict, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get(
        "classes",
        config["data"]["commands"] + [config["data"]["unknown_label"]],
    )
    model = build_model(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    feature_extractor = build_logmel_extractor(config).to(device)
    return model, feature_extractor, class_names


def evaluate_split(
    config: dict,
    split: str,
    model: nn.Module,
    feature_extractor: nn.Module,
    device: torch.device,
) -> tuple[object, list[int], torch.Tensor]:
    dataset = create_dataset(config, split)  # type: ignore[arg-type]
    loader = build_loader(dataset, config, shuffle=False, device=device)
    y_true, probabilities = collect_probabilities(
        model,
        feature_extractor,
        loader,
        device,
        description=f"Scoring {split}",
    )
    return dataset, y_true, probabilities


def save_thresholded_results(
    config: dict,
    prefix: str,
    y_true: list[int],
    probabilities: torch.Tensor,
    class_names: list[str],
    unknown_index: int,
    threshold: float,
) -> dict:
    predictions = apply_confidence_threshold(probabilities, unknown_index, threshold)
    metrics, report_text = compute_classification_metrics(y_true, predictions, class_names)
    metrics["threshold"] = threshold
    save_metrics(config["outputs"]["metrics_dir"], metrics, report_text, prefix=prefix)
    save_confusion_matrix(
        y_true,
        predictions,
        class_names,
        Path(config["outputs"]["figures_dir"]) / f"{prefix}_confusion_matrix.png",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the inference confidence threshold on validation and evaluate it on test."
    )
    parser.add_argument("--config", default="configs/cnn_gru.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--min-threshold", type=float, default=0.00)
    parser.add_argument("--max-threshold", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--target-unknown-recall", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or config["training"]["checkpoint_path"]
    thresholds = make_thresholds(args.min_threshold, args.max_threshold, args.step)

    model, feature_extractor, class_names = load_model_and_features(config, checkpoint_path, device)
    unknown_label = config["data"]["unknown_label"]
    if unknown_label not in class_names:
        raise ValueError(f"Checkpoint classes do not contain the unknown label: {unknown_label}")
    unknown_index = class_names.index(unknown_label)

    validation_dataset, validation_true, validation_probabilities = evaluate_split(
        config, "validation", model, feature_extractor, device
    )
    if validation_dataset.classes != class_names:
        raise ValueError("Current config class order does not match the checkpoint class order.")

    best_threshold, curve = tune_threshold(
        validation_true,
        validation_probabilities,
        unknown_index,
        thresholds,
    )
    safety_threshold = select_safety_threshold(curve, args.target_unknown_recall)
    raw_validation = save_thresholded_results(
        config,
        "validation_argmax",
        validation_true,
        validation_probabilities,
        class_names,
        unknown_index,
        threshold=0.0,
    )
    tuned_validation = save_thresholded_results(
        config,
        "validation_thresholded",
        validation_true,
        validation_probabilities,
        class_names,
        unknown_index,
        threshold=best_threshold,
    )
    safety_validation = save_thresholded_results(
        config,
        "validation_safety_thresholded",
        validation_true,
        validation_probabilities,
        class_names,
        unknown_index,
        threshold=safety_threshold,
    )

    test_dataset, test_true, test_probabilities = evaluate_split(
        config, "testing", model, feature_extractor, device
    )
    if test_dataset.classes != class_names:
        raise ValueError("Current config class order does not match the checkpoint class order.")
    raw_test = save_thresholded_results(
        config,
        "test_argmax",
        test_true,
        test_probabilities,
        class_names,
        unknown_index,
        threshold=0.0,
    )
    tuned_test = save_thresholded_results(
        config,
        "test_thresholded",
        test_true,
        test_probabilities,
        class_names,
        unknown_index,
        threshold=best_threshold,
    )
    safety_test = save_thresholded_results(
        config,
        "test_safety_thresholded",
        test_true,
        test_probabilities,
        class_names,
        unknown_index,
        threshold=safety_threshold,
    )

    summary = {
        "selection_split": "validation",
        "objective": "macro_f1",
        "best_threshold": best_threshold,
        "safety_objective": {"minimum_unknown_recall": args.target_unknown_recall},
        "safety_threshold": safety_threshold,
        "validation": {
            "argmax": raw_validation,
            "thresholded": tuned_validation,
            "safety_thresholded": safety_validation,
        },
        "test": {
            "argmax": raw_test,
            "thresholded": tuned_test,
            "safety_thresholded": safety_test,
        },
        "curve": curve,
    }
    summary_path = Path(config["outputs"]["metrics_dir"]) / "threshold_tuning.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"best_threshold={best_threshold:.2f} selected_on=validation objective=macro_f1")
    print(
        f"validation_macro_f1={tuned_validation['macro_f1']:.4f} "
        f"validation_unknown_recall={tuned_validation['per_class'][unknown_label]['recall']:.4f}"
    )
    print(
        f"test_accuracy={tuned_test['accuracy']:.4f} "
        f"test_macro_f1={tuned_test['macro_f1']:.4f} "
        f"test_unknown_recall={tuned_test['per_class'][unknown_label]['recall']:.4f}"
    )
    print(
        f"safety_threshold={safety_threshold:.2f} "
        f"validation_unknown_recall={safety_validation['per_class'][unknown_label]['recall']:.4f}"
    )
    print(
        f"safety_test_accuracy={safety_test['accuracy']:.4f} "
        f"safety_test_macro_f1={safety_test['macro_f1']:.4f} "
        f"safety_test_unknown_recall={safety_test['per_class'][unknown_label]['recall']:.4f}"
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
