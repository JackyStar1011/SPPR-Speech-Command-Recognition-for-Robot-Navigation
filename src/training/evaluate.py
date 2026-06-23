from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataset
from src.features.logmel import build_logmel_extractor
from src.models.cnn_gru import build_model
from src.utils.config import load_config
from src.utils.metrics import (
    compute_classification_metrics,
    save_confusion_matrix,
    save_metrics,
)
from src.utils.seed import resolve_device, set_seed


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    feature_extractor: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    feature_extractor.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    iterator = tqdm(data_loader, desc="Evaluating", leave=False) if show_progress else data_loader
    for waveforms, labels in iterator:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        features = feature_extractor(waveforms)
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, y_true, y_pred


def evaluate_checkpoint(config: dict, checkpoint_path: str, split: str = "testing", prefix: str = "test") -> dict:
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    dataset = create_dataset(config, split)  # type: ignore[arg-type]
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("classes", dataset.classes)
    model = build_model(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    feature_extractor = build_logmel_extractor(config).to(device)
    criterion = nn.CrossEntropyLoss()

    loss, accuracy, y_true, y_pred = evaluate_model(model, feature_extractor, loader, criterion, device)
    metrics, report_text = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = loss
    metrics["accuracy"] = accuracy

    save_metrics(config["outputs"]["metrics_dir"], metrics, report_text, prefix=prefix)
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        Path(config["outputs"]["figures_dir"]) / f"{prefix}_confusion_matrix.png",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Speech Command CNN-GRU checkpoint.")
    parser.add_argument("--config", default="configs/cnn_gru.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="testing", choices=["training", "validation", "testing"])
    parser.add_argument("--prefix", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or config["training"]["checkpoint_path"]
    metrics = evaluate_checkpoint(config, checkpoint, split=args.split, prefix=args.prefix)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
