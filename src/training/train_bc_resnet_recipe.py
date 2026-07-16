from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SpeechCommandsRobotDataset, create_datasets
from src.features.bc_resnet_recipe import BCResNetRecipePreprocessor, build_recipe_preprocessor
from src.models import build_model
from src.utils.config import load_config
from src.utils.metrics import compute_classification_metrics, save_confusion_matrix, save_metrics
from src.utils.seed import resolve_device, set_seed


def paper_learning_rate(
    iteration: int,
    total_iterations: int,
    warmup_iterations: int,
    initial_learning_rate: float,
    minimum_learning_rate: float = 0.0,
) -> float:
    """Per-iteration warm-up and cosine schedule from the reference trainer."""
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if not 0 <= warmup_iterations < total_iterations:
        raise ValueError("warmup_iterations must be in [0, total_iterations)")
    if not 1 <= iteration <= total_iterations:
        raise ValueError("iteration must be in [1, total_iterations]")
    if iteration < warmup_iterations:
        return initial_learning_rate * iteration / warmup_iterations
    denominator = total_iterations - warmup_iterations
    progress = (iteration - warmup_iterations) / denominator
    return minimum_learning_rate + 0.5 * (
        initial_learning_rate - minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def build_loader(
    dataset: SpeechCommandsRobotDataset,
    config: dict,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )


def find_background_noise_paths(
    dataset: SpeechCommandsRobotDataset,
    config: dict,
) -> list[Path]:
    configured = config.get("augmentation", {}).get("background_noise_dir")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    dataset_path = getattr(dataset.base, "_path", None)
    if dataset_path:
        candidates.append(Path(dataset_path) / "_background_noise_")

    for directory in candidates:
        paths = sorted(directory.glob("*.wav"))
        if paths:
            return paths
    checked = ", ".join(str(path) for path in candidates) or "no candidate directory"
    raise FileNotFoundError(f"No background-noise WAV files found; checked: {checked}")


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def train_one_epoch(
    model: nn.Module,
    preprocessor: BCResNetRecipePreprocessor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    iteration: int,
    total_iterations: int,
    warmup_iterations: int,
    initial_learning_rate: float,
    minimum_learning_rate: float,
) -> tuple[float, float, int, float]:
    model.train()
    preprocessor.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    learning_rate = 0.0

    for waveforms, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        iteration += 1
        learning_rate = paper_learning_rate(
            iteration,
            total_iterations,
            warmup_iterations,
            initial_learning_rate,
            minimum_learning_rate,
        )
        set_optimizer_learning_rate(optimizer, learning_rate)

        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = preprocessor(waveforms, augment=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        iteration,
        learning_rate,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    preprocessor: BCResNetRecipePreprocessor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    preprocessor.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(preprocessor(waveforms, augment=False))
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, y_true, y_pred


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    classes: list[str],
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    val_macro_f1: float,
    iteration: int,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "classes": classes,
            "epoch": epoch,
            "iteration": iteration,
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_macro_f1": float(val_macro_f1),
            "recipe": "qualcomm_bc_resnet_200_epoch_adapted_6class",
        },
        checkpoint_path,
    )


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC-ResNet with the official recipe.")
    parser.add_argument(
        "--config",
        default="configs/models/bc_resnet_1_5_paper_6classes.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    # Waveform augmentation belongs to the GPU preprocessor for this recipe,
    # so it must not also run inside dataset workers.
    dataset_config = copy.deepcopy(config)
    dataset_config["augmentation"] = {"enabled": False}
    train_dataset, val_dataset, test_dataset = create_datasets(dataset_config)
    train_loader = build_loader(train_dataset, config, shuffle=True, device=device)
    val_loader = build_loader(val_dataset, config, shuffle=False, device=device)
    test_loader = build_loader(test_dataset, config, shuffle=False, device=device)

    noise_paths = find_background_noise_paths(train_dataset, config)
    preprocessor = build_recipe_preprocessor(config, noise_paths).to(device)
    model = build_model(config, num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    training = config["training"]
    optimizer_config = training["optimizer"]
    if optimizer_config.get("type", "sgd").lower() != "sgd":
        raise ValueError("The official BC-ResNet recipe requires the SGD optimizer")
    initial_learning_rate = float(optimizer_config["learning_rate"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
        momentum=float(optimizer_config.get("momentum", 0.9)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-3)),
    )

    epochs = int(training["epochs"])
    scheduler = training["scheduler"]
    warmup_epochs = int(scheduler.get("warmup_epochs", 5))
    minimum_learning_rate = float(scheduler.get("minimum_learning_rate", 0.0))
    total_iterations = len(train_loader) * epochs
    warmup_iterations = len(train_loader) * warmup_epochs
    if warmup_iterations >= total_iterations:
        raise ValueError("warmup_epochs must be less than total epochs")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} trainable_parameters={parameter_count:,} "
        f"background_noise_files={len(noise_paths)}"
    )

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    checkpoint_path = str(training["checkpoint_path"])
    metrics_dir = Path(config["outputs"]["metrics_dir"])
    history_path = metrics_dir / "training_history.json"
    history: list[dict] = []
    iteration = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, iteration, learning_rate = train_one_epoch(
            model,
            preprocessor,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            iteration,
            total_iterations,
            warmup_iterations,
            initial_learning_rate,
            minimum_learning_rate,
        )
        val_loss, val_accuracy, y_true, y_pred = evaluate_model(
            model,
            preprocessor,
            val_loader,
            criterion,
            device,
        )
        val_metrics, _ = compute_classification_metrics(
            y_true,
            y_pred,
            train_dataset.classes,
        )
        val_macro_f1 = float(val_metrics["macro_f1"])
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
        }
        history.append(epoch_metrics)
        save_history(history_path, history)
        print(
            f"epoch={epoch:03d} lr={learning_rate:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        is_best = val_macro_f1 > best_val_macro_f1 or (
            val_macro_f1 == best_val_macro_f1 and val_loss < best_val_loss
        )
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                train_dataset.classes,
                epoch,
                val_loss,
                val_accuracy,
                val_macro_f1,
                iteration,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, y_true, y_pred = evaluate_model(
        model,
        preprocessor,
        test_loader,
        criterion,
        device,
    )
    test_metrics, report_text = compute_classification_metrics(
        y_true,
        y_pred,
        train_dataset.classes,
    )
    test_metrics["loss"] = test_loss
    test_metrics["accuracy"] = test_accuracy
    save_metrics(metrics_dir, test_metrics, report_text, prefix="test")
    save_confusion_matrix(
        y_true,
        y_pred,
        train_dataset.classes,
        Path(config["outputs"]["figures_dir"]) / "test_confusion_matrix.png",
    )
    print(
        f"best_checkpoint={checkpoint_path} epoch={checkpoint['epoch']} "
        f"test_accuracy={test_accuracy:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
