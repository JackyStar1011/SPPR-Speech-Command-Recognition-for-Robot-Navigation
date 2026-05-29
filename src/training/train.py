from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_datasets
from src.features.logmel import build_logmel_extractor
from src.models.cnn import build_model
from src.training.evaluate import evaluate_checkpoint, evaluate_model
from src.utils.config import load_config
from src.utils.seed import resolve_device, set_seed


def train_one_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    feature_extractor.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for waveforms, labels in tqdm(data_loader, desc=f"Epoch {epoch}", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = feature_extractor(waveforms)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def build_loader(dataset, config: dict, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    classes: list[str],
    epoch: int,
    val_loss: float,
    val_accuracy: float,
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
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline Speech Command CNN.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))

    train_dataset, val_dataset, _ = create_datasets(config)
    train_loader = build_loader(train_dataset, config, shuffle=True, device=device)
    val_loader = build_loader(val_dataset, config, shuffle=False, device=device)

    model = build_model(config, num_classes=len(train_dataset.classes)).to(device)
    feature_extractor = build_logmel_extractor(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    checkpoint_path = config["training"]["checkpoint_path"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            feature_extractor,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        val_loss, val_accuracy, _, _ = evaluate_model(
            model,
            feature_extractor,
            val_loader,
            criterion,
            device,
            show_progress=False,
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        is_best = val_accuracy > best_val_accuracy or (
            val_accuracy == best_val_accuracy and val_loss < best_val_loss
        )
        if is_best:
            best_val_accuracy = val_accuracy
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
            )

    print(f"best_checkpoint={checkpoint_path} val_acc={best_val_accuracy:.4f}")
    test_metrics = evaluate_checkpoint(config, checkpoint_path, split="testing", prefix="test")
    print(f"test_accuracy={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
