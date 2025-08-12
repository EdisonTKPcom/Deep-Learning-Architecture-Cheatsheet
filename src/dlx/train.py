import argparse
import time
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dlx.registry import get, available

# Import all modules to register them
import dlx.data.vision.cifar10
import dlx.models.vision.cnn_small

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(outputs, targets) * bs
        count += bs
    return {"loss": total_loss / count, "acc": total_acc / count}

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, desc="Val") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(outputs, targets) * bs
        count += bs
    return {"loss": total_loss / count, "acc": total_acc / count}

def main():
    parser = argparse.ArgumentParser(description="DLX: Train deep learning models with pluggable datasets/models")
    parser.add_argument("--dataset", type=str, default="cifar10", help=f"Dataset name. Available: {available('dataset')}")
    parser.add_argument("--model", type=str, default="cnn_small", help=f"Model name. Available: {available('model')}")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Instantiate dataset
    DatasetClass = get("dataset", args.dataset)
    dm = DatasetClass(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()
    train_loader, val_loader, test_loader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()

    # Instantiate model
    ModelClass = get("model", args.model)
    try:
        model = ModelClass(num_classes=dm.num_classes)
    except TypeError:
        model = ModelClass()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Val")

        best_val_acc = max(best_val_acc, val_metrics["acc"])

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
            f"Best Val acc {best_val_acc:.4f}"
        )

    test_metrics = evaluate(model, test_loader, criterion, device, desc="Test")
    elapsed = time.time() - start_time
    print(f"Test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f} | Elapsed {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()