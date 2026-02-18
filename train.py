#!/usr/bin/env python3
"""
Train MobileViT variants on CIFAR-10.

Usage:
    python train.py --model xxs --epochs 200 --lr 0.002
    python train.py --model s  --epochs 200 --lr 0.001 --amp

Hyperparameter defaults follow the MobileViT paper where applicable
(cosine LR, label smoothing 0.1, AdamW), adapted for CIFAR-10's
32×32 resolution.  We resize to 256×256 to match the paper's setup.
"""

import time
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from mobilevit import MobileViT


# ------------------------------------------------------------------ #
#  Reproducibility                                                     #
# ------------------------------------------------------------------ #

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
#  Data                                                                #
# ------------------------------------------------------------------ #

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_loaders(data_dir: str, img_size: int, batch_size: int, workers: int):
    """CIFAR-10 train/test with standard augmentation + resize."""

    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomCrop(img_size, padding=img_size // 8),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, test_loader


# ------------------------------------------------------------------ #
#  Cosine schedule with linear warmup                                  #
# ------------------------------------------------------------------ #

class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6,
                 last_epoch=-1):
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup:
            alpha = epoch / max(1, self.warmup)
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            alpha = 0.5 * (1 + np.cos(np.pi * progress))
        return [self.min_lr + (base - self.min_lr) * alpha
                for base in self.base_lrs]

    # override to avoid "Detected call of lr_scheduler.step()
    # before optimizer.step()" warnings in some PyTorch versions
    def _get_closed_form_lr(self):
        return self.get_lr()


# ------------------------------------------------------------------ #
#  Train / eval loops                                                  #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, 100.0 * correct / total


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # --- data --------------------------------------------------------
    train_loader, test_loader = get_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers)
    print(f"[data] CIFAR-10  train={len(train_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  img_size={args.img_size}")

    # --- model -------------------------------------------------------
    model = MobileViT(args.model, num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] MobileViT-{args.model.upper()}  params={n_params:.2f}M")

    # --- optimiser + scheduler ---------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineWarmupScheduler(
        optimizer, args.warmup, args.epochs, min_lr=args.min_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # --- output dir --------------------------------------------------
    run_dir = Path(args.output) / f"mobilevit_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- training loop -----------------------------------------------
    best_acc = 0.0
    best_epoch = 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()

        improved = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "config": args.model,
            }, run_dir / "best.pth")
            improved = " *"

        # ensure all values are plain floats for JSON serialization
        history.append({
            "epoch": epoch,
            "lr": float(lr_now),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        })

        print(f"[{epoch:03d}/{args.epochs}]  lr={lr_now:.2e}  "
              f"train {train_loss:.4f} / {train_acc:.2f}%  "
              f"test {test_loss:.4f} / {test_acc:.2f}%{improved}")

        # early stopping on long plateaus
        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    elapsed = time.perf_counter() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)

    # --- save summary ------------------------------------------------
    summary = {
        "model": f"MobileViT-{args.model.upper()}",
        "params_M": round(n_params, 3),
        "best_test_acc": round(best_acc, 2),
        "total_epochs": len(history),
        "wall_time": f"{h}h {m}m {s}s",
        "args": vars(args),
        "history": history,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Best test accuracy : {best_acc:.2f}%")
    print(f"  Training time      : {h}h {m}m {s}s")
    print(f"  Checkpoint saved   : {run_dir / 'best.pth'}")
    print(f"{'='*50}")

    return best_acc


def parse_args():
    p = argparse.ArgumentParser(
        description="Train MobileViT on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model", choices=["xxs", "xs", "s"], default="xxs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=0.01,
                   help="AdamW weight decay")
    p.add_argument("--warmup", type=int, default=10,
                   help="linear warmup epochs")
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--img-size", type=int, default=256,
                   help="input image resolution")
    p.add_argument("--amp", action="store_true",
                   help="use mixed-precision training")
    p.add_argument("--patience", type=int, default=30,
                   help="early stopping patience (0=off)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="./checkpoints")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
