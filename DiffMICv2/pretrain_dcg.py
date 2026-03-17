"""
DCG (Dual-granularity Conditional Guidance) Pretraining Script for HAM10000.

This script pretrains the DCG module with BottleneckKANHead guidance heads
before diffuser training. It uses 80/20 train-test split from GroundTruth.csv.

Features:
    - Checkpointing every 10 epochs
    - F1, AUC, Accuracy, Precision, Recall, Cohen Kappa metrics  
    - No progress bar — prints epoch + loss only
    - FP16-safe (no DataParallel)

Usage:
    python pretrain_dcg.py --config configs/ham10000.yml
    python pretrain_dcg.py --config configs/ham10000.yml --dry-run
    
References:
    - FastKAN (ZiyaoLi): https://github.com/ZiyaoLi/fast-kan
    - E-BayesSAM (MICCAI 2025): https://link.springer.com/chapter/10.1007/978-3-032-05185-1_13
    - DiffMIC (MICCAI 2023): https://arxiv.org/abs/2303.10610
    - EfficientSAM: https://arxiv.org/abs/2312.00863
    - FastKAN paper: https://arxiv.org/abs/2404.19756
"""

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict

from pretraining.dcg import DCG
from utils import get_dataset, get_optimizer, compute_isic_metrics
from utils import cast_label_to_one_hot_and_prototype


def parse_args():
    parser = argparse.ArgumentParser(description='DCG Pretraining for HAM10000')
    parser.add_argument('--config', type=str, default='configs/ham10000.yml',
                        help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of pretraining epochs (default: from config)')
    parser.add_argument('--ckpt-dir', type=str, default='pretraining/ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--ckpt-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just load config + create model, no training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def validate(model, val_loader, criterion, device, config):
    """Run validation and compute comprehensive metrics."""
    model.eval()
    val_loss = 0.0
    all_gts = []
    all_preds = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_labels = y_batch.to(device)
            y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_labels, config)
            y_one_hot = y_one_hot.to(device)

            y_fusion, y_global, y_local, _, _, _ = model(x_batch)

            loss_fusion = criterion(y_fusion, y_labels)
            loss_global = criterion(y_global, y_labels)
            loss_local = criterion(y_local, y_labels)
            loss = loss_fusion + 0.5 * loss_global + 0.5 * loss_local

            val_loss += loss.item() * x_batch.size(0)
            all_gts.append(y_one_hot)
            all_preds.append(y_fusion)

    val_loss /= len(val_loader.dataset)
    gt = torch.cat(all_gts)
    pred = torch.cat(all_preds)

    ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

    return {
        'loss': val_loss,
        'accuracy': ACC,
        'balanced_accuracy': BACC,
        'precision': Prec,
        'recall': Rec,
        'f1': F1,
        'auc': AUC_ovo,
        'kappa': kappa
    }


def train_one_epoch(model, train_loader, optimizer, criterion, device, config):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_labels = y_batch.to(device)

        optimizer.zero_grad()

        y_fusion, y_global, y_local, _, _, _ = model(x_batch)

        # Multi-task loss: fusion + global + local
        loss_fusion = criterion(y_fusion, y_labels)
        loss_global = criterion(y_global, y_labels)
        loss_local = criterion(y_local, y_labels)
        loss = loss_fusion + 0.5 * loss_global + 0.5 * loss_local

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Set seed
    set_seed(args.seed)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create model
    model = DCG(config)
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count KAN head parameters specifically
    kan_params = 0
    for name, p in model.named_parameters():
        if 'kan' in name.lower() or 'bottleneck' in name.lower():
            kan_params += p.numel()
    print(f"BottleneckKAN head parameters: {kan_params:,}")

    if args.dry_run:
        print("\n[DRY RUN] Model created successfully. Exiting.")
        print(model)
        return

    # Determine epochs
    n_epochs = args.epochs if args.epochs is not None else config.aux_cls.n_pretrain_epochs
    print(f"\nPretraining DCG for {n_epochs} epochs on {config.data.dataset}")

    # Get dataset
    _, train_dataset, test_dataset = get_dataset(config)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Optimizer + scheduler
    optimizer = get_optimizer(config.aux_optim, filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=config.aux_optim.lr * 0.01
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Checkpointing setup
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_name = f"{config.data.dataset.lower()}_aux_model"

    # Training loop
    best_f1 = 0.0
    print(f"\n{'='*70}")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'F1':<8} {'AUC':<8} {'Acc':<8} {'Kappa':<8}")
    print(f"{'='*70}")

    for epoch in range(1, n_epochs + 1):
        start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start

        # Print metrics
        print(f"[{epoch:>3}/{n_epochs}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  "
              f"AUC={val_metrics['auc']:.4f}  "
              f"Acc={val_metrics['accuracy']:.4f}  "
              f"Kappa={val_metrics['kappa']:.4f}  "
              f"({elapsed:.1f}s)")

        # Checkpoint every N epochs
        if epoch % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"{ckpt_name}_epoch{epoch}.pth")
            torch.save([model.state_dict(), optimizer.state_dict(), epoch], ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = os.path.join(args.ckpt_dir, f"{ckpt_name}.pth")
            torch.save([model.state_dict(), optimizer.state_dict(), epoch], best_path)
            print(f"  -> Best model saved (F1={best_f1:.4f}): {best_path}")

    print(f"\n{'='*70}")
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Best checkpoint: {os.path.join(args.ckpt_dir, f'{ckpt_name}.pth')}")


if __name__ == '__main__':
    main()
