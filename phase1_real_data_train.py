#!/usr/bin/env python
"""
Phase 1: Real LISS-IV Data Training
Trains U-Net on actual cloud/cloud-free image pairs from data/raw/

Usage:
  python phase1_real_data_train.py --epochs 30 --batch_size 16
"""

import argparse
import yaml
from pathlib import Path
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--data_dir", default="data/processed_real", help="Processed data directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override from CLI
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["batch_size"] = args.batch_size

    # Verify data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Run: python -m src.preprocessing.real_data_loader")
        return

    clear_count = len(list((data_path / "clear").glob("*.npy")))
    print(f"\n{'='*60}")
    print(f"Phase 1 Training: Real LISS-IV Data")
    print(f"{'='*60}")
    print(f"[OK] Data directory: {args.data_dir}")
    print(f"[OK] Training pairs: {clear_count}")
    print(f"[OK] Epochs: {args.epochs}")
    print(f"[OK] Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    # Train
    trainer = Trainer(cfg)
    history = trainer.train(args.data_dir)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"  Final PSNR: {history['val_psnr'][-1]:.2f} dB")
    print(f"  Final SSIM: {history['val_ssim'][-1]:.4f}")
    print(f"  Final MAE: {history['val_mae'][-1]:.4f}")
    print(f"\nModel saved: models/best_model.pth")
    print(f"Metrics saved: results/metrics/training_history.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
