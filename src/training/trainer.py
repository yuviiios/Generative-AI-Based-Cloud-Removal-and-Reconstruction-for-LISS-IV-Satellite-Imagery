"""
Training loop for LISS-IV cloud removal U-Net.
"""

import torch
import torch.optim as optim
import numpy as np
import os
import yaml
import time
import csv
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from src.model.unet import build_model
from src.model.losses import build_loss
from src.training.dataset import build_dataloaders
from src.evaluation.metrics import compute_metrics


class Trainer:
    """Manages training, validation, and checkpointing."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = self._get_device()
        self.model_dir = Path(cfg["paths"]["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(cfg["paths"]["results_dir"]) / "metrics"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model     = build_model(cfg).to(self.device)
        self.criterion = build_loss(cfg).to(self.device)

        train_cfg = cfg["training"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.epochs  = train_cfg["epochs"]
        self.patience = train_cfg.get("patience", 15)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_psnr": [], "val_ssim": [], "val_mae": []
        }

    def _get_device(self) -> torch.device:
        setting = self.cfg.get("device", "auto")
        if setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(setting)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for cloudy, clear, mask in loader:
            cloudy, clear, mask = cloudy.to(self.device), clear.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(cloudy)
            loss = self.criterion(pred, clear, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_metrics = {"psnr": [], "ssim": [], "mae": []}
        for cloudy, clear, mask in loader:
            cloudy, clear, mask = cloudy.to(self.device), clear.to(self.device), mask.to(self.device)
            pred = self.model(cloudy)
            loss = self.criterion(pred, clear, mask)
            total_loss += loss.item()
            m = compute_metrics(pred.cpu().numpy(), clear.cpu().numpy())
            for k in all_metrics:
                all_metrics[k].append(m[k])
        avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        return total_loss / len(loader), avg_metrics

    def _save_checkpoint(self, epoch: int, val_loss: float):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "cfg": self.cfg,
        }
        path = self.model_dir / "best_model.pth"
        torch.save(ckpt, path)

    def _save_history(self):
        csv_path = self.results_dir / "training_history.csv"
        keys = list(self.history.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
            writer.writeheader()
            for i in range(len(self.history["train_loss"])):
                row = {"epoch": i + 1}
                for k in keys:
                    row[k] = round(self.history[k][i], 6)
                writer.writerow(row)
        print(f"[Trainer] History saved to {csv_path}")

    def train(self, data_dir: str):
        data_cfg = self.cfg["data"]
        train_loader, val_loader, _ = build_dataloaders(
            data_dir,
            train_split=data_cfg["train_split"],
            val_split=data_cfg["val_split"],
            batch_size=data_cfg.get("batch_size", self.cfg["training"]["batch_size"]),
            seed=self.cfg["project"]["seed"],
        )

        print(f"\n[Trainer] Device: {self.device}")
        print(f"[Trainer] Model parameters: {self.model.count_parameters():,}")
        print(f"[Trainer] Starting training for {self.epochs} epochs...\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_psnr"].append(val_metrics["psnr"])
            self.history["val_ssim"].append(val_metrics["ssim"])
            self.history["val_mae"].append(val_metrics["mae"])

            print(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f}dB | "
                f"SSIM: {val_metrics['ssim']:.4f} | "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                print(f"  [NEW_BEST] Model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[Trainer] Early stopping at epoch {epoch}")
                    break

        self._save_history()
        print(f"\n[Trainer] Training complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history


def run_training(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    data_dir = cfg["data"]["processed_dir"]
    trainer.train(data_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_training(args.config)
