"""
Visualization tools for cloud removal results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
from typing import Optional


def save_comparison_grid(
    cloudy: np.ndarray,
    predicted: np.ndarray,
    clear: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    title: str = "Cloud Removal Results",
    n_samples: int = 4,
):
    """
    Save a side-by-side comparison grid.
    Inputs: (B, C, H, W) numpy arrays
    """
    B = min(n_samples, cloudy.shape[0])
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))

    if B == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Cloudy Input", "Predicted (Cloud-free)", "Ground Truth", "Cloud Mask"]

    for i in range(B):
        # Convert (C, H, W) → (H, W, C) for display
        c   = np.clip(cloudy[i].transpose(1, 2, 0)[:, :, :3], 0, 1)
        p   = np.clip(predicted[i].transpose(1, 2, 0)[:, :, :3], 0, 1)
        gt  = np.clip(clear[i].transpose(1, 2, 0)[:, :, :3], 0, 1)
        m   = mask[i].squeeze()

        axes[i, 0].imshow(c)
        axes[i, 1].imshow(p)
        axes[i, 2].imshow(gt)
        axes[i, 3].imshow(m, cmap="gray_r")

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12, fontweight="bold", pad=8)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved grid to {save_path}")


def save_training_curves(history: dict, save_dir: str):
    """Plot and save training/validation loss and metric curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_title("Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PSNR
    axes[1].plot(epochs, history["val_psnr"], color="#4CAF50")
    axes[1].set_title("Validation PSNR (dB)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    # SSIM
    axes[2].plot(epochs, history["val_ssim"], color="#FF9800")
    axes[2].set_title("Validation SSIM", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Training curves saved to {out}")


def run_inference_and_visualize(
    model,
    test_loader,
    device: torch.device,
    save_dir: str,
    n_samples: int = 4,
) -> dict:
    """Run inference on test loader and save visualizations."""
    from src.evaluation.metrics import evaluate_model

    model.eval()
    all_cloudy, all_pred, all_clear, all_mask = [], [], [], []

    with torch.no_grad():
        for cloudy, clear, mask in test_loader:
            cloudy_d = cloudy.to(device)
            pred = model(cloudy_d).cpu()
            all_cloudy.append(cloudy.numpy())
            all_pred.append(pred.numpy())
            all_clear.append(clear.numpy())
            all_mask.append(mask.numpy())
            if sum(a.shape[0] for a in all_cloudy) >= n_samples:
                break

    cloudy_arr  = np.concatenate(all_cloudy)[:n_samples]
    pred_arr    = np.concatenate(all_pred)[:n_samples]
    clear_arr   = np.concatenate(all_clear)[:n_samples]
    mask_arr    = np.concatenate(all_mask)[:n_samples]

    save_comparison_grid(
        cloudy_arr, pred_arr, clear_arr, mask_arr,
        save_path=str(Path(save_dir) / "comparison_grid.png"),
        n_samples=n_samples
    )

    metrics = evaluate_model(model, test_loader, device)
    return metrics
