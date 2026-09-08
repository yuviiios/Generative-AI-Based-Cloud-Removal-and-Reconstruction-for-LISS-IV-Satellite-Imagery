"""
LISS-IV Cloud Removal — Single-command demo runner.
Usage: python run_demo.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    print("\n" + "=" * 65)
    print("  🛰️  LISS-IV Cloud Removal — Generative AI Framework")
    print("       Prototype v1.0  |  Step 1 Demo")
    print("=" * 65 + "\n")


def step(n: int, msg: str):
    print(f"\n{'─'*65}")
    print(f"  Step {n}: {msg}")
    print("─" * 65)


def main():
    print_banner()

    # ── Load config ──────────────────────────────────────────────
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    data_dir = cfg["data"]["processed_dir"]
    results_dir = Path(cfg["paths"]["results_dir"])
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate Synthetic Data ──────────────────────────
    step(1, "Generating Synthetic LISS-IV Imagery")
    from src.preprocessing.data_generator import generate_dataset
    n_imgs = cfg["data"]["n_synthetic_images"]
    generate_dataset(
        data_dir,
        n_images=n_imgs,
        patch_size=cfg["data"]["patch_size"],
        seed=cfg["project"]["seed"],
    )

    # ── Step 2: Train Model ───────────────────────────────────────
    step(2, "Training U-Net Cloud Removal Model")
    from src.training.trainer import Trainer
    trainer = Trainer(cfg)
    history = trainer.train(data_dir)

    # ── Step 3: Save Training Curves ──────────────────────────────
    step(3, "Saving Training Curves")
    from src.evaluation.visualizer import save_training_curves
    save_training_curves(history, str(viz_dir))

    # ── Step 4: Load Best Model & Run Inference ───────────────────
    step(4, "Running Inference on Test Set")
    from src.model.unet import build_model
    from src.training.dataset import build_dataloaders

    model = build_model(cfg).to(device)
    ckpt_path = Path(cfg["paths"]["model_dir"]) / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best model from epoch {ckpt['epoch']}")
    else:
        print("  Warning: No checkpoint found, using last model weights")

    _, _, test_loader = build_dataloaders(
        data_dir,
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["project"]["seed"],
    )

    # ── Step 5: Evaluate & Visualize ──────────────────────────────
    step(5, "Evaluating & Visualizing Results")
    from src.evaluation.visualizer import run_inference_and_visualize

    metrics = run_inference_and_visualize(
        model, test_loader, device,
        save_dir=str(viz_dir),
        n_samples=4
    )

    # ── Step 6: Print Summary ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊  EVALUATION RESULTS (Test Set)")
    print("=" * 65)
    print(f"  PSNR  : {metrics['psnr']:.2f} dB  (higher is better)")
    print(f"  SSIM  : {metrics['ssim']:.4f}    (higher is better, max=1)")
    print(f"  MAE   : {metrics['mae']:.4f}    (lower is better)")
    print(f"  SAM   : {metrics['sam']:.2f}°    (lower is better)")
    print("=" * 65)

    print("\n  📁  Output Files:")
    print(f"     • Comparison grid : results/visualizations/comparison_grid.png")
    print(f"     • Training curves : results/visualizations/training_curves.png")
    print(f"     • Training CSV    : results/metrics/training_history.csv")
    print(f"     • Best model      : models/best_model.pth")

    print("\n  🌐  Launch Dashboard:")
    print("     streamlit run app.py")
    print("\n" + "=" * 65)
    print("  ✅  Prototype Demo Complete!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
