"""
Streamlit Dashboard for LISS-IV Cloud Removal Prototype.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import torch
import yaml
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="LISS-IV Cloud Removal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛰️ LISS-IV Cloud Removal — Generative AI Framework")
st.markdown("**Prototype v1.0** | Deep Learning-Based Satellite Cloud Reconstruction")
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_mode = st.radio("Data Source", ["Real Data", "Synthetic"], index=0)
    cfg["data"]["mode"] = "real" if data_mode == "Real Data" else "synthetic"

    if data_mode == "Synthetic":
        n_demo_images = st.slider("Demo Samples to Generate", 10, 50, 20)
        cfg["data"]["n_synthetic_images"] = n_demo_images

    epochs = st.slider("Training Epochs", 3, 20, cfg["training"]["epochs"])
    cfg["training"]["epochs"] = epochs

    patch_size = st.selectbox("Patch Size", [128, 256], index=1)
    cfg["data"]["patch_size"] = patch_size

    st.markdown("---")
    st.markdown("### 📋 Project Phases")
    st.success("✅ Phase 1: Prototype (Now)")
    st.info("🔵 Phase 2: Real LISS-IV Data")
    st.info("🔵 Phase 3: Diffusion/Transformer")
    st.info("🔵 Phase 4: Production Deploy")


# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Data Simulation",
    "🏋️ Train Model",
    "📊 Results",
    "📖 About"
])


# ── Tab 1: Data Simulation ───────────────────────────────────────
with tab1:
    if data_mode == "Real Data":
        st.header("Real LISS-IV Dataset Info")
        st.info("✓ Dataset loaded from data/raw/ (2,447 training patches)")
        st.markdown("""
        **Dataset Overview:**
        - Clear images: 10 scenes → 3,771 patches
        - Cloudy images: 10 scenes → 2,447 training triplets
        - Patch size: 256×256
        - Cloud coverage: 5-85% per patch
        """)
        if st.button("📊 Show Sample Patch", type="primary"):
            from pathlib import Path
            import random
            patches = sorted(Path("data/processed_real/cloudy").glob("*.npy"))
            if patches:
                idx = random.randint(0, len(patches)-1)
                cloudy = np.load(patches[idx])
                clear = np.load(f"data/processed_real/clear/clear_{idx:04d}.npy")
                mask = np.load(f"data/processed_real/masks/mask_{idx:04d}.npy")
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(np.clip(cloudy[:, :, :3], 0, 1))
                axes[0].set_title(f"Cloudy Input", fontsize=12, fontweight="bold")
                axes[0].axis("off")
                axes[1].imshow(np.clip(clear[:, :, :3], 0, 1))
                axes[1].set_title("Reference Clear", fontsize=12, fontweight="bold")
                axes[1].axis("off")
                axes[2].imshow(mask, cmap="gray_r")
                axes[2].set_title(f"Cloud Mask ({mask.mean()*100:.1f}%)", fontsize=12, fontweight="bold")
                axes[2].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.header("Synthetic LISS-IV Data Generation")
        st.markdown("""
        Generates synthetic 3-band (Green, Red, NIR) satellite images simulating
        LISS-IV characteristics with realistic cloud contamination.
        """)

        col1, col2 = st.columns(2)
        with col1:
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)
            cloud_cov = st.slider("Target Cloud Coverage", 0.1, 0.8, 0.4, 0.05)

        if st.button("🎲 Generate Sample Pair", type="primary"):
            with st.spinner("Generating..."):
                from src.preprocessing.data_generator import generate_liss4_scene
                from src.preprocessing.cloud_simulator import CloudSimulator

                clear = generate_liss4_scene(size=patch_size, seed=int(seed))
                sim = CloudSimulator(
                    patch_size=patch_size
                )
                mask, shadow = sim.generate_cloud_mask(seed=int(seed))
                cloudy = sim.apply_clouds(clear, mask, shadow)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(np.clip(clear[:, :, :3], 0, 1))
                axes[0].set_title("Clear Image (Ground Truth)", fontsize=12, fontweight="bold")
                axes[0].axis("off")
                axes[1].imshow(np.clip(cloudy[:, :, :3], 0, 1))
                axes[1].set_title("Cloudy Image (Input)", fontsize=12, fontweight="bold")
                axes[1].axis("off")
                axes[2].imshow(mask, cmap="gray_r")
                axes[2].set_title(f"Cloud Mask ({mask.mean()*100:.1f}% coverage)", fontsize=12, fontweight="bold")
                axes[2].axis("off")
                plt.tight_layout()
                st.pyplot(fig)

                # Band stats
                st.subheader("Band Statistics")
                col_a, col_b, col_c = st.columns(3)
                bands = ["Green", "Red", "NIR"]
                for i, (col, band) in enumerate(zip([col_a, col_b, col_c], bands)):
                    with col:
                        st.metric(f"{band} Mean (Clear)", f"{clear[:,:,i].mean():.3f}")
                        st.metric(f"{band} Mean (Cloudy)", f"{cloudy[:,:,i].mean():.3f}")


# ── Tab 2: Train ─────────────────────────────────────────────────
with tab2:
    st.header("Train U-Net Cloud Removal Model")

    if data_mode == "Real Data":
        st.markdown(f"Training on real LISS-IV data for **{epochs}** epochs.")
        st.info("✓ Using 2,447 real training triplets from data/processed_real")
        data_dir_to_use = "data/processed_real"
    else:
        st.markdown(f"Generates synthetic data and trains the model end-to-end.")
        st.info(f"Will generate **{n_demo_images}** image pairs and train for **{epochs}** epochs.")
        data_dir_to_use = cfg["data"]["processed_dir"]

    if st.button("🚀 Run Full Training Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        # Step 1: Generate or use real data
        if data_mode != "Real Data":
            status.text("📦 Generating synthetic data...")
            progress_bar.progress(10)
            from src.preprocessing.data_generator import generate_dataset
            generate_dataset(
                data_dir_to_use,
                n_images=n_demo_images,
                patch_size=patch_size,
                seed=42
            )
            progress_bar.progress(30)
            status.text("✅ Data generated!")
        else:
            status.text("📦 Loading real data...")
            progress_bar.progress(20)
            status.text("✅ Real data ready!")

        # Step 2: Train
        status.text("🏋️ Training model...")
        from src.training.trainer import Trainer
        trainer = Trainer(cfg)
        history = trainer.train(data_dir_to_use)
        progress_bar.progress(85)

        # Step 3: Plot curves
        status.text("📈 Saving training curves...")
        from src.evaluation.visualizer import save_training_curves
        save_training_curves(history, "results/visualizations")
        progress_bar.progress(100)
        status.text("✅ Training complete!")

        # Show curves
        st.subheader("Training Curves")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="Train", color="#2196F3")
        axes[0].plot(epochs_range, history["val_loss"], label="Val", color="#F44336")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs_range, history["val_psnr"], color="#4CAF50")
        axes[1].set_title("PSNR (dB)"); axes[1].grid(alpha=0.3)
        axes[2].plot(epochs_range, history["val_ssim"], color="#FF9800")
        axes[2].set_title("SSIM"); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.success("Model saved to models/best_model.pth")


# ── Tab 3: Results ───────────────────────────────────────────────
with tab3:
    st.header("Inference Results & Metrics")

    ckpt_path = Path("models/best_model.pth")
    if not ckpt_path.exists():
        st.warning("⚠️ No trained model found. Run training first (Tab 2).")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.model.unet import build_model
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Generate a test pair
        seed_test = st.number_input("Test Seed", value=999, min_value=0)
        if st.button("🔍 Run Inference", type="primary"):
            from src.preprocessing.data_generator import generate_liss4_scene
            from src.preprocessing.cloud_simulator import CloudSimulator

            clear = generate_liss4_scene(size=patch_size, seed=int(seed_test))
            sim = CloudSimulator(patch_size=patch_size)
            mask, shadow = sim.generate_cloud_mask(seed=int(seed_test))
            cloudy = sim.apply_clouds(clear, mask, shadow)

            # Inference
            inp = torch.from_numpy(cloudy.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).cpu().numpy()[0].transpose(1, 2, 0)

            from src.evaluation.metrics import psnr, ssim, mae, sam

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("PSNR",  f"{psnr(pred.transpose(2,0,1), clear.transpose(2,0,1)):.2f} dB")
            col2.metric("SSIM",  f"{ssim(pred.transpose(2,0,1), clear.transpose(2,0,1)):.4f}")
            col3.metric("MAE",   f"{mae(pred, clear):.4f}")
            col4.metric("SAM",   f"{sam(pred.transpose(2,0,1), clear.transpose(2,0,1)):.2f}°")

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            imgs = [cloudy, pred, clear, np.stack([mask]*3, axis=-1)]
            titles = ["Cloudy Input", "Predicted", "Ground Truth", "Cloud Mask"]
            for ax, img, title in zip(axes, imgs, titles):
                ax.imshow(np.clip(img[:,:,:3] if img.shape[-1]>=3 else img, 0, 1))
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)


# ── Tab 4: About ─────────────────────────────────────────────────
with tab4:
    st.header("About This Project")
    st.markdown("""
    ### Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery

    **Problem**: Persistent cloud cover over North Eastern India reduces usability of
    LISS-IV optical imagery for land use mapping, disaster monitoring, and environmental assessment.

    **Solution**: U-Net-based generative framework to reconstruct cloud-free imagery
    while preserving spatial and spectral fidelity.

    ---
    ### Architecture (Prototype)
    - **Model**: U-Net with skip connections
    - **Loss**: L1 + SSIM combined loss
    - **Metrics**: PSNR, SSIM, MAE, SAM

    ### Roadmap
    | Phase | Status | Description |
    |-------|--------|-------------|
    | 1 | ✅ Now | Prototype with synthetic data |
    | 2 | 🔵 Next | Real LISS-IV + Sentinel fusion |
    | 3 | 🔵 Later | Diffusion / Transformer models |
    | 4 | 🔵 Final | Production deployment + API |

    ### Tech Stack
    `PyTorch` · `NumPy` · `OpenCV` · `scikit-image` · `Streamlit` · `Matplotlib`
    """)
