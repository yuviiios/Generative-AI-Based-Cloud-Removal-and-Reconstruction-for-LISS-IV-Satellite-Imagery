from setuptools import setup, find_packages

setup(
    name="liss4_cloud_removal",
    version="1.0.0",
    description="Generative AI-based Cloud Removal for LISS-IV Satellite Imagery",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "scikit-image>=0.20.0",
        "albumentations>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "streamlit>=1.22.0",
        "Pillow>=9.5.0",
        "pandas>=2.0.0",
    ],
)
