# setup.py
from setuptools import setup, find_packages

setup(
    name="act_test",
    version="0.1.0",
    packages=find_packages(),   # will pick up the act_test directory
    python_requires=">=3.10",  # Specify Python 3.10 minimum requirement
    install_requires=[
        # Core ML/AI packages
        "torch==2.6.0",
        "torchaudio==2.6.0", 
        "torchvision==0.21.0",
        "numpy==2.2.3",
        "einops==0.8.1",
        
        # Computer Vision
        "opencv-python==4.11.0.86",
        "pillow==11.1.0",
        
        # ML Libraries
        "huggingface-hub==0.31.2",
        "timm==1.0.15",
        "safetensors==0.5.3",
        
        # Data handling
        "h5py==3.13.0",
        "webdataset==0.2.111",
        "PyYAML==6.0.2",
        
        # Visualization
        "matplotlib==3.10.1",
        
        # Utilities
        "tqdm==4.67.1",
        "requests==2.32.3",
        "click==8.1.8",
        "pydantic==2.10.6",
        
        # Monitoring/Logging
        "wandb==0.19.7",
        
        # Development tools (optional - you might want to move these to dev dependencies)
        "ipython==8.32.0",
    ],
)
