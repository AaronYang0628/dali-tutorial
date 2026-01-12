#!/bin/bash

# Update package lists
apt-get update && apt-get upgrade -y

# Install system dependencies for DALI
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# Install Python development tools
python3 -m pip install --upgrade pip setuptools wheel

# Install Nvidia DALI (if not already installed)
# 注意：默认安装 CPU 版本，如果需要 GPU 支持，请使用: pip install nvidia-dali-cuda120
# Since it's already installed in Dockerfile, skip or update if needed
# pip install nvidia-dali-cuda120

# Install additional useful packages for learning
pip install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    jupyter \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    opencv-python \
    scikit-learn \
    torch \
    torchvision \
    torchaudio \
    black \
    pylint \
    ipython \
    nvidia-dali-cuda120

# Install Claude extension dependencies (if needed)
pip install anthropic

# Create notebooks directory
mkdir -p /workspaces/dali-tutorial/notebooks

echo "DALI development environment setup completed successfully!"
