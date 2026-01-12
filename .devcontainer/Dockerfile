# 基于 NVIDIA CUDA 镜像，支持 GPU 计算
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置时区
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装基础工具和 Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 为默认版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 安装 DALI 和相关依赖
RUN pip install \
    nvidia-dali-cuda120 \
    jupyter \
    jupyter-lab \
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
    anthropic

# 创建非 root 用户
RUN useradd -m -s /bin/bash vscode

# 设置工作目录
WORKDIR /workspace

# 切换到 vscode 用户
USER vscode

# 默认命令
CMD ["/bin/bash"]
