# Use runtime image instead of devel to reduce size (4GB vs 10GB)
FROM m.daocloud.io/docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Add deadsnakes PPA for Python 3.11 and install base dependencies
# Clear proxy settings to avoid connection issues during build
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set NVIDIA library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set Python 3.11 as default version
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install DALI and minimal packages for MCP development
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    nvidia-dali-cuda120 \
    numpy \
    ipython

# Create non-root user
RUN useradd -m -s /bin/bash vscode && \
    mkdir -p /workspace && \
    chown -R vscode:vscode /workspace

WORKDIR /workspace
USER vscode

CMD ["/bin/bash"]
