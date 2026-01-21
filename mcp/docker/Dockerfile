# DALI MCP Server Dockerfile
# Build a containerized DALI MCP Server with CUDA support

FROM m.daocloud.io/docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY scripts/dali_mcp_server_http.py .

# Create directory for temporary files
RUN mkdir -p /tmp/dali_datasets

# Set the entrypoint
ENTRYPOINT ["python", "dali_mcp_server_http.py"]

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import nvidia.dali as dali; import mcp; print('OK')" || exit 1

# Metadata
LABEL maintainer="DALI MCP Server" \
      description="Model Context Protocol Server for NVIDIA DALI" \
      version="0.2.0"
