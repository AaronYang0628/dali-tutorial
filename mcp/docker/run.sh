#!/bin/bash
# Run script for DALI MCP Server

set -e

IMAGE_NAME="dali-mcp-server:latest"

# Check if podman or docker is available
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither podman nor docker is installed"
    exit 1
fi

echo "=========================================="
echo "Running DALI MCP Server"
echo "=========================================="
echo "Container runtime: ${CONTAINER_CMD}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Check if image exists
if ! ${CONTAINER_CMD} image inspect "${IMAGE_NAME}" &> /dev/null; then
    echo "Error: Image ${IMAGE_NAME} not found"
    echo "Please build the image first using: ./build.sh"
    exit 1
fi

# Run the container
echo "Starting container..."
${CONTAINER_CMD} run \
    --rm \
    -it \
    --gpus all \
    --name dali-mcp-server \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$(pwd)/datasets:/data:ro" \
    "${IMAGE_NAME}"
