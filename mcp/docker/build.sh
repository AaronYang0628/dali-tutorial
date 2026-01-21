#!/bin/bash
# Build script for DALI MCP Server Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="dali-mcp-server"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building DALI MCP Server Docker Image"
echo "=========================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Check if podman or docker is available
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither podman nor docker is installed"
    exit 1
fi

echo "Using container runtime: ${CONTAINER_CMD}"
echo ""

# Build the image
echo "Building image..."
${CONTAINER_CMD} build -f Dockerfile -t "${FULL_IMAGE_NAME}" ..

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo ""
echo "To run the server:"
echo "  ${CONTAINER_CMD} run --rm -it --gpus all ${FULL_IMAGE_NAME}"
echo ""
echo "Or use docker-compose:"
echo "  ${CONTAINER_CMD}-compose up"
echo ""
