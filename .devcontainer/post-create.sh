#!/bin/bash

# Clear proxy settings (should already be cleared by containerEnv, but double-check)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Install Python dependencies (MCP SDK and Anthropic)
pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    mcp \
    anthropic

# Install Claude CLI
npm install -g @anthropic-ai/claude-code

# Create working directories
mkdir -p /workspace/scripts

echo "✅ DALI environment with Claude CLI setup completed!"
