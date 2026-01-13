#!/bin/bash

# Clear proxy settings (should already be cleared by containerEnv, but double-check)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Install MCP SDK and minimal dependencies
pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    mcp \
    anthropic

curl -fsSL https://claude.ai/install.sh | bash

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Create working directories
mkdir -p /workspace/scripts

echo "✅ DALI  environment setup completed!"