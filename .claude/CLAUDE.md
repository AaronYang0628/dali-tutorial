# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning environment for NVIDIA DALI (Data Loading Library), configured as a GPU-enabled Dev Container with Python 3.11, CUDA 12.1, and comprehensive data processing tools.

## Development Environment

### Container Setup
- **Base Image**: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
- **Python Version**: 3.11
- **CUDA Version**: 12.1
- **User**: vscode (non-root)
- **Working Directory**: /workspace (mapped to project root)

### Starting the Environment
```bash
# In VS Code
# Ctrl+Shift+P -> "Dev Containers: Reopen in Container"

# Using podman/docker CLI
podman build -f .devcontainer/Dockerfile -t dali-dev .
podman run -it --rm --gpus all -v $(pwd):/workspace dali-dev
```

### Rebuilding the Container
```bash
# In VS Code
# Ctrl+Shift+P -> "Dev Containers: Rebuild Container"

# CLI cleanup and rebuild
docker system prune -a
```

## Common Commands

### Running Jupyter Lab
```bash
jupyter lab --ip=0.0.0.0 --no-browser
```

### Running Python Scripts
```bash
python scripts/your_script.py
python your_script.py
```

### Checking GPU Access
```bash
nvidia-smi
python -c "import nvidia.dali as dali; print(dali.__version__)"
```

### Code Formatting and Linting
```bash
black scripts/ notebooks/
pylint scripts/
```

## Architecture

### Directory Structure
- `.devcontainer/` - Dev Container configuration
  - `Dockerfile` - Container image definition with CUDA, Python 3.11, DALI
  - `devcontainer.json` - VS Code container settings, extensions, GPU mounts
  - `post-create.sh` - Post-creation setup script (pip installs, directory creation)
- `notebooks/` - Jupyter notebooks for DALI tutorials and experiments
- `scripts/` - Python scripts for DALI examples and utilities
- `.claude/` - Claude Code configuration

### Key Dependencies
- **nvidia-dali-cuda120** - DALI library for CUDA 12.0
- **torch/torchvision/torchaudio** - PyTorch ecosystem
- **numpy/pandas** - Data manipulation
- **matplotlib** - Visualization
- **opencv-python** - Image processing
- **jupyterlab** - Interactive development

### GPU Configuration
The Dev Container is configured with:
- NVIDIA device access (/dev/nvidiactl, /dev/nvidia0, etc.)
- Host library mounts for NVIDIA drivers (read-only)
- Environment variables: CUDA_VISIBLE_DEVICES=0, NVIDIA_VISIBLE_DEVICES=all
- LD_LIBRARY_PATH configured for CUDA and host NVIDIA libraries

### Container Customization

**Python Version**: Edit `Dockerfile` line 10 (e.g., change `python3.11` to `python3.10`)

**CUDA Version**: Edit `Dockerfile` FROM line and corresponding DALI package:
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Then in post-create.sh or Dockerfile:
pip install nvidia-dali-cuda118
```

**Python Packages**: Add to `post-create.sh` pip install section or `Dockerfile` RUN pip install

**VS Code Extensions**: Add extension IDs to `devcontainer.json` extensions array

## Development Principles

### Core Principles
- Pragmatic over dogmatic
- Single responsibility per function/class
- Minimal comment style
- After modification/refactoring/generation codes: no code summaries, only minimal change overviews
- No summary.md or similar files in interactive mode
- Designing and Discussing before Implementing large tasks
- Show design without detail codes

### Code Restrictions
- Comments only in English
- Single function < 200 lines
- Loop nesting <= 3 levels
- Shell scripts must be invokable via absolute paths (no working directory dependencies)

### Testing
- Ask before running tests

### Commit Messages
- No info beyond changes (e.g., no author attribution)

### MCP
- serena: Priority use, unless alternative methods can significantly reduce token consumption

### Token Efficiency
- Use `mv` command instead of creating then removing old files, unless alternatives significantly reduce token consumption

## Shell Scripts Best Practices

All shell scripts in this repository should be written to:
- Use absolute paths or handle paths relative to script location
- Not depend on current working directory
- Be invokable from any location

## Notes

- SSH keys are mounted read-only at `/home/vscode/.ssh`
- Container uses vscode user (non-root) for security
- GPU support requires nvidia-docker/nvidia-container-toolkit on host
- Project uses Tsinghua PyPI mirror in post-create.sh for faster downloads in China
