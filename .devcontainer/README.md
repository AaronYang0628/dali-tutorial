# DALI 学习环境设置指南

## 概述

本项目包含一个完整的 Dev Container 配置，用于学习和开发 Nvidia DALI（Data Loading Library）。

### 包含内容

- **CUDA 12.1 支持** - GPU 计算能力
- **Python 3.11** - 最新的 Python 版本
- **Nvidia DALI** - 数据加载库
- **Claude Code 插件** - AI 代码助手
- **Jupyter Lab** - 交互式笔记本环境
- **完整的 Python 开发工具** - Pylint, Black, etc.

## 前置要求

1. **VS Code** - 最新版本
2. **Docker** 或 **Docker Desktop** - 用于容器化环境
3. **GPU（可选）** - 如果使用 GPU 版本，需要 NVIDIA 显卡和 nvidia-docker

## 快速开始

### 方式 1：使用 VS Code Dev Containers（推荐）

1. 在 VS Code 中安装 **Remote - Containers** 扩展
2. 打开项目文件夹
3. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P`）打开命令面板
4. 输入 "Dev Containers: Reopen in Container"
5. 等待容器构建完成（首次会需要几分钟）

### 方式 2：使用命令行

```bash
# 构建镜像
podman build -f .devcontainer/Dockerfile -t dali-dev .

# 运行容器（CPU 版本）
podman run -it --rm -v $(pwd):/workspace dali-dev

# 运行容器（GPU 版本，需要 nvidia-docker）
podman run -it --rm --gpus all -v $(pwd):/workspace dali-dev
```

## 使用环境

### 启动 Jupyter Lab

在容器内运行：
```bash
jupyter lab --ip=0.0.0.0 --no-browser
```

然后在浏览器中访问显示的 URL。

### Python 脚本

创建 `.py` 文件并运行：
```bash
python your_script.py
```

### VS Code 特性

- **Python IntelliSense** - 智能代码补全
- **Debugging** - 调试支持
- **Claude Code** - AI 驱动的代码助手
- **Git 集成** - 版本控制

## 目录结构

```
.devcontainer/
├── devcontainer.json    # Dev Container 主配置文件
├── Dockerfile           # Docker 镜像定义
├── post-create.sh       # 容器创建后运行的初始化脚本
└── .dockerignore        # Docker 构建时忽略的文件

notebooks/               # Jupyter 笔记本存储目录（自动创建）
```

## 自定义配置

### 修改 Python 版本

编辑 `Dockerfile` 中的 `python3.11`，改为其他版本（如 `python3.10`）

### 安装额外的 Python 包

1. 编辑 `post-create.sh` 的 `pip install` 部分
2. 重新构建容器：`Dev Containers: Rebuild Container`

### 修改 VS Code 扩展

在 `devcontainer.json` 的 `extensions` 部分添加或移除扩展 ID

### 启用 GPU 支持

默认已启用。如果需要特定的 CUDA 版本，编辑 `Dockerfile` 的 FROM 行：

```dockerfile
# 例如，使用 CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

然后在 `post-create.sh` 中修改：
```bash
pip install nvidia-dali-cuda118
```

## 常见问题

### Q: 如何清除容器并重新开始？
A: 
```bash
# VS Code 方式
# 按 Ctrl+Shift+P，选择 "Dev Containers: Rebuild Container"

# 命令行方式
docker system prune -a
```

### Q: 容器无法访问 GPU？
A: 
1. 确认已安装 `nvidia-docker`
2. 检查 `nvidia-smi` 是否能识别 GPU
3. 确认 Docker daemon 已正确配置

### Q: 如何在本地和容器间共享文件？
A: 默认工作目录映射到 `/workspace`，所有文件自动同步。SSH 密钥也已挂载（只读）。

### Q: 如何使用 Claude Code 插件？
A: 
1. 插件会自动安装
2. 按 `Ctrl+K`（或 `Cmd+K`）打开 Claude 对话框
3. 开始提问或请求代码建议

## 学习资源

- [Nvidia DALI 官方文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [DALI 示例](https://github.com/NVIDIA/DALI)
- [Python 官方文档](https://docs.python.org/3/)

## 故障排除

如果遇到任何问题，请：

1. 检查 Docker 是否正确安装和运行
2. 尝试重建容器（`Dev Containers: Rebuild Container`）
3. 查看 Dev Container 日志（VS Code 输出面板）
4. 清理 Docker 资源（`docker system prune -a`）

---
