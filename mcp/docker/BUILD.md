# DALI MCP Server - 构建和部署指南

本文档介绍如何构建和部署 DALI MCP Server 的 Docker 镜像。

## 目录

- [前置要求](#前置要求)
- [快速开始](#快速开始)
- [构建选项](#构建选项)
- [运行服务](#运行服务)
- [配置说明](#配置说明)
- [故障排查](#故障排查)

## 前置要求

### 系统要求

- Linux 操作系统（推荐 Ubuntu 22.04+）
- NVIDIA GPU 及驱动（用于 CUDA 支持）
- 容器运行时（选择其一）：
  - Docker 20.10+
  - Podman 3.0+

### 安装容器运行时

**Docker:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 nvidia-container-toolkit (GPU 支持)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Podman:**
```bash
# Ubuntu/Debian
sudo apt-get install -y podman

# Fedora/RHEL
sudo dnf install -y podman
```

### 验证环境

```bash
# 检查 GPU
nvidia-smi

# 检查容器运行时
docker --version
# 或
podman --version
```

## 快速开始

### 方式 1: 使用构建脚本（推荐）

```bash
cd /workspaces/dali-tutorial/mcp

# 构建镜像
./build.sh

# 运行服务
./run.sh
```

### 方式 2: 使用 docker-compose

```bash
cd /workspaces/dali-tutorial/mcp

# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d

# 停止服务
docker-compose down
```

### 方式 3: 手动构建和运行

```bash
cd /workspaces/dali-tutorial/mcp

# 构建镜像
docker build -t dali-mcp-server:latest .

# 运行容器
docker run --rm -it --gpus all dali-mcp-server:latest
```

## 构建选项

### 自定义镜像标签

```bash
# 构建指定版本
./build.sh v0.2.0

# 或使用 docker 命令
docker build -t dali-mcp-server:v0.2.0 .
```

### 使用国内镜像源

如果在中国大陆，基础镜像已配置使用 DaoCloud 镜像源：

```dockerfile
FROM m.daocloud.io/docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

### 自定义 Python 版本

编辑 `Dockerfile`，修改 Python 版本：

```dockerfile
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    ...
```

### 自定义 CUDA 版本

编辑 `Dockerfile` 和 `requirements.txt`：

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

**requirements.txt:**
```
nvidia-dali-cuda118>=1.50.0
```

## 运行服务

### 基本运行

```bash
docker run --rm -it --gpus all dali-mcp-server:latest
```

### 挂载本地数据集

```bash
docker run --rm -it \
    --gpus all \
    -v /path/to/local/data:/data:ro \
    dali-mcp-server:latest
```

### 配置 AWS S3 凭证

```bash
docker run --rm -it \
    --gpus all \
    -e AWS_ACCESS_KEY_ID="your_access_key" \
    -e AWS_SECRET_ACCESS_KEY="your_secret_key" \
    dali-mcp-server:latest
```

### 持久化临时文件

```bash
docker run --rm -it \
    --gpus all \
    -v dali-temp:/tmp/dali_datasets \
    dali-mcp-server:latest
```

### 后台运行

```bash
docker run -d \
    --gpus all \
    --name dali-mcp-server \
    --restart unless-stopped \
    dali-mcp-server:latest
```

### 查看日志

```bash
# 实时查看日志
docker logs -f dali-mcp-server

# 查看最近 100 行
docker logs --tail 100 dali-mcp-server
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见的 GPU 设备 ID | `0` |
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA 设备可见性 | `all` |
| `AWS_ACCESS_KEY_ID` | AWS S3 访问密钥 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS S3 密钥 | - |
| `PYTHONUNBUFFERED` | Python 输出不缓冲 | `1` |

### 卷挂载

| 容器路径 | 说明 | 推荐挂载 |
|----------|------|----------|
| `/data` | 本地数据集目录 | `-v /host/data:/data:ro` |
| `/tmp/dali_datasets` | 临时文件目录 | `-v dali-temp:/tmp/dali_datasets` |

### GPU 资源限制

```bash
# 使用特定 GPU
docker run --gpus '"device=0,1"' dali-mcp-server:latest

# 限制 GPU 内存
docker run --gpus 'all,capabilities=compute' \
    --memory=8g \
    dali-mcp-server:latest
```

## 故障排查

### 问题 1: GPU 不可用

**错误信息:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**解决方案:**
```bash
# 检查 nvidia-container-toolkit 是否安装
dpkg -l | grep nvidia-container-toolkit

# 重新安装
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 问题 2: 镜像拉取失败

**错误信息:**
```
Error response from daemon: Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout
```

**解决方案:**
- 使用国内镜像源（Dockerfile 已配置）
- 或配置 Docker 镜像加速器

### 问题 3: Python 依赖安装失败

**错误信息:**
```
ERROR: Could not find a version that satisfies the requirement nvidia-dali-cuda120
```

**解决方案:**
```bash
# 检查 CUDA 版本与 DALI 包是否匹配
# 编辑 requirements.txt 使用对应版本
nvidia-dali-cuda118>=1.50.0  # 对于 CUDA 11.8
nvidia-dali-cuda120>=1.50.0  # 对于 CUDA 12.0
```

### 问题 4: 服务启动后无响应

**检查步骤:**

1. **查看容器日志:**
```bash
docker logs dali-mcp-server
```

2. **检查容器状态:**
```bash
docker ps -a
```

3. **进入容器调试:**
```bash
docker exec -it dali-mcp-server bash
python -c "import nvidia.dali; print(dali.__version__)"
```

### 问题 5: 权限问题

**错误信息:**
```
PermissionError: [Errno 13] Permission denied: '/tmp/dali_datasets'
```

**解决方案:**
```bash
# 修改挂载卷的权限
sudo chown -R 1000:1000 /path/to/volume

# 或在 Dockerfile 中设置正确的用户权限
```

## 测试服务

### 方式 1: 使用示例客户端

```bash
# 在宿主机上
cd /workspaces/dali-tutorial/mcp
python example_client.py
```

### 方式 2: 使用 Claude Desktop

参考 `README.md` 中的 Claude Desktop 集成部分。

### 方式 3: 手动测试

```bash
# 启动服务
docker run --rm -it --gpus all dali-mcp-server:latest

# 在另一个终端，发送 MCP 请求
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | \
    docker exec -i dali-mcp-server python dali_mcp_server.py
```

## 生产部署建议

### 1. 使用固定版本标签

```bash
docker build -t dali-mcp-server:v0.2.0 .
```

### 2. 配置健康检查

在 `docker-compose.yml` 中已配置：

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import nvidia.dali as dali; import mcp; print('OK')"]
  interval: 30s
  timeout: 10s
  start_period: 5s
  retries: 3
```

### 3. 资源限制

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 4. 日志管理

```bash
# 配置日志驱动
docker run \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    dali-mcp-server:latest
```

### 5. 安全加固

- 使用非 root 用户运行容器
- 限制容器权限（`--cap-drop=ALL`）
- 使用只读文件系统（`--read-only`）
- 扫描镜像漏洞（`docker scan`）

## 性能优化

### 1. 多阶段构建

当前 Dockerfile 使用 runtime 镜像，已优化大小。如需进一步优化：

```dockerfile
# 构建阶段
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as builder
# 安装依赖...

# 运行阶段
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
```

### 2. 缓存优化

构建时使用 BuildKit：

```bash
DOCKER_BUILDKIT=1 docker build -t dali-mcp-server:latest .
```

### 3. 镜像大小优化

```bash
# 查看镜像层
docker history dali-mcp-server:latest

# 清理不必要的文件
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
```

## 更多资源

- [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU 支持](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Podman GPU 支持](https://github.com/containers/podman/blob/main/docs/tutorials/podman-for-windows.md#gpu-support)

## 相关文件

- `Dockerfile` - 镜像定义
- `docker-compose.yml` - Compose 配置
- `.dockerignore` - 构建忽略文件
- `build.sh` - 构建脚本
- `run.sh` - 运行脚本
- `requirements.txt` - Python 依赖
- `dali_mcp_server.py` - 服务器代码
