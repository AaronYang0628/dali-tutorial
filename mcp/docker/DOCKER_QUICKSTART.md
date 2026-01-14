# DALI MCP Server - Docker 部署快速参考

## 新增文件

本次更新添加了以下文件，用于 Docker 化部署 DALI MCP Server：

```
mcp/
├── Dockerfile              # Docker 镜像定义
├── docker-compose.yml      # Docker Compose 配置
├── .dockerignore          # 构建忽略文件
├── build.sh               # 构建脚本（可执行）
├── run.sh                 # 运行脚本（可执行）
└── BUILD.md               # 完整的构建和部署指南
```

## 服务器增强

`dali_mcp_server.py` 已增强日志功能：
- ✅ 启动时显示版本信息
- ✅ 工具调用时记录日志
- ✅ 错误时详细追踪
- ✅ 日志输出到 stderr（不影响 MCP 协议）

## 快速使用

### 本地测试（无需 Docker）

```bash
# 启动服务器（带日志）
python dali_mcp_server.py

# 测试服务器
python example_client.py
```

### Docker 部署

**方式 1: 使用脚本（推荐）**
```bash
cd mcp/

# 构建镜像
./build.sh

# 运行服务
./run.sh
```

**方式 2: 使用 docker-compose**
```bash
cd mcp/

# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d

# 停止
docker-compose down
```

**方式 3: 手动命令**
```bash
cd mcp/

# 构建
docker build -t dali-mcp-server:latest .

# 运行
docker run --rm -it --gpus all dali-mcp-server:latest
```

## 日志示例

启动服务时会看到类似输出（输出到 stderr）：

```
2026-01-14 10:05:40,404 - dali-mcp-server - INFO - ============================================================
2026-01-14 10:05:40,404 - dali-mcp-server - INFO - DALI MCP Server Starting...
2026-01-14 10:05:40,404 - dali-mcp-server - INFO - DALI Version: 1.53.0
2026-01-14 10:05:40,404 - dali-mcp-server - INFO - Python Version: 3.11.14 (main, Oct 10 2025, 08:54:03) [GCC 11.4.0]
2026-01-14 10:05:40,404 - dali-mcp-server - INFO - ============================================================
2026-01-14 10:05:40,407 - dali-mcp-server - INFO - Server initialized, waiting for connections...
```

工具调用时：
```
2026-01-14 02:08:16,182 - dali-mcp-server - INFO - Tool called: create_test_dataset
```

## 配置选项

### 环境变量

```bash
# GPU 配置
-e CUDA_VISIBLE_DEVICES=0
-e NVIDIA_VISIBLE_DEVICES=all

# AWS S3 凭证
-e AWS_ACCESS_KEY_ID="your_key"
-e AWS_SECRET_ACCESS_KEY="your_secret"
```

### 挂载卷

```bash
# 挂载本地数据集
-v /path/to/data:/data:ro

# 持久化临时文件
-v dali-temp:/tmp/dali_datasets
```

## 验证部署

1. **检查镜像是否构建成功**
   ```bash
   docker images | grep dali-mcp-server
   ```

2. **测试服务器启动**
   ```bash
   timeout 3 docker run --rm --gpus all dali-mcp-server:latest 2>&1 | head -10
   ```

3. **运行完整测试**
   ```bash
   python example_client.py
   ```

## 故障排查

### 问题: 没有 GPU 支持

如果 `--gpus` 参数不可用，服务器会自动降级到 CPU 模式运行。

### 问题: 容器运行时未安装

在 devcontainer 中无法安装 Docker/Podman，请在宿主机上操作：
```bash
# 退出 devcontainer
# 在宿主机上运行
cd /path/to/project/mcp
./build.sh
```

### 问题: 权限错误

确保脚本有执行权限：
```bash
chmod +x build.sh run.sh
```

## 下一步

- 📖 详细部署指南: 查看 `BUILD.md`
- 🚀 使用教程: 查看 `README.md`
- 💡 快速上手: 查看 `QUICKSTART.md`
- 🧪 测试示例: 运行 `example_client.py`

## 测试结果

✅ 所有功能测试通过：
- 创建测试数据集
- 创建基础 Pipeline
- 创建增强 Pipeline
- 运行 Pipeline 并获取统计
- 列出数据集和 Pipeline
- 日志输出正常

服务器已就绪，可以部署使用！
