# N8N MCP Client 配置指南

## 概述

这份指南说明如何在 n8n 中使用 MCP Client 节点连接到 DALI MCP 服务器。

## 重要：传输协议

n8n MCP Client 支持以下传输协议：
- ✅ **HTTP Streamable**
- ✅ **Server Sent Event (SSE)**
- ❌ STDIO (不支持)

因此，你需要使用 **HTTP/SSE 版本的服务器**，而不是 STDIO 版本。

## 服务器选择

| 文件 | 传输模式 | 端口 | 适用于 n8n |
|------|---------|------|-----------|
| `dali_mcp_server.py` | STDIO | 无 | ❌ 不适用 |
| `dali_mcp_server_http.py` | HTTP + SSE | 8000 (默认) | ✅ 适用 |

## 第一步：启动 HTTP/SSE 服务器

### 方式 1：默认配置启动

```bash
cd /workspaces/dali-tutorial/mcp/scripts
python dali_mcp_server_http.py
```

服务器将在 `http://0.0.0.0:8000` 上监听。

### 方式 2：自定义端口

```bash
python dali_mcp_server_http.py --port 8888
```

### 方式 3：后台运行

```bash
nohup python dali_mcp_server_http.py --port 8000 > server.log 2>&1 &
```

### 启动成功日志示例

```
============================================================
DALI MCP Server Starting (HTTP/SSE Transport)
DALI Version: 1.53.0
Python Version: 3.11.14
============================================================

Transport Mode: HTTP + Server-Sent Events (SSE)
Listening on: http://0.0.0.0:8000

Endpoints:
  - Root:     http://0.0.0.0:8000/
  - Health:   http://0.0.0.0:8000/health
  - SSE:      http://0.0.0.0:8000/sse
  - Messages: http://0.0.0.0:8000/messages/

Compatible with:
  ✓ n8n MCP Client (HTTP/SSE transport)
  ✓ Claude Desktop
  ✓ Custom HTTP/SSE clients

Configuration for n8n:
  Server URL: http://0.0.0.0:8000
  Transport: "Server Sent Event"
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 第二步：配置 n8n MCP Client

### 1. 添加 MCP Client 节点

在 n8n 工作流中：
1. 点击 "+" 添加节点
2. 搜索 "MCP Client"
3. 选择 "MCP Client" 节点

### 2. 配置连接

在 MCP Client 节点配置界面：

#### 基本设置

| 字段 | 值 | 说明 |
|------|-----|------|
| **Server URL** | `http://localhost:8000/sse` | ⚠️ 必须包含 `/sse` 端点 |
| | 或 `http://<server-ip>:8000/sse` | 如果服务器在不同机器上 |
| **Transport** | `Server Sent Event` | 选择 SSE 传输模式 |

**重要**：Server URL 必须是完整的 SSE 端点 URL（包含 `/sse`），不能只填根 URL。

#### Docker 环境注意事项

如果 n8n 运行在 Docker 容器中，而服务器运行在宿主机上：

```yaml
# docker-compose.yml
services:
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    extra_hosts:
      - "host.docker.internal:host-gateway"  # 允许访问宿主机
```

然后在 n8n 中使用：
- Server URL: `http://host.docker.internal:8000`

### 3. 测试连接

1. 保存 MCP Client 节点配置
2. 执行工作流
3. 检查节点是否成功连接

## 第三步：使用可用工具

DALI MCP 服务器提供以下工具：

### 1. create_test_dataset

创建测试图像数据集。

```json
{
  "tool": "create_test_dataset",
  "arguments": {
    "name": "my_dataset",
    "num_images": 20,
    "image_size": 512
  }
}
```

### 2. import_local_dataset

从本地目录导入图像。

```json
{
  "tool": "import_local_dataset",
  "arguments": {
    "dataset_name": "local_images",
    "local_path": "/workspaces/dali-tutorial/sample_images"
  }
}
```

### 3. create_pipeline

创建数据处理管道。

```json
{
  "tool": "create_pipeline",
  "arguments": {
    "name": "my_pipeline",
    "dataset_name": "my_dataset",
    "pipeline_type": "augmentation",
    "batch_size": 8,
    "target_size": 224
  }
}
```

### 4. run_pipeline

运行管道并获取统计信息。

```json
{
  "tool": "run_pipeline",
  "arguments": {
    "pipeline_name": "my_pipeline",
    "num_iterations": 3
  }
}
```

### 5. list_datasets

列出所有数据集。

```json
{
  "tool": "list_datasets",
  "arguments": {}
}
```

### 6. list_pipelines

列出所有管道。

```json
{
  "tool": "list_pipelines",
  "arguments": {}
}
```

### 7. import_s3_dataset

从 S3 导入数据集。

```json
{
  "tool": "import_s3_dataset",
  "arguments": {
    "dataset_name": "s3_images",
    "s3_uri": "s3://my-bucket/images/",
    "download": true
  }
}
```

## 完整 n8n 工作流示例

### 示例 1：创建和运行基础数据集

```
Start Node
  ↓
MCP Client (create_test_dataset)
  - name: "test_images"
  - num_images: 10
  ↓
MCP Client (create_pipeline)
  - name: "basic_pipe"
  - dataset_name: "test_images"
  - pipeline_type: "basic"
  ↓
MCP Client (run_pipeline)
  - pipeline_name: "basic_pipe"
  - num_iterations: 2
  ↓
Output Node
```

### 示例 2：导入本地数据并应用增强

```
Start Node
  ↓
MCP Client (import_local_dataset)
  - dataset_name: "my_photos"
  - local_path: "/path/to/images"
  ↓
MCP Client (create_pipeline)
  - name: "augment_pipe"
  - dataset_name: "my_photos"
  - pipeline_type: "augmentation"
  ↓
MCP Client (run_pipeline)
  - pipeline_name: "augment_pipe"
  ↓
Process Results
```

## 验证服务器状态

### 检查健康状态

使用浏览器或 curl：

```bash
curl http://localhost:8000/health
```

预期响应：

```json
{
  "status": "healthy",
  "server": "dali-mcp-server-http",
  "version": "1.0.0",
  "dali_version": "1.53.0",
  "transport": "HTTP/SSE",
  "endpoints": {
    "sse": "/sse",
    "messages": "/messages/",
    "health": "/health"
  }
}
```

### 查看服务器信息

```bash
curl http://localhost:8000/
```

## 故障排除

### 问题 1：连接失败

**症状**：n8n 无法连接到服务器

**解决方法**：
1. 确认服务器正在运行：`ps aux | grep dali_mcp_server_http`
2. 检查端口是否开放：`netstat -tulpn | grep 8000`
3. 测试端点：`curl http://localhost:8000/health`
4. 检查防火墙设置

### 问题 2：工具调用失败

**症状**：工具调用返回错误

**解决方法**：
1. 检查服务器日志：`tail -f server.log`
2. 验证参数格式是否正确
3. 确认数据集/管道是否存在（使用 `list_datasets` 或 `list_pipelines`）

### 问题 3：Docker 网络问题

**症状**：Docker 中的 n8n 无法访问宿主机服务器

**解决方法**：
1. 使用 `host.docker.internal` 而不是 `localhost`
2. 或者配置 Docker 网络：
   ```bash
   docker network create mcp-network
   ```
3. 将两个容器连接到同一网络

### 问题 4：传输协议不匹配

**症状**："Transport not supported" 错误

**解决方法**：
- ✅ 使用 `dali_mcp_server_http.py`（支持 SSE）
- ❌ 不要使用 `dali_mcp_server.py`（仅支持 STDIO）

## 监控和日志

### 查看实时日志

```bash
# 服务器日志
tail -f server.log

# 或者如果前台运行，直接查看 stderr 输出
```

### 工具调用日志示例

```
============================================================
Incoming tool call: create_test_dataset
Arguments: {
  "name": "test_images",
  "num_images": 10,
  "image_size": 256
}
============================================================
✓ Tool 'create_test_dataset' completed successfully
```

## 安全注意事项

1. **生产环境**：
   - 使用 `--host 127.0.0.1` 限制只允许本地访问
   - 配置反向代理（nginx）和 HTTPS
   - 添加认证中间件

2. **文件路径**：
   - 只使用绝对路径导入本地数据集
   - 验证路径权限

3. **S3 凭证**：
   - 使用环境变量而不是直接传递凭证
   - 定期轮换访问密钥

## 性能优化

1. **批次大小**：根据可用 GPU 内存调整 `batch_size`
2. **工作线程**：服务器默认使用 2 个工作线程
3. **并发请求**：服务器支持多个并发 SSE 连接

## 相关文件

- `dali_mcp_server_http.py` - HTTP/SSE 服务器实现
- `dali_mcp_server.py` - STDIO 服务器（不适用于 n8n）
- `dali_mcp_client.py` - Python 测试客户端
- `README_MCP_SERVER.md` - 服务器说明文档

## 更多资源

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [n8n 文档](https://docs.n8n.io/)
- [NVIDIA DALI 文档](https://docs.nvidia.com/deeplearning/dali/)
