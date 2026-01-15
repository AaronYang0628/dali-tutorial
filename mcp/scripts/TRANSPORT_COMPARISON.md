# DALI MCP Server - 传输模式对比

## 快速选择

| 使用场景 | 推荐服务器 | 启动命令 |
|---------|-----------|---------|
| n8n MCP Client | `dali_mcp_server_http.py` | `./start_http_server.sh` |
| Claude Desktop | `dali_mcp_server.py` | `python dali_mcp_server.py` |
| 自定义 Python 客户端 | 两者皆可 | 根据传输模式选择 |

## 两种服务器对比

### 1. STDIO 版本 (`dali_mcp_server.py`)

**传输模式**: STDIO (标准输入输出)

**特点**:
- ✅ 轻量级，无需 HTTP 服务器
- ✅ 适用于进程间通信
- ✅ Claude Desktop 原生支持
- ❌ **不支持 n8n** (n8n 不支持 STDIO)
- ❌ 没有 HTTP 端点
- ❌ 不能通过浏览器访问

**启动方式**:
```bash
python dali_mcp_server.py
```

**配置示例** (Claude Desktop):
```json
{
  "mcpServers": {
    "dali-server": {
      "command": "python",
      "args": ["/path/to/dali_mcp_server.py"]
    }
  }
}
```

**日志输出**:
```
Transport Mode: STDIO (Standard Input/Output)
Communication: JSON-RPC over stdin/stdout
This server does NOT use HTTP ports.
```

---

### 2. HTTP/SSE 版本 (`dali_mcp_server_http.py`)

**传输模式**: HTTP + Server-Sent Events (SSE)

**特点**:
- ✅ **完全支持 n8n**
- ✅ 支持 HTTP 端点和浏览器访问
- ✅ 健康检查端点 `/health`
- ✅ 支持多个并发客户端
- ✅ 可配置端口和主机
- ⚠️ 需要更多资源（HTTP 服务器）

**启动方式**:

使用启动脚本（推荐）:
```bash
./start_http_server.sh           # 默认端口 8000
./start_http_server.sh 8888      # 自定义端口
./start_http_server.sh 8000 debug # 调试模式
```

或直接运行:
```bash
python dali_mcp_server_http.py --port 8000 --host 0.0.0.0
```

**配置示例** (n8n):
```
Server URL: http://localhost:8000
Transport: Server Sent Event
```

**日志输出**:
```
Transport Mode: HTTP + Server-Sent Events (SSE)
Listening on: http://0.0.0.0:8000
Endpoints:
  - Root:     http://0.0.0.0:8000/
  - Health:   http://0.0.0.0:8000/health
  - SSE:      http://0.0.0.0:8000/sse
  - Messages: http://0.0.0.0:8000/messages/
```

**端点访问**:
```bash
# 健康检查
curl http://localhost:8000/health

# 服务器信息
curl http://localhost:8000/

# 响应示例
{
  "status": "healthy",
  "server": "dali-mcp-server-http",
  "version": "1.0.0",
  "dali_version": "1.53.0",
  "transport": "HTTP/SSE"
}
```

---

## n8n 用户必读

### ⚠️ 重要提示

**n8n MCP Client 只支持以下传输模式：**
- ✅ HTTP Streamable
- ✅ Server Sent Event (SSE)

**不支持：**
- ❌ STDIO

### 正确配置

#### ✅ 正确 - 使用 HTTP/SSE 服务器

```bash
# 1. 启动 HTTP/SSE 服务器
./start_http_server.sh

# 2. 在 n8n 中配置
Server URL: http://localhost:8000
Transport: Server Sent Event
```

#### ❌ 错误 - 使用 STDIO 服务器

```bash
# 这个不能用于 n8n！
python dali_mcp_server.py  # ❌ n8n 无法连接
```

### 快速开始 (n8n)

1. **启动服务器**
   ```bash
   cd /workspaces/dali-tutorial/mcp/scripts
   ./start_http_server.sh
   ```

2. **配置 n8n**
   - 添加 "MCP Client" 节点
   - Server URL: `http://localhost:8000`
   - Transport: `Server Sent Event`

3. **测试连接**
   ```bash
   curl http://localhost:8000/health
   ```

4. **开始使用工具**
   - `create_test_dataset`
   - `create_pipeline`
   - `run_pipeline`
   - 等等...

## 功能对比表

| 功能 | STDIO 版本 | HTTP/SSE 版本 |
|-----|-----------|--------------|
| MCP 工具 | ✅ 完全相同 | ✅ 完全相同 |
| DALI Pipeline | ✅ | ✅ |
| S3 导入 | ✅ | ✅ |
| n8n 支持 | ❌ | ✅ |
| Claude Desktop | ✅ | ✅ |
| HTTP 端点 | ❌ | ✅ |
| 健康检查 | ❌ | ✅ `/health` |
| 并发客户端 | 单个 | 多个 |
| 端口配置 | 无 | ✅ |
| 浏览器访问 | ❌ | ✅ |

## 文档索引

| 文档 | 说明 |
|-----|------|
| `N8N_SETUP_GUIDE.md` | **n8n 用户必读** - 详细配置指南 |
| `README_MCP_SERVER.md` | STDIO 版本说明文档 |
| `start_http_server.sh` | HTTP/SSE 服务器启动脚本 |
| 本文件 | 传输模式对比和快速选择指南 |

## 常见问题

### Q: 我应该使用哪个服务器？

**A**:
- 使用 n8n → `dali_mcp_server_http.py`
- 使用 Claude Desktop → `dali_mcp_server.py`
- 需要 HTTP API → `dali_mcp_server_http.py`

### Q: n8n 能用 STDIO 版本吗？

**A**: 不能。n8n MCP Client 不支持 STDIO 传输，必须使用 HTTP/SSE 版本。

### Q: 为什么 STDIO 版本没有端口？

**A**: STDIO 传输通过标准输入输出通信，不使用网络端口。客户端通过启动服务器进程并使用管道通信。

### Q: HTTP 版本能用于 Claude Desktop 吗？

**A**: 可以，但 STDIO 版本更简单高效，推荐使用 STDIO 版本。

### Q: 两个版本的功能有区别吗？

**A**: 没有。两个版本提供完全相同的 MCP 工具和 DALI 功能，只是传输方式不同。

### Q: 能同时运行两个服务器吗？

**A**: 可以。它们使用不同的传输方式，不会冲突。

## 迁移指南

### 从 STDIO 迁移到 HTTP/SSE

如果你之前使用 STDIO 版本，现在想切换到 HTTP/SSE：

1. **停止 STDIO 服务器**
   ```bash
   # 找到并停止进程
   pkill -f dali_mcp_server.py
   ```

2. **启动 HTTP/SSE 服务器**
   ```bash
   ./start_http_server.sh
   ```

3. **更新客户端配置**
   - 从进程启动改为 HTTP URL
   - 添加端口号
   - 选择 SSE 传输

4. **测试连接**
   ```bash
   curl http://localhost:8000/health
   ```

### 注意事项

- ⚠️ 数据集和管道状态不会迁移（它们在内存中）
- ⚠️ 需要重新创建数据集和管道
- ✅ 临时文件会自动清理

## 总结

- **n8n 用户**: 使用 `dali_mcp_server_http.py` + 阅读 `N8N_SETUP_GUIDE.md`
- **Claude Desktop 用户**: 使用 `dali_mcp_server.py` + 阅读 `README_MCP_SERVER.md`
- **两个版本功能完全相同**，只是传输方式不同
- **HTTP/SSE 版本更通用**，支持更多客户端类型

---

**需要帮助？**
- 查看详细文档：`N8N_SETUP_GUIDE.md`
- 测试健康状态：`curl http://localhost:8000/health`
- 查看服务器日志以排查问题
