# N8N MCP Client 连接故障排除

## 问题：Could not connect to your MCP server

### 症状
- HTTP 节点可以访问服务器（如 `/health` 端点）
- MCP Client 节点报错 "Could not connect to your MCP server"
- 服务器日志显示没有连接尝试

### 原因分析

MCP Client 和普通 HTTP 请求的区别：

| 特性 | HTTP 节点 | MCP Client 节点 |
|------|----------|----------------|
| 连接类型 | 短连接 | 长连接（SSE） |
| 端点 | 任意 HTTP 端点 | 必须是 `/sse` |
| 协议 | HTTP GET/POST | Server-Sent Events |
| 响应 | 一次性响应 | 持续的事件流 |

### 解决方案

#### 方案 1：检查 URL 格式（最常见）

n8n MCP Client 的 Server URL 配置有两种可能的格式：

**格式 A：根 URL（推荐）**
```
http://192.168.31.111:8000
```

**格式 B：完整 SSE 端点 URL**
```
http://192.168.31.111:8000/sse
```

**测试步骤：**
1. 先尝试格式 A（根 URL）
2. 如果不行，尝试格式 B（完整端点）
3. 确保 Transport 选择 "Server Sent Event"

#### 方案 2：检查服务器绑定地址

如果服务器启动时使用了 `--host 0.0.0.0`，确认它确实在监听所有网络接口。

**检查命令：**
```bash
# 查看服务器进程
ps aux | grep dali_mcp_server_http

# 查看端口监听状态
netstat -tulpn | grep 8000
# 或
ss -tulpn | grep 8000
```

**预期输出：**
```
tcp  0  0  0.0.0.0:8000  0.0.0.0:*  LISTEN  12345/python
```

如果显示 `127.0.0.1:8000` 而不是 `0.0.0.0:8000`，说明服务器只监听本地回环接口。

**解决方法：**
```bash
# 重新启动服务器，明确指定 host
python dali_mcp_server_http.py --host 0.0.0.0 --port 8000
```

#### 方案 3：防火墙检查

检查防火墙是否阻止了 8000 端口。

**Ubuntu/Debian：**
```bash
# 查看防火墙状态
sudo ufw status

# 如果启用了防火墙，允许 8000 端口
sudo ufw allow 8000/tcp
```

**CentOS/RHEL：**
```bash
# 查看防火墙状态
sudo firewall-cmd --list-all

# 允许 8000 端口
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

