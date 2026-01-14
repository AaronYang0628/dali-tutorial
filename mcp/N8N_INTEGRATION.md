# DALI HTTP API Server - N8N Integration Guide

## 概述

DALI HTTP API Server 提供RESTful API接口，让N8N等自动化工具可以通过HTTP调用NVIDIA DALI的图像处理功能。

## 快速开始

### 1. 安装依赖

```bash
cd /workspaces/dali-tutorial/mcp/scripts
pip install -r requirements.txt
```

### 2. 启动HTTP服务器

```bash
# 基础启动
python dali_http_server.py

# 自定义端口
python dali_http_server.py --port 8080

# 开发模式（自动重载）
python dali_http_server.py --reload
```

服务器默认运行在 `http://localhost:8000`

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看所有端点
curl http://localhost:8000/
```

## API端点

### 基础端点

| 方法 | 路径 | 说明 |
|-----|------|------|
| GET | `/` | 服务信息和端点列表 |
| GET | `/health` | 健康检查 |

### 数据集管理

| 方法 | 路径 | 说明 |
|-----|------|------|
| POST | `/api/dataset/create` | 创建测试数据集 |
| POST | `/api/dataset/import/local` | 导入本地数据集 |
| POST | `/api/dataset/import/s3` | 导入S3数据集 |
| GET | `/api/dataset/list` | 列出所有数据集 |

### Pipeline管理

| 方法 | 路径 | 说明 |
|-----|------|------|
| POST | `/api/pipeline/create` | 创建Pipeline |
| POST | `/api/pipeline/run` | 运行Pipeline |
| GET | `/api/pipeline/list` | 列出所有Pipeline |

## N8N 配置示例

### 示例 1: 创建测试数据集

在N8N中添加 **HTTP Request** 节点：

**配置:**
- Method: `POST`
- URL: `http://localhost:8000/api/dataset/create`
- Body Content Type: `JSON`

**请求体:**
```json
{
  "name": "test_dataset_001",
  "num_images": 20,
  "image_size": 256
}
```

**响应示例:**
```json
{
  "dataset_name": "test_dataset_001",
  "dataset_path": "/tmp/dali_dataset_test_dataset_001_xyz",
  "num_files": 20,
  "image_size": 256,
  "file_list": [
    "/tmp/dali_dataset_test_dataset_001_xyz/image_0000.jpg",
    "/tmp/dali_dataset_test_dataset_001_xyz/image_0001.jpg",
    "..."
  ]
}
```

### 示例 2: 导入本地数据集

**配置:**
- Method: `POST`
- URL: `http://localhost:8000/api/dataset/import/local`

**请求体:**
```json
{
  "dataset_name": "my_photos",
  "local_path": "/path/to/your/images",
  "supported_formats": ["jpg", "jpeg", "png"]
}
```

### 示例 3: 创建图像处理Pipeline

**配置:**
- Method: `POST`
- URL: `http://localhost:8000/api/pipeline/create`

**请求体 - 基础处理:**
```json
{
  "name": "basic_pipeline",
  "dataset_name": "test_dataset_001",
  "pipeline_type": "basic",
  "batch_size": 8,
  "target_size": 224
}
```

**请求体 - 数据增强:**
```json
{
  "name": "augmentation_pipeline",
  "dataset_name": "test_dataset_001",
  "pipeline_type": "augmentation",
  "batch_size": 16,
  "target_size": 224
}
```

### 示例 4: 运行Pipeline

**配置:**
- Method: `POST`
- URL: `http://localhost:8000/api/pipeline/run`

**请求体:**
```json
{
  "pipeline_name": "basic_pipeline",
  "num_iterations": 5
}
```

**响应示例:**
```json
{
  "pipeline_name": "basic_pipeline",
  "iterations": 5,
  "batch_size": 8,
  "batches": [
    {
      "iteration": 1,
      "num_outputs": 2,
      "shapes": ["[(224, 224, 3)]", "[1]"]
    }
  ],
  "status": "completed"
}
```

### 示例 5: 列出所有资源

**列出数据集:**
- Method: `GET`
- URL: `http://localhost:8000/api/dataset/list`

**列出Pipeline:**
- Method: `GET`
- URL: `http://localhost:8000/api/pipeline/list`

## N8N 工作流示例

### 完整图像处理工作流

```
[开始]
  → [创建数据集]
  → [创建Pipeline]
  → [运行Pipeline]
  → [处理结果]
```

**节点配置:**

#### 1. 创建数据集节点
- 名称: Create Dataset
- Type: HTTP Request
- Method: POST
- URL: `http://localhost:8000/api/dataset/create`
- Body:
```json
{
  "name": "{{ $json.dataset_name }}",
  "num_images": {{ $json.num_images }},
  "image_size": 256
}
```

#### 2. 创建Pipeline节点
- 名称: Create Pipeline
- Type: HTTP Request
- Method: POST
- URL: `http://localhost:8000/api/pipeline/create`
- Body:
```json
{
  "name": "{{ $json.pipeline_name }}",
  "dataset_name": "{{ $node['Create Dataset'].json['dataset_name'] }}",
  "pipeline_type": "augmentation",
  "batch_size": 8,
  "target_size": 224
}
```

#### 3. 运行Pipeline节点
- 名称: Run Pipeline
- Type: HTTP Request
- Method: POST
- URL: `http://localhost:8000/api/pipeline/run`
- Body:
```json
{
  "pipeline_name": "{{ $node['Create Pipeline'].json['pipeline_name'] }}",
  "num_iterations": 3
}
```

## S3数据集导入（MinIO/AWS）

### 从MinIO导入

```json
{
  "dataset_name": "minio_dataset",
  "s3_uri": "s3://my-bucket/images",
  "endpoint_url": "http://minio-server:9000",
  "access_key": "minioadmin",
  "secret_key": "minioadmin",
  "download": true,
  "supported_formats": ["jpg", "png"]
}
```

### 从AWS S3导入

```json
{
  "dataset_name": "aws_dataset",
  "s3_uri": "s3://my-aws-bucket/images",
  "download": true,
  "supported_formats": ["jpg", "png"]
}
```

注意：AWS凭证会从环境变量读取（`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`）

## 错误处理

### 常见错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 409 | 资源已存在（如数据集/Pipeline名称重复） |
| 500 | 服务器内部错误 |
| 501 | 功能未实现（如S3支持未安装） |

### N8N错误处理配置

在HTTP Request节点中启用 **Continue On Fail** 选项，并添加错误处理分支：

```
[HTTP Request]
  → Success → [继续处理]
  → Error → [错误通知/日志]
```

## 性能优化建议

### 1. 批处理
使用较大的 `batch_size` 来提高GPU利用率：
```json
{
  "batch_size": 32  // 根据GPU内存调整
}
```

### 2. 异步处理
N8N中使用 **Split In Batches** 节点处理大量数据集。

### 3. 数据预热
创建Pipeline后运行一次空迭代来预热DALI：
```json
{
  "num_iterations": 1
}
```

## 部署建议

### Docker部署

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dali-http-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["python", "dali_http_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### 生产环境配置

```bash
# 使用Gunicorn + Uvicorn workers
pip install gunicorn

gunicorn dali_http_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

## API文档

启动服务器后访问自动生成的API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 测试脚本

### cURL测试

```bash
# 创建数据集
curl -X POST http://localhost:8000/api/dataset/create \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "num_images": 10, "image_size": 256}'

# 创建Pipeline
curl -X POST http://localhost:8000/api/pipeline/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "pipe1",
    "dataset_name": "test",
    "pipeline_type": "basic",
    "batch_size": 4,
    "target_size": 224
  }'

# 运行Pipeline
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "pipe1", "num_iterations": 2}'

# 列出所有资源
curl http://localhost:8000/api/dataset/list
curl http://localhost:8000/api/pipeline/list
```

### Python测试脚本

```python
import requests

BASE_URL = "http://localhost:8000"

# 创建数据集
response = requests.post(
    f"{BASE_URL}/api/dataset/create",
    json={
        "name": "test_dataset",
        "num_images": 20,
        "image_size": 256
    }
)
print(response.json())

# 创建Pipeline
response = requests.post(
    f"{BASE_URL}/api/pipeline/create",
    json={
        "name": "test_pipeline",
        "dataset_name": "test_dataset",
        "pipeline_type": "augmentation",
        "batch_size": 8,
        "target_size": 224
    }
)
print(response.json())

# 运行Pipeline
response = requests.post(
    f"{BASE_URL}/api/pipeline/run",
    json={
        "pipeline_name": "test_pipeline",
        "num_iterations": 3
    }
)
print(response.json())
```

## 安全建议

### 1. 身份验证
生产环境应添加API密钥或OAuth2认证：

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/dataset/create", dependencies=[Depends(security)])
async def create_dataset(...):
    ...
```

### 2. CORS配置
限制允许的源：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-n8n-instance.com"],
    ...
)
```

### 3. 速率限制
使用slowapi或类似库限制请求频率。

## 故障排除

### 问题 1: CUDA不可用
**错误:** "CUDA not available"

**解决方案:**
- 检查GPU驱动: `nvidia-smi`
- 确认DALI版本与CUDA版本匹配
- Pipeline会自动降级到CPU模式

### 问题 2: 端口被占用
**错误:** "Address already in use"

**解决方案:**
```bash
# 使用不同端口
python dali_http_server.py --port 8001
```

### 问题 3: S3连接失败
**错误:** "S3 error: ..."

**解决方案:**
- 检查endpoint_url是否正确
- 验证access_key和secret_key
- 确认网络连接

## 支持

如有问题请查看：
- API文档: http://localhost:8000/docs
- DALI官方文档: https://docs.nvidia.com/deeplearning/dali/
- N8N文档: https://docs.n8n.io/

## 许可证

与DALI MCP Server相同
