# DALI MCP Server

一个基于 Model Context Protocol (MCP) 的 NVIDIA DALI 服务器，允许 AI Agent 通过标准协议调用 DALI 进行数据生成和处理。

## 功能特性

### 当前功能（v0.2）

- ✅ **数据集创建**：生成测试图像数据集
- ✅ **本地数据导入**：从本地目录导入真实图像数据
- ✅ **S3 数据导入**：从 AWS S3 或 MinIO 等兼容存储导入数据
- ✅ **Pipeline 管理**：创建和管理多个 DALI Pipeline
- ✅ **基础处理**：图像解码、缩放、裁剪
- ✅ **数据增强**：随机裁剪、翻转、颜色调整
- ✅ **统计分析**：获取处理结果的统计信息

### 规划功能（未来版本）

- 🔲 自定义 Pipeline 配置
- 🔲 支持更多数据格式（视频、音频）
- 🔲 性能分析和优化建议
- 🔲 与 PyTorch/TensorFlow 集成
- 🔲 分布式处理支持

## 安装

### 前置要求

```bash
# 1. Python 3.8+
python --version

# 2. NVIDIA DALI
pip install nvidia-dali-cuda120

# 3. MCP SDK
pip install mcp

# 4. 其他依赖
pip install numpy pillow
```

### 快速安装

```bash
cd /workspaces/dali-tutorial/mcp
pip install -r requirements.txt
```

## 使用方式

### 方式 1: 命令行客户端

运行示例客户端：

```bash
python example_client.py
```

### 方式 2: Claude Desktop 集成

1. **复制配置到 Claude Desktop**

   ```bash
   # macOS
   cat claude_desktop_config.json >> ~/Library/Application\ Support/Claude/claude_desktop_config.json

   # Windows
   type claude_desktop_config.json >> %APPDATA%\Claude\claude_desktop_config.json

   # Linux
   cat claude_desktop_config.json >> ~/.config/Claude/claude_desktop_config.json
   ```

2. **重启 Claude Desktop**

3. **在对话中使用工具**

   现在可以直接在 Claude Desktop 中使用 DALI 工具：

   ```
   User: 帮我创建一个包含 50 张图片的数据集，然后用数据增强 pipeline 处理它们

   Claude: 我来帮你操作：
   1. 首先创建数据集...
   [调用 create_test_dataset]
   2. 创建数据增强 pipeline...
   [调用 create_pipeline]
   3. 运行 pipeline...
   [调用 run_pipeline]
   ```

### 方式 3: Python API

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_dali_server():
    server_params = StdioServerParameters(
        command="python",
        args=["dali_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 创建数据集
            result = await session.call_tool(
                "create_test_dataset",
                arguments={"name": "my_data", "num_images": 100}
            )

            # ... 更多操作
```

## 可用工具

### 1. create_test_dataset

创建测试图像数据集。

**参数**：
- `name` (string, required): 数据集名称
- `num_images` (integer, optional): 图像数量，默认 10
- `image_size` (integer, optional): 图像尺寸，默认 256

**示例**：
```json
{
  "name": "training_data",
  "num_images": 1000,
  "image_size": 512
}
```

**返回**：
```json
{
  "dataset_name": "training_data",
  "dataset_path": "/tmp/dali_dataset_training_data_xxx",
  "num_files": 1000,
  "image_size": 512,
  "file_list": ["...", "..."]
}
```

### 2. create_pipeline

创建 DALI 数据处理 Pipeline。

**参数**：
- `name` (string, required): Pipeline 名称
- `dataset_name` (string, required): 数据集名称
- `pipeline_type` (string, optional): 类型 ('basic' 或 'augmentation')，默认 'basic'
- `batch_size` (integer, optional): 批次大小，默认 4
- `target_size` (integer, optional): 目标尺寸，默认 224

**示例**：
```json
{
  "name": "train_pipeline",
  "dataset_name": "training_data",
  "pipeline_type": "augmentation",
  "batch_size": 32,
  "target_size": 224
}
```

**返回**：
```json
{
  "pipeline_name": "train_pipeline",
  "pipeline_type": "augmentation",
  "batch_size": 32,
  "target_size": 224,
  "dataset_name": "training_data",
  "num_files": 1000,
  "status": "created and built"
}
```

### 3. run_pipeline

运行 Pipeline 并获取统计信息。

**参数**：
- `pipeline_name` (string, required): Pipeline 名称
- `num_iterations` (integer, optional): 迭代次数，默认 1

**示例**：
```json
{
  "pipeline_name": "train_pipeline",
  "num_iterations": 5
}
```

**返回**：
```json
{
  "pipeline_name": "train_pipeline",
  "pipeline_type": "augmentation",
  "batch_size": 32,
  "statistics": {
    "iterations": 5,
    "batches": [
      {
        "iteration": 1,
        "batch_size": 32,
        "shapes": ["(3, 224, 224)", "..."],
        "dtype": "DALIDataType.FLOAT",
        "sample_stats": {
          "min": 0.0,
          "max": 1.0,
          "mean": 0.48,
          "std": 0.25
        }
      }
    ]
  }
}
```

### 4. list_datasets

列出所有已创建的数据集。

**参数**：无

**返回**：
```json
{
  "count": 2,
  "datasets": [
    {
      "name": "training_data",
      "path": "/tmp/dali_dataset_training_data_xxx",
      "num_files": 1000
    },
    {
      "name": "validation_data",
      "path": "/tmp/dali_dataset_validation_data_xxx",
      "num_files": 200
    }
  ]
}
```

### 5. list_pipelines

列出所有已创建的 Pipeline。

**参数**：无

**返回**：
```json
{
  "count": 2,
  "pipelines": [
    {
      "name": "train_pipeline",
      "type": "augmentation",
      "batch_size": 32,
      "target_size": 224,
      "dataset_name": "training_data",
      "num_files": 1000
    },
    {
      "name": "val_pipeline",
      "type": "basic",
      "batch_size": 64,
      "target_size": 224,
      "dataset_name": "validation_data",
      "num_files": 200
    }
  ]
}
```

### 6. import_local_dataset

从本地文件目录导入图像数据集。相比 `create_test_dataset` 生成随机图像，此工具用于导入你自己的真实数据。

**参数**：
- `dataset_name` (string, required): 数据集名称，用于后续引用
- `local_path` (string, required): 本地目录的绝对路径
- `supported_formats` (array, optional): 支持的文件格式，默认 `["jpg", "jpeg", "png"]`

**示例**：
```json
{
  "dataset_name": "my_photos",
  "local_path": "/data/photos",
  "supported_formats": ["jpg", "png"]
}
```

**返回**：
```json
{
  "dataset_name": "my_photos",
  "dataset_path": "/data/photos",
  "num_files": 1250,
  "supported_formats": ["jpg", "png"],
  "file_list": [
    "/data/photos/photo_001.jpg",
    "/data/photos/photo_002.jpg",
    "..."
  ]
}
```

**常见场景**：
- 导入自己的图像数据集用于训练
- 支持多种图像格式自动扫描
- 与 `create_pipeline` 无缝集成

**错误处理**：
- 路径不存在时返回错误
- 路径必须是绝对路径
- 找不到支持格式的图像时返回错误
- 数据集名称重复时返回错误

### 7. import_s3_dataset

从 S3 兼容存储（AWS S3、MinIO 等）导入图像数据集。支持列举或下载两种模式。

**参数**：
- `dataset_name` (string, required): 数据集名称
- `s3_uri` (string, required): S3 URI，格式 `s3://bucket/prefix` 或 `s3://bucket`
- `endpoint_url` (string, optional): S3 端点 URL（用于 MinIO 等兼容存储）
- `access_key` (string, optional): AWS access key（优先从环境变量 `AWS_ACCESS_KEY_ID` 读取）
- `secret_key` (string, optional): AWS secret key（优先从环境变量 `AWS_SECRET_ACCESS_KEY` 读取）
- `download` (boolean, optional): 是否下载到本地，默认 `false`
- `supported_formats` (array, optional): 支持的文件格式，默认 `["jpg", "jpeg", "png"]`

**示例 1：AWS S3 + 下载**：
```json
{
  "dataset_name": "s3_training_data",
  "s3_uri": "s3://my-bucket/datasets/training",
  "download": true,
  "supported_formats": ["jpg", "png"]
}
```

**示例 2：MinIO + 流式读取**：
```json
{
  "dataset_name": "minio_data",
  "s3_uri": "s3://data-bucket/images",
  "endpoint_url": "http://minio:9000",
  "access_key": "minioadmin",
  "secret_key": "minioadmin",
  "download": false
}
```

**返回（下载模式）**：
```json
{
  "dataset_name": "s3_training_data",
  "s3_uri": "s3://my-bucket/datasets/training",
  "num_files": 5000,
  "local_path": "/tmp/dali_s3_dataset_s3_training_data_xxx",
  "status": "downloaded",
  "file_list": ["image_001.jpg", "image_002.jpg", "..."]
}
```

**返回（流式读取模式）**：
```json
{
  "dataset_name": "minio_data",
  "s3_uri": "s3://data-bucket/images",
  "num_files": 3000,
  "status": "listed",
  "note": "Files not downloaded. Use download=true to download to local directory before creating pipeline.",
  "file_list": ["img_001.jpg", "img_002.jpg", "..."]
}
```

**凭证管理**：
1. **优先级顺序**：
   - 环境变量 `AWS_ACCESS_KEY_ID` 和 `AWS_SECRET_ACCESS_KEY` （推荐）
   - 函数参数中的 `access_key` 和 `secret_key` （备选）

2. **设置环境变量**：
   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   ```

**两种使用模式**：

**模式 1：下载（`download=true`）**
- 适合：中等大小的数据集，需要 Pipeline 处理
- 优点：与本地数据集完全相同的使用体验
- 缺点：占用本地磁盘空间
- 自动清理：下载的文件在服务器关闭时自动清理

**模式 2：流式读取（`download=false`）**
- 适合：仅查看文件列表，评估数据
- 优点：节省磁盘空间
- 缺点：不能用于 Pipeline（Pipeline 需要本地文件路径）
- 用途：数据探索和规划

**常见场景**：
- 导入 AWS S3 上的大规模数据集
- 与 MinIO 私有存储集成
- 支持多个数据源共存（本地+S3）

**错误处理**：
- 凭证无效时返回认证错误
- Bucket 不存在时返回 "NoSuchBucket" 错误
- 权限不足时返回 "AccessDenied" 错误
- 不支持的格式的文件被自动过滤
- boto3 未安装时提示安装

## Pipeline 类型说明

### basic (基础处理)

执行操作：
1. 图像解码（JPEG）
2. 缩放到目标尺寸（保持宽高比）
3. 中心裁剪到目标尺寸
4. 输出 HWC 格式，uint8 类型

适用场景：
- 验证集/测试集处理
- 推理阶段
- 不需要数据增强的场景

### augmentation (数据增强)

执行操作：
1. 图像解码（JPEG）
2. 随机缩放裁剪（8%-100% 面积）
3. 随机水平翻转（50% 概率）
4. 亮度和对比度调整（±20%）
5. 归一化到 [0, 1]
6. 转换为 CHW 格式（PyTorch 兼容）

适用场景：
- 训练集处理
- 需要数据增强的场景
- 与 PyTorch 集成

## 完整使用流程示例

### 场景：训练数据准备

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def prepare_training_data():
    server_params = StdioServerParameters(
        command="python",
        args=["dali_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. 创建训练数据集
            await session.call_tool(
                "create_test_dataset",
                arguments={
                    "name": "train_set",
                    "num_images": 1000,
                    "image_size": 512
                }
            )

            # 2. 创建验证数据集
            await session.call_tool(
                "create_test_dataset",
                arguments={
                    "name": "val_set",
                    "num_images": 200,
                    "image_size": 512
                }
            )

            # 3. 创建训练 Pipeline（数据增强）
            await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "train_pipe",
                    "dataset_name": "train_set",
                    "pipeline_type": "augmentation",
                    "batch_size": 32,
                    "target_size": 224
                }
            )

            # 4. 创建验证 Pipeline（基础处理）
            await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "val_pipe",
                    "dataset_name": "val_set",
                    "pipeline_type": "basic",
                    "batch_size": 64,
                    "target_size": 224
                }
            )

            # 5. 测试运行
            train_stats = await session.call_tool(
                "run_pipeline",
                arguments={
                    "pipeline_name": "train_pipe",
                    "num_iterations": 10
                }
            )

            print("训练数据准备完成！")
            print(train_stats.content[0].text)

asyncio.run(prepare_training_data())
```

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client (Agent)                    │
│  (Claude Desktop / Python Script / Custom Application)  │
└──────────────────────┬──────────────────────────────────┘
                       │ MCP Protocol (JSON-RPC)
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  DALI MCP Server                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Tool Handlers                                   │   │
│  │  ├─ create_test_dataset                         │   │
│  │  ├─ create_pipeline                             │   │
│  │  ├─ run_pipeline                                │   │
│  │  ├─ list_datasets                               │   │
│  │  └─ list_pipelines                              │   │
│  └─────────────────────────────────────────────────┘   │
│                        │                                 │
│  ┌─────────────────────▼─────────────────────────┐     │
│  │  State Management                              │     │
│  │  ├─ datasets: Dict[name -> path]              │     │
│  │  ├─ pipelines: Dict[name -> pipeline]         │     │
│  │  └─ temp_dirs: List[path]                     │     │
│  └────────────────────────────────────────────────┘     │
│                        │                                 │
│  ┌─────────────────────▼─────────────────────────┐     │
│  │  DALI Pipelines                                │     │
│  │  ├─ basic_image_pipeline                      │     │
│  │  └─ augmentation_pipeline                     │     │
│  └────────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               NVIDIA DALI Library                        │
│  (GPU-accelerated data loading and preprocessing)       │
└──────────────────────────────────────────────────────────┘
```

## 故障排查

### 问题 1: MCP SDK 导入错误

**错误信息**：
```
ModuleNotFoundError: No module named 'mcp'
```

**解决方案**：
```bash
pip install mcp
```

### 问题 2: DALI 未安装或版本不兼容

**错误信息**：
```
ModuleNotFoundError: No module named 'nvidia.dali'
```

**解决方案**：
```bash
pip install nvidia-dali-cuda120
# 或根据你的 CUDA 版本选择
pip install nvidia-dali-cuda118
```

### 问题 3: GPU 不可用

**错误信息**：
```
RuntimeError: CUDA not available
```

**解决方案**：
服务器会自动使用 CPU 模式，但性能会降低。确保：
1. 安装了正确的 CUDA 版本
2. GPU 驱动正常工作
3. `nvidia-smi` 命令可用

### 问题 4: Pipeline 构建失败

**错误信息**：
```
RuntimeError: Critical error when building pipeline
```

**解决方案**：
1. 检查数据集是否正确创建
2. 确认文件列表不为空
3. 检查 GPU 内存是否足够
4. 尝试减小 batch_size

## 扩展开发

### 添加新的 Pipeline 类型

1. **定义 Pipeline 函数**：

```python
@pipeline_def
def custom_pipeline(file_list, **kwargs):
    images, labels = fn.readers.file(files=file_list)
    # 你的自定义处理
    images = fn.your_custom_operation(images)
    return images, labels
```

2. **在 create_pipeline 中注册**：

```python
async def handle_create_pipeline(arguments: Dict[str, Any]):
    # ...
    elif pipeline_type == "custom":
        pipe = custom_pipeline(
            file_list=file_list,
            **custom_kwargs
        )
    # ...
```

3. **更新工具描述**：

在 `list_tools()` 中添加新的 pipeline_type 到 enum。

### 添加新工具

1. **定义工具描述**：

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... 现有工具
        Tool(
            name="your_new_tool",
            description="工具描述",
            inputSchema={
                # JSON Schema
            }
        )
    ]
```

2. **实现处理函数**：

```python
async def handle_your_new_tool(arguments: Dict[str, Any]):
    # 实现逻辑
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]
```

3. **在 call_tool 中注册**：

```python
@app.call_tool()
async def call_tool(name: str, arguments: Any):
    if name == "your_new_tool":
        return await handle_your_new_tool(arguments)
    # ...
```

## 性能优化建议

1. **批次大小**：根据 GPU 内存调整 batch_size
2. **线程数**：`num_threads` 设置为 CPU 核心数的 2-4 倍
3. **预取**：DALI 会自动预取数据，无需手动配置
4. **混合设备**：使用 `device="mixed"` 进行 GPU 解码

## 贡献指南

欢迎贡献新功能！请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License

## 相关链接

- [NVIDIA DALI 官方文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## 更新日志

### v0.1.0 (2026-01-13)

- ✨ 初始版本发布
- ✅ 支持数据集创建
- ✅ 基础和增强 Pipeline
- ✅ 统计信息输出
- ✅ Claude Desktop 集成
