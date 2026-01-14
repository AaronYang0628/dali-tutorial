# DALI MCP Server - 快速开始指南

## ✅ 快速验证

首先验证服务器是否可以正常工作：

```bash
cd /workspaces/dali-tutorial/mcp
python test_server.py
```

如果所有测试都通过了，你可以继续使用服务器。

## 📦 安装依赖

如果尚未安装依赖：

```bash
pip install -r requirements.txt
```

## 🚀 三种使用方式

### 方式 1: 命令行快速体验（推荐新手）

最简单的方式，运行完整示例：

```bash
cd /workspaces/dali-tutorial/mcp
python example_client.py
```

**输出示例**：
```
============================================================
DALI MCP Server 使用示例
============================================================

📋 步骤 1: 列出可用工具
------------------------------------------------------------
可用工具数量: 5
  - create_test_dataset: 创建测试图像数据集
  - create_pipeline: 创建 DALI 数据处理 Pipeline
  - run_pipeline: 运行 DALI Pipeline 并获取处理结果统计
  - list_datasets: 列出所有已创建的数据集
  - list_pipelines: 列出所有已创建的 Pipeline

📸 步骤 2: 创建测试数据集
------------------------------------------------------------
{
  "dataset_name": "my_dataset",
  "dataset_path": "/tmp/dali_dataset_my_dataset_xxx",
  "num_files": 20,
  "image_size": 256,
  "file_list": [...]
}

... [更多输出] ...
```

### 方式 2: Python 脚本使用（推荐开发者）

在自己的 Python 脚本中使用服务器：

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["/workspaces/dali-tutorial/mcp/dali_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 创建数据集
            print("创建数据集...")
            result = await session.call_tool(
                "create_test_dataset",
                arguments={
                    "name": "my_data",
                    "num_images": 100,
                    "image_size": 256
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 创建 Pipeline
            print("\n创建 Pipeline...")
            result = await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "my_pipe",
                    "dataset_name": "my_data",
                    "pipeline_type": "basic",
                    "batch_size": 8,
                    "target_size": 224
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 运行 Pipeline
            print("\n运行 Pipeline...")
            result = await session.call_tool(
                "run_pipeline",
                arguments={
                    "pipeline_name": "my_pipe",
                    "num_iterations": 3
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

asyncio.run(main())
```

保存为 `my_script.py`，运行：
```bash
python my_script.py
```

### 方式 3: Claude Desktop 集成（推荐 AI 交互）

将 MCP 服务器集成到 Claude Desktop，使用自然语言与 DALI 交互。

#### 安装步骤

**macOS**：
```bash
# 1. 打开 Claude Desktop 配置文件
open ~/Library/Application\ Support/Claude/claude_desktop_config.json

# 2. 追加以下内容（保留 JSON 格式）
{
  "mcpServers": {
    "dali-server": {
      "command": "python",
      "args": [
        "/workspaces/dali-tutorial/mcp/dali_mcp_server.py"
      ]
    }
  }
}

# 3. 重启 Claude Desktop
```

**Windows**：
```bash
# 1. 打开配置文件
notepad %APPDATA%\Claude\claude_desktop_config.json

# 2. 追加 dali-server 配置
# 3. 重启 Claude Desktop
```

**Linux**：
```bash
# 1. 打开配置文件
nano ~/.config/Claude/claude_desktop_config.json

# 2. 追加 dali-server 配置
# 3. 重启 Claude Desktop
```

#### 在 Claude Desktop 中使用

重启后，在 Claude 的对话框中就可以使用 DALI 工具了。例如：

**例子 1: 基础使用**
```
User: 帮我创建一个包含 50 张图片的数据集

Claude: 我来帮你创建这个数据集。
[调用 create_test_dataset 工具]
{
  "dataset_name": "dataset",
  "num_files": 50,
  "image_size": 256,
  ...
}
```

**例子 2: 完整工作流**
```
User: 我需要：
1. 创建 1000 张图像的训练数据集
2. 创建一个数据增强 pipeline，batch size 32
3. 运行 pipeline 5 次看看效果

Claude: 我来帮你完成这个工作流...
[依次调用对应工具]
```

**例子 3: 数据分析**
```
User: 创建 500 张图像的数据集，然后运行一个 augmentation pipeline，
      告诉我处理后图像的统计特性

Claude: 好的，让我创建数据集和 pipeline，然后运行它...
[调用工具并分析结果]
```

## 📝 常见使用场景

### 场景 1: 测试 DALI 配置

```python
async def test_dali_setup():
    # 创建小型数据集
    dataset = await session.call_tool(
        "create_test_dataset",
        arguments={"name": "test", "num_images": 10}
    )

    # 创建 pipeline
    pipe = await session.call_tool(
        "create_pipeline",
        arguments={
            "name": "test_pipe",
            "dataset_name": "test",
            "pipeline_type": "basic"
        }
    )

    # 快速运行
    results = await session.call_tool(
        "run_pipeline",
        arguments={"pipeline_name": "test_pipe"}
    )
```

### 场景 2: 性能基准测试

```python
async def benchmark():
    # 创建大数据集
    await session.call_tool(
        "create_test_dataset",
        arguments={"name": "bench", "num_images": 10000}
    )

    # 创建不同 batch size 的 pipeline
    for bs in [8, 16, 32, 64]:
        await session.call_tool(
            "create_pipeline",
            arguments={
                "name": f"pipe_bs{bs}",
                "dataset_name": "bench",
                "batch_size": bs
            }
        )

        # 运行并比较性能
        results = await session.call_tool(
            "run_pipeline",
            arguments={
                "pipeline_name": f"pipe_bs{bs}",
                "num_iterations": 100
            }
        )
```

### 场景 3: 导入本地数据集

```python
async def import_local_data():
    # 1. 从本地目录导入真实数据
    dataset = await session.call_tool(
        "import_local_dataset",
        arguments={
            "dataset_name": "my_photos",
            "local_path": "/home/user/dataset/photos",
            "supported_formats": ["jpg", "png"]
        }
    )
    print(f"导入 {dataset['num_files']} 张图像")

    # 2. 创建处理 pipeline
    pipe = await session.call_tool(
        "create_pipeline",
        arguments={
            "name": "photo_pipe",
            "dataset_name": "my_photos",
            "pipeline_type": "augmentation",
            "batch_size": 16
        }
    )

    # 3. 运行并获取结果
    results = await session.call_tool(
        "run_pipeline",
        arguments={"pipeline_name": "photo_pipe", "num_iterations": 5}
    )
```

### 场景 4: 导入 S3 数据集

```python
async def import_s3_data():
    # 1. 从 AWS S3 导入数据（下载到本地）
    import os

    # 设置凭证（或使用环境变量）
    os.environ["AWS_ACCESS_KEY_ID"] = "your_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"

    dataset = await session.call_tool(
        "import_s3_dataset",
        arguments={
            "dataset_name": "s3_training",
            "s3_uri": "s3://my-bucket/training-data",
            "download": True,  # 下载到本地
            "supported_formats": ["jpg", "png"]
        }
    )
    print(f"下载了 {dataset['num_files']} 张图像到 {dataset['local_path']}")

    # 2. 创建处理 pipeline
    pipe = await session.call_tool(
        "create_pipeline",
        arguments={
            "name": "s3_pipe",
            "dataset_name": "s3_training",
            "pipeline_type": "augmentation",
            "batch_size": 32
        }
    )

    # 3. 运行 pipeline
    results = await session.call_tool(
        "run_pipeline",
        arguments={"pipeline_name": "s3_pipe", "num_iterations": 10}
    )
```

### 场景 5: MinIO 私有存储集成

```python
async def import_minio_data():
    # 从 MinIO 导入数据
    dataset = await session.call_tool(
        "import_s3_dataset",
        arguments={
            "dataset_name": "minio_data",
            "s3_uri": "s3://private-bucket/images",
            "endpoint_url": "http://minio-server:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            "download": True
        }
    )
    print(f"从 MinIO 导入 {dataset['num_files']} 张图像")
```

### 场景 6: 数据预处理工作流

```python
async def prepare_data():
    # 1. 从本地导入训练集
    train = await session.call_tool(
        "import_local_dataset",
        arguments={
            "dataset_name": "train",
            "local_path": "/data/train"
        }
    )

    # 2. 从 S3 导入验证集
    val = await session.call_tool(
        "import_s3_dataset",
        arguments={
            "dataset_name": "val",
            "s3_uri": "s3://my-bucket/validation",
            "download": True
        }
    )

    # 3. 创建训练 pipeline（数据增强）
    train_pipe = await session.call_tool(
        "create_pipeline",
        arguments={
            "name": "train_pipe",
            "dataset_name": "train",
            "pipeline_type": "augmentation",
            "batch_size": 32
        }
    )

    # 4. 创建验证 pipeline（基础处理）
    val_pipe = await session.call_tool(
        "create_pipeline",
        arguments={
            "name": "val_pipe",
            "dataset_name": "val",
            "pipeline_type": "basic",
            "batch_size": 64
        }
    )

    # 5. 获取统计信息
    stats = await session.call_tool(
        "run_pipeline",
        arguments={"pipeline_name": "train_pipe"}
    )
```

## 🔧 工具参考

### 7 个核心工具

| 工具名 | 功能 | 用途 |
|-------|------|------|
| `create_test_dataset` | 生成测试图像 | 快速创建数据集 |
| `import_local_dataset` | 导入本地数据 | 导入自己的图像数据 |
| `import_s3_dataset` | 导入 S3 数据 | 从 AWS S3 或 MinIO 导入数据 |
| `create_pipeline` | 创建 DALI Pipeline | 配置数据处理流程 |
| `run_pipeline` | 运行 Pipeline | 执行数据处理和获取统计 |
| `list_datasets` | 列出数据集 | 查看已创建的数据集 |
| `list_pipelines` | 列出 Pipeline | 查看已创建的 Pipeline |

## 📊 理解返回数据

### create_test_dataset 返回值

```json
{
  "dataset_name": "my_dataset",      // 数据集名称
  "dataset_path": "/tmp/...",        // 物理路径
  "num_files": 20,                   // 文件数量
  "image_size": 256,                 // 图像尺寸
  "file_list": ["img1.jpg", ...]     // 文件列表（示例）
}
```

### run_pipeline 返回值（关键部分）

```json
{
  "pipeline_name": "my_pipe",
  "statistics": {
    "batches": [
      {
        "iteration": 1,
        "batch_size": 32,
        "sample_stats": {
          "min": 0.0,                // 最小像素值
          "max": 1.0,                // 最大像素值
          "mean": 0.48,              // 平均像素值
          "std": 0.25                // 标准差
        }
      }
    ]
  }
}
```

## 🎯 下一步

### 了解更多

- 📖 查看完整 README: `README.md`
- 🔧 学习如何扩展: README.md 中的 "扩展开发" 章节
- 📝 查看源代码: `dali_mcp_server.py`

### 尝试更多功能

1. 修改 `example_client.py` 进行实验
2. 创建自己的 pipeline
3. 集成到你的项目中

### 获取帮助

如果遇到问题：
1. 运行 `test_server.py` 检查环境
2. 查看错误信息的详细描述
3. 检查 README.md 中的 "故障排查" 部分

## 🚦 状态指示

| 符号 | 含义 |
|------|------|
| ✅ | 成功 |
| ❌ | 失败 |
| ⚠️ | 警告 |
| 📸 | 数据相关 |
| 🔧 | Pipeline 相关 |
| ▶️ | 执行 |
| 📊 | 统计/列表 |

## 💡 Tips

1. **保存数据集名称**：创建数据集后记住名称，后续需要用到
2. **Pipeline 复用**：同一个 pipeline 可以多次运行，无需重复创建
3. **批次大小选择**：
   - 小数据集（<100）：batch_size = 4-8
   - 中数据集（100-10K）：batch_size = 16-32
   - 大数据集（>10K）：batch_size = 64-128
4. **内存管理**：数据集和 pipeline 信息存储在内存中，程序退出后自动清理

---

祝你使用愉快！🎉
