#!/usr/bin/env python3
"""
DALI MCP Server 使用示例

演示如何通过 MCP 客户端调用 DALI 服务器
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_example():
    """运行完整的示例流程"""

    # 服务器参数
    server_params = StdioServerParameters(
        command="python",
        args=["/workspaces/dali-tutorial/mcp/scripts/dali_mcp_server.py"],
        env=None
    )

    print("=" * 60)
    print("DALI MCP Server 使用示例")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 初始化会话
            await session.initialize()

            # 1. 列出可用工具
            print("\n📋 步骤 1: 列出可用工具")
            print("-" * 60)
            tools = await session.list_tools()
            print(f"可用工具数量: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description.split(chr(10))[0]}")

            # 2. 创建测试数据集
            print("\n📸 步骤 2: 创建测试数据集")
            print("-" * 60)
            result = await session.call_tool(
                "create_test_dataset",
                arguments={
                    "name": "my_dataset",
                    "num_images": 20,
                    "image_size": 256
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 3. 创建基础 Pipeline
            print("\n🔧 步骤 3: 创建基础图像处理 Pipeline")
            print("-" * 60)
            result = await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "basic_pipe",
                    "dataset_name": "my_dataset",
                    "pipeline_type": "basic",
                    "batch_size": 4,
                    "target_size": 224
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 4. 运行基础 Pipeline
            print("\n▶️  步骤 4: 运行基础 Pipeline")
            print("-" * 60)
            result = await session.call_tool(
                "run_pipeline",
                arguments={
                    "pipeline_name": "basic_pipe",
                    "num_iterations": 2
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 5. 创建数据增强 Pipeline
            print("\n🎨 步骤 5: 创建数据增强 Pipeline")
            print("-" * 60)
            result = await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "aug_pipe",
                    "dataset_name": "my_dataset",
                    "pipeline_type": "augmentation",
                    "batch_size": 8,
                    "target_size": 224
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 6. 运行数据增强 Pipeline
            print("\n▶️  步骤 6: 运行数据增强 Pipeline")
            print("-" * 60)
            result = await session.call_tool(
                "run_pipeline",
                arguments={
                    "pipeline_name": "aug_pipe",
                    "num_iterations": 3
                }
            )
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 7. 列出所有数据集
            print("\n📊 步骤 7: 列出所有数据集")
            print("-" * 60)
            result = await session.call_tool("list_datasets", arguments={})
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 8. 列出所有 Pipeline
            print("\n📊 步骤 8: 列出所有 Pipeline")
            print("-" * 60)
            result = await session.call_tool("list_pipelines", arguments={})
            print(json.dumps(json.loads(result.content[0].text), indent=2))

            # 9. 本地数据集导入示例
            print("\n📁 步骤 9: 导入本地数据集")
            print("-" * 60)
            print("💡 演示：从本地目录导入真实的图像数据集")
            print("   使用方式：")
            print("   await session.call_tool(")
            print('       "import_local_dataset",')
            print('       arguments={')
            print('           "dataset_name": "my_local_data",')
            print('           "local_path": "/path/to/your/images",')
            print('           "supported_formats": ["jpg", "jpeg", "png"]')
            print("       }")
            print("   )")
            print()
            print("   这将：")
            print("   1. 扫描指定目录下的所有jpg/jpeg/png图像")
            print("   2. 注册为名为'my_local_data'的数据集")
            print("   3. 可用于创建Pipeline处理")

            # 10. S3 数据集导入示例
            print("\n☁️  步骤 10: 导入 S3 数据集（示例）")
            print("-" * 60)
            print("💡 演示：从 S3 兼容存储导入图像数据集")
            print("   使用方式（AWS S3）:")
            print("   await session.call_tool(")
            print('       "import_s3_dataset",')
            print('       arguments={')
            print('           "dataset_name": "s3_dataset",')
            print('           "s3_uri": "s3://my-bucket/images",')
            print('           "download": True,  # 下载到本地')
            print('           "supported_formats": ["jpg", "png"]')
            print("       }")
            print("   )")
            print()
            print("   使用方式（MinIO 兼容）:")
            print("   await session.call_tool(")
            print('       "import_s3_dataset",')
            print('       arguments={')
            print('           "dataset_name": "minio_dataset",')
            print('           "s3_uri": "s3://my-bucket/images",')
            print('           "endpoint_url": "http://minio-server:9000",')
            print('           "access_key": "minioadmin",')
            print('           "secret_key": "minioadmin",')
            print('           "download": True')
            print("       }")
            print("   )")
            print()
            print("   特点：")
            print("   1. 支持 AWS S3 和 MinIO 等兼容存储")
            print("   2. 凭证优先从环境变量读取（AWS_ACCESS_KEY_ID等）")
            print("   3. 可选择下载到本地或仅列举文件")
            print("   4. 下载的文件自动保存到临时目录")

            print("\n" + "=" * 60)
            print("✅ 示例完成！")
            print("=" * 60)
            print()
            print("📖 更多信息请查看:")
            print("   - README.md 中的「可用工具」部分")
            print("   - QUICKSTART.md 中的「常见场景」部分")


if __name__ == "__main__":
    asyncio.run(run_example())
