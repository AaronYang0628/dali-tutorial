#!/usr/bin/env python3
"""
测试本地数据集导入功能
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_local_import():
    """测试本地导入功能"""

    server_params = StdioServerParameters(
        command="python",
        args=["/workspaces/dali-tutorial/mcp/scripts/dali_mcp_server.py"]
    )

    print("=" * 70)
    print("测试本地数据集导入功能")
    print("=" * 70)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 测试 1: 导入本地数据集
            print("\n✅ 测试 1: 导入本地数据集")
            print("-" * 70)
            result = await session.call_tool(
                "import_local_dataset",
                arguments={
                    "dataset_name": "local_test",
                    "local_path": "/tmp/test_images",
                    "supported_formats": ["jpg", "jpeg", "png"]
                }
            )
            import_result = json.loads(result.content[0].text)
            print(json.dumps(import_result, indent=2))

            if "error" in import_result:
                print("❌ 导入失败！")
                return

            print(f"✅ 成功导入 {import_result['num_files']} 张图像")

            # 测试 2: 使用导入的数据集创建 Pipeline
            print("\n✅ 测试 2: 使用导入的数据集创建 Pipeline")
            print("-" * 70)
            result = await session.call_tool(
                "create_pipeline",
                arguments={
                    "name": "local_basic",
                    "dataset_name": "local_test",
                    "pipeline_type": "basic",
                    "batch_size": 2,
                    "target_size": 128
                }
            )
            pipeline_result = json.loads(result.content[0].text)
            print(json.dumps(pipeline_result, indent=2))

            if "error" in pipeline_result:
                print("❌ Pipeline 创建失败！")
                return

            print(f"✅ 成功创建 Pipeline")

            # 测试 3: 运行 Pipeline
            print("\n✅ 测试 3: 运行 Pipeline")
            print("-" * 70)
            result = await session.call_tool(
                "run_pipeline",
                arguments={
                    "pipeline_name": "local_basic",
                    "num_iterations": 1
                }
            )
            run_result = json.loads(result.content[0].text)
            print(json.dumps(run_result, indent=2))

            if "error" in run_result:
                print("❌ Pipeline 执行失败！")
                return

            print(f"✅ 成功运行 Pipeline")

            # 测试 4: 列出数据集
            print("\n✅ 测试 4: 列出所有数据集")
            print("-" * 70)
            result = await session.call_tool("list_datasets", arguments={})
            datasets_result = json.loads(result.content[0].text)
            print(json.dumps(datasets_result, indent=2))

            # 测试 5: 错误处理 - 重复导入
            print("\n✅ 测试 5: 错误处理 - 重复导入相同名称")
            print("-" * 70)
            result = await session.call_tool(
                "import_local_dataset",
                arguments={
                    "dataset_name": "local_test",
                    "local_path": "/tmp/test_images"
                }
            )
            error_result = json.loads(result.content[0].text)
            print(json.dumps(error_result, indent=2))
            if "error" in error_result and "already exists" in error_result["error"]:
                print("✅ 正确检测到重复数据集名称")
            else:
                print("❌ 未正确处理重复名称")

            # 测试 6: 错误处理 - 无效路径
            print("\n✅ 测试 6: 错误处理 - 无效路径")
            print("-" * 70)
            result = await session.call_tool(
                "import_local_dataset",
                arguments={
                    "dataset_name": "invalid",
                    "local_path": "/nonexistent/path"
                }
            )
            error_result = json.loads(result.content[0].text)
            print(json.dumps(error_result, indent=2))
            if "error" in error_result and "does not exist" in error_result["error"]:
                print("✅ 正确检测到无效路径")
            else:
                print("❌ 未正确处理无效路径")

            print("\n" + "=" * 70)
            print("✅ 所有测试完成！")
            print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_local_import())
