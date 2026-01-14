#!/usr/bin/env python3
"""
快速测试 DALI MCP Server 是否正常工作
"""

import subprocess
import sys
import time


def test_imports():
    """测试必要的依赖是否安装"""
    print("=" * 60)
    print("测试 1: 检查依赖")
    print("=" * 60)

    required_modules = [
        ("mcp", "MCP SDK"),
        ("nvidia.dali", "NVIDIA DALI"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow")
    ]

    all_ok = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name:20s} - 已安装")
        except ImportError:
            print(f"❌ {name:20s} - 未安装")
            all_ok = False

    return all_ok


def test_server_startup():
    """测试服务器是否可以启动"""
    print("\n" + "=" * 60)
    print("测试 2: 服务器启动")
    print("=" * 60)

    try:
        # 尝试导入服务器模块
        import dali_mcp_server
        print("✅ 服务器模块加载成功")
        return True
    except Exception as e:
        print(f"❌ 服务器模块加载失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试 3: 基本功能")
    print("=" * 60)

    try:
        import nvidia.dali as dali
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        import numpy as np
        from PIL import Image
        import tempfile
        import os
        import glob

        # 创建临时目录和测试图像
        with tempfile.TemporaryDirectory() as tmpdir:
            # 生成测试图像
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            img.save(os.path.join(tmpdir, "test.jpg"))

            # 获取文件列表
            file_list = glob.glob(os.path.join(tmpdir, "*.jpg"))

            # 定义简单 Pipeline
            @pipeline_def
            def test_pipeline(file_list):
                images, labels = fn.readers.file(files=file_list)
                images = fn.decoders.image(images, device="mixed")
                images = fn.resize(images, size=32)
                return images, labels

            # 构建并运行
            pipe = test_pipeline(file_list=file_list, batch_size=1, num_threads=1, device_id=0)
            pipe.build()
            outputs = pipe.run()

            if len(outputs) >= 1:
                print("✅ DALI Pipeline 运行成功")
                print(f"   输出形状: {outputs[0].shape()}")
                return True
            else:
                print("❌ DALI Pipeline 输出为空")
                return False

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DALI MCP Server - 快速测试")
    print("="*60 + "\n")

    results = []

    # 测试 1: 依赖检查
    results.append(("依赖检查", test_imports()))

    # 测试 2: 服务器启动
    results.append(("服务器启动", test_server_startup()))

    # 测试 3: 基本功能
    results.append(("基本功能", test_basic_functionality()))

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过！服务器可以正常使用。")
        print("\n下一步:")
        print("  python example_client.py")
    else:
        print("❌ 部分测试失败。请检查依赖安装。")
        print("\n安装缺失的依赖:")
        print("  pip install -r requirements.txt")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
