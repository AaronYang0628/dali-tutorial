#!/usr/bin/env python3
"""
测试 DALI HTTP API Server
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_response(response):
    """打印响应"""
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")


def test_health():
    """测试健康检查"""
    print_section("测试 1: 健康检查")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print_response(response)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器。请确保服务器正在运行。")
        print(f"   启动命令: python dali_http_server.py")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_root():
    """测试根端点"""
    print_section("测试 2: 根端点 (端点列表)")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_create_dataset():
    """测试创建数据集"""
    print_section("测试 3: 创建测试数据集")
    try:
        response = requests.post(
            f"{BASE_URL}/api/dataset/create",
            json={
                "name": "test_dataset",
                "num_images": 10,
                "image_size": 128
            },
            timeout=30
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_list_datasets():
    """测试列出数据集"""
    print_section("测试 4: 列出所有数据集")
    try:
        response = requests.get(f"{BASE_URL}/api/dataset/list", timeout=5)
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_create_pipeline():
    """测试创建Pipeline"""
    print_section("测试 5: 创建基础 Pipeline")
    try:
        response = requests.post(
            f"{BASE_URL}/api/pipeline/create",
            json={
                "name": "basic_pipeline",
                "dataset_name": "test_dataset",
                "pipeline_type": "basic",
                "batch_size": 4,
                "target_size": 224
            },
            timeout=30
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_run_pipeline():
    """测试运行Pipeline"""
    print_section("测试 6: 运行 Pipeline")
    try:
        response = requests.post(
            f"{BASE_URL}/api/pipeline/run",
            json={
                "pipeline_name": "basic_pipeline",
                "num_iterations": 2
            },
            timeout=30
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_list_pipelines():
    """测试列出Pipeline"""
    print_section("测试 7: 列出所有 Pipeline")
    try:
        response = requests.get(f"{BASE_URL}/api/pipeline/list", timeout=5)
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_augmentation_pipeline():
    """测试数据增强Pipeline"""
    print_section("测试 8: 创建数据增强 Pipeline")
    try:
        response = requests.post(
            f"{BASE_URL}/api/pipeline/create",
            json={
                "name": "aug_pipeline",
                "dataset_name": "test_dataset",
                "pipeline_type": "augmentation",
                "batch_size": 8,
                "target_size": 224
            },
            timeout=30
        )
        print_response(response)

        if response.status_code == 200:
            # 运行增强Pipeline
            print("\n运行数据增强 Pipeline...")
            response2 = requests.post(
                f"{BASE_URL}/api/pipeline/run",
                json={
                    "pipeline_name": "aug_pipeline",
                    "num_iterations": 1
                },
                timeout=30
            )
            print_response(response2)
            return response2.status_code == 200
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print_section("测试 9: 错误处理")

    # 测试重复数据集名称
    print("\n9.1: 测试重复数据集名称")
    try:
        response = requests.post(
            f"{BASE_URL}/api/dataset/create",
            json={
                "name": "test_dataset",  # 已存在
                "num_images": 5,
                "image_size": 128
            },
            timeout=10
        )
        print_response(response)
        duplicate_ok = response.status_code == 409
        print(f"{'✅' if duplicate_ok else '❌'} 重复名称检测: {'通过' if duplicate_ok else '失败'}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        duplicate_ok = False

    # 测试不存在的数据集
    print("\n9.2: 测试使用不存在的数据集创建Pipeline")
    try:
        response = requests.post(
            f"{BASE_URL}/api/pipeline/create",
            json={
                "name": "invalid_pipeline",
                "dataset_name": "nonexistent_dataset",
                "pipeline_type": "basic",
                "batch_size": 4
            },
            timeout=10
        )
        print_response(response)
        notfound_ok = response.status_code == 404
        print(f"{'✅' if notfound_ok else '❌'} 不存在数据集检测: {'通过' if notfound_ok else '失败'}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        notfound_ok = False

    return duplicate_ok and notfound_ok


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("  DALI HTTP API Server - 测试套件")
    print("="*70)
    print(f"\n服务器地址: {BASE_URL}")
    print("确保服务器正在运行: python dali_http_server.py\n")

    time.sleep(1)

    # 运行测试
    tests = [
        ("健康检查", test_health),
        ("根端点", test_root),
        ("创建数据集", test_create_dataset),
        ("列出数据集", test_list_datasets),
        ("创建Pipeline", test_create_pipeline),
        ("运行Pipeline", test_run_pipeline),
        ("列出Pipeline", test_list_pipelines),
        ("数据增强Pipeline", test_augmentation_pipeline),
        ("错误处理", test_error_handling),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if not result:
                # 如果是连接失败，停止后续测试
                if name == "健康检查":
                    break
        except KeyboardInterrupt:
            print("\n\n测试中断")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            results.append((name, False))

    # 汇总结果
    print_section("测试结果汇总")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有测试通过！HTTP API Server 正常工作。")
        print("\n下一步:")
        print("  1. 查看 N8N 集成文档: N8N_INTEGRATION.md")
        print("  2. 访问 API 文档: http://localhost:8000/docs")
        print("  3. 在 N8N 中配置 HTTP Request 节点")
    else:
        print("❌ 部分测试失败。请检查服务器日志。")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
