#!/usr/bin/env python3
"""
DALI Agent - Natural Language Data Processing Agent

This agent understands natural language requests and automatically
calls the appropriate DALI HTTP API endpoints to configure data processing.

Usage:
    python dali_agent.py

Then interact in natural language:
    > 我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转
"""

import json
import re
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import requests


# ============================================================
# Configuration
# ============================================================

DALI_API_BASE = os.environ.get("DALI_API_BASE", "http://localhost:8000")
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 224
DEFAULT_SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]


# ============================================================
# DALI API Client
# ============================================================

class DALIClient:
    """Client for DALI HTTP API"""

    def __init__(self, base_url: str = DALI_API_BASE):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict[str, Any]:
        """Check if API is available"""
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def create_dataset(self, name: str, num_images: int, image_size: int) -> Dict[str, Any]:
        """Create test dataset"""
        response = requests.post(
            f"{self.base_url}/api/dataset/create",
            json={
                "name": name,
                "num_images": num_images,
                "image_size": image_size
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def import_local_dataset(
        self,
        dataset_name: str,
        local_path: str,
        supported_formats: List[str] = None
    ) -> Dict[str, Any]:
        """Import local dataset"""
        if supported_formats is None:
            supported_formats = DEFAULT_SUPPORTED_FORMATS

        response = requests.post(
            f"{self.base_url}/api/dataset/import/local",
            json={
                "dataset_name": dataset_name,
                "local_path": local_path,
                "supported_formats": supported_formats
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def import_s3_dataset(
        self,
        dataset_name: str,
        s3_uri: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        download: bool = True,
        supported_formats: List[str] = None
    ) -> Dict[str, Any]:
        """Import S3 dataset"""
        if supported_formats is None:
            supported_formats = DEFAULT_SUPPORTED_FORMATS

        payload = {
            "dataset_name": dataset_name,
            "s3_uri": s3_uri,
            "download": download,
            "supported_formats": supported_formats
        }

        if endpoint_url:
            payload["endpoint_url"] = endpoint_url
        if access_key:
            payload["access_key"] = access_key
        if secret_key:
            payload["secret_key"] = secret_key

        response = requests.post(
            f"{self.base_url}/api/dataset/import/s3",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def create_pipeline(
        self,
        name: str,
        dataset_name: str,
        pipeline_type: str,
        batch_size: int,
        target_size: int
    ) -> Dict[str, Any]:
        """Create pipeline"""
        response = requests.post(
            f"{self.base_url}/api/pipeline/create",
            json={
                "name": name,
                "dataset_name": dataset_name,
                "pipeline_type": pipeline_type,
                "batch_size": batch_size,
                "target_size": target_size
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def run_pipeline(
        self,
        pipeline_name: str,
        num_iterations: int = 1
    ) -> Dict[str, Any]:
        """Run pipeline"""
        response = requests.post(
            f"{self.base_url}/api/pipeline/run",
            json={
                "pipeline_name": pipeline_name,
                "num_iterations": num_iterations
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def list_datasets(self) -> Dict[str, Any]:
        """List all datasets"""
        response = requests.get(f"{self.base_url}/api/dataset/list", timeout=5)
        response.raise_for_status()
        return response.json()

    def list_pipelines(self) -> Dict[str, Any]:
        """List all pipelines"""
        response = requests.get(f"{self.base_url}/api/pipeline/list", timeout=5)
        response.raise_for_status()
        return response.json()


# ============================================================
# Natural Language Parser
# ============================================================

class NLParser:
    """Parse natural language requests into structured parameters"""

    # Keywords for augmentation detection
    AUGMENTATION_KEYWORDS = [
        # Chinese
        "增强", "裁剪", "翻转", "旋转", "亮度", "对比度",
        "随机", "数据增强",
        # English
        "augment", "augmentation", "crop", "flip", "rotate",
        "brightness", "contrast", "random"
    ]

    # Keywords for basic processing
    BASIC_KEYWORDS = [
        # Chinese
        "基础", "简单", "仅", "只", "不需要增强",
        # English
        "basic", "simple", "only", "just", "no augment"
    ]

    @staticmethod
    def extract_path(text: str) -> Optional[str]:
        """Extract file path from text"""
        # Match paths like /data/imagenet, /path/to/data, etc.
        patterns = [
            r'(?:数据在|data at|path|from)\s+([/\w\-_.]+)',
            r'([/]\w+[/\w\-_.]*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                path = match.group(1)
                # Clean up path
                path = path.rstrip('，。,.')
                return path

        return None

    @staticmethod
    def extract_s3_uri(text: str) -> Optional[str]:
        """Extract S3 URI from text"""
        match = re.search(r's3://[\w\-./]+', text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def extract_batch_size(text: str) -> int:
        """Extract batch size from text"""
        patterns = [
            r'批次[大小]*[：:]*\s*(\d+)',
            r'batch\s*(?:size)?[：:]*\s*(\d+)',
            r'批[：:]*\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return DEFAULT_BATCH_SIZE

    @staticmethod
    def extract_image_size(text: str) -> int:
        """Extract image size from text"""
        patterns = [
            r'(?:图像)?尺寸[：:]*\s*(\d+)',
            r'(?:image)?\s*size[：:]*\s*(\d+)',
            r'(\d+)x\1',  # 224x224
            r'(\d+)\s*[×x]\s*\1',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return DEFAULT_IMAGE_SIZE

    @staticmethod
    def extract_num_images(text: str) -> Optional[int]:
        """Extract number of test images to create"""
        patterns = [
            r'(\d+)\s*张',
            r'(\d+)\s*(?:images?|pics?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    @staticmethod
    def detect_pipeline_type(text: str) -> str:
        """Detect if user wants basic or augmentation pipeline"""
        text_lower = text.lower()

        # Check for explicit basic keywords first
        if any(keyword in text_lower for keyword in NLParser.BASIC_KEYWORDS):
            return "basic"

        # Check for augmentation keywords
        if any(keyword in text_lower for keyword in NLParser.AUGMENTATION_KEYWORDS):
            return "augmentation"

        # Default to basic
        return "basic"

    @staticmethod
    def detect_data_source(text: str) -> Tuple[str, Optional[str]]:
        """
        Detect data source type and location

        Returns:
            (source_type, location)
            source_type: "local", "s3", "test"
            location: path, s3_uri, or None for test
        """
        # Check for S3
        s3_uri = NLParser.extract_s3_uri(text)
        if s3_uri:
            return ("s3", s3_uri)

        # Check for local path
        local_path = NLParser.extract_path(text)
        if local_path:
            return ("local", local_path)

        # Check for test data keywords
        test_keywords = ["测试", "test", "synthetic", "生成"]
        if any(keyword in text.lower() for keyword in test_keywords):
            return ("test", None)

        # Default to local with no path specified
        return ("local", None)

    @staticmethod
    def parse_request(text: str) -> Dict[str, Any]:
        """Parse complete request into parameters"""
        source_type, location = NLParser.detect_data_source(text)

        params = {
            "source_type": source_type,
            "location": location,
            "batch_size": NLParser.extract_batch_size(text),
            "image_size": NLParser.extract_image_size(text),
            "pipeline_type": NLParser.detect_pipeline_type(text),
            "num_images": NLParser.extract_num_images(text),
        }

        return params


# ============================================================
# DALI Agent
# ============================================================

class DALIAgent:
    """Main agent that orchestrates API calls based on natural language"""

    def __init__(self, api_base: str = DALI_API_BASE):
        self.client = DALIClient(api_base)
        self.parser = NLParser()

    def check_api_availability(self) -> bool:
        """Check if DALI API is available"""
        try:
            self.client.health_check()
            return True
        except Exception as e:
            print(f"❌ 无法连接到 DALI API 服务器: {e}")
            print(f"   请确保服务器运行在: {self.client.base_url}")
            print(f"   启动命令: python dali_http_server.py")
            return False

    def generate_dataset_name(self, location: Optional[str]) -> str:
        """Generate meaningful dataset name"""
        if location:
            if location.startswith("s3://"):
                # Extract bucket and prefix
                parts = location.replace("s3://", "").split("/")
                return f"s3_{parts[0]}"
            else:
                # Extract directory name
                name = os.path.basename(location.rstrip("/"))
                return f"{name}_dataset" if name else "local_dataset"
        else:
            return "test_dataset"

    def generate_pipeline_name(
        self,
        dataset_name: str,
        pipeline_type: str,
        batch_size: int
    ) -> str:
        """Generate meaningful pipeline name"""
        return f"{dataset_name}_{pipeline_type}_{batch_size}"

    def process_request(self, user_input: str) -> None:
        """Process user request and execute workflow"""

        print("\n" + "="*70)
        print("  正在分析您的需求...")
        print("="*70 + "\n")

        # Parse request
        params = self.parser.parse_request(user_input)

        print(f"📋 检测到的参数:")
        print(f"   - 数据源: {params['source_type']}")
        if params['location']:
            print(f"   - 位置: {params['location']}")
        print(f"   - 批次大小: {params['batch_size']}")
        print(f"   - 图像尺寸: {params['image_size']}x{params['image_size']}")
        print(f"   - Pipeline类型: {params['pipeline_type']}")
        if params['num_images']:
            print(f"   - 图像数量: {params['num_images']}")

        # Step 1: Import or create dataset
        print(f"\n{'='*70}")
        print("  步骤 1: 准备数据集")
        print("="*70)

        dataset_name = self.generate_dataset_name(params['location'])

        try:
            if params['source_type'] == "test":
                # Create test dataset
                num_images = params['num_images'] or 50
                print(f"正在创建 {num_images} 张测试图像...")
                result = self.client.create_dataset(
                    name=dataset_name,
                    num_images=num_images,
                    image_size=params['image_size']
                )
                print(f"✅ 测试数据集创建成功")
                print(f"   - 数据集名称: {result['dataset_name']}")
                print(f"   - 图像数量: {result['num_files']}")
                print(f"   - 存储路径: {result['dataset_path']}")

            elif params['source_type'] == "local":
                # Import local dataset
                if not params['location']:
                    print("❌ 错误: 未指定数据路径")
                    print("   示例: 数据在 /data/imagenet")
                    return

                print(f"正在导入本地数据集: {params['location']}...")
                result = self.client.import_local_dataset(
                    dataset_name=dataset_name,
                    local_path=params['location']
                )
                print(f"✅ 本地数据集导入成功")
                print(f"   - 数据集名称: {result['dataset_name']}")
                print(f"   - 图像数量: {result['num_files']:,}")
                print(f"   - 数据路径: {result['dataset_path']}")

            elif params['source_type'] == "s3":
                # Import S3 dataset
                print(f"正在从 S3 导入数据集: {params['location']}...")
                result = self.client.import_s3_dataset(
                    dataset_name=dataset_name,
                    s3_uri=params['location'],
                    download=True
                )
                print(f"✅ S3 数据集导入成功")
                print(f"   - 数据集名称: {result['dataset_name']}")
                print(f"   - 图像数量: {result['num_files']:,}")
                print(f"   - S3 URI: {result['s3_uri']}")
                if result.get('downloaded'):
                    print(f"   - 本地路径: {result['dataset_path']}")

        except requests.exceptions.HTTPError as e:
            print(f"❌ 数据集导入失败: {e.response.json().get('detail', str(e))}")
            return
        except Exception as e:
            print(f"❌ 数据集导入失败: {e}")
            return

        # Step 2: Create pipeline
        print(f"\n{'='*70}")
        print("  步骤 2: 创建 Pipeline")
        print("="*70)

        pipeline_name = self.generate_pipeline_name(
            dataset_name,
            params['pipeline_type'],
            params['batch_size']
        )

        try:
            print(f"正在创建 {params['pipeline_type']} Pipeline...")
            result = self.client.create_pipeline(
                name=pipeline_name,
                dataset_name=dataset_name,
                pipeline_type=params['pipeline_type'],
                batch_size=params['batch_size'],
                target_size=params['image_size']
            )

            print(f"✅ Pipeline 创建成功")
            print(f"   - Pipeline名称: {result['pipeline_name']}")
            print(f"   - 类型: {result['pipeline_type']}")
            print(f"   - 批次大小: {result['batch_size']}")
            print(f"   - 目标尺寸: {result['target_size']}x{result['target_size']}")

            if params['pipeline_type'] == "augmentation":
                print(f"   - 增强操作: 随机裁剪、水平翻转、旋转、亮度/对比度调整")

        except requests.exceptions.HTTPError as e:
            print(f"❌ Pipeline 创建失败: {e.response.json().get('detail', str(e))}")
            return
        except Exception as e:
            print(f"❌ Pipeline 创建失败: {e}")
            return

        # Summary
        print(f"\n{'='*70}")
        print("  ✅ 配置完成！")
        print("="*70)
        print(f"\n**数据集:** {dataset_name}")
        print(f"**Pipeline:** {pipeline_name}")
        print(f"**状态:** 准备就绪，可以开始训练\n")
        print(f"💡 提示:")
        print(f"   - 运行测试: python -c 'agent.run_pipeline(\"{pipeline_name}\")'")
        print(f"   - 查看所有: agent.list_resources()")
        print(f"   - 在训练代码中引用 Pipeline: '{pipeline_name}'")
        print()

    def run_pipeline_test(self, pipeline_name: str, iterations: int = 1) -> None:
        """Run pipeline test"""
        print(f"\n运行 Pipeline 测试: {pipeline_name}...")
        try:
            result = self.client.run_pipeline(pipeline_name, iterations)
            print(f"✅ Pipeline 运行成功")
            print(f"   - 迭代次数: {result['iterations']}")
            print(f"   - 批次大小: {result['batch_size']}")
            for batch in result['batches'][:3]:  # Show first 3 batches
                print(f"   - Batch {batch['iteration']}: {batch['shapes']}")
        except Exception as e:
            print(f"❌ Pipeline 运行失败: {e}")

    def list_resources(self) -> None:
        """List all datasets and pipelines"""
        print("\n" + "="*70)
        print("  资源列表")
        print("="*70)

        # List datasets
        try:
            datasets = self.client.list_datasets()
            print(f"\n📦 数据集 ({datasets['count']}):")
            for ds in datasets['datasets']:
                print(f"   - {ds['name']}: {ds['path']}")
        except Exception as e:
            print(f"❌ 无法获取数据集列表: {e}")

        # List pipelines
        try:
            pipelines = self.client.list_pipelines()
            print(f"\n🔧 Pipeline ({pipelines['count']}):")
            for pipe in pipelines['pipelines']:
                print(f"   - {pipe['name']}: {pipe['type']} (batch={pipe['batch_size']})")
        except Exception as e:
            print(f"❌ 无法获取 Pipeline 列表: {e}")

        print()


# ============================================================
# Interactive Mode
# ============================================================

def interactive_mode():
    """Run agent in interactive mode"""
    print("="*70)
    print("  DALI Agent - 自然语言数据处理助手")
    print("="*70)
    print()
    print("我可以帮你配置图像数据集的处理流程。")
    print("用自然语言描述你的需求，我会自动调用 DALI API。")
    print()
    print("示例:")
    print('  > 我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，')
    print('    批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转')
    print()
    print('  > Create a test dataset with 100 images, batch 16, size 128x128')
    print()
    print('  > 从 s3://my-bucket/images 导入数据，batch 64，做数据增强')
    print()
    print("命令:")
    print("  - list: 列出所有资源")
    print("  - test <pipeline_name>: 测试运行 pipeline")
    print("  - quit: 退出")
    print("="*70)
    print()

    agent = DALIAgent()

    # Check API availability
    if not agent.check_api_availability():
        return

    print("✅ DALI API 服务器连接成功\n")

    while True:
        try:
            user_input = input("👤 > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break

            if user_input.lower() == 'list':
                agent.list_resources()
                continue

            if user_input.lower().startswith('test '):
                pipeline_name = user_input[5:].strip()
                agent.run_pipeline_test(pipeline_name)
                continue

            # Process natural language request
            agent.process_request(user_input)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Direct command mode
        agent = DALIAgent()
        if not agent.check_api_availability():
            sys.exit(1)
        agent.process_request(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        interactive_mode()
