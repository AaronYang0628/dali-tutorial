#!/usr/bin/env python3
"""
DALI MCP Server - HTTP/SSE Transport Version

支持 HTTP 和 Server-Sent Events (SSE) 传输的 DALI MCP 服务器
专为 n8n、Claude Desktop 等客户端设计

功能：
1. 生成测试图像数据
2. 创建和运行 DALI Pipeline
3. 数据增强操作
4. 图像预处理
5. S3 和本地数据集导入

传输模式：HTTP + SSE
默认端口：8000
"""

import os
import sys
import json
import glob
import tempfile
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Web framework imports
try:
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import Response, JSONResponse
    import uvicorn
except ImportError:
    print("Error: Web framework not installed. Run: pip install starlette uvicorn sse-starlette", file=sys.stderr)
    sys.exit(1)

# DALI imports
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
from PIL import Image

# S3 and cloud storage imports
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = None


# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("dali-mcp-server-http")


# ============================================================
# Global State Management
# ============================================================

class DALIServerState:
    """管理 DALI MCP 服务器的全局状态"""
    def __init__(self):
        self.datasets: Dict[str, str] = {}  # name -> path
        self.pipelines: Dict[str, Any] = {}  # name -> pipeline
        self.temp_dirs: List[str] = []  # 临时目录列表

    def cleanup(self):
        """清理临时资源"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


state = DALIServerState()


# ============================================================
# DALI Pipeline Definitions
# ============================================================

@pipeline_def
def basic_image_pipeline(file_list, target_size=224, device_mode="cpu"):
    """基础图像处理 Pipeline"""
    images, labels = fn.readers.file(files=file_list, random_shuffle=False)
    images = fn.decoders.image(images, device=device_mode, output_type=types.RGB)
    images = fn.resize(images, size=target_size, mode="not_smaller")
    images = fn.crop(images, crop=(target_size, target_size), crop_pos_x=0.5, crop_pos_y=0.5)
    return images, labels


@pipeline_def
def augmentation_pipeline(file_list, target_size=224, device_mode="cpu"):
    """数据增强 Pipeline"""
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device=device_mode, output_type=types.RGB)

    images = fn.random_resized_crop(
        images,
        size=target_size,
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.33]
    )

    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),
        contrast=fn.random.uniform(range=[0.8, 1.2])
    )

    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


# ============================================================
# Helper Functions
# ============================================================

def create_test_images(output_dir: str, num_images: int = 10, image_size: int = 256) -> List[str]:
    """创建测试图像"""
    os.makedirs(output_dir, exist_ok=True)
    file_list = []

    for i in range(num_images):
        img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        filename = os.path.join(output_dir, f"image_{i:04d}.jpg")
        img.save(filename, quality=95)
        file_list.append(filename)

    return file_list


def get_pipeline_stats(pipeline, num_iterations: int = 1) -> Dict[str, Any]:
    """运行 pipeline 并获取统计信息"""
    stats = {
        "iterations": num_iterations,
        "batches": []
    }

    for i in range(num_iterations):
        outputs = pipeline.run()
        images_batch = outputs[0]
        labels_batch = outputs[1] if len(outputs) > 1 else None

        batch_info = {
            "iteration": i + 1,
            "batch_size": len(images_batch),
            "shapes": [str(shape) for shape in images_batch.shape()],
            "dtype": str(images_batch.dtype)
        }

        if len(images_batch) > 0:
            first_img = np.array(images_batch.as_cpu()[0])
            batch_info["sample_stats"] = {
                "min": float(np.min(first_img)),
                "max": float(np.max(first_img)),
                "mean": float(np.mean(first_img)),
                "std": float(np.std(first_img))
            }

        stats["batches"].append(batch_info)

    return stats


def save_pipeline_results_local(pipeline, output_dir: str, num_iterations: int = 1,
                                output_format: str = "jpg") -> Dict[str, Any]:
    """运行 pipeline 并保存结果到本地目录"""
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    total_images = 0

    for iteration in range(num_iterations):
        outputs = pipeline.run()
        images_batch = outputs[0]

        # 将图像从 GPU 移到 CPU 并转换为 numpy 数组
        images_cpu = images_batch.as_cpu()

        for idx in range(len(images_cpu)):
            img_array = np.array(images_cpu[idx])

            # 确保数据范围在 [0, 255]
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

            # 处理不同的图像格式
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # RGB 图像
                img = Image.fromarray(img_array, mode='RGB')
            elif len(img_array.shape) == 2:
                # 灰度图像
                img = Image.fromarray(img_array, mode='L')
            else:
                # 其他格式，尝试转换
                img = Image.fromarray(img_array)

            # 保存图像
            filename = f"batch{iteration:04d}_img{idx:04d}.{output_format}"
            filepath = os.path.join(output_dir, filename)

            if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
                img.save(filepath, 'JPEG', quality=95)
            else:
                img.save(filepath, output_format.upper())

            saved_files.append(filename)
            total_images += 1

    return {
        "output_location": output_dir,
        "saved_files": saved_files[:10],  # 只返回前10个文件名
        "total_images": total_images,
        "output_format": output_format
    }


def save_pipeline_results_s3(pipeline, s3_uri: str, num_iterations: int = 1,
                             output_format: str = "jpg") -> Dict[str, Any]:
    """运行 pipeline 并上传结果到 S3"""
    if boto3 is None:
        raise ImportError("boto3 not installed. Run: pip install boto3")

    # 解析 S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")

    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # 创建临时目录保存图像
    temp_dir = tempfile.mkdtemp(prefix="dali_s3_upload_")

    try:
        # 先保存到本地临时目录
        local_result = save_pipeline_results_local(
            pipeline, temp_dir, num_iterations, output_format
        )

        # 上传到 S3
        s3_client = boto3.client('s3')
        uploaded_files = []

        for filename in os.listdir(temp_dir):
            local_path = os.path.join(temp_dir, filename)
            s3_key = f"{prefix}/{filename}" if prefix else filename

            s3_client.upload_file(local_path, bucket, s3_key)
            uploaded_files.append(s3_key)

        return {
            "output_location": s3_uri,
            "bucket": bucket,
            "prefix": prefix,
            "uploaded_files": uploaded_files[:10],
            "total_images": local_result["total_images"],
            "output_format": output_format
        }
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Tool Handlers
# ============================================================

async def handle_create_dataset(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理创建数据集请求"""
    name = arguments["name"]
    num_images = arguments.get("num_images", 10)
    image_size = arguments.get("image_size", 256)

    temp_dir = tempfile.mkdtemp(prefix=f"dali_dataset_{name}_")
    state.temp_dirs.append(temp_dir)

    file_list = create_test_images(temp_dir, num_images, image_size)
    state.datasets[name] = temp_dir

    result = {
        "dataset_name": name,
        "dataset_path": temp_dir,
        "num_files": len(file_list),
        "image_size": image_size,
        "file_list": file_list[:5]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_create_pipeline(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理创建 Pipeline 请求"""
    name = arguments["name"]
    dataset_name = arguments["dataset_name"]
    pipeline_type = arguments.get("pipeline_type", "basic")
    batch_size = arguments.get("batch_size", 4)
    target_size = arguments.get("target_size", 224)

    if dataset_name not in state.datasets:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Dataset '{dataset_name}' not found",
                "available_datasets": list(state.datasets.keys())
            }, indent=2)
        )]

    dataset_path = state.datasets[dataset_name]
    file_list = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))

    if not file_list:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"No images found in dataset '{dataset_name}'"}, indent=2)
        )]

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        gpu_available = False

    device_id = 0 if gpu_available else None
    device_mode = "mixed" if gpu_available else "cpu"

    if pipeline_type == "basic":
        pipe = basic_image_pipeline(
            file_list=file_list,
            target_size=target_size,
            device_mode=device_mode,
            batch_size=batch_size,
            num_threads=2,
            device_id=device_id
        )
    elif pipeline_type == "augmentation":
        pipe = augmentation_pipeline(
            file_list=file_list,
            target_size=target_size,
            device_mode=device_mode,
            batch_size=batch_size,
            num_threads=2,
            device_id=device_id,
            seed=42
        )
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown pipeline type: {pipeline_type}"}, indent=2)
        )]

    pipe.build()

    state.pipelines[name] = {
        "pipeline": pipe,
        "type": pipeline_type,
        "batch_size": batch_size,
        "target_size": target_size,
        "dataset_name": dataset_name,
        "num_files": len(file_list),
        "device": "GPU" if gpu_available else "CPU"
    }

    result = {
        "pipeline_name": name,
        "pipeline_type": pipeline_type,
        "batch_size": batch_size,
        "target_size": target_size,
        "dataset_name": dataset_name,
        "num_files": len(file_list),
        "device": "GPU" if gpu_available else "CPU",
        "status": "created and built"
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_run_pipeline(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理运行 Pipeline 请求"""
    pipeline_name = arguments["pipeline_name"]
    num_iterations = arguments.get("num_iterations", 1)

    if pipeline_name not in state.pipelines:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Pipeline '{pipeline_name}' not found",
                "available_pipelines": list(state.pipelines.keys())
            }, indent=2)
        )]

    pipe_info = state.pipelines[pipeline_name]
    pipe = pipe_info["pipeline"]

    stats = get_pipeline_stats(pipe, num_iterations)

    result = {
        "pipeline_name": pipeline_name,
        "pipeline_type": pipe_info["type"],
        "batch_size": pipe_info["batch_size"],
        "statistics": stats
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_save_results(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理保存 Pipeline 结果请求"""
    pipeline_name = arguments["pipeline_name"]
    output_path = arguments["output_path"]
    num_iterations = arguments.get("num_iterations", 1)
    output_format = arguments.get("output_format", "jpg")

    if pipeline_name not in state.pipelines:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Pipeline '{pipeline_name}' not found",
                "available_pipelines": list(state.pipelines.keys())
            }, indent=2)
        )]

    pipe_info = state.pipelines[pipeline_name]
    pipe = pipe_info["pipeline"]

    try:
        # 判断是本地路径还是 S3 URI
        if output_path.startswith("s3://"):
            result = save_pipeline_results_s3(pipe, output_path, num_iterations, output_format)
        else:
            result = save_pipeline_results_local(pipe, output_path, num_iterations, output_format)

        result["pipeline_name"] = pipeline_name
        result["pipeline_type"] = pipe_info["type"]
        result["batch_size"] = pipe_info["batch_size"]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "pipeline_name": pipeline_name
            }, indent=2)
        )]


async def handle_list_datasets(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理列出数据集请求"""
    datasets = []
    for name, path in state.datasets.items():
        num_files = len(glob.glob(os.path.join(path, "*.jpg")))
        datasets.append({"name": name, "path": path, "num_files": num_files})

    result = {"count": len(datasets), "datasets": datasets}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_list_pipelines(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理列出 Pipeline 请求"""
    pipelines = []
    for name, info in state.pipelines.items():
        pipelines.append({
            "name": name,
            "type": info["type"],
            "batch_size": info["batch_size"],
            "target_size": info["target_size"],
            "dataset_name": info["dataset_name"],
            "num_files": info["num_files"],
            "device": info.get("device", "CPU")
        })

    result = {"count": len(pipelines), "pipelines": pipelines}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_import_local_dataset(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理本地数据集导入请求"""
    dataset_name = arguments["dataset_name"]
    local_path = arguments["local_path"]
    supported_formats = arguments.get("supported_formats", ["jpg", "jpeg", "png"])

    if dataset_name in state.datasets:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Dataset name '{dataset_name}' already exists",
                "available_datasets": list(state.datasets.keys())
            }, indent=2)
        )]

    if not os.path.isabs(local_path):
        return [TextContent(
            type="text",
            text=json.dumps({"error": "Path must be absolute"}, indent=2)
        )]

    if not os.path.isdir(local_path):
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Local path does not exist: {local_path}"}, indent=2)
        )]

    file_list = []
    for fmt in supported_formats:
        pattern = os.path.join(local_path, f"*.{fmt}")
        file_list.extend(glob.glob(pattern))

    file_list = sorted(list(set(file_list)))

    if not file_list:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"No images found in {local_path} with supported formats: {supported_formats}"
            }, indent=2)
        )]

    state.datasets[dataset_name] = local_path

    result = {
        "dataset_name": dataset_name,
        "dataset_path": local_path,
        "num_files": len(file_list),
        "supported_formats": supported_formats,
        "file_list": file_list[:5]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_import_s3_dataset(arguments: Dict[str, Any]) -> list[TextContent]:
    """处理 S3 数据集导入请求"""
    if boto3 is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "boto3 not installed. Run: pip install boto3 botocore"}, indent=2)
        )]

    dataset_name = arguments["dataset_name"]
    s3_uri = arguments["s3_uri"]
    endpoint_url = arguments.get("endpoint_url")
    download = arguments.get("download", False)
    supported_formats = arguments.get("supported_formats", ["jpg", "jpeg", "png"])

    if dataset_name in state.datasets:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Dataset name '{dataset_name}' already exists",
                "available_datasets": list(state.datasets.keys())
            }, indent=2)
        )]

    try:
        if s3_uri.startswith("s3://"):
            s3_uri = s3_uri[5:]

        parts = s3_uri.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        access_key = arguments.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = arguments.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")

        s3_kwargs = {}
        if access_key:
            s3_kwargs["aws_access_key_id"] = access_key
        if secret_key:
            s3_kwargs["aws_secret_access_key"] = secret_key
        if endpoint_url:
            s3_kwargs["endpoint_url"] = endpoint_url

        s3 = boto3.client("s3", **s3_kwargs)

        file_list = []
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                for fmt in supported_formats:
                    if key.lower().endswith(f".{fmt}"):
                        file_list.append(key)
                        break

        if not file_list:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"No images found in s3://{bucket}/{prefix} with supported formats"
                }, indent=2)
            )]

        if download:
            temp_dir = tempfile.mkdtemp(prefix=f"dali_s3_dataset_{dataset_name}_")
            state.temp_dirs.append(temp_dir)

            for key in file_list:
                local_file = os.path.join(temp_dir, os.path.basename(key))
                try:
                    s3.download_file(bucket, key, local_file)
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Failed to download {key}: {str(e)}"}, indent=2)
                    )]

            state.datasets[dataset_name] = temp_dir

            result = {
                "dataset_name": dataset_name,
                "s3_uri": f"s3://{bucket}/{prefix}",
                "num_files": len(file_list),
                "local_path": temp_dir,
                "status": "downloaded",
                "file_list": file_list[:5]
            }
        else:
            state.datasets[dataset_name] = f"s3://{bucket}/{prefix}"

            result = {
                "dataset_name": dataset_name,
                "s3_uri": f"s3://{bucket}/{prefix}",
                "num_files": len(file_list),
                "status": "listed",
                "note": "Files not downloaded. Use download=true to download to local directory before creating pipeline.",
                "file_list": file_list[:5]
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchBucket":
            error_msg = f"S3 bucket '{bucket}' not found"
        elif error_code == "AccessDenied":
            error_msg = f"Access denied to S3 bucket '{bucket}'"
        else:
            error_msg = f"S3 error: {str(e)}"

        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Failed to import from S3: {str(e)}"}, indent=2)
        )]


# ============================================================
# MCP Server Setup
# ============================================================

app = Server("dali-mcp-server-http")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的工具"""
    return [
        Tool(
            name="create_test_dataset",
            description="创建测试图像数据集",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "数据集名称"},
                    "num_images": {"type": "integer", "description": "生成图像数量", "default": 10},
                    "image_size": {"type": "integer", "description": "图像尺寸", "default": 256}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="create_pipeline",
            description="创建 DALI 数据处理 Pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Pipeline 名称"},
                    "dataset_name": {"type": "string", "description": "数据集名称"},
                    "pipeline_type": {
                        "type": "string",
                        "enum": ["basic", "augmentation"],
                        "description": "Pipeline 类型",
                        "default": "basic"
                    },
                    "batch_size": {"type": "integer", "description": "批次大小", "default": 4},
                    "target_size": {"type": "integer", "description": "目标图像尺寸", "default": 224}
                },
                "required": ["name", "dataset_name"]
            }
        ),
        Tool(
            name="run_pipeline",
            description="运行 DALI Pipeline 并获取处理结果统计",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_name": {"type": "string", "description": "Pipeline 名称"},
                    "num_iterations": {"type": "integer", "description": "运行迭代次数", "default": 1}
                },
                "required": ["pipeline_name"]
            }
        ),
        Tool(
            name="list_datasets",
            description="列出所有已创建的数据集",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="list_pipelines",
            description="列出所有已创建的 Pipeline",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="import_local_dataset",
            description="从本地文件目录导入图像数据集",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "数据集名称"},
                    "local_path": {"type": "string", "description": "本地目录的绝对路径"},
                    "supported_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "支持的文件格式",
                        "default": ["jpg", "jpeg", "png"]
                    }
                },
                "required": ["dataset_name", "local_path"]
            }
        ),
        Tool(
            name="import_s3_dataset",
            description="从 S3 兼容存储导入图像数据集",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "数据集名称"},
                    "s3_uri": {"type": "string", "description": "S3 URI，格式: s3://bucket/prefix"},
                    "endpoint_url": {"type": "string", "description": "S3 端点 URL（可选）"},
                    "access_key": {"type": "string", "description": "AWS access key（可选）"},
                    "secret_key": {"type": "string", "description": "AWS secret key（可选）"},
                    "download": {"type": "boolean", "description": "是否下载到本地", "default": False},
                    "supported_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["jpg", "jpeg", "png"]
                    }
                },
                "required": ["dataset_name", "s3_uri"]
            }
        ),
        Tool(
            name="save_results",
            description="运行 Pipeline 并保存处理后的图像到本地或 S3",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_name": {"type": "string", "description": "Pipeline 名称"},
                    "output_path": {"type": "string", "description": "输出路径（本地目录或 s3://bucket/prefix）"},
                    "num_iterations": {"type": "integer", "description": "运行迭代次数", "default": 1},
                    "output_format": {
                        "type": "string",
                        "enum": ["jpg", "jpeg", "png"],
                        "description": "输出图像格式",
                        "default": "jpg"
                    }
                },
                "required": ["pipeline_name", "output_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """处理工具调用"""
    logger.info("="*60)
    logger.info(f"Incoming tool call: {name}")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
    logger.info("="*60)

    try:
        if name == "create_test_dataset":
            result = await handle_create_dataset(arguments)
        elif name == "create_pipeline":
            result = await handle_create_pipeline(arguments)
        elif name == "run_pipeline":
            result = await handle_run_pipeline(arguments)
        elif name == "list_datasets":
            result = await handle_list_datasets(arguments)
        elif name == "list_pipelines":
            result = await handle_list_pipelines(arguments)
        elif name == "import_local_dataset":
            result = await handle_import_local_dataset(arguments)
        elif name == "import_s3_dataset":
            result = await handle_import_s3_dataset(arguments)
        elif name == "save_results":
            result = await handle_save_results(arguments)
        else:
            result = [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
            )]

        logger.info(f"✓ Tool '{name}' completed successfully")
        return result

    except Exception as e:
        logger.error(f"✗ Error in tool '{name}': {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }, indent=2)
        )]


# ============================================================
# Web Server Setup
# ============================================================

def create_web_app(host: str = "0.0.0.0", port: int = 8000):
    """创建 Starlette Web 应用"""

    # 创建 SSE 传输
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        """处理 SSE 连接"""
        logger.info(f"New SSE connection from {request.client.host}")
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )
        logger.info("SSE connection closed")
        return Response()

    async def handle_health(request):
        """健康检查端点"""
        return JSONResponse({
            "status": "healthy",
            "server": "dali-mcp-server-http",
            "version": "1.0.0",
            "dali_version": dali.__version__,
            "transport": "HTTP/SSE",
            "endpoints": {
                "sse": "/sse",
                "messages": "/messages/",
                "health": "/health"
            }
        })

    async def handle_root(request):
        """根路径处理"""
        return JSONResponse({
            "name": "DALI MCP Server",
            "version": "1.0.0",
            "transport": "HTTP + SSE",
            "endpoints": {
                "sse": f"http://{host}:{port}/sse",
                "messages": f"http://{host}:{port}/messages/",
                "health": f"http://{host}:{port}/health"
            },
            "usage": {
                "sse_endpoint": "Connect to /sse with GET request for Server-Sent Events",
                "messages_endpoint": "POST client messages to /messages/?session_id=<id>",
                "health_endpoint": "GET /health for server health status"
            }
        })

    # 创建路由
    routes = [
        Route("/", endpoint=handle_root, methods=["GET"]),
        Route("/health", endpoint=handle_health, methods=["GET"]),
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    return Starlette(routes=routes)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """启动 HTTP/SSE MCP 服务器"""
    import argparse

    parser = argparse.ArgumentParser(description="DALI MCP Server - HTTP/SSE Transport")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")

    args = parser.parse_args()

    # 设置日志级别
    logger.setLevel(args.log_level.upper())

    logger.info("="*60)
    logger.info("DALI MCP Server Starting (HTTP/SSE Transport)")
    logger.info(f"DALI Version: {dali.__version__}")
    logger.info(f"Python Version: {sys.version}")
    logger.info("="*60)
    logger.info("")
    logger.info("Transport Mode: HTTP + Server-Sent Events (SSE)")
    logger.info(f"Listening on: http://{args.host}:{args.port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  - Root:     http://{args.host}:{args.port}/")
    logger.info(f"  - Health:   http://{args.host}:{args.port}/health")
    logger.info(f"  - SSE:      http://{args.host}:{args.port}/sse")
    logger.info(f"  - Messages: http://{args.host}:{args.port}/messages/")
    logger.info("")
    logger.info("Compatible with:")
    logger.info("  ✓ n8n MCP Client (HTTP/SSE transport)")
    logger.info("  ✓ Claude Desktop")
    logger.info("  ✓ Custom HTTP/SSE clients")
    logger.info("")
    logger.info("Configuration for n8n:")
    logger.info(f'  Server URL: http://{args.host}:{args.port}')
    logger.info('  Transport: "Server Sent Event"')
    logger.info("="*60)

    try:
        starlette_app = create_web_app(args.host, args.port)
        uvicorn.run(
            starlette_app,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Cleaning up resources...")
        state.cleanup()
        logger.info("DALI MCP Server stopped")


if __name__ == "__main__":
    main()
