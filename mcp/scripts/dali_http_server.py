#!/usr/bin/env python3
"""
DALI HTTP API Server - FastAPI wrapper for DALI MCP Server

提供REST API端点，让N8N等工具可以通过HTTP调用DALI服务
"""

import os
import sys
import json
import glob
import tempfile
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dali-http-server")


# ============================================================
# Global State Management
# ============================================================

class DALIServerState:
    """管理 DALI HTTP 服务器的全局状态"""
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
    images = fn.crop_mirror_normalize(
        images,
        crop=(target_size, target_size),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    return images, labels


@pipeline_def
def augmentation_pipeline(file_list, target_size=224, device_mode="cpu"):
    """数据增强 Pipeline"""
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device=device_mode, output_type=types.RGB)
    images = fn.random_resized_crop(images, size=target_size)
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    images = fn.rotate(images, angle=fn.random.uniform(range=(-15, 15)))
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=(0.8, 1.2)),
        contrast=fn.random.uniform(range=(0.8, 1.2))
    )
    images = fn.crop_mirror_normalize(
        images,
        crop=(target_size, target_size),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    return images, labels


# ============================================================
# Helper Functions
# ============================================================

def create_test_images(output_dir: str, num_images: int, image_size: int) -> List[str]:
    """生成测试图像"""
    file_list = []
    for i in range(num_images):
        img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        file_path = os.path.join(output_dir, f"image_{i:04d}.jpg")
        img.save(file_path, quality=95)
        file_list.append(file_path)
    return file_list


# ============================================================
# Pydantic Models
# ============================================================

class CreateDatasetRequest(BaseModel):
    name: str = Field(..., description="数据集名称")
    num_images: int = Field(10, description="生成图像数量", ge=1)
    image_size: int = Field(256, description="图像尺寸", ge=32)


class CreatePipelineRequest(BaseModel):
    name: str = Field(..., description="Pipeline名称")
    dataset_name: str = Field(..., description="数据集名称")
    pipeline_type: str = Field("basic", description="Pipeline类型: basic 或 augmentation")
    batch_size: int = Field(4, description="批次大小", ge=1)
    target_size: int = Field(224, description="目标图像尺寸", ge=32)


class RunPipelineRequest(BaseModel):
    pipeline_name: str = Field(..., description="Pipeline名称")
    num_iterations: int = Field(1, description="运行迭代次数", ge=1)


class ImportLocalDatasetRequest(BaseModel):
    dataset_name: str = Field(..., description="数据集名称")
    local_path: str = Field(..., description="本地目录路径")
    supported_formats: List[str] = Field(["jpg", "jpeg", "png"], description="支持的文件格式")


class ImportS3DatasetRequest(BaseModel):
    dataset_name: str = Field(..., description="数据集名称")
    s3_uri: str = Field(..., description="S3 URI")
    endpoint_url: Optional[str] = Field(None, description="S3端点URL")
    access_key: Optional[str] = Field(None, description="Access Key")
    secret_key: Optional[str] = Field(None, description="Secret Key")
    download: bool = Field(False, description="是否下载到本地")
    supported_formats: List[str] = Field(["jpg", "jpeg", "png"], description="支持的文件格式")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="DALI HTTP API Server",
    description="REST API for NVIDIA DALI image processing",
    version="1.0.0"
)

# CORS配置 - 允许N8N调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "ok",
        "service": "DALI HTTP API Server",
        "version": "1.0.0",
        "endpoints": {
            "create_dataset": "/api/dataset/create",
            "import_local": "/api/dataset/import/local",
            "import_s3": "/api/dataset/import/s3",
            "list_datasets": "/api/dataset/list",
            "create_pipeline": "/api/pipeline/create",
            "run_pipeline": "/api/pipeline/run",
            "list_pipelines": "/api/pipeline/list"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查DALI是否可用
        _ = dali.__version__
        return {"status": "healthy", "dali_version": dali.__version__}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/api/dataset/create")
async def create_dataset(request: CreateDatasetRequest):
    """创建测试数据集"""
    try:
        # 检查数据集是否已存在
        if request.name in state.datasets:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Dataset '{request.name}' already exists"
            )

        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix=f"dali_dataset_{request.name}_")
        state.temp_dirs.append(temp_dir)

        # 生成测试图像
        file_list = create_test_images(temp_dir, request.num_images, request.image_size)

        # 保存数据集信息
        state.datasets[request.name] = temp_dir

        return {
            "dataset_name": request.name,
            "dataset_path": temp_dir,
            "num_files": len(file_list),
            "image_size": request.image_size,
            "file_list": file_list[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/dataset/import/local")
async def import_local_dataset(request: ImportLocalDatasetRequest):
    """从本地目录导入数据集"""
    try:
        # 检查数据集名称是否已存在
        if request.dataset_name in state.datasets:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Dataset '{request.dataset_name}' already exists"
            )

        # 检查路径是否存在
        if not os.path.exists(request.local_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path does not exist: {request.local_path}"
            )

        # 扫描文件
        file_list = []
        for ext in request.supported_formats:
            pattern = os.path.join(request.local_path, f"*.{ext}")
            file_list.extend(glob.glob(pattern))

        if not file_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No image files found in {request.local_path}"
            )

        # 保存数据集信息
        state.datasets[request.dataset_name] = request.local_path

        return {
            "dataset_name": request.dataset_name,
            "dataset_path": request.local_path,
            "num_files": len(file_list),
            "file_list": file_list[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing local dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/dataset/import/s3")
async def import_s3_dataset(request: ImportS3DatasetRequest):
    """从S3导入数据集"""
    try:
        if boto3 is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="S3 support not available. Install boto3: pip install boto3"
            )

        # 检查数据集名称
        if request.dataset_name in state.datasets:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Dataset '{request.dataset_name}' already exists"
            )

        # 解析S3 URI
        if not request.s3_uri.startswith("s3://"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid S3 URI format. Expected: s3://bucket/prefix"
            )

        uri_parts = request.s3_uri[5:].split("/", 1)
        bucket = uri_parts[0]
        prefix = uri_parts[1] if len(uri_parts) > 1 else ""

        # 创建S3客户端
        s3_config = {}
        if request.endpoint_url:
            s3_config["endpoint_url"] = request.endpoint_url
        if request.access_key:
            s3_config["aws_access_key_id"] = request.access_key
        if request.secret_key:
            s3_config["aws_secret_access_key"] = request.secret_key

        s3_client = boto3.client("s3", **s3_config)

        # 列举对象
        file_list = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if any(key.lower().endswith(f".{fmt}") for fmt in request.supported_formats):
                    file_list.append(key)

        if not file_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No image files found in {request.s3_uri}"
            )

        # 如果需要下载
        if request.download:
            temp_dir = tempfile.mkdtemp(prefix=f"dali_s3_{request.dataset_name}_")
            state.temp_dirs.append(temp_dir)

            downloaded_files = []
            for key in file_list[:100]:  # 限制下载数量
                local_path = os.path.join(temp_dir, os.path.basename(key))
                s3_client.download_file(bucket, key, local_path)
                downloaded_files.append(local_path)

            state.datasets[request.dataset_name] = temp_dir
            dataset_path = temp_dir
            result_files = downloaded_files
        else:
            # 仅保存S3信息
            state.datasets[request.dataset_name] = request.s3_uri
            dataset_path = request.s3_uri
            result_files = file_list

        return {
            "dataset_name": request.dataset_name,
            "dataset_path": dataset_path,
            "s3_uri": request.s3_uri,
            "num_files": len(result_files),
            "file_list": result_files[:5],
            "downloaded": request.download
        }

    except HTTPException:
        raise
    except ClientError as e:
        logger.error(f"S3 error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error importing S3 dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/dataset/list")
async def list_datasets():
    """列出所有数据集"""
    datasets = []
    for name, path in state.datasets.items():
        datasets.append({
            "name": name,
            "path": path
        })
    return {"datasets": datasets, "count": len(datasets)}


@app.post("/api/pipeline/create")
async def create_pipeline(request: CreatePipelineRequest):
    """创建DALI Pipeline"""
    try:
        # 检查数据集是否存在
        if request.dataset_name not in state.datasets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{request.dataset_name}' not found"
            )

        # 检查pipeline是否已存在
        if request.name in state.pipelines:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Pipeline '{request.name}' already exists"
            )

        # 获取文件列表
        dataset_path = state.datasets[request.dataset_name]
        file_list = glob.glob(os.path.join(dataset_path, "*.jpg"))
        file_list.extend(glob.glob(os.path.join(dataset_path, "*.jpeg")))
        file_list.extend(glob.glob(os.path.join(dataset_path, "*.png")))

        if not file_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No image files found in dataset '{request.dataset_name}'"
            )

        # 创建Pipeline
        # 尝试使用GPU，如果不可用则降级到CPU
        try:
            device_mode = "mixed"
            device_id = 0
        except:
            device_mode = "cpu"
            device_id = -1

        if request.pipeline_type == "basic":
            pipe = basic_image_pipeline(
                file_list=file_list,
                target_size=request.target_size,
                device_mode=device_mode,
                batch_size=request.batch_size,
                num_threads=2,
                device_id=device_id
            )
        elif request.pipeline_type == "augmentation":
            pipe = augmentation_pipeline(
                file_list=file_list,
                target_size=request.target_size,
                device_mode=device_mode,
                batch_size=request.batch_size,
                num_threads=2,
                device_id=device_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid pipeline_type: {request.pipeline_type}"
            )

        pipe.build()

        # 保存Pipeline
        state.pipelines[request.name] = {
            "pipeline": pipe,
            "type": request.pipeline_type,
            "batch_size": request.batch_size,
            "dataset": request.dataset_name
        }

        return {
            "pipeline_name": request.name,
            "pipeline_type": request.pipeline_type,
            "batch_size": request.batch_size,
            "target_size": request.target_size,
            "dataset": request.dataset_name,
            "num_files": len(file_list),
            "status": "created"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/pipeline/run")
async def run_pipeline(request: RunPipelineRequest):
    """运行Pipeline"""
    try:
        # 检查Pipeline是否存在
        if request.pipeline_name not in state.pipelines:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline '{request.pipeline_name}' not found"
            )

        pipeline_info = state.pipelines[request.pipeline_name]
        pipe = pipeline_info["pipeline"]

        # 运行Pipeline
        batches = []
        for i in range(request.num_iterations):
            outputs = pipe.run()
            batch_info = {
                "iteration": i + 1,
                "num_outputs": len(outputs),
                "shapes": [str(output.shape()) for output in outputs]
            }
            batches.append(batch_info)

        return {
            "pipeline_name": request.pipeline_name,
            "iterations": request.num_iterations,
            "batch_size": pipeline_info["batch_size"],
            "batches": batches,
            "status": "completed"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/pipeline/list")
async def list_pipelines():
    """列出所有Pipeline"""
    pipelines = []
    for name, info in state.pipelines.items():
        pipelines.append({
            "name": name,
            "type": info["type"],
            "batch_size": info["batch_size"],
            "dataset": info["dataset"]
        })
    return {"pipelines": pipelines, "count": len(pipelines)}


# ============================================================
# Startup and Shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():
    """服务启动"""
    logger.info("DALI HTTP API Server starting...")
    logger.info(f"DALI version: {dali.__version__}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭"""
    logger.info("DALI HTTP API Server shutting down...")
    state.cleanup()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DALI HTTP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        "dali_http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
