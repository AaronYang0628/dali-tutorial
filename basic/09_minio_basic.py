"""
09 - MinIO 基础集成

学习目标：
1. 理解对象存储的基本概念
2. 掌握 MinIO 客户端配置
3. 学习从 MinIO 读取文件列表
4. 使用 external_source 读取 MinIO 数据

核心概念：
- MinIO: S3 兼容的对象存储
- Bucket: 存储桶，类似文件系统的根目录
- Object: 对象，存储的文件
- Presigned URL: 临时访问 URL
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import numpy as np
from minio import Minio
from minio.error import S3Error
import io
from PIL import Image
import os


def setup_minio_client(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
):
    """
    设置 MinIO 客户端

    Args:
        endpoint: MinIO 服务地址
        access_key: 访问密钥
        secret_key: 密钥
        secure: 是否使用 HTTPS

    Returns:
        MinIO 客户端对象
    """
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

    print(f"✓ Connected to MinIO at {endpoint}")
    return client


def create_sample_bucket(client, bucket_name="dali-tutorial", num_images=50):
    """
    创建示例 bucket 并上传测试图像

    Args:
        client: MinIO 客户端
        bucket_name: bucket 名称
        num_images: 要创建的图像数量
    """
    # 创建 bucket（如果不存在）
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✓ Created bucket: {bucket_name}")
        else:
            print(f"✓ Bucket already exists: {bucket_name}")
    except S3Error as e:
        print(f"Error creating bucket: {e}")
        return

    # 上传示例图像
    print(f"Uploading {num_images} sample images to MinIO...")

    for i in range(num_images):
        # 创建随机图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        # 转换为字节流
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)

        # 上传到 MinIO
        object_name = f"images/image_{i:04d}.jpg"
        client.put_object(
            bucket_name,
            object_name,
            img_bytes,
            length=img_bytes.getbuffer().nbytes,
            content_type='image/jpeg'
        )

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{num_images} images")

    print(f"✓ Uploaded {num_images} images to {bucket_name}")


def list_objects(client, bucket_name, prefix=""):
    """
    列出 bucket 中的对象

    Args:
        client: MinIO 客户端
        bucket_name: bucket 名称
        prefix: 对象前缀（类似目录）

    Returns:
        对象名称列表
    """
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    object_names = [obj.object_name for obj in objects]

    print(f"Found {len(object_names)} objects in {bucket_name}/{prefix}")
    return object_names


class MinIODataSource:
    """
    MinIO 数据源

    从 MinIO 读取图像数据供 DALI 使用
    """
    def __init__(self, client, bucket_name, object_names):
        """
        Args:
            client: MinIO 客户端
            bucket_name: bucket 名称
            object_names: 对象名称列表
        """
        self.client = client
        self.bucket_name = bucket_name
        self.object_names = object_names
        self.num_samples = len(object_names)

        print(f"Initialized MinIODataSource: {self.num_samples} samples from {bucket_name}")

    def __call__(self, sample_info):
        """
        DALI 回调函数：读取单个样本

        Args:
            sample_info: DALI 提供的样本信息

        Returns:
            图像数据和标签
        """
        idx = sample_info.idx_in_epoch

        if idx >= self.num_samples:
            raise StopIteration

        # 获取对象名称
        object_name = self.object_names[idx]

        try:
            # 从 MinIO 读取对象
            response = self.client.get_object(self.bucket_name, object_name)
            image_data = response.read()
            response.close()
            response.release_conn()

            # 将 JPEG 字节转换为 NumPy 数组
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img, dtype=np.uint8)

            # 生成标签（这里简化为基于索引）
            label = idx % 10

            return img_array, np.array([label], dtype=np.int32)

        except S3Error as e:
            print(f"Error reading object {object_name}: {e}")
            # 返回空图像作为后备
            return np.zeros((224, 224, 3), dtype=np.uint8), np.array([0], dtype=np.int32)


@pipeline_def
def minio_pipeline(minio_source):
    """
    从 MinIO 读取数据的 DALI Pipeline

    Args:
        minio_source: MinIO 数据源对象
    """
    # 使用 external_source 读取 MinIO 数据
    images, labels = fn.external_source(
        source=minio_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=False
    )

    # 图像预处理
    images = fn.cast(images, dtype=types.FLOAT) / 255.0

    # 归一化
    images = fn.normalize(
        images,
        mean=[0.485, 0.456, 0.406],
        stddev=[0.229, 0.224, 0.225],
        axes=(2,)
    )

    # 转换为 CHW 格式
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_minio_setup():
    """演示 MinIO 设置和数据上传"""
    print("\n" + "="*60)
    print("Demo 1: MinIO Setup and Data Upload")
    print("="*60)

    # 连接到 MinIO
    # 注意：确保 MinIO 服务正在运行
    # 可以使用 Docker 启动：
    # docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
    try:
        client = setup_minio_client()
    except Exception as e:
        print(f"\n⚠️  Warning: Could not connect to MinIO")
        print(f"   Error: {e}")
        print(f"   Please ensure MinIO is running on localhost:9000")
        print(f"   You can start MinIO with Docker:")
        print(f"   docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ':9001'")
        return None

    # 创建 bucket 并上传数据
    bucket_name = "dali-tutorial"
    create_sample_bucket(client, bucket_name, num_images=20)

    # 列出对象
    objects = list_objects(client, bucket_name, prefix="images/")

    return client, bucket_name, objects


def demo_minio_pipeline(client, bucket_name, objects):
    """演示从 MinIO 读取数据的 Pipeline"""
    if client is None:
        print("\n⚠️  Skipping MinIO pipeline demo (no connection)")
        return

    print("\n" + "="*60)
    print("Demo 2: DALI Pipeline with MinIO")
    print("="*60)

    # 创建 MinIO 数据源
    minio_source = MinIODataSource(client, bucket_name, objects)

    # 创建 Pipeline
    pipe = minio_pipeline(
        minio_source=minio_source,
        batch_size=4,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    print(f"\n✓ Pipeline built")

    # 创建迭代器
    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        size=len(objects),
        auto_reset=True
    )

    # 迭代数据
    print(f"\nIterating through MinIO data:")
    for i, batch in enumerate(dali_iter):
        data = batch[0]
        images = data["images"]
        labels = data["labels"]

        print(f"  Batch {i+1}: images {images.shape}, labels {labels.shape}")

        if i >= 2:  # 只显示前 3 批
            break

    print(f"\n✓ Successfully loaded data from MinIO!")


def demo_presigned_url(client, bucket_name, objects):
    """演示使用预签名 URL"""
    if client is None or not objects:
        print("\n⚠️  Skipping presigned URL demo")
        return

    print("\n" + "="*60)
    print("Demo 3: Presigned URLs")
    print("="*60)

    # 生成预签名 URL（临时访问链接）
    from datetime import timedelta

    object_name = objects[0]
    url = client.presigned_get_object(
        bucket_name,
        object_name,
        expires=timedelta(hours=1)
    )

    print(f"\nGenerated presigned URL for: {object_name}")
    print(f"URL (valid for 1 hour):")
    print(f"  {url}")
    print(f"\nThis URL can be used to download the object without credentials")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 09: MinIO Basic Integration")
    print("="*60)

    # 运行演示
    result = demo_minio_setup()

    if result:
        client, bucket_name, objects = result
        demo_minio_pipeline(client, bucket_name, objects)
        demo_presigned_url(client, bucket_name, objects)
    else:
        print("\n" + "="*60)
        print("MinIO Setup Instructions")
        print("="*60)
        print("\n1. Install MinIO (via Docker):")
        print("   docker run -d -p 9000:9000 -p 9001:9001 \\")
        print("     -e MINIO_ROOT_USER=minioadmin \\")
        print("     -e MINIO_ROOT_PASSWORD=minioadmin \\")
        print("     minio/minio server /data --console-address ':9001'")
        print("\n2. Install Python MinIO client:")
        print("   pip install minio")
        print("\n3. Rerun this script")

    print("\n" + "="*60)
    print("✓ Tutorial 09 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. MinIO 是 S3 兼容的对象存储")
    print("2. 使用 minio 库连接和操作 MinIO")
    print("3. 可以通过 external_source 集成到 DALI")
    print("4. 适合大规模分布式数据存储")
    print("\nNext: 10_minio_production_pipeline.py - 生产级 Pipeline")


if __name__ == "__main__":
    main()
