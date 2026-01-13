"""
10 - 生产级 MinIO Pipeline

学习目标：
1. 构建生产级数据加载流水线
2. 实现错误处理和重试机制
3. 添加性能监控和日志
4. 实现缓存和预取策略
5. 与完整训练循环集成

核心概念：
- Production-ready pipeline
- Error handling and retry
- Performance monitoring
- Caching strategies
- Multi-GPU support
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import torch
import torch.nn as nn
import numpy as np
from minio import Minio
from minio.error import S3Error
import io
from PIL import Image
import time
import logging
from collections import deque
import threading


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinIODataSourceWithCache:
    """
    带缓存的生产级 MinIO 数据源

    Features:
    - 错误处理和重试
    - LRU 缓存
    - 性能监控
    - 线程安全
    """
    def __init__(
        self,
        client,
        bucket_name,
        object_names,
        cache_size=100,
        max_retries=3,
        retry_delay=1.0
    ):
        """
        Args:
            client: MinIO 客户端
            bucket_name: bucket 名称
            object_names: 对象名称列表
            cache_size: 缓存大小
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.client = client
        self.bucket_name = bucket_name
        self.object_names = object_names
        self.num_samples = len(object_names)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # LRU 缓存
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = deque(maxlen=cache_size)

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'retries': 0
        }

        # 线程锁
        self.lock = threading.Lock()

        logger.info(f"Initialized MinIODataSource: {self.num_samples} samples, cache_size={cache_size}")

    def _get_from_cache(self, key):
        """从缓存获取数据"""
        with self.lock:
            if key in self.cache:
                self.stats['cache_hits'] += 1
                # 更新 LRU 顺序
                self.cache_order.remove(key)
                self.cache_order.append(key)
                return self.cache[key]

            self.stats['cache_misses'] += 1
            return None

    def _add_to_cache(self, key, value):
        """添加数据到缓存"""
        with self.lock:
            # 如果缓存已满，移除最旧的项
            if len(self.cache) >= self.cache_size and key not in self.cache:
                if self.cache_order:
                    oldest_key = self.cache_order.popleft()
                    del self.cache[oldest_key]

            self.cache[key] = value
            if key not in self.cache_order:
                self.cache_order.append(key)

    def _download_with_retry(self, object_name):
        """
        带重试的下载

        Args:
            object_name: 对象名称

        Returns:
            图像数据（NumPy array）

        Raises:
            Exception: 如果所有重试都失败
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.get_object(self.bucket_name, object_name)
                image_data = response.read()
                response.close()
                response.release_conn()

                # 解码图像
                img = Image.open(io.BytesIO(image_data))
                img_array = np.array(img, dtype=np.uint8)

                return img_array

            except S3Error as e:
                self.stats['retries'] += 1
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {object_name}: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.stats['errors'] += 1
                    raise

    def __call__(self, sample_info):
        """DALI 回调函数"""
        idx = sample_info.idx_in_epoch

        if idx >= self.num_samples:
            raise StopIteration

        self.stats['total_requests'] += 1
        object_name = self.object_names[idx]

        # 尝试从缓存获取
        cached_data = self._get_from_cache(object_name)
        if cached_data is not None:
            return cached_data

        # 下载数据
        try:
            img_array = self._download_with_retry(object_name)

            # 生成标签（实际应用中从元数据读取）
            label = idx % 10

            result = (img_array, np.array([label], dtype=np.int32))

            # 添加到缓存
            self._add_to_cache(object_name, result)

            return result

        except Exception as e:
            logger.error(f"Failed to load {object_name}: {e}")
            # 返回空图像作为后备
            return (
                np.zeros((224, 224, 3), dtype=np.uint8),
                np.array([0], dtype=np.int32)
            )

    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            stats = self.stats.copy()

        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
        else:
            stats['cache_hit_rate'] = 0.0

        return stats

    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            self.stats = {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'retries': 0
            }


@pipeline_def
def production_minio_pipeline(minio_source, image_size=224, is_training=True):
    """
    生产级 MinIO Pipeline

    Args:
        minio_source: MinIO 数据源
        image_size: 图像大小
        is_training: 是否为训练模式
    """
    # 读取数据
    images, labels = fn.external_source(
        source=minio_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=False
    )

    if is_training:
        # 训练模式：数据增强
        images = fn.random_resized_crop(
            images,
            size=image_size,
            random_area=[0.08, 1.0],
            random_aspect_ratio=[0.75, 1.33]
        )

        images = fn.flip(
            images,
            horizontal=fn.random.coin_flip(probability=0.5)
        )

        images = fn.brightness_contrast(
            images,
            brightness=fn.random.uniform(range=[0.8, 1.2]),
            contrast=fn.random.uniform(range=[0.8, 1.2])
        )

    else:
        # 验证模式：中心裁剪
        images = fn.resize(images, size=int(image_size * 1.14))
        images = fn.crop(images, crop=image_size, crop_pos_x=0.5, crop_pos_y=0.5)

    # 归一化
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.normalize(
        images,
        mean=[0.485, 0.456, 0.406],
        stddev=[0.229, 0.224, 0.225],
        axes=(2,)
    )

    # CHW 格式
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_production_pipeline():
    """演示生产级 Pipeline"""
    print("\n" + "="*60)
    print("Production Pipeline Demo")
    print("="*60)

    # 设置 MinIO 客户端
    try:
        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        logger.info("Connected to MinIO")
    except Exception as e:
        logger.error(f"Could not connect to MinIO: {e}")
        print("\n⚠️  Please ensure MinIO is running on localhost:9000")
        return

    # 获取对象列表
    bucket_name = "dali-tutorial"
    try:
        objects = list(client.list_objects(bucket_name, prefix="images/", recursive=True))
        object_names = [obj.object_name for obj in objects]

        if not object_names:
            logger.warning(f"No objects found in {bucket_name}/images/")
            print("\n⚠️  Please run 09_minio_basic.py first to create sample data")
            return

        logger.info(f"Found {len(object_names)} objects")

    except S3Error as e:
        logger.error(f"Error accessing bucket: {e}")
        return

    # 创建数据源
    data_source = MinIODataSourceWithCache(
        client=client,
        bucket_name=bucket_name,
        object_names=object_names,
        cache_size=50,
        max_retries=3
    )

    # 创建 Pipeline
    pipe = production_minio_pipeline(
        minio_source=data_source,
        image_size=224,
        is_training=True,
        batch_size=8,
        num_threads=4,
        device_id=0,
        prefetch_queue_depth=2  # 预取深度
    )
    pipe.build()

    logger.info("Pipeline built successfully")

    # 创建迭代器
    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        size=len(object_names),
        auto_reset=True
    )

    # 性能测试
    print("\nPerformance Test:")
    num_epochs = 2
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        for i, batch in enumerate(dali_iter):
            data = batch[0]
            images = data["images"]
            labels = data["labels"]

            # 模拟训练
            time.sleep(0.001)

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")

    total_time = time.time() - start_time
    total_images = len(object_names) * num_epochs
    throughput = total_images / total_time

    print(f"\nResults:")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Throughput: {throughput:.0f} images/sec")

    # 打印统计信息
    stats = data_source.get_stats()
    print(f"\nCache Statistics:")
    print(f"  - Total requests: {stats['total_requests']}")
    print(f"  - Cache hits: {stats['cache_hits']}")
    print(f"  - Cache misses: {stats['cache_misses']}")
    print(f"  - Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  - Errors: {stats['errors']}")
    print(f"  - Retries: {stats['retries']}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 10: Production MinIO Pipeline")
    print("="*60)

    demo_production_pipeline()

    print("\n" + "="*60)
    print("✓ Tutorial 10 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. 生产环境需要完善的错误处理和重试机制")
    print("2. 缓存可以显著提升性能")
    print("3. 监控和日志对于问题排查很重要")
    print("4. 预取和多线程可以提高吞吐量")
    print("5. DALI 可以高效处理对象存储数据")
    print("\n🎉 恭喜！你已完成 DALI 基础教程")
    print("现在你可以构建高性能的数据流水线从 MinIO 加载数据了！")


if __name__ == "__main__":
    main()
