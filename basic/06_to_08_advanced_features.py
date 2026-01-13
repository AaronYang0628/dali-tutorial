"""
06-08 - 高级特性合集

包含三个高级主题：
- 06: 并行处理和性能优化
- 07: 多 GPU 支持
- 08: 动态 Pipeline 配置
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import numpy as np
import os
import time
from PIL import Image


def create_large_dataset(output_dir="large_dataset", num_images=1000):
    """创建大型测试数据集"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating {num_images} images...")

    for i in range(num_images):
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f"img_{i:05d}.jpg"))

        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{num_images}")

    return output_dir


# ============================================================================
# 06 - 并行处理和性能优化
# ============================================================================

@pipeline_def
def optimized_pipeline(data_dir, num_threads=4, prefetch_depth=2):
    """
    优化的 Pipeline：并行处理、预取、设备亲和性

    关键参数：
    - num_threads: 数据加载线程数（通常等于 CPU 核数）
    - prefetch_queue_depth: 预取队列深度（增加内存消耗但提高吞吐）
    - device_id: GPU 设备 ID
    """
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)

    # GPU 解码
    images = fn.decoders.image(images, device="mixed")

    # GPU 操作
    images = fn.resize(images, size=224)
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_parallel_pipeline():
    """演示并行处理性能差异"""
    print("\n" + "="*60)
    print("Demo 06: Parallel Processing and Optimization")
    print("="*60)

    data_dir = create_large_dataset(num_images=200)

    batch_size = 32
    num_iterations = 10

    # 测试不同的线程数
    thread_configs = [1, 2, 4, 8]

    print(f"\nPerformance vs number of threads:")
    for num_threads in thread_configs:
        pipe = optimized_pipeline(
            data_dir=data_dir,
            num_threads=num_threads,
            prefetch_depth=2,
            batch_size=batch_size,
            device_id=0
        )
        pipe.build()

        start = time.time()
        for _ in range(num_iterations):
            pipe.run()
        elapsed = time.time() - start

        throughput = (batch_size * num_iterations) / elapsed
        print(f"  num_threads={num_threads}: {throughput:.0f} images/sec")


# ============================================================================
# 07 - 多 GPU 支持（数据分片）
# ============================================================================

@pipeline_def
def sharded_pipeline(data_dir, shard_id=0, num_shards=1):
    """
    支持多 GPU 的 Pipeline

    每个 GPU 处理数据集的一部分

    Args:
        shard_id: 当前 GPU 的分片 ID
        num_shards: 总 GPU 数量
    """
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards
    )

    images = fn.decoders.image(images, device="mixed")
    images = fn.resize(images, size=224)
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_multi_gpu():
    """演示多 GPU 支持"""
    print("\n" + "="*60)
    print("Demo 07: Multi-GPU Support")
    print("="*60)

    data_dir = create_large_dataset(num_images=300)

    # 模拟 2 个 GPU
    num_gpus = 2

    print(f"\nSimulating {num_gpus} GPUs with data sharding:")

    for gpu_id in range(num_gpus):
        pipe = sharded_pipeline(
            data_dir=data_dir,
            shard_id=gpu_id,
            num_shards=num_gpus,
            batch_size=16,
            device_id=0,  # 实际应用中使用 gpu_id
            num_threads=2
        )
        pipe.build()

        outputs = pipe.run()
        images = outputs[0]

        print(f"  GPU {gpu_id}: batch {images.shape()}")


# ============================================================================
# 08 - 动态 Pipeline 配置
# ============================================================================

@pipeline_def
def configurable_pipeline(
    data_dir,
    image_size=224,
    augmentation_strength=1.0,
    enable_blur=False
):
    """
    可配置的 Pipeline

    参数可以根据训练阶段动态调整
    """
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)

    images = fn.decoders.image(images, device="mixed")

    # 根据强度调整增强
    images = fn.resize(images, size=int(image_size * 1.2))

    # 随机裁剪，强度越高裁剪越激进
    min_crop_ratio = 0.08 / augmentation_strength
    images = fn.random_resized_crop(
        images,
        size=image_size,
        random_area=[min(min_crop_ratio, 1.0), 1.0]
    )

    # 条件模糊
    if enable_blur:
        images = fn.gaussian_blur(images, window_size=5, sigma=1.0)

    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_dynamic_config():
    """演示动态配置"""
    print("\n" + "="*60)
    print("Demo 08: Dynamic Pipeline Configuration")
    print("="*60)

    data_dir = create_large_dataset(num_images=150)

    print(f"\nSimulating training stages with different augmentation:")

    # 模拟不同训练阶段
    stages = [
        {"name": "Warmup", "strength": 0.5, "blur": False},
        {"name": "Main", "strength": 1.0, "blur": True},
        {"name": "Fine-tune", "strength": 0.3, "blur": False},
    ]

    for stage_config in stages:
        print(f"\n  Stage: {stage_config['name']}")

        pipe = configurable_pipeline(
            data_dir=data_dir,
            image_size=224,
            augmentation_strength=stage_config['strength'],
            enable_blur=stage_config['blur'],
            batch_size=8,
            num_threads=2,
            device_id=0
        )
        pipe.build()

        # 运行一个迭代
        outputs = pipe.run()
        images = outputs[0]

        print(f"    Output shape: {images.shape()}")
        print(f"    Augmentation strength: {stage_config['strength']}")
        print(f"    Blur enabled: {stage_config['blur']}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorials 06-08: Advanced Features")
    print("="*60)

    demo_parallel_pipeline()
    demo_multi_gpu()
    demo_dynamic_config()

    print("\n" + "="*60)
    print("✓ Tutorials 06-08 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("06. 多线程和预取对性能有显著影响")
    print("07. 使用 shard_id/num_shards 支持多 GPU")
    print("08. Pipeline 参数可以动态配置")
    print("\nNext: 09_minio_basic.py - 学习 MinIO 集成")


if __name__ == "__main__":
    main()
