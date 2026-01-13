"""
05 - 外部数据源

学习目标：
1. 掌握 fn.external_source 的使用
2. 学习自定义数据加载逻辑
3. 理解回调函数的编写
4. 掌握批次和样本迭代模式

核心概念：
- fn.external_source: 接入自定义数据
- Callback function: 数据生成函数
- batch vs no_copy 模式
- 与其他数据源集成
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import numpy as np
import os


class CustomDataSource:
    """
    自定义数据源示例

    模拟从数据库、网络或其他存储读取数据
    """
    def __init__(self, num_samples=100, image_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.current_index = 0

        print(f"Initialized CustomDataSource: {num_samples} samples")

    def __iter__(self):
        """迭代器协议：返回自身"""
        self.current_index = 0
        return self

    def __next__(self):
        """
        迭代器协议：返回下一个样本

        DALI 会重复调用此方法获取单个样本
        """
        if self.current_index >= self.num_samples:
            raise StopIteration

        # 生成一个随机图像
        image = np.random.randint(
            0, 255,
            (*self.image_size, 3),
            dtype=np.uint8
        )

        # 生成对应的标签
        label = self.current_index % 10

        self.current_index += 1

        return image, label

    def __call__(self, sample_info):
        """
        回调函数模式：DALI 调用此方法获取样本

        Args:
            sample_info: 包含当前迭代信息的对象
                - idx_in_epoch: 样本在 epoch 中的索引
                - idx_in_batch: 样本在批次中的索引
                - iteration: 当前迭代编号

        Returns:
            单个样本或批次数据
        """
        idx = sample_info.idx_in_epoch

        if idx >= self.num_samples:
            raise StopIteration

        image = np.random.randint(
            0, 255,
            (*self.image_size, 3),
            dtype=np.uint8
        )
        label = idx % 10

        return image, label


@pipeline_def
def external_source_pipeline(source, image_size=64):
    """
    使用外部数据源的 Pipeline

    Args:
        source: 数据源对象（callable）
        image_size: 图像大小
    """
    # fn.external_source: 从外部数据源读取
    # - source: 回调函数或迭代器
    # - num_outputs: 输出数量（如果返回元组）
    # - batch: 是否批量读取
    # - no_copy: 是否避免拷贝（返回 numpy array）
    images, labels = fn.external_source(
        source=source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=False  # 每次返回单个样本
    )

    # 处理图像
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])  # HWC -> CHW

    return images, labels


class BatchDataSource:
    """批量数据源：一次返回整批数据"""
    def __init__(self, batch_size, num_batches=10, image_size=(64, 64)):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.image_size = image_size
        self.current_batch = 0

    def __call__(self, sample_info):
        """
        返回整批数据

        当 batch=True 时，DALI 期望返回整批数据
        """
        if self.current_batch >= self.num_batches:
            self.current_batch = 0
            raise StopIteration

        # 生成一批图像
        batch_images = []
        batch_labels = []

        for i in range(self.batch_size):
            image = np.random.randint(
                0, 255,
                (*self.image_size, 3),
                dtype=np.uint8
            )
            label = (self.current_batch * self.batch_size + i) % 10

            batch_images.append(image)
            batch_labels.append(label)

        self.current_batch += 1

        return batch_images, np.array(batch_labels, dtype=np.int32)


@pipeline_def
def batch_external_source_pipeline(source):
    """使用批量外部数据源的 Pipeline"""
    images, labels = fn.external_source(
        source=source,
        num_outputs=2,
        batch=True  # 批量读取模式
    )

    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_basic_external_source():
    """演示基本的外部数据源"""
    print("\n" + "="*60)
    print("Demo 1: Basic External Source")
    print("="*60)

    # 创建数据源
    data_source = CustomDataSource(num_samples=50, image_size=(64, 64))

    # 创建 Pipeline
    pipe = external_source_pipeline(
        source=data_source,
        batch_size=8,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    print(f"\n✓ Pipeline built with external source")

    # 迭代几批数据
    print(f"\nIterating through data:")
    for i in range(3):
        outputs = pipe.run()
        images, labels = outputs

        print(f"  Batch {i+1}: images {images.shape()}, labels {labels.shape()}")


def demo_batch_external_source():
    """演示批量外部数据源"""
    print("\n" + "="*60)
    print("Demo 2: Batch External Source")
    print("="*60)

    batch_size = 16
    data_source = BatchDataSource(
        batch_size=batch_size,
        num_batches=5,
        image_size=(64, 64)
    )

    pipe = batch_external_source_pipeline(
        source=data_source,
        batch_size=batch_size,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    print(f"\n✓ Pipeline built with batch external source")

    print(f"\nIterating through batches:")
    for i in range(3):
        outputs = pipe.run()
        images, labels = outputs

        print(f"  Batch {i+1}: images {images.shape()}")


def demo_numpy_array_source():
    """演示从 NumPy 数组读取数据"""
    print("\n" + "="*60)
    print("Demo 3: NumPy Array Source")
    print("="*60)

    # 预先生成所有数据（模拟从文件加载）
    num_samples = 100
    all_images = np.random.randint(
        0, 255,
        (num_samples, 64, 64, 3),
        dtype=np.uint8
    )
    all_labels = np.random.randint(0, 10, num_samples, dtype=np.int32)

    print(f"Loaded {num_samples} images into memory")

    class NumpyArraySource:
        """从 NumPy 数组提供数据"""
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            self.num_samples = len(images)

        def __call__(self, sample_info):
            idx = sample_info.idx_in_epoch
            if idx >= self.num_samples:
                raise StopIteration
            return self.images[idx], self.labels[idx]

    data_source = NumpyArraySource(all_images, all_labels)

    pipe = external_source_pipeline(
        source=data_source,
        batch_size=16,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    print(f"\n✓ Pipeline built")

    # 测试迭代
    outputs = pipe.run()
    images, labels = outputs

    print(f"\nFirst batch:")
    print(f"  - Images shape: {images.shape()}")
    print(f"  - Labels shape: {labels.shape()}")
    print(f"  - Sample labels: {labels.as_array().flatten()[:8]}")


def demo_mixed_source():
    """演示混合数据源（文件 + 外部源）"""
    print("\n" + "="*60)
    print("Demo 4: Mixed Sources")
    print("="*60)

    # 创建一些测试图像
    from PIL import Image
    test_dir = "mixed_source_data"
    os.makedirs(test_dir, exist_ok=True)

    for i in range(20):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(test_dir, f"img_{i:03d}.jpg"))

    # 外部元数据源
    class MetadataSource:
        """提供额外的元数据"""
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __call__(self, sample_info):
            idx = sample_info.idx_in_epoch
            if idx >= self.num_samples:
                raise StopIteration

            # 返回额外的元数据（例如：bbox, mask 等）
            metadata = {
                'confidence': np.random.rand(),
                'class_weight': np.random.rand(10)
            }
            return np.array([metadata['confidence']], dtype=np.float32)

    @pipeline_def
    def mixed_pipeline(data_dir, metadata_source):
        # 从文件读取图像
        images, labels = fn.readers.file(file_root=data_dir)
        images = fn.decoders.image(images, device="mixed")
        images = fn.resize(images, size=64)

        # 从外部源读取元数据
        metadata = fn.external_source(source=metadata_source, batch=False)

        images = fn.cast(images, dtype=types.FLOAT) / 255.0
        images = fn.transpose(images, perm=[2, 0, 1])

        return images, labels, metadata

    metadata_src = MetadataSource(num_samples=20)

    pipe = mixed_pipeline(
        data_dir=test_dir,
        metadata_source=metadata_src,
        batch_size=4,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    outputs = pipe.run()
    images, labels, metadata = outputs

    print(f"\n✓ Mixed pipeline executed")
    print(f"  - Images: {images.shape()}")
    print(f"  - Labels: {labels.shape()}")
    print(f"  - Metadata: {metadata.shape()}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 05: External Source")
    print("="*60)

    demo_basic_external_source()
    demo_batch_external_source()
    demo_numpy_array_source()
    demo_mixed_source()

    print("\n" + "="*60)
    print("✓ Tutorial 05 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. fn.external_source 可以接入任何自定义数据源")
    print("2. 支持单样本和批量两种模式")
    print("3. 可以与文件读取等其他数据源混合使用")
    print("4. 适合从数据库、网络等非文件系统读取数据")
    print("\nNext: 06_parallel_pipeline.py - 学习并行处理")


if __name__ == "__main__":
    main()
