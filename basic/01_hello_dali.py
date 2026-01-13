"""
01 - Hello DALI: 第一个 DALI 程序

学习目标：
1. 理解 DALI Pipeline 的基本结构
2. 学习使用 @pipeline_def 装饰器
3. 了解 Pipeline 的构建和运行流程
4. 掌握基本的文件读取操作

核心概念：
- Pipeline: 数据处理流水线，定义了数据的处理流程
- @pipeline_def: 装饰器，用于定义 Pipeline
- fn.readers.file: 文件读取器
- pipeline.build(): 构建计算图
- pipeline.run(): 执行一次迭代
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os
import glob


def create_sample_data(output_dir="sample_data", num_images=10):
    """创建示例图像数据用于测试"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_images} sample images in {output_dir}/")

    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Please run: pip install Pillow")
        return output_dir

    for i in range(num_images):
        # 创建随机图像 (224x224, RGB)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # 转换为 PIL Image
        img = Image.fromarray(img_array, mode='RGB')

        # 保存为 JPEG
        img.save(f"{output_dir}/image_{i:03d}.jpg", quality=95)

    print(f"✓ Created {num_images} sample images")
    return output_dir


@pipeline_def
def simple_pipeline(file_list):
    """
    最简单的 DALI Pipeline

    Args:
        file_list: 文件路径列表

    Returns:
        读取的文件内容
    """
    # fn.readers.file: 从文件列表读取文件,只读取原始文件的字节数，没有解码成为图像
    # - files: 文件路径列表
    # - random_shuffle: 是否随机打乱
    images, labels = fn.readers.file(
        files=file_list,
        random_shuffle=True
    )

    # 返回数据（此时还是原始字节）
    return images, labels


@pipeline_def
def simple_image_pipeline(file_list):
    """
    带图像解码的 Pipeline

    增加了图像解码步骤，将原始字节解码为图像张量
    """
    # 读取文件
    images, labels = fn.readers.file(
        files=file_list,
        random_shuffle=True
    )

    # fn.decoders.image: 解码图像
    # - device: 解码设备 ('cpu' 或 'mixed')
    # - output_type: 输出类型 (RGB 或 GRAY)
    images = fn.decoders.image(
        images,
        device="mixed",
        output_type=types.RGB
    )

    return images, labels


def demo_simple_pipeline():
    """演示最简单的 Pipeline 使用"""
    print("\n" + "="*60)
    print("Demo 1: Simple File Reading Pipeline")
    print("="*60)

    # 创建测试数据
    data_dir = create_sample_data(num_images=5)
    
    # 获取文件列表
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not file_list:
        print("Error: No image files found!")
        return
    
    print(f"Found {len(file_list)} image files")

    # 创建 Pipeline 实例
    # - batch_size: 每批次样本数
    # - num_threads: 数据加载线程数
    # - device_id: GPU 设备 ID
    pipe = simple_pipeline(
        file_list=file_list,
        batch_size=3,
        num_threads=2,
        device_id=0
    )

    # 构建 Pipeline（编译计算图）
    pipe.build()

    print(f"\n✓ Pipeline built successfully")
    print(f"  - Batch size: {pipe.max_batch_size}")
    print(f"  - Num threads: {pipe.num_threads}")
    print(f"  - Device ID: {pipe.device_id}")

    # 运行一次迭代
    print(f"\nRunning pipeline first iteration...")
    outputs = pipe.run()

    # outputs 是一个列表，包含 Pipeline 返回的所有输出
    # images 是文件在列表中的字节数
    # labels 是文件在列表中的索引：
    #   - image_000.jpg → label = 0
    #   - image_001.jpg → label = 1
    #   - image_002.jpg → label = 2
    images_batch, labels_batch = outputs

    print(f"\n✓ Pipeline executed successfully")
    print("  - Index of iteration: 1")
    print(f"  - Number of outputs: {len(outputs)}")
    print(f"  - Images batch shape: {images_batch.shape()}")
    print(f"  - Labels batch shape: {labels_batch.shape()}")

    # 访问单个样本
    print(f"\nFirst sample in batch:")
    print(f"  - Image data type: {type(images_batch.at(0))}")
    print(f"  - Label: {labels_batch.at(0)}")


def demo_multiple_iterations():
    """演示多次迭代"""
    print("\n" + "="*60)
    print("Demo 2: Multiple Iterations")
    print("="*60)

    data_dir = create_sample_data(num_images=8)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

    pipe = simple_pipeline(
        file_list=file_list,
        batch_size=3,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    # 运行多个迭代
    num_iterations = 3
    print(f"\nRunning {num_iterations} iterations:")

    for i in range(num_iterations):
        outputs = pipe.run()
        images_batch, labels_batch = outputs

        batch_size = len(images_batch)
        print(f"  Iteration {i+1}: Got {batch_size} samples")


def demo_pipeline_reset():
    """演示 Pipeline 重置"""
    print("\n" + "="*60)
    print("Demo 3: Pipeline Reset")
    print("="*60)

    data_dir = create_sample_data(num_images=6)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

    pipe = simple_pipeline(
        file_list=file_list,
        batch_size=2,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    print(f"\nFirst epoch:")
    for i in range(3):
        outputs = pipe.run()
        print(f"  Iteration {i+1}")

    # 重置 Pipeline（从头开始）
    print(f"\nResetting pipeline...")
    pipe.reset()

    print(f"\nSecond epoch (after reset):")
    for i in range(3):
        outputs = pipe.run()
        print(f"  Iteration {i+1}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 01: Hello DALI")
    print("="*60)

    # 检查 DALI 版本
    print(f"\nNVIDIA DALI version: {dali.__version__}")

    # 运行演示
    demo_simple_pipeline()
    demo_multiple_iterations()
    demo_pipeline_reset()

    print("\n" + "="*60)
    print("✓ Tutorial 01 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Pipeline 使用 @pipeline_def 装饰器定义")
    print("2. 需要先 build() 再 run()")
    print("3. run() 返回一批数据")
    print("4. 可以通过 reset() 重新开始迭代")
    print("\nNext: 02_basic_image_processing.py - 学习图像处理操作")


if __name__ == "__main__":
    main()
