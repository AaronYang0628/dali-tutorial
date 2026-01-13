"""
02 - 基础图像处理

学习目标：
1. 掌握图像解码操作
2. 学习基本的图像变换 (resize, crop, flip)
3. 理解 CPU vs GPU 操作的区别
4. 了解数据类型和输出格式

核心概念：
- fn.decoders.image: 图像解码
- fn.resize: 图像缩放
- fn.crop: 图像裁剪
- fn.flip: 图像翻转
- Device: CPU 或 GPU 执行设备
- Output layout: 数据排列格式 (HWC vs CHW)
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os
import glob
from PIL import Image


def create_sample_images(output_dir="sample_images", num_images=10):
    """创建真实的 JPEG 图像用于测试"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_images} sample JPEG images...")

    for i in range(num_images):
        # 创建随机 RGB 图像
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')

        # 保存为 JPEG
        filename = os.path.join(output_dir, f"image_{i:03d}.jpg")
        img.save(filename, quality=95)

    print(f"✓ Created {num_images} JPEG images in {output_dir}/")
    return output_dir


@pipeline_def
def decode_and_resize_pipeline(file_list, target_size=224):
    """
    基础图像处理 Pipeline：解码 → 缩放

    Args:
        file_list: 图像文件路径列表
        target_size: 目标大小

    Returns:
        解码和缩放后的图像，以及标签
    """
    # 读取 JPEG 文件
    images, labels = fn.readers.file(
        files=file_list,
        random_shuffle=True
    )

    # fn.decoders.image: 解码 JPEG 图像
    # - device: 'cpu' 在 CPU 解码，'mixed' 使用 nvJPEG (GPU)
    # - output_type: 输出格式 (RGB, GRAY, BGR 等)
    images = fn.decoders.image(
        images,
        device="mixed",  # 使用硬件加速解码
        output_type=types.RGB
    )

    # fn.resize: 调整图像大小
    # - resize_x, resize_y: 目标大小，或使用单一值
    # - resize_longer: 保持宽高比，调整较长边到指定大小
    images = fn.resize(
        images,
        size=target_size,
        mode="not_smaller",  # 至少这么大
        interp_type=types.INTERP_LINEAR
    )

    return images, labels


@pipeline_def
def augmentation_pipeline(file_list, target_size=224):
    """
    包含多个增强操作的 Pipeline

    Args:
        file_list: 图像文件路径列表
        target_size: 目标大小
    """
    images, labels = fn.readers.file(
        files=file_list,
        random_shuffle=True
    )

    # 解码
    images = fn.decoders.image(
        images,
        device="mixed",
        output_type=types.RGB
    )

    # 缩放到目标大小
    images = fn.resize(
        images,
        size=target_size,
        mode="not_smaller"
    )

    # fn.crop: 中心裁剪
    # - crop: 裁剪大小 (height, width)
    # - crop_pos_x, crop_pos_y: 裁剪位置 (0-1 范围内)
    images = fn.crop(
        images,
        crop=(target_size, target_size),  # (height, width)
        crop_pos_x=0.5,  # 从中心开始
        crop_pos_y=0.5
    )

    # fn.flip: 随机翻转
    # - horizontal: 水平翻转
    # - vertical: 垂直翻转
    images = fn.flip(
        images,
        horizontal=True,
        vertical=False
    )

    # fn.cast: 数据类型转换
    # 将 uint8 (0-255) 转换为 float32 并归一化到 [0, 1]
    images = fn.cast(
        images,
        dtype=types.FLOAT
    )

    # 简单归一化：除以 255 将范围转到 [0, 1]
    images = images / 255.0

    # fn.transpose: 数据转置
    # - perm: 维度排列 (0,1,2) -> (2,0,1) 从 HWC 转到 CHW
    images = fn.transpose(
        images,
        perm=[2, 0, 1]  # HWC -> CHW
    )

    return images, labels


def demo_decode_pipeline():
    """演示图像解码"""
    print("\n" + "="*60)
    print("Demo 1: Image Decoding")
    print("="*60)

    data_dir = create_sample_images(num_images=5)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = decode_and_resize_pipeline(
        file_list=file_list,
        target_size=224,
        batch_size=2,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    outputs = pipe.run()
    images_batch, labels_batch = outputs

    print(f"\n✓ Images decoded and resized")
    print(f"  - Output shape: {images_batch.shape()}")
    print(f"  - Output dtype: {images_batch.dtype}")
    device_name = "GPU" if "GPU" in str(type(images_batch)) else "CPU"
    print(f"  - Output device: {device_name}")

    # 获取第一张图像的统计信息
    # 将 GPU tensor 转到 CPU 并转换为 numpy array
    img = np.array(images_batch.as_cpu()[0])
    print(f"\nFirst image statistics:")
    print(f"  - Min value: {np.min(img)}")
    print(f"  - Max value: {np.max(img)}")
    print(f"  - Mean value: {np.mean(img):.2f}")


def demo_augmentation_pipeline():
    """演示数据增强"""
    print("\n" + "="*60)
    print("Demo 2: Image Augmentation")
    print("="*60)

    data_dir = create_sample_images(num_images=8)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = augmentation_pipeline(
        file_list=file_list,
        target_size=224,
        batch_size=4,
        num_threads=2,
        device_id=0
    )
    pipe.build()

    outputs = pipe.run()
    images_batch, labels_batch = outputs

    print(f"\n✓ Images augmented and normalized")
    print(f"  - Output shape: {images_batch.shape()}")
    print(f"  - Output dtype: {images_batch.dtype}")
    print(f"  - Layout: CHW (Channel, Height, Width)")

    # 检查归一化结果
    # 将 GPU tensor 转到 CPU 并转换为 numpy array
    img = np.array(images_batch.as_cpu()[0])
    print(f"\nFirst image after normalization:")
    print(f"  - Min value: {np.min(img):.4f}")
    print(f"  - Max value: {np.max(img):.4f}")
    print(f"  - Mean value: {np.mean(img):.4f}")


def demo_device_comparison():
    """比较 CPU 和 GPU 解码的性能"""
    print("\n" + "="*60)
    print("Demo 3: CPU vs GPU Decoding Performance")
    print("="*60)

    import time

    data_dir = create_sample_images(num_images=100)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    # CPU 解码版本
    @pipeline_def
    def cpu_decode_pipeline(file_list):
        images, labels = fn.readers.file(files=file_list, random_shuffle=False)
        images = fn.decoders.image(images, device="cpu", output_type=types.RGB)
        images = fn.resize(images, size=224)
        return images, labels

    # GPU 解码版本 (mixed device)
    @pipeline_def
    def gpu_decode_pipeline(file_list):
        images, labels = fn.readers.file(files=file_list, random_shuffle=False)
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, size=224)
        return images, labels

    batch_size = 32
    num_iterations = 3

    # 测试 CPU 解码
    print(f"\nCPU decoding (batch_size={batch_size}):")
    pipe_cpu = cpu_decode_pipeline(
        file_list=file_list,
        batch_size=batch_size,
        num_threads=4,
        device_id=0
    )
    pipe_cpu.build()

    start = time.time()
    for _ in range(num_iterations):
        pipe_cpu.run()
    cpu_time = time.time() - start
    print(f"  - Time: {cpu_time:.3f}s")
    print(f"  - Throughput: {(batch_size * num_iterations / cpu_time):.0f} images/sec")

    # 测试 GPU 解码
    print(f"\nGPU decoding (mixed device, batch_size={batch_size}):")
    pipe_gpu = gpu_decode_pipeline(
        file_list=file_list,
        batch_size=batch_size,
        num_threads=4,
        device_id=0
    )
    pipe_gpu.build()

    start = time.time()
    for _ in range(num_iterations):
        pipe_gpu.run()
    gpu_time = time.time() - start
    print(f"  - Time: {gpu_time:.3f}s")
    print(f"  - Throughput: {(batch_size * num_iterations / gpu_time):.0f} images/sec")

    print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 02: Basic Image Processing")
    print("="*60)

    # 运行演示
    # demo_decode_pipeline()
    # demo_augmentation_pipeline()
    demo_device_comparison()

    print("\n" + "="*60)
    print("✓ Tutorial 02 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. fn.decoders.image 解码 JPEG 图像")
    print("2. fn.resize/crop/flip 进行基本的几何变换")
    print("3. 'mixed' device 使用硬件加速，比 'cpu' 快得多")
    print("4. fn.cast/normalize 进行数据预处理")
    print("5. fn.transpose 转换数据布局")
    print("\nNext: 03_augmentation.py - 学习高级数据增强")


if __name__ == "__main__":
    main()
