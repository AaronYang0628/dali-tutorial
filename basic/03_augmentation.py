"""
03 - 数据增强

学习目标：
1. 掌握常用的数据增强技术
2. 理解随机增强操作
3. 学习组合多个增强操作
4. 了解增强参数的随机化

核心概念：
- fn.random_resized_crop: 随机裁剪
- fn.color_twist: 颜色抖动
- fn.brightness_contrast: 亮度对比度调整
- fn.rotate: 旋转
- fn.gaussian_blur: 高斯模糊
- fn.coin_flip: 随机决策
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as math
import numpy as np
import os
import glob
from PIL import Image


def create_varied_images(output_dir="varied_images", num_images=20):
    """创建不同内容的测试图像"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_images} varied images...")

    for i in range(num_images):
        # 创建渐变图像便于观察增强效果
        height, width = 256, 256
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # 红色通道：水平渐变
        img_array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]

        # 绿色通道：垂直渐变
        img_array[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]

        # 蓝色通道：随机值
        img_array[:, :, 2] = np.random.randint(0, 255, (height, width), dtype=np.uint8)

        img = Image.fromarray(img_array, mode='RGB')
        img.save(os.path.join(output_dir, f"image_{i:03d}.jpg"), quality=95)

    print(f"✓ Created {num_images} varied images in {output_dir}/")
    return output_dir


@pipeline_def
def random_crop_pipeline(file_list):
    """
    随机裁剪 Pipeline

    fn.random_resized_crop 是最常用的训练时数据增强
    """
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # fn.random_resized_crop: 随机大小和宽高比裁剪
    # - size: 输出尺寸
    # - random_area: 裁剪面积范围 [min, max]，相对于原图
    # - random_aspect_ratio: 宽高比范围 [min, max]
    images = fn.random_resized_crop(
        images,
        size=224,
        random_area=[0.08, 1.0],  # 裁剪 8%-100% 的面积
        random_aspect_ratio=[0.75, 1.33]  # 3:4 到 4:3 的宽高比
    )

    return images, labels


@pipeline_def
def color_augmentation_pipeline(file_list):
    """
    颜色增强 Pipeline

    包含亮度、对比度、饱和度、色调调整
    """
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=224)

    # fn.brightness_contrast: 调整亮度和对比度
    # - brightness: 亮度调整范围
    # - contrast: 对比度调整范围
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),  # ±20%
        contrast=fn.random.uniform(range=[0.8, 1.2])  # ±20%
    )

    # fn.saturation: 饱和度调整
    images = fn.saturation(
        images,
        saturation=fn.random.uniform(range=[0.5, 1.5])  # 50%-150%
    )

    # fn.hue: 色调调整
    # hue 值范围 [-180, 180] 度
    images = fn.hue(
        images,
        hue=fn.random.uniform(range=[-20, 20])
    )

    return images, labels


@pipeline_def
def geometric_augmentation_pipeline(file_list):
    """
    几何增强 Pipeline

    包含旋转、翻转、仿射变换
    """
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=256)

    # fn.rotate: 随机旋转
    # - angle: 旋转角度
    # - fill_value: 填充值
    images = fn.rotate(
        images,
        angle=fn.random.uniform(range=[-15, 15]),  # ±15 度
        fill_value=0,
        keep_size=True
    )

    # 中心裁剪到 224x224
    images = fn.crop(images, crop=(224, 224), crop_pos_x=0.5, crop_pos_y=0.5)

    # fn.flip: 随机水平翻转
    # 使用 coin_flip 以 0.5 概率翻转
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5)
    )

    return images, labels


@pipeline_def
def advanced_augmentation_pipeline(file_list):
    """
    高级增强 Pipeline

    组合多种增强技术，模拟真实训练场景
    """
    images, labels = fn.readers.file(files=file_list, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # Step 1: 随机裁剪和缩放
    images = fn.random_resized_crop(
        images,
        size=224,
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.33]
    )

    # Step 2: 随机水平翻转
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5)
    )

    # Step 3: 颜色抖动（ColorJitter）
    # 随机调整亮度、对比度、饱和度
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=[0.6, 1.4]),
        contrast=fn.random.uniform(range=[0.6, 1.4])
    )

    images = fn.saturation(
        images,
        saturation=fn.random.uniform(range=[0.6, 1.4])
    )

    images = fn.hue(
        images,
        hue=fn.random.uniform(range=[-30, 30])
    )

    # Step 4: 随机高斯模糊（以概率 0.1 应用）
    should_blur = fn.random.coin_flip(probability=0.1)

    # 注意：DALI 的条件执行需要使用 if/else
    # 这里使用 gaussian_blur 的窗口大小为 0 表示不模糊
    blur_sigma = fn.random.uniform(range=[0.1, 2.0])
    images = fn.gaussian_blur(
        images,
        window_size=5 * should_blur,  # 0 或 5
        sigma=blur_sigma
    )

    # Step 5: 转换为 float 并归一化
    images = fn.cast(images, dtype=types.FLOAT) / 255.0

    # Step 6: 转换为 CHW 格式
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def demo_random_crop():
    """演示随机裁剪"""
    print("\n" + "="*60)
    print("Demo 1: Random Resized Crop")
    print("="*60)

    data_dir = create_varied_images(num_images=10)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = random_crop_pipeline(
        file_list=file_list,
        batch_size=4,
        num_threads=2,
        device_id=0,
        seed=42  # 设置随机种子便于复现
    )
    pipe.build()

    print(f"\nRunning 3 iterations to see randomness:")
    for i in range(3):
        outputs = pipe.run()
        images_batch = outputs[0]

        # 获取第一张图像的统计信息
        img = np.array(images_batch.as_cpu()[0])
        print(f" First Image In Iteration {i+1}: shape={img.shape}, "
              f"mean={np.mean(img):.1f}, std={np.std(img):.1f}")


def demo_color_augmentation():
    """演示颜色增强"""
    print("\n" + "="*60)
    print("Demo 2: Color Augmentation")
    print("="*60)

    data_dir = create_varied_images(num_images=10)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = color_augmentation_pipeline(
        file_list=file_list,
        batch_size=4,
        num_threads=2,
        device_id=0,
        seed=123
    )
    pipe.build()

    print(f"\nOriginal and augmented color statistics:")

    for i in range(2):
        outputs = pipe.run()
        images_batch = outputs[0]

        img = np.array(images_batch.as_cpu()[0])
        # 按通道统计
        r_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        b_mean = np.mean(img[:, :, 2])

        print(f"  Iteration {i+1}: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")


def demo_geometric_augmentation():
    """演示几何增强"""
    print("\n" + "="*60)
    print("Demo 3: Geometric Augmentation")
    print("="*60)

    data_dir = create_varied_images(num_images=10)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = geometric_augmentation_pipeline(
        file_list=file_list,
        batch_size=4,
        num_threads=2,
        device_id=0,
        seed=456
    )
    pipe.build()

    outputs = pipe.run()
    images_batch = outputs[0]

    print(f"\n✓ Applied rotation and flip augmentation")
    print(f"  - Output shape: {images_batch.shape()}")


def demo_full_augmentation():
    """演示完整的增强 Pipeline"""
    print("\n" + "="*60)
    print("Demo 4: Full Augmentation Pipeline")
    print("="*60)

    data_dir = create_varied_images(num_images=20)
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(file_list)} image files")

    pipe = advanced_augmentation_pipeline(
        file_list=file_list,
        batch_size=8,
        num_threads=4,
        device_id=0,
        seed=789
    )
    pipe.build()

    import time

    # 性能测试
    num_iterations = 10
    print(f"\nPerformance test ({num_iterations} iterations):")

    start = time.time()
    for _ in range(num_iterations):
        pipe.run()
    elapsed = time.time() - start

    total_images = num_iterations * 8
    print(f"  - Total time: {elapsed:.3f}s")
    print(f"  - Throughput: {total_images / elapsed:.0f} images/sec")

    # 检查输出
    outputs = pipe.run()
    images_batch = outputs[0]

    print(f"\n✓ Full augmentation pipeline")
    print(f"  - Output shape: {images_batch.shape()}")
    print(f"  - Output dtype: {images_batch.dtype}")
    img_sample = np.array(images_batch.as_cpu()[0])
    print(f"  - Output range: [{np.min(img_sample):.3f}, "
          f"{np.max(img_sample):.3f}]")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 03: Data Augmentation")
    print("="*60)

    # 运行演示
    demo_random_crop()
    # demo_color_augmentation()
    # demo_geometric_augmentation()
    # demo_full_augmentation()

    print("\n" + "="*60)
    print("✓ Tutorial 03 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. fn.random_resized_crop 用于训练时的随机裁剪")
    print("2. 颜色增强包括亮度、对比度、饱和度、色调调整")
    print("3. 几何增强包括旋转、翻转、仿射变换")
    print("4. 使用 fn.random.uniform/coin_flip 生成随机参数")
    print("5. 组合多种增强技术可以提升模型泛化能力")
    print("\nNext: 04_pytorch_integration.py - 学习与 PyTorch 集成")


if __name__ == "__main__":
    main()
