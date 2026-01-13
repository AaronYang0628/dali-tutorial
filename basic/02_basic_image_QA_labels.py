"""
演示 fn.readers.file 的 labels 生成机制
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import numpy as np
import os
from PIL import Image

def create_sample_data():
    """创建示例数据"""
    os.makedirs("demo_data", exist_ok=True)

    for i in range(5):
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(f"demo_data/image_{i:03d}.jpg", quality=95)

    print("✓ Created 5 sample images in demo_data/")


@pipeline_def
def file_reader_pipeline(file_list):
    """简单的文件读取 pipeline"""
    images, labels = fn.readers.file(
        files=file_list,
        random_shuffle=False  # 关闭随机，方便观察
    )
    return images, labels


def demo_labels():
    """演示 labels 的内容"""
    print("\n" + "="*60)
    print("Demo: What are the labels?")
    print("="*60)

    create_sample_data()

    # 获取文件列表
    import glob
    file_list = sorted(glob.glob("demo_data/*.jpg"))

    print(f"\nFile list:")
    for i, f in enumerate(file_list):
        print(f"  [{i}] {f}")

    # 创建 pipeline
    pipe = file_reader_pipeline(
        file_list=file_list,
        batch_size=3,
        num_threads=1,
        device_id=0
    )
    pipe.build()

    # 运行第一个 batch
    print(f"\n--- Batch 1 ---")
    outputs = pipe.run()
    images_batch, labels_batch = outputs

    print(f"Batch size: {len(labels_batch)}")

    for i in range(len(labels_batch)):
        # 将 GPU tensor 转到 CPU
        label = np.array(labels_batch.as_cpu()[i])
        print(f"  Sample {i}: label = {label[0]} (对应文件: {file_list[label[0]]})")

    # 运行第二个 batch
    print(f"\n--- Batch 2 ---")
    outputs = pipe.run()
    images_batch, labels_batch = outputs

    print(f"Batch size: {len(labels_batch)}")

    for i in range(len(labels_batch)):
        label = np.array(labels_batch.as_cpu()[i])
        print(f"  Sample {i}: label = {label[0]} (对应文件: {file_list[label[0]]})")

    print(f"\n" + "="*60)
    print("结论: labels 是文件在 file_list 中的索引 (0-based)")
    print("="*60)


def demo_labels_with_shuffle():
    """演示启用 random_shuffle 时的情况"""
    print("\n" + "="*60)
    print("Demo: Labels with random_shuffle=True")
    print("="*60)

    import glob
    file_list = sorted(glob.glob("demo_data/*.jpg"))

    @pipeline_def
    def shuffled_pipeline(file_list):
        images, labels = fn.readers.file(
            files=file_list,
            random_shuffle=True  # 启用随机
        )
        return images, labels

    pipe = shuffled_pipeline(
        file_list=file_list,
        batch_size=3,
        num_threads=1,
        device_id=0
    )
    pipe.build()

    print(f"\n原始文件列表顺序:")
    for i, f in enumerate(file_list):
        print(f"  [{i}] {f}")

    print(f"\n启用随机打乱后:")
    outputs = pipe.run()
    images_batch, labels_batch = outputs

    for i in range(len(labels_batch)):
        label = np.array(labels_batch.as_cpu()[i])
        print(f"  Sample {i}: label = {label[0]} (对应文件: {file_list[label[0]]})")

    print(f"\n注意: 虽然文件顺序被打乱了，但 label 仍然是原始索引")


def demo_custom_labels():
    """演示如何使用自定义标签"""
    print("\n" + "="*60)
    print("Demo: Using Custom Labels")
    print("="*60)

    print("\n如果你想使用真实的分类标签（如猫=0，狗=1），有两种方式：")
    print("\n方式1: 使用目录结构")
    print("  data/")
    print("    ├── cat/          # 类别 0")
    print("    │   ├── img1.jpg")
    print("    └── dog/          # 类别 1")
    print("        └── img2.jpg")
    print("\n  images, labels = fn.readers.file(file_root='data/')")

    print("\n方式2: 手动创建标签映射")
    print("  # 创建自己的标签列表")
    print("  custom_labels = [0, 1, 0, 1, 2]  # 对应每个文件的真实标签")
    print("  # 在 pipeline 中使用 ExternalSource 传入")

    print("\n方式3: 忽略自动标签，使用外部标注文件")
    print("  # 读取标注文件（如 CSV, JSON）")
    print("  # 在 pipeline 中与图像配对")


if __name__ == "__main__":
    demo_labels()
    demo_labels_with_shuffle()
    demo_custom_labels()

    print("\n" + "="*60)
    print("✓ Demo completed!")
    print("="*60)
