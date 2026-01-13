"""
04 - PyTorch 集成

学习目标：
1. 掌握 DALIGenericIterator 的使用
2. 理解 DALI 与 PyTorch DataLoader 的集成
3. 学习在训练循环中使用 DALI
4. 对比 DALI 与原生 PyTorch 的性能

核心概念：
- DALIGenericIterator: DALI 的 PyTorch 迭代器
- 与 torch.utils.data.DataLoader 的对比
- Epoch 管理和重置
- 性能优化技巧
"""

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import os
import time
from PIL import Image


def create_classification_dataset(output_dir="classification_data", num_images=100, num_classes=10):
    """创建分类任务的数据集"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating classification dataset: {num_images} images, {num_classes} classes")

    # 创建类别目录
    for class_id in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        # 每个类别创建若干图像
        images_per_class = num_images // num_classes
        for i in range(images_per_class):
            # 为每个类别创建不同颜色的图像
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_array[:, :, class_id % 3] = (class_id * 25) % 256  # 特征化颜色

            img = Image.fromarray(img_array, mode='RGB')
            filename = os.path.join(class_dir, f"img_{i:04d}.jpg")
            img.save(filename, quality=95)

    print(f"✓ Created {num_images} images in {num_classes} classes")
    return output_dir


@pipeline_def
def training_pipeline(data_dir, image_size=224):
    """
    训练用的 DALI Pipeline

    包含完整的数据增强
    """
    # readers.file 可以自动从目录结构推断标签
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        name="Reader"
    )

    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # 训练增强
    images = fn.random_resized_crop(
        images,
        size=image_size,
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.33]
    )

    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),
        contrast=fn.random.uniform(range=[0.8, 1.2])
    )

    # 归一化
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.normalize(
        images,
        mean=[0.485, 0.456, 0.406],
        stddev=[0.229, 0.224, 0.225],
        axes=(2,)
    )

    images = fn.transpose(images, perm=[2, 0, 1])  # CHW

    return images, labels


def demo_dali_iterator():
    """演示 DALIGenericIterator 的基本使用"""
    print("\n" + "="*60)
    print("Demo 1: DALIGenericIterator Basics")
    print("="*60)

    data_dir = create_classification_dataset(num_images=50, num_classes=5)

    # 创建 Pipeline
    pipe = training_pipeline(
        data_dir=data_dir,
        batch_size=8,
        num_threads=4,
        device_id=0,
        seed=42
    )
    pipe.build()

    # 创建 DALIGenericIterator
    # - pipelines: Pipeline 列表（可以多个）
    # - output_map: 输出名称映射
    # - size: 数据集大小（用于进度条）
    # - last_batch_policy: 最后一批的处理方式
    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        size=50,  # 数据集大小
        last_batch_policy=LastBatchPolicy.PARTIAL,  # 保留不完整的最后一批
        auto_reset=True  # epoch 结束后自动重置
    )

    print(f"\n✓ Created DALIGenericIterator")
    print(f"  - Dataset size: {len(dali_iter)}")
    print(f"  - Batch size: 8")

    # 迭代数据
    print(f"\nIterating through one epoch:")
    for i, batch in enumerate(dali_iter):
        # batch 是一个列表，每个 pipeline 一个元素
        data = batch[0]

        images = data["images"]  # PyTorch Tensor
        labels = data["labels"]  # PyTorch Tensor

        print(f"  Batch {i+1}: images {images.shape}, labels {labels.shape}")

        if i >= 2:  # 只显示前 3 个批次
            print(f"  ... ({len(dali_iter) - 3} more batches)")
            break


def demo_training_loop():
    """演示在训练循环中使用 DALI"""
    print("\n" + "="*60)
    print("Demo 2: Training Loop with DALI")
    print("="*60)

    data_dir = create_classification_dataset(num_images=80, num_classes=5)

    # 简单的 CNN 模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = SimpleCNN(num_classes=5).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建 DALI Pipeline 和 Iterator
    pipe = training_pipeline(
        data_dir=data_dir,
        image_size=64,
        batch_size=16,
        num_threads=4,
        device_id=0
    )
    pipe.build()

    train_loader = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        size=80,
        auto_reset=True
    )

    # 训练循环
    num_epochs = 2
    print(f"\nTraining for {num_epochs} epochs:")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            data = batch[0]
            images = data["images"]
            labels = data["labels"].squeeze().long()

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

    print(f"\n✓ Training completed")


class PyTorchDataset(Dataset):
    """PyTorch 原生 Dataset 用于性能对比"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # 收集所有图像路径和标签
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        class_idx
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def demo_performance_comparison():
    """对比 DALI 和 PyTorch 原生 DataLoader 的性能"""
    print("\n" + "="*60)
    print("Demo 3: Performance Comparison")
    print("="*60)

    data_dir = create_classification_dataset(num_images=200, num_classes=10)
    batch_size = 32
    num_iterations = 20

    # PyTorch 原生 DataLoader
    print(f"\n1. PyTorch DataLoader:")
    transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pytorch_dataset = PyTorchDataset(data_dir, transform=transform)
    pytorch_loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 预热
    for _ in pytorch_loader:
        break

    start = time.time()
    count = 0
    for i, (images, labels) in enumerate(pytorch_loader):
        images = images.cuda()
        labels = labels.cuda()
        count += 1
        if count >= num_iterations:
            break
    pytorch_time = time.time() - start

    print(f"  - Time: {pytorch_time:.3f}s")
    print(f"  - Throughput: {(batch_size * num_iterations / pytorch_time):.0f} images/sec")

    # DALI
    print(f"\n2. DALI Pipeline:")
    pipe = training_pipeline(
        data_dir=data_dir,
        image_size=64,
        batch_size=batch_size,
        num_threads=4,
        device_id=0
    )
    pipe.build()

    dali_loader = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "labels"],
        size=200,
        auto_reset=True
    )

    # 预热
    for _ in dali_loader:
        break
    dali_loader.reset()

    start = time.time()
    count = 0
    for batch in dali_loader:
        data = batch[0]
        images = data["images"]
        labels = data["labels"]
        count += 1
        if count >= num_iterations:
            break
    dali_time = time.time() - start

    print(f"  - Time: {dali_time:.3f}s")
    print(f"  - Throughput: {(batch_size * num_iterations / dali_time):.0f} images/sec")

    print(f"\nSpeedup: {pytorch_time/dali_time:.2f}x")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DALI Tutorial 04: PyTorch Integration")
    print("="*60)

    # 运行演示
    demo_dali_iterator()
    demo_training_loop()
    demo_performance_comparison()

    print("\n" + "="*60)
    print("✓ Tutorial 04 completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. DALIGenericIterator 封装 DALI Pipeline 为 PyTorch 迭代器")
    print("2. output_map 指定输出的名称")
    print("3. auto_reset=True 自动处理 epoch 重置")
    print("4. DALI 通常比原生 DataLoader 快 2-5x")
    print("5. DALI 的数据已经在 GPU 上，无需额外传输")
    print("\nNext: 05_conditional_pipeline.py - 学习条件执行")


if __name__ == "__main__":
    main()
