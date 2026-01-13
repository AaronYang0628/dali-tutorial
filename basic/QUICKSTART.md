# DALI 快速入门指南

10 分钟快速上手 NVIDIA DALI！

## 目标

从 MinIO 对象存储读取图像数据，构建高性能数据流水线用于深度学习训练。

## 环境要求

```bash
# 检查环境
python -c "import nvidia.dali as dali; print(f'DALI: {dali.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
nvidia-smi
```

## 学习路径（3 小时）

### 第一步：基础（30 分钟）

```bash
# 运行基础示例
python basic/01_hello_dali.py          # 15 min - 理解 Pipeline 概念
python basic/02_basic_image_processing.py  # 15 min - 图像处理操作
```

**关键概念：**
- Pipeline：定义数据处理流程
- fn.readers.file：读取文件
- fn.decoders.image：解码图像
- fn.resize/crop/flip：基本图像操作

### 第二步：数据增强（30 分钟）

```bash
python basic/03_augmentation.py        # 30 min - 数据增强技术
```

**关键概念：**
- fn.random_resized_crop：随机裁剪
- fn.brightness_contrast：颜色调整
- fn.rotate/flip：几何变换
- fn.random.uniform：随机参数

### 第三步：PyTorch 集成（30 分钟）

```bash
python basic/04_pytorch_integration.py  # 30 min - 与 PyTorch 集成
```

**关键概念：**
- DALIGenericIterator：PyTorch 迭代器
- 替代 DataLoader
- 训练循环集成
- 性能对比

### 第四步：高级特性（30 分钟）

```bash
python basic/05_external_source.py     # 15 min - 自定义数据源
python basic/06_to_08_advanced_features.py  # 15 min - 并行处理、多 GPU
```

**关键概念：**
- fn.external_source：接入自定义数据
- num_threads：并行线程数
- prefetch_queue_depth：预取深度
- shard_id/num_shards：多 GPU 支持

### 第五步：MinIO 集成（60 分钟）

#### 5.1 启动 MinIO（10 分钟）

```bash
# 使用 Docker 启动 MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# 访问 Web UI
# http://localhost:9001
# 用户名: minioadmin
# 密码: minioadmin
```

#### 5.2 基础集成（20 分钟）

```bash
# 安装 MinIO 客户端
pip install minio

# 运行 MinIO 基础示例
python basic/09_minio_basic.py         # 20 min - MinIO 基础
```

**关键概念：**
- MinIO 客户端配置
- Bucket 和 Object
- 与 external_source 集成
- 从对象存储读取图像

#### 5.3 生产级 Pipeline（30 分钟）

```bash
python basic/10_minio_production_pipeline.py  # 30 min - 生产级实现
```

**关键概念：**
- 错误处理和重试
- LRU 缓存
- 性能监控
- 与训练循环集成

## 快速参考

### 最简 Pipeline

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def simple_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir)
    images = fn.decoders.image(images, device="mixed")
    images = fn.resize(images, size=224)
    return images, labels

pipe = simple_pipeline(data_dir="/path/to/data", batch_size=32, num_threads=4, device_id=0)
pipe.build()
outputs = pipe.run()
```

### 训练用 Pipeline

```python
@pipeline_def
def training_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed")

    # 数据增强
    images = fn.random_resized_crop(images, size=224, random_area=[0.08, 1.0])
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    images = fn.brightness_contrast(images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),
        contrast=fn.random.uniform(range=[0.8, 1.2])
    )

    # 归一化
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    images = fn.normalize(images,
        mean=[0.485, 0.456, 0.406],
        stddev=[0.229, 0.224, 0.225],
        axes=(2,)
    )
    images = fn.transpose(images, perm=[2, 0, 1])  # CHW

    return images, labels
```

### PyTorch 集成

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = training_pipeline(data_dir="/data", batch_size=64, num_threads=8, device_id=0)
pipe.build()

train_loader = DALIGenericIterator(
    pipelines=[pipe],
    output_map=["images", "labels"],
    size=num_samples,
    auto_reset=True
)

for batch in train_loader:
    data = batch[0]
    images = data["images"]  # PyTorch Tensor on GPU
    labels = data["labels"]
    # 训练代码...
```

### MinIO 数据源

```python
from minio import Minio

# 连接 MinIO
client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

# 自定义数据源
class MinIOSource:
    def __init__(self, client, bucket, objects):
        self.client = client
        self.bucket = bucket
        self.objects = objects

    def __call__(self, sample_info):
        idx = sample_info.idx_in_epoch
        obj = self.objects[idx]

        # 从 MinIO 读取
        response = self.client.get_object(self.bucket, obj)
        data = response.read()

        # 解码图像
        img = Image.open(io.BytesIO(data))
        return np.array(img), label

@pipeline_def
def minio_pipeline(source):
    images, labels = fn.external_source(source=source, num_outputs=2, batch=False)
    # 处理...
    return images, labels
```

## 常见问题

**Q: DALI 比 PyTorch DataLoader 快多少？**
A: 通常 2-5x，取决于数据增强复杂度和硬件配置。

**Q: 何时使用 device="cpu" vs "mixed"？**
A: "mixed" 使用 GPU 硬件加速解码，通常更快。CPU 解码适合特殊格式或调试。

**Q: 如何调试 Pipeline？**
A: 使用小 batch_size，检查输出形状，使用 fn.dump_image 保存中间结果。

**Q: 内存不足怎么办？**
A: 减少 batch_size、prefetch_queue_depth，或使用更小的图像尺寸。

**Q: MinIO 性能优化？**
A: 使用缓存、增加线程数、启用预取、考虑数据本地性。

## 性能优化检查清单

- [ ] 使用 device="mixed" 进行 GPU 解码
- [ ] 设置合适的 num_threads（通常 4-8）
- [ ] 启用 prefetch_queue_depth（2-3）
- [ ] 使用 GPU 操作而非 CPU
- [ ] 多 GPU 时使用 sharding
- [ ] MinIO 启用缓存
- [ ] 监控 GPU 利用率（目标 >95%）

## 下一步

完成基础教程后：
1. 阅读完整的 [DALI 文档](https://docs.nvidia.com/deeplearning/dali/)
2. 探索 [官方示例](https://github.com/NVIDIA/DALI/tree/main/docs/examples)
3. 尝试视频、音频数据处理
4. 开发自定义 Operator

## 资源

- [DALI GitHub](https://github.com/NVIDIA/DALI)
- [DALI API 文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations.html)
- [MinIO 文档](https://min.io/docs/minio/linux/index.html)
- [PyTorch 官方文档](https://pytorch.org/docs/)

## 支持

遇到问题？
- 查看 README.md 学习大纲
- 阅读示例代码注释
- 搜索 [DALI Issues](https://github.com/NVIDIA/DALI/issues)
- 提问到社区论坛

**祝学习愉快！🚀**
