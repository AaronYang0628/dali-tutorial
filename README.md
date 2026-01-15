# NVIDIA DALI 学习教程

从零开始学习 NVIDIA DALI，掌握如何从 MinIO 构建高性能深度学习数据流水线。

## 项目简介

本项目提供完整的 NVIDIA DALI 学习材料，包括：
- **系统的学习大纲**：从基础到高级，循序渐进
- **10+ 个实战示例**：每个示例都可独立运行，包含详细注释
- **MinIO 集成**：学习如何从对象存储加载数据
- **生产级代码**：错误处理、缓存、性能监控等实用特性
- **快速入门指南**：10 分钟快速上手

## 快速开始

### 1. 环境要求

- Python 3.8+
- NVIDIA GPU with CUDA 12.1+
- 8GB+ GPU Memory

### 2. 检查环境

```bash
python basic/check_environment.py
```

### 3. 开始学习

**快速路径（1 小时）：**
```bash
# 阅读快速入门
cat basic/QUICKSTART.md

# 运行核心示例
python basic/01_hello_dali.py
python basic/03_augmentation.py
python basic/04_pytorch_integration.py
python basic/09_minio_basic.py
```

**完整路径（3 小时）：**
```bash
# 阅读完整大纲
cat basic/README.md

# 按顺序学习所有示例
python basic/01_hello_dali.py
python basic/02_basic_image_processing.py
python basic/03_augmentation.py
python basic/04_pytorch_integration.py
python basic/05_external_source.py
python basic/06_to_08_advanced_features.py
python basic/09_minio_basic.py
python basic/10_minio_production_pipeline.py
```

## 学习内容

### 基础篇（01-04）

| 示例 | 内容 | 时长 | 难度 |
|------|------|------|------|
| [01_hello_dali.py](basic/01_hello_dali.py) | Pipeline 基础 | 15 min | ⭐ |
| [02_basic_image_processing.py](basic/02_basic_image_processing.py) | 图像处理操作 | 15 min | ⭐⭐ |
| [03_augmentation.py](basic/03_augmentation.py) | 数据增强 | 30 min | ⭐⭐⭐ |
| [04_pytorch_integration.py](basic/04_pytorch_integration.py) | PyTorch 集成 | 30 min | ⭐⭐⭐ |

### 进阶篇（05-08）

| 示例 | 内容 | 时长 | 难度 |
|------|------|------|------|
| [05_external_source.py](basic/05_external_source.py) | 外部数据源 | 15 min | ⭐⭐⭐ |
| [06_to_08_advanced_features.py](basic/06_to_08_advanced_features.py) | 并行、多GPU、动态配置 | 30 min | ⭐⭐⭐⭐ |

### MinIO 集成篇（09-10）

| 示例 | 内容 | 时长 | 难度 |
|------|------|------|------|
| [09_minio_basic.py](basic/09_minio_basic.py) | MinIO 基础 | 20 min | ⭐⭐⭐ |
| [10_minio_production_pipeline.py](basic/10_minio_production_pipeline.py) | 生产级 Pipeline | 30 min | ⭐⭐⭐⭐⭐ |

## 学习目标

完成本教程后，你将能够：

✅ 理解 DALI 的核心概念和工作原理

✅ 构建高效的数据预处理 Pipeline

✅ 使用各种图像增强操作

✅ 将 DALI 与 PyTorch 集成

✅ 从对象存储（MinIO）读取数据

✅ 构建生产级数据流水线

✅ 优化数据加载性能（2-5x 提升）

## 项目结构

```
dali-tutorial/
├── README.md                          # 本文件
├── .devcontainer/                     # Dev Container 配置
│   ├── Dockerfile
│   ├── devcontainer.json
│   └── post-create.sh
├── basic/                             # 学习材料
│   ├── README.md                      # 完整学习大纲
│   ├── QUICKSTART.md                  # 快速入门指南
│   ├── INDEX.md                       # 详细索引
│   ├── requirements.txt               # Python 依赖
│   ├── check_environment.py           # 环境检查
│   ├── 01_hello_dali.py              # 示例 1
│   ├── ...                            # 更多示例
│   └── 10_minio_production_pipeline.py
├── notebooks/                         # Jupyter 笔记本
├── scripts/                           # 工具脚本
└── .claude/                          # Claude Code 配置
    └── CLAUDE.md
```

## 文档导航

- **[完整学习大纲](basic/README.md)** - 详细的学习路径和概念解释
- **[快速入门指南](basic/QUICKSTART.md)** - 10 分钟快速上手
- **[详细索引](basic/INDEX.md)** - 所有示例的详细说明和使用指南

## 核心特性

### 🚀 高性能

- GPU 加速的图像解码（nvJPEG）
- 并行数据加载（多线程）
- 预取机制（prefetch）
- 通常比 PyTorch DataLoader 快 2-5x

### 🎨 丰富的数据增强

- 随机裁剪和缩放
- 颜色抖动（亮度、对比度、饱和度、色调）
- 几何变换（旋转、翻转、仿射）
- 高斯模糊、噪声等

### 🔌 易于集成

- 无缝集成 PyTorch
- 支持 TensorFlow
- 自定义数据源（external_source）
- 对象存储支持（MinIO/S3）

### 💪 生产就绪

- 错误处理和重试机制
- 性能监控和日志
- LRU 缓存
- 多 GPU 支持

## MinIO 集成

本教程重点介绍如何从 MinIO 对象存储读取数据：

### 启动 MinIO

```bash
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### 基础使用

```python
from minio import Minio
import nvidia.dali.fn as fn

# 连接 MinIO
client = Minio("localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# 自定义数据源
class MinIOSource:
    def __call__(self, sample_info):
        # 从 MinIO 读取图像
        obj = self.client.get_object(bucket, object_name)
        data = obj.read()
        # 解码并返回
        return image_array, label

# 集成到 DALI
@pipeline_def
def minio_pipeline(source):
    images, labels = fn.external_source(source=source, num_outputs=2)
    # 处理...
    return images, labels
```

详见：[09_minio_basic.py](basic/09_minio_basic.py) 和 [10_minio_production_pipeline.py](basic/10_minio_production_pipeline.py)

## 性能对比

典型性能提升（vs PyTorch DataLoader）：

| 场景 | 图像大小 | Batch Size | 提升 |
|------|----------|------------|------|
| 基础增强 | 224x224 | 64 | 2.5x |
| 复杂增强 | 224x224 | 64 | 3.8x |
| 大图像 | 512x512 | 32 | 4.2x |
| 高分辨率 | 1024x1024 | 16 | 5.1x |

## 常见问题

**Q: 需要什么硬件？**
A: NVIDIA GPU（计算能力 6.0+），建议 8GB+ 显存。

**Q: 支持哪些数据格式？**
A: JPEG、PNG、TIFF、WebP 等常见图像格式，以及视频、音频等。

**Q: 可以用于非深度学习任务吗？**
A: 可以！DALI 适合任何需要高性能数据加载的场景。

**Q: 如何调试 Pipeline？**
A: 使用小 batch_size，打印中间结果 shape，使用 fn.dump_image 保存图像。

**Q: 性能没有提升？**
A: 检查是否使用 GPU 解码（device="mixed"），增加线程数，启用预取。

## 依赖安装

```bash
# 核心依赖
pip install nvidia-dali-cuda120
pip install torch torchvision

# MinIO 集成
pip install minio

# 其他工具
pip install Pillow numpy pandas matplotlib
```

或一键安装：

```bash
pip install -r basic/requirements.txt
```

## 贡献

欢迎提 Issue 和 PR！

## 资源

- [DALI 官方文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [DALI GitHub](https://github.com/NVIDIA/DALI)
- [MinIO 文档](https://min.io/docs/)
- [PyTorch 文档](https://pytorch.org/docs/)

## 许可

本项目仅用于学习目的。

## 致谢

- NVIDIA DALI 团队
- MinIO 项目
- PyTorch 社区

---

**开始学习：** `python basic/check_environment.py`

**快速入门：** `cat basic/QUICKSTART.md`

**祝学习愉快！🚀**
