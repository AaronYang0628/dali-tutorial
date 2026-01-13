# DALI fn.resize 参数详解

## 1. `mode` 参数 - 缩放模式

控制当输入图像尺寸与目标尺寸不匹配时的行为。

### 可选值

#### `"not_smaller"` (最常用)
- 确保缩放后的图像**至少**和目标尺寸一样大
- 保持宽高比，较短的边会被缩放到 `target_size`，较长的边会更大
- 示例：
  ```
  输入: 512×256 (宽×高)
  target_size=224
  输出: 448×224  # 短边224，长边按比例
  ```

#### `"not_larger"`
- 确保缩放后的图像**不超过**目标尺寸
- 保持宽高比，较长的边会被缩放到 `target_size`，较短的边会更小
- 示例：
  ```
  输入: 512×256
  target_size=224
  输出: 224×112  # 长边224，短边按比例
  ```

#### `"stretch"`
- 不保持宽高比，强制拉伸到目标尺寸
- 可能导致图像变形
- 示例：
  ```
  输入: 512×256
  target_size=224
  输出: 224×224  # 强制拉伸
  ```

#### `"default"`
- 精确缩放到指定尺寸（需要同时指定 `resize_x` 和 `resize_y`）

### 实际使用场景

```python
# 场景1: 训练 - 先放大后裁剪
images = fn.resize(images, size=256, mode="not_smaller")  # 保证至少256
images = fn.crop(images, crop=(224, 224))  # 裁剪到224×224

# 场景2: 推理 - 保持原比例
images = fn.resize(images, size=224, mode="not_larger")  # 最大边224
```

---

## 2. `interp_type` 参数 - 插值算法

控制像素重采样时使用的插值算法（影响图像质量和速度）。

### 可选值

#### `types.INTERP_LINEAR` (最常用，默认)
- 双线性插值
- 质量和速度的良好平衡
- 适合大多数场景

#### `types.INTERP_NN` (最快)
- Nearest Neighbor（最近邻）
- 速度最快但质量最差
- 适合不需要平滑的场景（如语义分割的 label map）

#### `types.INTERP_CUBIC` (高质量)
- 双三次插值
- 质量最好但速度较慢
- 适合需要高质量缩放的场景

#### `types.INTERP_LANCZOS3` (最高质量)
- Lanczos3 插值
- 质量最佳但计算最慢
- 适合照片级图像处理

#### `types.INTERP_GAUSSIAN`
- 高斯插值
- 平滑效果好

#### `types.INTERP_TRIANGULAR`
- 三角插值

### 视觉对比

将 512×512 缩小到 64×64 再放大到 512×512：

- **INTERP_NN**: 马赛克效果，有明显锯齿
- **INTERP_LINEAR**: 较平滑，可能略微模糊
- **INTERP_CUBIC**: 清晰锐利
- **INTERP_LANCZOS3**: 最佳质量

### 性能对比

| 算法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| INTERP_NN | ⚡⚡⚡⚡⚡ | ⭐ | Label map, 实时推理 |
| INTERP_LINEAR | ⚡⚡⚡⚡ | ⭐⭐⭐ | **通用（推荐）** |
| INTERP_CUBIC | ⚡⚡⚡ | ⭐⭐⭐⭐ | 高质量图像 |
| INTERP_LANCZOS3 | ⚡⚡ | ⭐⭐⭐⭐⭐ | 照片编辑 |

---

## 代码示例对比

### 示例1: 训练时的典型用法
```python
images = fn.resize(
    images,
    size=256,
    mode="not_smaller",  # 保证足够大，后续裁剪
    interp_type=types.INTERP_LINEAR  # 平衡质量和速度
)
images = fn.crop(images, crop=(224, 224))
```

### 示例2: 处理标签图（语义分割）
```python
labels = fn.resize(
    labels,
    size=224,
    mode="stretch",  # 精确匹配尺寸
    interp_type=types.INTERP_NN  # 避免插值导致类别混淆
)
```

### 示例3: 高质量图像处理
```python
images = fn.resize(
    images,
    size=512,
    mode="not_larger",  # 保持比例
    interp_type=types.INTERP_LANCZOS3  # 最佳质量
)
```

---

## 总结

- **训练场景**: 使用 `mode="not_smaller"` + `INTERP_LINEAR`
- **推理场景**: 使用 `mode="not_larger"` + `INTERP_LINEAR`
- **标签处理**: 使用 `mode="stretch"` + `INTERP_NN`
- **高质量处理**: 使用 `mode="not_larger"` + `INTERP_LANCZOS3`

---

# fn.readers.file 的 labels 详解

## 核心问题

在使用 `fn.readers.file` 时，会返回两个值：

```python
images, labels = fn.readers.file(files=file_list)
```

**这个 `labels` 是什么？它不是我们提供的，那它从哪来？**

## 关键结论

**`labels` 并不是分类标签（如猫=0，狗=1），而是文件在输入列表中的索引！**

---

## labels 生成规则

### 规则 1: 使用 `files` 参数（传入文件列表）

当传入文件列表时，labels 是文件的索引（0-based）：

```python
file_list = [
    "sample_images/image_000.jpg",  # 索引 0
    "sample_images/image_001.jpg",  # 索引 1
    "sample_images/image_002.jpg",  # 索引 2
    "sample_images/image_003.jpg",  # 索引 3
    "sample_images/image_004.jpg",  # 索引 4
]

images, labels = fn.readers.file(files=file_list, random_shuffle=False)
```

**输出结果：**
```
Batch 1 (batch_size=3):
  Sample 0: label = 0  (对应 image_000.jpg)
  Sample 1: label = 1  (对应 image_001.jpg)
  Sample 2: label = 2  (对应 image_002.jpg)

Batch 2:
  Sample 0: label = 3  (对应 image_003.jpg)
  Sample 1: label = 4  (对应 image_004.jpg)
  Sample 2: label = 0  (对应 image_000.jpg, 循环读取)
```

### 规则 2: 使用 `file_root` 参数（目录结构推断）

使用目录结构时，labels 根据**子目录名称按字母顺序**自动分配：

```
dataset/
  ├── cat/          # 类别 0 (字母顺序第一)
  │   ├── img1.jpg
  │   └── img2.jpg
  ├── dog/          # 类别 1 (字母顺序第二)
  │   ├── img3.jpg
  │   └── img4.jpg
  └── bird/         # 类别 2 (字母顺序: bird < cat < dog, 实际是 bird=0)
      └── img5.jpg
```

```python
images, labels = fn.readers.file(file_root="dataset/")
```

**输出结果：**
- `bird/` 目录的所有图片 → label = 0
- `cat/` 目录的所有图片 → label = 1
- `dog/` 目录的所有图片 → label = 2

---

## 实际演示结果

运行以下代码：

```python
file_list = [
    "demo_data/image_000.jpg",
    "demo_data/image_001.jpg",
    "demo_data/image_002.jpg",
    "demo_data/image_003.jpg",
    "demo_data/image_004.jpg"
]

pipe = file_reader_pipeline(
    file_list=file_list,
    batch_size=3,
    num_threads=1,
    device_id=0
)
pipe.build()

outputs = pipe.run()
images_batch, labels_batch = outputs

for i in range(len(labels_batch)):
    label = np.array(labels_batch.as_cpu()[i])
    print(f"Sample {i}: label = {label[0]}")
```

**输出：**
```
Sample 0: label = 0 (对应文件: demo_data/image_000.jpg)
Sample 1: label = 1 (对应文件: demo_data/image_001.jpg)
Sample 2: label = 2 (对应文件: demo_data/image_002.jpg)
```

---

## 实际使用场景

### 场景 1: 不需要标签（仅处理图像）

直接忽略 labels：

```python
# 方法1: 使用下划线忽略
images, _ = fn.readers.file(files=file_list)

# 方法2: 只取第一个输出
outputs = pipe.run()
images_batch = outputs[0]
```

### 场景 2: 需要真实分类标签

#### 方法 1 - 使用目录结构（推荐）

这是最简单的方式，适合 ImageNet 风格的数据集：

```python
# 数据结构:
# dataset/
#   ├── cat/    # 标签自动为 0
#   ├── dog/    # 标签自动为 1
#   └── bird/   # 标签自动为 2

images, labels = fn.readers.file(file_root="dataset/")
# labels 会自动根据子目录分配: bird=0, cat=1, dog=2 (按字母顺序)
```

#### 方法 2 - 使用 ExternalSource 传入自定义标签

适合标签来自外部文件（如 CSV, JSON）的情况：

```python
# 准备数据
file_list = ["cat1.jpg", "dog1.jpg", "cat2.jpg"]
real_labels = [0, 1, 0]  # 真实标签

@pipeline_def
def custom_label_pipeline():
    images, _ = fn.readers.file(files=file_list)  # 忽略自动标签

    # 方式A: 使用 fn.external_source
    labels = fn.external_source(source=lambda: real_labels, batch=False)

    # 方式B: 在 Python 侧处理
    # 读取标注文件，在训练循环中与图像配对

    return images, labels
```

#### 方法 3 - 创建索引到标签的映射

适合标签与文件名有复杂关系的情况：

```python
# 创建文件索引到真实标签的映射字典
index_to_label = {
    0: 5,   # image_000.jpg 实际是类别 5
    1: 3,   # image_001.jpg 实际是类别 3
    2: 5,   # image_002.jpg 实际是类别 5
    3: 7,   # image_003.jpg 实际是类别 7
}

# 在训练时转换
for images, file_indices in dataloader:
    file_indices_cpu = [np.array(file_indices.as_cpu()[i])[0]
                        for i in range(len(file_indices))]
    real_labels = [index_to_label[idx] for idx in file_indices_cpu]

    # 使用 real_labels 进行训练
```

---

## 使用方式对比

| 使用方式 | labels 含义 | 适用场景 | 优点 | 缺点 |
|---------|------------|---------|------|------|
| `files=[...]` | 文件索引 (0, 1, 2...) | 无监督学习、图像处理 | 简单直接 | 没有真实标签 |
| `file_root="data/"` (有子目录) | 子目录编号 (按字母顺序) | 分类任务 (ImageNet 格式) | 自动推断标签 | 需要目录结构 |
| 配合 ExternalSource | 自定义标签 | 复杂标注、多任务 | 最灵活 | 实现复杂 |
| 索引映射 | 运行时转换 | 标签来自外部文件 | 灵活可控 | 需要额外处理 |

---

## 典型错误理解

### 错误理解 1
```python
# ❌ 错误：以为 labels 是图片内容的分类标签
images, labels = fn.readers.file(files=["cat.jpg", "dog.jpg"])
# labels 不是 [0, 1]（猫、狗），而是 [0, 1]（文件索引）
```

### 错误理解 2
```python
# ❌ 错误：以为可以传入自定义标签
images, labels = fn.readers.file(
    files=file_list,
    labels=[5, 3, 7]  # ❌ 没有这个参数！
)
```

### 正确做法
```python
# ✅ 方法1: 使用目录结构
images, labels = fn.readers.file(file_root="dataset/")

# ✅ 方法2: 忽略自动标签，使用外部标签
images, _ = fn.readers.file(files=file_list)
labels = fn.external_source(source=your_label_source)

# ✅ 方法3: 不需要标签时直接忽略
images, _ = fn.readers.file(files=file_list)
```

---

## 完整示例：ImageNet 风格数据集

```python
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def imagenet_pipeline(data_dir):
    """
    处理 ImageNet 风格的数据集

    数据结构:
    data_dir/
      ├── n01440764/  # 类别 0: tench (梭鱼)
      ├── n01443537/  # 类别 1: goldfish (金鱼)
      └── ...
    """
    # 自动根据子目录分配标签
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True
    )

    # 解码和增强
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=256, mode="not_smaller")
    images = fn.crop(images, crop=(224, 224), crop_pos_x=0.5, crop_pos_y=0.5)

    # labels 已经是正确的类别 ID (0, 1, 2, ...)
    return images, labels

# 使用
pipe = imagenet_pipeline(
    data_dir="/path/to/imagenet/train",
    batch_size=32,
    num_threads=4,
    device_id=0
)
pipe.build()

for i in range(10):
    images, labels = pipe.run()
    # labels 是类别 ID，可以直接用于训练
```

---

## 总结

1. **`fn.readers.file` 的 labels 是文件索引或子目录编号**，不是自定义的分类标签
2. **使用 `files` 参数**：labels = 文件在列表中的索引（0, 1, 2, ...）
3. **使用 `file_root` 参数**：labels = 子目录编号（按字母顺序排序）
4. **如果不需要标签**：直接用 `_` 忽略
5. **如果需要自定义标签**：
   - 简单场景：使用目录结构
   - 复杂场景：使用 ExternalSource 或索引映射
