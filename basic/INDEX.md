# DALI 学习材料索引

## 文件结构

```
basic/
├── README.md                          # 完整学习大纲
├── QUICKSTART.md                      # 10 分钟快速入门
├── INDEX.md                           # 本文件：索引和导航
├── requirements.txt                   # Python 依赖
├── check_environment.py               # 环境检查脚本
│
├── 01_hello_dali.py                  # 第一个 DALI 程序
├── 02_basic_image_processing.py      # 基础图像处理
├── 03_augmentation.py                # 数据增强
├── 04_pytorch_integration.py         # PyTorch 集成
├── 05_external_source.py             # 外部数据源
├── 06_to_08_advanced_features.py     # 高级特性（并行、多GPU、动态配置）
├── 09_minio_basic.py                 # MinIO 基础集成
└── 10_minio_production_pipeline.py   # 生产级 MinIO Pipeline
```

## 使用指南

### 1. 环境准备

```bash
# 检查环境
python basic/check_environment.py

# 如果有缺失，安装依赖
pip install -r basic/requirements.txt
```

### 2. 学习路径选择

#### 快速路径（1 小时）
**适合：**已有深度学习经验，想快速了解 DALI

```bash
# 1. 阅读快速入门
cat basic/QUICKSTART.md

# 2. 运行核心示例
python basic/01_hello_dali.py          # 理解 Pipeline
python basic/03_augmentation.py        # 数据增强
python basic/04_pytorch_integration.py  # PyTorch 集成
python basic/09_minio_basic.py         # MinIO 集成
```

#### 完整路径（3 小时）
**适合：**系统学习 DALI，掌握所有特性

```bash
# 1. 阅读学习大纲
cat basic/README.md

# 2. 按顺序运行所有示例
for script in basic/0*.py; do
    echo "Running $script..."
    python $script
done
```

#### 实战路径（边学边做）
**适合：**有具体项目需求

直接跳到相关章节：
- 需要从本地文件加载 → `01_hello_dali.py`
- 需要数据增强 → `03_augmentation.py`
- 需要与 PyTorch 集成 → `04_pytorch_integration.py`
- 需要自定义数据源 → `05_external_source.py`
- 需要性能优化 → `06_to_08_advanced_features.py`
- 需要从 MinIO 加载 → `09_minio_basic.py` + `10_minio_production_pipeline.py`

### 3. 代码模板

每个示例都是完整可运行的，可以直接复制代码段：

**基础 Pipeline 模板：**
```python
# 从 01_hello_dali.py
@pipeline_def
def simple_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir)
    images = fn.decoders.image(images, device="mixed")
    images = fn.resize(images, size=224)
    return images, labels
```

**训练 Pipeline 模板：**
```python
# 从 03_augmentation.py
@pipeline_def
def training_pipeline(data_dir):
    # ... 完整的训练数据增强
```

**MinIO Pipeline 模板：**
```python
# 从 10_minio_production_pipeline.py
# 带缓存、错误处理的生产级实现
```

## 学习目标检查清单

完成学习后，你应该能够：

**基础知识：**
- [ ] 理解 DALI Pipeline 的概念和工作原理
- [ ] 使用 @pipeline_def 装饰器定义 Pipeline
- [ ] 调用 build() 和 run() 执行 Pipeline
- [ ] 理解 batch 和 iteration 的概念

**图像处理：**
- [ ] 使用 fn.readers.file 读取文件
- [ ] 使用 fn.decoders.image 解码图像
- [ ] 应用基本图像操作（resize, crop, flip）
- [ ] 理解 device="cpu" vs "mixed" 的区别

**数据增强：**
- [ ] 实现随机裁剪（random_resized_crop）
- [ ] 应用颜色增强（brightness, contrast, saturation）
- [ ] 使用几何变换（rotate, flip）
- [ ] 使用 fn.random.uniform 和 coin_flip 生成随机参数

**PyTorch 集成：**
- [ ] 使用 DALIGenericIterator 创建迭代器
- [ ] 在训练循环中使用 DALI
- [ ] 理解 auto_reset 和 epoch 管理
- [ ] 对比 DALI 与 DataLoader 的性能差异

**高级特性：**
- [ ] 使用 fn.external_source 接入自定义数据
- [ ] 配置 num_threads 和 prefetch_queue_depth
- [ ] 使用 shard_id/num_shards 支持多 GPU
- [ ] 动态调整 Pipeline 参数

**MinIO 集成：**
- [ ] 配置 MinIO 客户端
- [ ] 从 MinIO 读取图像数据
- [ ] 实现错误处理和重试机制
- [ ] 使用缓存优化性能
- [ ] 构建生产级数据流水线

## 示例说明

### 01_hello_dali.py
**时长：** 15 分钟
**难度：** ⭐
**内容：**
- Pipeline 基本结构
- fn.readers.file 使用
- build() 和 run() 流程
- 多次迭代和重置

**关键代码：**
```python
@pipeline_def
def simple_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir)
    return images, labels

pipe = simple_pipeline(data_dir="data", batch_size=8, num_threads=2, device_id=0)
pipe.build()
outputs = pipe.run()
```

### 02_basic_image_processing.py
**时长：** 15 分钟
**难度：** ⭐⭐
**内容：**
- 图像解码（fn.decoders.image）
- 基本操作（resize, crop, flip）
- CPU vs GPU 性能对比
- 数据类型和格式转换

**性能提示：**
- 使用 `device="mixed"` 获得最佳性能
- GPU 解码通常比 CPU 快 2-5x

### 03_augmentation.py
**时长：** 30 分钟
**难度：** ⭐⭐⭐
**内容：**
- 随机裁剪（random_resized_crop）
- 颜色增强（brightness, contrast, saturation, hue）
- 几何变换（rotate, flip）
- 组合多种增强

**最佳实践：**
- 训练时使用激进的增强
- 验证时只使用中心裁剪
- 合理控制增强强度

### 04_pytorch_integration.py
**时长：** 30 分钟
**难度：** ⭐⭐⭐
**内容：**
- DALIGenericIterator 使用
- 训练循环集成
- 性能对比测试

**性能提升：**
- DALI 数据已在 GPU 上，无需 `tensor.cuda()`
- 通常比 DataLoader 快 2-5x

### 05_external_source.py
**时长：** 15 分钟
**难度：** ⭐⭐⭐
**内容：**
- fn.external_source 使用
- 回调函数编写
- 批量 vs 单样本模式
- 与文件读取混合使用

**使用场景：**
- 从数据库读取数据
- 从网络流读取
- 实时数据生成
- 与 MinIO 等对象存储集成

### 06_to_08_advanced_features.py
**时长：** 30 分钟
**难度：** ⭐⭐⭐⭐
**内容：**
- 并行处理（num_threads）
- 预取优化（prefetch_queue_depth）
- 多 GPU 支持（shard_id/num_shards）
- 动态 Pipeline 配置

**性能调优：**
- num_threads: 通常设为 CPU 核数的 1-2 倍
- prefetch_queue_depth: 2-3 通常足够
- 监控 GPU 利用率，目标 >95%

### 09_minio_basic.py
**时长：** 20 分钟
**难度：** ⭐⭐⭐
**前置条件：** MinIO 服务运行
**内容：**
- MinIO 客户端配置
- 创建 bucket 和上传数据
- 使用 external_source 读取 MinIO 数据
- 预签名 URL

**环境准备：**
```bash
# 启动 MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### 10_minio_production_pipeline.py
**时长：** 30 分钟
**难度：** ⭐⭐⭐⭐⭐
**内容：**
- 生产级实现
- 错误处理和重试
- LRU 缓存
- 性能监控
- 完整训练集成

**生产特性：**
- 自动重试（最多 3 次）
- LRU 缓存（减少重复下载）
- 详细的性能统计
- 线程安全

## 常见问题

**Q: 应该从哪里开始？**
A: 先运行 `check_environment.py` 检查环境，然后阅读 `QUICKSTART.md`。

**Q: 我只想学 MinIO 集成，可以跳过前面吗？**
A: 建议至少学习 01、05 示例理解基础概念，然后直接跳到 09、10。

**Q: 代码运行出错怎么办？**
A:
1. 检查环境：`python check_environment.py`
2. 查看代码注释中的说明
3. 减小 batch_size 测试
4. 检查 GPU 内存是否足够

**Q: 性能没有提升？**
A:
1. 确认使用 `device="mixed"` 解码
2. 增加 `num_threads`（4-8）
3. 启用 `prefetch_queue_depth`（2-3）
4. 检查 GPU 利用率（`nvidia-smi -l 1`）

**Q: 如何调试 Pipeline？**
A:
1. 使用小的 batch_size（2-4）
2. 打印中间结果的 shape
3. 使用 `fn.dump_image` 保存中间图像
4. 检查数据范围是否合理

## 下一步学习

完成本教程后，可以：

1. **深入官方文档**
   - [DALI User Guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
   - [Operations Reference](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations.html)

2. **探索高级主题**
   - 视频数据处理
   - 音频数据处理
   - 自定义 Operator 开发
   - TensorFlow 集成

3. **实战项目**
   - 图像分类（ImageNet）
   - 目标检测（COCO）
   - 语义分割
   - 视频理解

4. **性能优化**
   - Profiling 和瓶颈分析
   - 多机多卡训练
   - 混合精度训练

## 反馈和贡献

遇到问题或有改进建议？
- 提 Issue 到项目仓库
- 查看 DALI GitHub Issues
- 参与社区讨论

**祝学习愉快！🎉**
