# NVIDIA DALI 学习大纲

从零开始学习 NVIDIA DALI，最终掌握如何从 MinIO 构建高性能数据流水线。

## 学习目标

完成本教程后，你将能够：
- 理解 DALI 的核心概念和工作原理
- 构建高效的数据预处理 Pipeline
- 使用各种图像增强操作
- 将 DALI 与 PyTorch/TensorFlow 集成
- 从对象存储（MinIO）读取数据构建生产级数据集

## 前置知识

- Python 基础
- NumPy 基础
- 基本的深度学习概念
- 了解图像数据格式

## 学习路径

### 第一部分：DALI 基础 (01-03)

#### 01_hello_dali.py - 第一个 DALI 程序
**学习内容：**
- DALI Pipeline 基本结构
- `@pipeline_def` 装饰器
- 简单的文件读取操作
- Pipeline 的构建和执行

**关键概念：**
- Pipeline：DALI 的核心抽象，定义数据处理流程
- Operator：数据处理的基本单元
- Iterator：从 Pipeline 获取数据

#### 02_basic_image_processing.py - 基础图像处理
**学习内容：**
- 图像解码 (`fn.decoders.image`)
- 基本图像操作（resize, crop, flip）
- 数据类型转换
- GPU vs CPU 操作

**关键概念：**
- Device：操作执行的位置（CPU/GPU）
- Mixed operators：部分在 CPU、部分在 GPU
- Output layout：数据格式（HWC, CHW）

#### 03_augmentation.py - 数据增强
**学习内容：**
- 常用数据增强操作
  - 随机裁剪 (RandomResizedCrop)
  - 颜色抖动 (ColorTwist)
  - 归一化 (Normalize)
  - 旋转和翻转
- 增强参数的随机化
- 组合多个增强操作

**关键概念：**
- Random operators：使用随机参数的操作
- Argument inputs：动态参数传递

### 第二部分：Pipeline 进阶 (04-06)

#### 04_pytorch_integration.py - PyTorch 集成
**学习内容：**
- DALIGenericIterator 使用
- 与 PyTorch DataLoader 的对比
- 训练循环中的集成
- 性能对比测试

**关键概念：**
- Iterator 重置和 epoch 管理
- 批次大小和设备管理
- 数据格式转换

#### 05_conditional_pipeline.py - 条件执行
**学习内容：**
- 条件操作符 (`if` statements)
- 不同数据路径
- 多输入源处理
- 动态 Pipeline 配置

**关键概念：**
- Conditional execution
- Multiple data sources
- Dynamic graph construction

#### 06_external_source.py - 外部数据源
**学习内容：**
- `fn.external_source` 使用
- 自定义数据加载逻辑
- 回调函数编写
- 批次数据生成

**关键概念：**
- External source：接入自定义数据
- Callback function：数据生成函数
- Batch vs sample iteration

### 第三部分：高级特性 (07-08)

#### 07_parallel_pipeline.py - 并行处理
**学习内容：**
- 多线程数据加载
- 预取机制 (prefetch)
- CPU/GPU 混合执行
- 性能调优参数

**关键概念：**
- `num_threads`：并行线程数
- `prefetch_queue_depth`：预取队列深度
- Device affinity：设备亲和性

#### 08_multi_gpu.py - 多 GPU 支持
**学习内容：**
- 多 GPU 数据分片
- 分布式训练数据准备
- Shard 索引和数量
- 跨 GPU 数据同步

**关键概念：**
- Sharding：数据分片
- `shard_id` 和 `num_shards`
- Data parallelism

### 第四部分：MinIO 集成 (09-10)

#### 09_minio_basic.py - MinIO 基础读取
**学习内容：**
- MinIO 客户端配置
- 从对象存储读取文件列表
- 使用 `fn.external_source` 读取 MinIO 数据
- URL 格式和访问控制

**关键概念：**
- Object storage：对象存储概念
- S3 协议兼容性
- Bucket 和 object key

#### 10_minio_production_pipeline.py - 生产级 MinIO Pipeline
**学习内容：**
- 完整的生产级数据流水线
- 错误处理和重试机制
- 性能监控
- 缓存策略
- 与训练循环集成

**关键概念：**
- Production-ready pipeline
- Error handling
- Performance monitoring
- Cache management

## 实践项目

### Project: ImageNet-Style 数据集构建
在 `project/` 目录下，使用所学知识构建一个完整的图像分类数据集 Pipeline：
- 从 MinIO 读取 ImageNet 格式数据
- 实现训练和验证的不同增强策略
- 支持多 GPU 训练
- 性能达到 GPU 利用率 > 95%

## 学习建议

1. **按顺序学习**：每个示例都基于前面的概念
2. **动手实践**：运行每个脚本，观察输出
3. **修改参数**：尝试不同的参数组合，理解影响
4. **阅读文档**：配合 [DALI 官方文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
5. **性能测试**：对比 DALI 与传统方法的性能差异

## 常用命令

```bash
# 运行单个示例
python basic/01_hello_dali.py

# 查看 GPU 使用情况
nvidia-smi -l 1

# 性能分析
python -m cProfile -o output.prof basic/04_pytorch_integration.py
```

## 参考资源

- [DALI 官方文档](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [DALI API 参考](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations.html)
- [NVIDIA DALI GitHub](https://github.com/NVIDIA/DALI)
- [示例集合](https://github.com/NVIDIA/DALI/tree/main/docs/examples)

## 下一步

完成基础学习后，可以探索：
- 视频数据处理
- 音频数据处理
- 自定义 Operator 开发
- TensorFlow 集成
- Triton Inference Server 集成
