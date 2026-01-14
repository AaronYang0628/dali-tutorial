# DALI Agent - Quick Start Guide

## 概述

DALI Agent 是一个自然语言数据处理助手，能够理解用户的自然语言需求，自动调用 DALI HTTP API 配置图像数据处理流程。

## 特性

✅ **自然语言理解** - 支持中文和英文输入
✅ **自动参数提取** - 智能识别数据路径、批次大小、图像尺寸等
✅ **智能决策** - 自动判断使用基础处理还是数据增强
✅ **多数据源** - 支持本地路径、S3/MinIO、测试数据生成
✅ **完整工作流** - 自动按顺序调用导入数据→创建Pipeline→配置完成
✅ **错误处理** - 友好的错误提示和建议

## 架构

```
用户自然语言输入
       ↓
   [NL Parser]  ← 提取参数、判断意图
       ↓
  [DALIAgent]   ← 编排API调用
       ↓
 [DALI HTTP API] ← 执行数据处理
       ↓
   格式化输出
```

## 快速开始

### 1. 启动 DALI HTTP 服务器

```bash
cd /workspaces/dali-tutorial/mcp/scripts
python dali_http_server.py
```

服务器运行在 http://localhost:8000

### 2. 运行 Agent

#### 交互式模式

```bash
python dali_agent.py
```

然后输入自然语言需求：

```
👤 > 我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转
```

#### 命令行模式

```bash
python dali_agent.py "创建测试数据集，50张图像，batch 16，尺寸 128x128"
```

## 使用示例

### 示例 1: ImageNet 训练数据

**输入（中文）:**
```
我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转
```

**Agent 理解:**
- 数据源: 本地路径 `/data/imagenet`
- 批次大小: 32
- 图像尺寸: 224x224
- Pipeline类型: 数据增强（检测到"随机裁剪"和"翻转"）

**执行步骤:**
1. 导入本地数据集 → `imagenet_dataset`
2. 创建增强Pipeline → `imagenet_dataset_augmentation_32`

### 示例 2: 测试数据生成

**输入（英文）:**
```
Create a test dataset with 100 images, batch size 16, size 128x128
```

**Agent 理解:**
- 数据源: 生成测试数据
- 图像数量: 100
- 批次大小: 16
- 图像尺寸: 128x128
- Pipeline类型: 基础处理（未提及增强）

**执行步骤:**
1. 创建测试数据集 → `test_dataset`
2. 创建基础Pipeline → `test_dataset_basic_16`

### 示例 3: S3 数据导入

**输入（中文）:**
```
从 s3://my-bucket/training-data 导入数据，batch 64，做数据增强
```

**Agent 理解:**
- 数据源: S3 存储
- S3 URI: s3://my-bucket/training-data
- 批次大小: 64
- Pipeline类型: 数据增强（检测到"增强"）

**执行步骤:**
1. 从S3导入数据集 → `s3_my-bucket`
2. 创建增强Pipeline → `s3_my-bucket_augmentation_64`

### 示例 4: 验证数据（无增强）

**输入（混合语言）:**
```
数据在 /data/val，batch 64, size 224x224, 只需要resize，不要augmentation
```

**Agent 理解:**
- 数据源: 本地路径 `/data/val`
- 批次大小: 64
- 图像尺寸: 224x224
- Pipeline类型: 基础处理（检测到"只需要"和"不要"）

**执行步骤:**
1. 导入本地数据集 → `val_dataset`
2. 创建基础Pipeline → `val_dataset_basic_64`

## 自然语言关键词

### 数据源识别

| 中文 | 英文 | 类型 |
|------|------|------|
| 数据在, 路径 | data at, path, from | 本地路径 |
| s3://, 云存储 | s3://, cloud storage | S3存储 |
| 测试, 生成 | test, synthetic, create | 测试数据 |

### 增强检测

**触发增强模式的关键词:**
- 中文: 增强, 裁剪, 翻转, 旋转, 亮度, 对比度, 随机
- 英文: augment, crop, flip, rotate, brightness, contrast, random

**触发基础模式的关键词:**
- 中文: 基础, 简单, 仅, 只, 不需要增强
- 英文: basic, simple, only, just, no augment

### 参数提取

| 参数 | 示例 |
|------|------|
| 批次大小 | "batch 32", "批次32", "批 32" |
| 图像尺寸 | "224x224", "尺寸 256", "size 128x128" |
| 图像数量 | "50张", "100 images", "30 pics" |

## Agent 命令

### list - 列出所有资源

```bash
👤 > list
```

显示所有已配置的数据集和Pipeline。

### test <pipeline_name> - 测试Pipeline

```bash
👤 > test imagenet_dataset_augmentation_32
```

运行指定Pipeline进行测试。

### quit - 退出

```bash
👤 > quit
```

## 集成到自己的应用

### 方式 1: 直接使用 Python 模块

```python
from dali_agent import DALIAgent

agent = DALIAgent()
agent.process_request("创建测试数据集，50张图像，batch 16")
```

### 方式 2: 集成 LLM API

#### 使用 Claude API

```python
import anthropic

client = anthropic.Anthropic()

# 加载 Agent Prompt
with open("AGENT_PROMPT.md", "r") as f:
    system_prompt = f.read()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    system=system_prompt,
    messages=[
        {"role": "user", "content": "我需要处理ImageNet数据，batch 32"}
    ]
)

print(message.content[0].text)
```

#### 使用 OpenAI API

```python
from openai import OpenAI

client = OpenAI()

with open("AGENT_PROMPT.md", "r") as f:
    system_prompt = f.read()

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Create test dataset, 100 images"}
    ]
)

print(response.choices[0].message.content)
```

### 方式 3: 在 N8N 中使用

1. 添加 **Execute Command** 节点
2. 命令: `python /path/to/dali_agent.py "{{ $json.user_request }}"`
3. 解析输出并继续工作流

## 配置

### 自定义 API 端点

```bash
export DALI_API_BASE="http://my-server:8080"
python dali_agent.py
```

或在代码中：

```python
from dali_agent import DALIAgent

agent = DALIAgent(api_base="http://my-server:8080")
```

### 默认参数

在 `dali_agent.py` 中修改：

```python
DEFAULT_BATCH_SIZE = 32      # 默认批次大小
DEFAULT_IMAGE_SIZE = 224     # 默认图像尺寸
DEFAULT_SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]  # 支持的格式
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `AGENT_PROMPT.md` | Agent系统提示词，用于LLM集成 |
| `dali_agent.py` | Agent实现（Python客户端） |
| `AGENT_EXAMPLES.md` | 详细示例和测试用例 |
| `dali_http_server.py` | DALI HTTP API服务器 |

## 工作流程

```
1. 用户输入自然语言
   ↓
2. NLParser 解析并提取参数
   - 数据源类型 (local/s3/test)
   - 数据位置
   - 批次大小
   - 图像尺寸
   - Pipeline类型
   ↓
3. DALIAgent 调用 HTTP API
   - 步骤1: 导入/创建数据集
   - 步骤2: 创建Pipeline
   ↓
4. 格式化输出结果
   - 数据集信息
   - Pipeline配置
   - 使用提示
```

## 故障排除

### 问题 1: 无法连接到API服务器

**错误:**
```
❌ 无法连接到 DALI API 服务器
```

**解决:**
1. 启动HTTP服务器: `python dali_http_server.py`
2. 检查端口: `curl http://localhost:8000/health`

### 问题 2: 路径识别失败

**症状:** Agent显示"未指定数据路径"

**解决:** 使用明确的路径格式
- ✅ "数据在 /data/imagenet"
- ✅ "data at /home/user/photos"
- ❌ "in folder data"

### 问题 3: Pipeline类型错误

**症状:** 需要增强但创建了basic pipeline

**解决:** 使用明确的增强关键词
- ✅ "需要数据增强"
- ✅ "with augmentation"
- ✅ "随机裁剪和翻转"

## 高级用法

### 批量处理

```python
from dali_agent import DALIAgent

agent = DALIAgent()

datasets = [
    ("训练集", "/data/train", 32, True),
    ("验证集", "/data/val", 64, False),
    ("测试集", "/data/test", 128, False),
]

for name, path, batch, augment in datasets:
    aug_text = "需要数据增强" if augment else "不要增强"
    request = f"{name}数据在 {path}，batch {batch}，{aug_text}"
    agent.process_request(request)
```

### 自定义解析规则

扩展 `NLParser` 类添加自定义规则：

```python
class CustomNLParser(NLParser):
    @staticmethod
    def extract_custom_param(text: str):
        # 添加自定义参数提取逻辑
        pass
```

## 性能建议

- **批次大小**: 训练用32-256，验证用64-128
- **图像尺寸**: ImageNet标准224x224，检测任务用640x640
- **Pipeline类型**: 训练用augmentation，验证/推理用basic

## 下一步

1. 阅读完整文档: `AGENT_PROMPT.md`
2. 查看示例对话: `AGENT_EXAMPLES.md`
3. 集成到你的工作流
4. 根据需求定制Agent

## 贡献

欢迎提交Issue和PR来改进Agent功能！

## 许可证

与DALI MCP Server相同
