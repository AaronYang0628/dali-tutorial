# DALI Agent - Example Conversations and Test Cases

## Quick Start

### 1. Start DALI HTTP Server

```bash
cd /workspaces/dali-tutorial/mcp/scripts
python dali_http_server.py
```

### 2. Run Agent

```bash
# Interactive mode
python dali_agent.py

# Direct command mode
python dali_agent.py "我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32"
```

---

## Example Conversations

### Example 1: Basic Image Classification (Chinese)

```
👤 > 我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转

======================================================================
  正在分析您的需求...
======================================================================

📋 检测到的参数:
   - 数据源: local
   - 位置: /data/imagenet
   - 批次大小: 32
   - 图像尺寸: 224x224
   - Pipeline类型: augmentation

======================================================================
  步骤 1: 准备数据集
======================================================================
正在导入本地数据集: /data/imagenet...
✅ 本地数据集导入成功
   - 数据集名称: imagenet_dataset
   - 图像数量: 1,281,167
   - 数据路径: /data/imagenet

======================================================================
  步骤 2: 创建 Pipeline
======================================================================
正在创建 augmentation Pipeline...
✅ Pipeline 创建成功
   - Pipeline名称: imagenet_dataset_augmentation_32
   - 类型: augmentation
   - 批次大小: 32
   - 目标尺寸: 224x224
   - 增强操作: 随机裁剪、水平翻转、旋转、亮度/对比度调整

======================================================================
  ✅ 配置完成！
======================================================================

**数据集:** imagenet_dataset
**Pipeline:** imagenet_dataset_augmentation_32
**状态:** 准备就绪，可以开始训练

💡 提示:
   - 运行测试: python -c 'agent.run_pipeline("imagenet_dataset_augmentation_32")'
   - 查看所有: agent.list_resources()
   - 在训练代码中引用 Pipeline: 'imagenet_dataset_augmentation_32'
```

---

### Example 2: Test Dataset Creation (English)

```
👤 > Create a test dataset with 100 images, batch size 16, size 128x128, no augmentation

======================================================================
  正在分析您的需求...
======================================================================

📋 检测到的参数:
   - 数据源: test
   - 批次大小: 16
   - 图像尺寸: 128x128
   - Pipeline类型: basic
   - 图像数量: 100

======================================================================
  步骤 1: 准备数据集
======================================================================
正在创建 100 张测试图像...
✅ 测试数据集创建成功
   - 数据集名称: test_dataset
   - 图像数量: 100
   - 存储路径: /tmp/dali_dataset_test_dataset_abc123

======================================================================
  步骤 2: 创建 Pipeline
======================================================================
正在创建 basic Pipeline...
✅ Pipeline 创建成功
   - Pipeline名称: test_dataset_basic_16
   - 类型: basic
   - 批次大小: 16
   - 目标尺寸: 128x128

======================================================================
  ✅ 配置完成！
======================================================================

**数据集:** test_dataset
**Pipeline:** test_dataset_basic_16
**状态:** 准备就绪，可以开始训练
```

---

### Example 3: S3 Dataset Import (Chinese)

```
👤 > 从 s3://my-bucket/training-data 导入数据，需要下载到本地，batch 64，做数据增强

======================================================================
  正在分析您的需求...
======================================================================

📋 检测到的参数:
   - 数据源: s3
   - 位置: s3://my-bucket/training-data
   - 批次大小: 64
   - 图像尺寸: 224x224
   - Pipeline类型: augmentation

======================================================================
  步骤 1: 准备数据集
======================================================================
正在从 S3 导入数据集: s3://my-bucket/training-data...
✅ S3 数据集导入成功
   - 数据集名称: s3_my-bucket
   - 图像数量: 45,892
   - S3 URI: s3://my-bucket/training-data
   - 本地路径: /tmp/dali_s3_my-bucket_xyz789

======================================================================
  步骤 2: 创建 Pipeline
======================================================================
正在创建 augmentation Pipeline...
✅ Pipeline 创建成功
   - Pipeline名称: s3_my-bucket_augmentation_64
   - 类型: augmentation
   - 批次大小: 64
   - 目标尺寸: 224x224
   - 增强操作: 随机裁剪、水平翻转、旋转、亮度/对比度调整

======================================================================
  ✅ 配置完成！
======================================================================

**数据集:** s3_my-bucket
**Pipeline:** s3_my-bucket_augmentation_64
**状态:** 准备就绪，可以开始训练
```

---

### Example 4: Simple Resize Only (Mixed Language)

```
👤 > 数据在 /home/user/photos，batch 8, size 256x256, 只需要resize，不要augmentation

======================================================================
  正在分析您的需求...
======================================================================

📋 检测到的参数:
   - 数据源: local
   - 位置: /home/user/photos
   - 批次大小: 8
   - 图像尺寸: 256x256
   - Pipeline类型: basic

======================================================================
  步骤 1: 准备数据集
======================================================================
正在导入本地数据集: /home/user/photos...
✅ 本地数据集导入成功
   - 数据集名称: photos_dataset
   - 图像数量: 523
   - 数据路径: /home/user/photos

======================================================================
  步骤 2: 创建 Pipeline
======================================================================
正在创建 basic Pipeline...
✅ Pipeline 创建成功
   - Pipeline名称: photos_dataset_basic_8
   - 类型: basic
   - 批次大小: 8
   - 目标尺寸: 256x256

======================================================================
  ✅ 配置完成！
======================================================================

**数据集:** photos_dataset
**Pipeline:** photos_dataset_basic_8
**状态:** 准备就绪，可以开始训练
```

---

### Example 5: Error Handling - Path Not Found

```
👤 > 数据在 /nonexistent/path，batch 32

======================================================================
  正在分析您的需求...
======================================================================

📋 检测到的参数:
   - 数据源: local
   - 位置: /nonexistent/path
   - 批次大小: 32
   - 图像尺寸: 224x224
   - Pipeline类型: basic

======================================================================
  步骤 1: 准备数据集
======================================================================
正在导入本地数据集: /nonexistent/path...
❌ 数据集导入失败: Path does not exist: /nonexistent/path
```

---

## Command Reference

### List Resources

```
👤 > list

======================================================================
  资源列表
======================================================================

📦 数据集 (3):
   - imagenet_dataset: /data/imagenet
   - test_dataset: /tmp/dali_dataset_test_dataset_abc123
   - photos_dataset: /home/user/photos

🔧 Pipeline (3):
   - imagenet_dataset_augmentation_32: augmentation (batch=32)
   - test_dataset_basic_16: basic (batch=16)
   - photos_dataset_basic_8: basic (batch=8)
```

### Test Pipeline

```
👤 > test imagenet_dataset_augmentation_32

运行 Pipeline 测试: imagenet_dataset_augmentation_32...
✅ Pipeline 运行成功
   - 迭代次数: 1
   - 批次大小: 32
   - Batch 1: ['[(3, 224, 224), ...]', '[(1,), ...]']
```

---

## Test Cases

### Test Case 1: ImageNet Standard Setup

**Input:**
```
我需要准备ImageNet训练数据，路径 /data/imagenet/train，batch 256，224x224，需要数据增强
```

**Expected Output:**
- Dataset: `train_dataset`
- Pipeline: `train_dataset_augmentation_256`
- Type: `augmentation`
- Batch: 256
- Size: 224x224

---

### Test Case 2: CIFAR-10 Style (Small Images)

**Input:**
```
Create test dataset, 50000 images, 32x32, batch 128, with augmentation
```

**Expected Output:**
- Dataset: `test_dataset`
- Pipeline: `test_dataset_augmentation_128`
- Type: `augmentation`
- Batch: 128
- Size: 32x32
- Images: 50000

---

### Test Case 3: Object Detection (Larger Images)

**Input:**
```
数据在 /data/coco，batch 16，尺寸 640x640，做增强
```

**Expected Output:**
- Dataset: `coco_dataset`
- Pipeline: `coco_dataset_augmentation_16`
- Type: `augmentation`
- Batch: 16
- Size: 640x640

---

### Test Case 4: Inference Only (No Augmentation)

**Input:**
```
Validation data at /data/imagenet/val, batch 64, 224x224, no augmentation
```

**Expected Output:**
- Dataset: `val_dataset`
- Pipeline: `val_dataset_basic_64`
- Type: `basic`
- Batch: 64
- Size: 224x224

---

### Test Case 5: MinIO/S3 Import

**Input:**
```
从 s3://ml-datasets/animals 导入，batch 32，增强
```

**Expected Output:**
- Dataset: `s3_ml-datasets`
- Pipeline: `s3_ml-datasets_augmentation_32`
- Type: `augmentation`
- Batch: 32
- S3 URI: `s3://ml-datasets/animals`

---

## Integration with LLM APIs

### Using with Claude API

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Load system prompt
with open("AGENT_PROMPT.md", "r") as f:
    system_prompt = f.read()

# User request
user_message = "我需要准备一个图像分类数据集，数据在 /data/imagenet 路径，批次大小32，图像尺寸224x224，需要随机裁剪和水平翻转"

# Call Claude
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    system=system_prompt,
    messages=[
        {"role": "user", "content": user_message}
    ]
)

print(message.content[0].text)
```

### Using with OpenAI API

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load system prompt
with open("AGENT_PROMPT.md", "r") as f:
    system_prompt = f.read()

# User request
user_message = "Create a test dataset with 100 images, batch 16, size 128x128"

# Call GPT-4
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
)

print(response.choices[0].message.content)
```

---

## Advanced Usage

### Batch Processing Multiple Requests

```python
from dali_agent import DALIAgent

agent = DALIAgent()

requests = [
    "训练数据在 /data/train，batch 32，增强",
    "验证数据在 /data/val，batch 64，不增强",
    "测试数据在 /data/test，batch 128，不增强"
]

for req in requests:
    agent.process_request(req)
```

### Custom API Endpoint

```python
import os

# Set custom API endpoint
os.environ["DALI_API_BASE"] = "http://my-server:8080"

from dali_agent import DALIAgent
agent = DALIAgent()
```

---

## Troubleshooting

### Issue 1: Agent can't connect to API

**Error:**
```
❌ 无法连接到 DALI API 服务器
```

**Solution:**
1. Start the HTTP server: `python dali_http_server.py`
2. Check if server is running: `curl http://localhost:8000/health`
3. Verify port (default: 8000)

### Issue 2: Path not recognized

**Problem:** Agent doesn't extract path correctly

**Example:**
```
Input: "process images in folder data"
Agent: ❌ 未指定数据路径
```

**Solution:** Use explicit path format:
- ✅ "数据在 /data/images"
- ✅ "data at /data/images"
- ❌ "in folder data"

### Issue 3: Wrong pipeline type detected

**Problem:** Agent creates `basic` when you want `augmentation`

**Solution:** Use explicit augmentation keywords:
- ✅ "需要数据增强" / "with augmentation"
- ✅ "随机裁剪" / "random crop"
- ✅ "翻转" / "flip"

---

## Performance Tips

1. **Batch Size:** Use larger batches (64, 128, 256) for better GPU utilization
2. **Image Size:** Common sizes: 224 (ImageNet), 256, 512, 640 (detection)
3. **Pipeline Type:**
   - Training: Use `augmentation`
   - Validation/Inference: Use `basic`

---

## Next Steps

1. Review the agent prompt: `AGENT_PROMPT.md`
2. Try the interactive mode: `python dali_agent.py`
3. Integrate with your LLM: See examples above
4. Customize for your use case: Modify `NLParser` class

---

## FAQ

**Q: Can I use this with other languages?**
A: Yes, the agent supports both Chinese and English. Extend `NLParser` for other languages.

**Q: How do I add custom augmentation options?**
A: Modify the DALI HTTP server (`dali_http_server.py`) to add new pipeline types, then update the agent prompt.

**Q: Can I use this in production?**
A: Yes, but add authentication and input validation to the HTTP API first.

**Q: How do I chain multiple operations?**
A: Use the agent multiple times or modify `process_request()` to handle complex workflows.

---

## License

Same as DALI MCP Server
