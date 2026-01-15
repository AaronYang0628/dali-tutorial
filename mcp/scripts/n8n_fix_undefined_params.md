# N8N Agent 参数提取问题修复指南

## 问题描述

当用户输入 "Prepare training data from /data/images with augmentation" 时,Agent调用 `import_local_dataset` API的参数为:

```json
{
  "dataset_name": "[undefined]",
  "local_path": "[undefined]",
  "supported_formats": "jpg,jpeg,png"
}
```

## 根本原因

N8N的AI Agent无法自动从用户输入中提取参数,需要在工具的**Description**字段中明确告诉LLM如何提取参数。

## 解决方案

### 方案1: 更新工具描述 (推荐)

在N8N的AI Agent节点中,更新 `import_local_dataset` 工具的配置:

**原来的Description**:
```
Import images from a local directory
```

**修改为**:
```
Import images from a local directory. Extract the dataset_name from the path (use the last folder name) and use the full path as local_path. For example, if path is '/data/images', set dataset_name to 'images' and local_path to '/data/images'.
```

### 方案2: 更新System Prompt

在N8N的AI Agent节点的**System Prompt**开头添加以下内容:

```markdown
## Parameter Extraction Guidelines

**CRITICAL**: When calling tools, you MUST extract parameters from the user's natural language input. DO NOT leave parameters as [undefined].

### How to Extract Parameters from User Input

**For IMPORT_LOCAL_DATASET**:
```
User Input: "Prepare training data from /data/images with augmentation"
Extract:
  - dataset_name: "images" (last folder in path)
  - local_path: "/data/images" (full path from input)

User Input: "Load images from /workspace/my_dataset"
Extract:
  - dataset_name: "my_dataset"
  - local_path: "/workspace/my_dataset"
```

**For CREATE_PIPELINE**:
```
User Input: "...with augmentation"
Extract:
  - pipeline_type: "augmentation"

User Input: "...for inference" or no mention of augmentation
Extract:
  - pipeline_type: "basic"
```

**General Rules**:
- Extract paths using patterns like "from X", "at X", "in X"
- Extract dataset_name as the last folder name from the path
- If path is not specified, ask the user
- Default values: batch_size=4, target_size=224, supported_formats="jpg,jpeg,png"
```

### 方案3: 同时应用方案1和方案2 (最佳)

为了获得最好的效果,建议同时更新工具描述和System Prompt。

## 完整的工具配置

### import_local_dataset 工具配置

```
Name: import_local_dataset

Description: Import images from a local directory. Extract the dataset_name from the path (use the last folder name) and use the full path as local_path. For example, if path is '/data/images', set dataset_name to 'images' and local_path to '/data/images'.

Method: POST

URL: http://localhost:8000/api/dataset/import/local

Body (JSON):
{
  "dataset_name": "{{$json.dataset_name}}",
  "local_path": "{{$json.local_path}}",
  "supported_formats": "{{$json.supported_formats || 'jpg,jpeg,png'}}"
}
```

### create_pipeline 工具配置

```
Name: create_pipeline

Description: Create an image processing pipeline. Extract pipeline_type: use 'augmentation' if user mentions augmentation/training/flip/crop/rotate, otherwise use 'basic'. Extract dataset_name from previous import step.

Method: POST

URL: http://localhost:8000/api/pipeline/create

Body (JSON):
{
  "name": "{{$json.name}}",
  "dataset_name": "{{$json.dataset_name}}",
  "pipeline_type": "{{$json.pipeline_type || 'basic'}}",
  "batch_size": "{{$json.batch_size || 4}}",
  "target_size": "{{$json.target_size || 224}}"
}
```

## 测试步骤

1. **更新配置**:在N8N中应用上述修改

2. **重新加载System Prompt**: 复制更新后的 `n8n_agent_prompt.md` 的完整内容到AI Agent节点的System Prompt字段

3. **测试用例1**:
   ```
   User: "Prepare training data from /data/images with augmentation"
   Expected:
     - import_local_dataset called with dataset_name="images", local_path="/data/images"
     - create_pipeline called with pipeline_type="augmentation"
   ```

4. **测试用例2**:
   ```
   User: "Load images from /workspace/cats for inference"
   Expected:
     - import_local_dataset called with dataset_name="cats", local_path="/workspace/cats"
     - create_pipeline called with pipeline_type="basic"
   ```

5. **测试用例3**:
   ```
   User: "Create a test dataset with 50 images"
   Expected:
     - create_dataset called with name="test_dataset", num_images=50
   ```

## 验证

在N8N的执行日志中,检查工具调用的参数应该类似:

```json
{
  "dataset_name": "images",
  "local_path": "/data/images",
  "supported_formats": "jpg,jpeg,png"
}
```

**不应该出现** `[undefined]` 值。

## 额外提示

### 如果问题仍然存在

1. **检查LLM模型**: 确保使用的是GPT-4或Claude 3.5,旧模型可能理解能力不足

2. **添加Few-Shot示例**: 在System Prompt中添加更多示例:
   ```markdown
   ## Example Conversations

   User: "Prepare training data from /data/images with augmentation"
   Agent Thought: I need to extract dataset_name="images" from path and local_path="/data/images"
   Agent Action: Call import_local_dataset with {"dataset_name": "images", "local_path": "/data/images", "supported_formats": "jpg,jpeg,png"}
   ```

3. **使用预处理Function节点**: 如果AI Agent仍然无法正确提取,可以在Agent之前添加Function节点:
   ```javascript
   // Pre-process user input
   const userMessage = $input.first().json.message;

   // Extract path
   const pathMatch = userMessage.match(/from\s+(\/[\w\/\-_]+)/i);
   const localPath = pathMatch ? pathMatch[1] : null;

   // Extract dataset name
   const datasetName = localPath ? localPath.split('/').pop() : null;

   // Detect augmentation
   const pipelineType = /augment|training|flip|crop|rotate/i.test(userMessage)
     ? 'augmentation'
     : 'basic';

   return {
     message: userMessage,
     extracted: {
       dataset_name: datasetName,
       local_path: localPath,
       pipeline_type: pipelineType
     }
   };
   ```

## 相关文件

- `/workspaces/dali-tutorial/mcp/scripts/n8n_agent_prompt.md` - 完整的Agent System Prompt
- `/workspaces/dali-tutorial/mcp/scripts/n8n_setup_guide.md` - N8N配置指南

## 支持

如果问题依然无法解决,请检查:
1. N8N Agent节点的LLM配置
2. System Prompt是否完整复制
3. 工具的Description字段是否更新
4. N8N的执行日志,查看Agent的推理过程
