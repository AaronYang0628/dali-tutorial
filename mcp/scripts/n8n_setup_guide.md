# N8N DALI Agent 设置指南

本指南说明如何在N8N中配置DALI数据准备Agent。

## 前置要求

1. 运行中的DALI HTTP服务器 (默认: http://localhost:8000)
2. N8N实例 (云端或自托管)
3. N8N中的AI Agent节点或Chat节点

## 快速启动

### 1. 启动DALI HTTP服务器

```bash
cd /workspaces/dali-tutorial/mcp/scripts
python dali_http_server.py --host 0.0.0.0 --port 8000
```

验证服务:
```bash
curl http://localhost:8000/health
```

### 2. N8N工作流配置

#### 方案A: 使用AI Agent节点 (推荐)

**工作流结构**:
```
[Webhook/Chat Trigger]
    → [AI Agent]
        → [HTTP Request Tools (多个)]
    → [Response]
```

**AI Agent节点配置**:

1. **System Prompt**: 复制完整的 `n8n_agent_prompt.md` 内容

2. **添加工具 (Tools)**:

   **工具1: create_dataset**
   - Name: `create_dataset`
   - Description: "Create a synthetic test dataset with random images"
   - Method: POST
   - URL: `http://localhost:8000/api/dataset/create`
   - Body (JSON):
     ```json
     {
       "name": "{{$json.name}}",
       "num_images": "{{$json.num_images || 10}}",
       "image_size": "{{$json.image_size || 256}}"
     }
     ```

   **工具2: import_local_dataset**
   - Name: `import_local_dataset`
   - Description: "Import images from a local directory. Extract the dataset_name from the path (use the last folder name) and use the full path as local_path. For example, if path is '/data/images', set dataset_name to 'images' and local_path to '/data/images'."
   - Method: POST
   - URL: `http://localhost:8000/api/dataset/import/local`
   - Body (JSON):
     ```json
     {
       "dataset_name": "{{$json.dataset_name}}",
       "local_path": "{{$json.local_path}}",
       "supported_formats": "{{$json.supported_formats || 'jpg,jpeg,png'}}"
     }
     ```

   **工具3: import_s3_dataset**
   - Name: `import_s3_dataset`
   - Description: "Import images from S3-compatible storage"
   - Method: POST
   - URL: `http://localhost:8000/api/dataset/import/s3`
   - Body (JSON):
     ```json
     {
       "dataset_name": "{{$json.dataset_name}}",
       "s3_uri": "{{$json.s3_uri}}",
       "endpoint_url": "{{$json.endpoint_url}}",
       "access_key": "{{$json.access_key}}",
       "secret_key": "{{$json.secret_key}}",
       "download": "{{$json.download || false}}",
       "supported_formats": "{{$json.supported_formats || ['jpg', 'jpeg', 'png']}}"
     }
     ```

   **工具4: list_datasets**
   - Name: `list_datasets`
   - Description: "List all registered datasets"
   - Method: GET
   - URL: `http://localhost:8000/api/dataset/list`

   **工具5: create_pipeline**
   - Name: `create_pipeline`
   - Description: "Create an image processing pipeline"
   - Method: POST
   - URL: `http://localhost:8000/api/pipeline/create`
   - Body (JSON):
     ```json
     {
       "name": "{{$json.name}}",
       "dataset_name": "{{$json.dataset_name}}",
       "pipeline_type": "{{$json.pipeline_type || 'basic'}}",
       "batch_size": "{{$json.batch_size || 4}}",
       "target_size": "{{$json.target_size || 224}}"
     }
     ```

   **工具6: run_pipeline**
   - Name: `run_pipeline`
   - Description: "Execute a processing pipeline"
   - Method: POST
   - URL: `http://localhost:8000/api/pipeline/run`
   - Body (JSON):
     ```json
     {
       "pipeline_name": "{{$json.pipeline_name}}",
       "num_iterations": "{{$json.num_iterations || 1}}"
     }
     ```

   **工具7: list_pipelines**
   - Name: `list_pipelines`
   - Description: "List all configured pipelines"
   - Method: GET
   - URL: `http://localhost:8000/api/pipeline/list`

   **工具8: health_check**
   - Name: `health_check`
   - Description: "Check DALI service health status"
   - Method: GET
   - URL: `http://localhost:8000/health`

#### 方案B: 使用传统HTTP请求节点

如果不使用AI Agent,可以构建如下工作流:

```
[Webhook/Chat]
    → [Function: Parse Intent]
    → [Switch: Route by Action]
        → [HTTP Request: Create Dataset]
        → [HTTP Request: Import Local]
        → [HTTP Request: Create Pipeline]
        → ...
    → [Function: Format Response]
    → [Respond]
```

### 3. 测试工作流

**测试用例1: 创建测试数据集**
```
User: "Create a test dataset with 50 images"
Expected: Agent calls create_dataset API and confirms creation
```

**测试用例2: 完整流程**
```
User: "Prepare training data from /data/images with augmentation"
Expected: Agent calls import_local → create_pipeline → run_pipeline
```

**测试用例3: 列出资源**
```
User: "What datasets do I have?"
Expected: Agent calls list_datasets and presents results
```

## 环境变量配置

在N8N中设置以下环境变量:

```bash
# DALI服务地址
DALI_API_BASE_URL=http://localhost:8000

# 可选: S3凭证 (如果需要S3导入)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_ENDPOINT=https://s3.amazonaws.com
```

然后在HTTP请求节点中使用: `{{$env.DALI_API_BASE_URL}}/api/...`

## 高级配置

### 1. 错误处理

在HTTP Request工具中添加错误处理:

```javascript
// 在"On Error"分支中
if ($json.statusCode === 404) {
  return { message: "Resource not found. Please check the name and try again." };
} else if ($json.statusCode === 409) {
  return { message: "Resource already exists. Please use a different name." };
} else {
  return { message: `API error: ${$json.error}` };
}
```

### 2. 响应格式化

在Agent之后添加Function节点来格式化响应:

```javascript
// 格式化Pipeline运行结果
const result = $input.first().json;
return {
  summary: `Processed ${result.iterations} batches`,
  details: result.batches.map(b =>
    `Batch ${b.iteration}: ${b.shapes.join(', ')}`
  ).join('\n')
};
```

### 3. 会话状态管理

使用N8N的Sticky Note或Database节点保存会话状态:

```javascript
// 保存最后使用的dataset和pipeline
$context.set('last_dataset', datasetName);
$context.set('last_pipeline', pipelineName);
```

## 生产环境最佳实践

1. **安全性**:
   - 使用环境变量存储敏感信息
   - 在DALI服务器前配置反向代理(Nginx)
   - 启用HTTPS和API认证

2. **性能优化**:
   - 使用HTTP连接池
   - 设置合理的超时时间
   - 对于大型dataset,使用异步处理

3. **监控**:
   - 添加Webhook节点记录API调用
   - 使用N8N的Error Workflow捕获失败
   - 监控DALI服务器的健康状态

4. **资源管理**:
   - 定期清理临时数据集
   - 限制并发Pipeline运行数量
   - 监控GPU内存使用

## 故障排查

### 问题: "Connection refused"
**解决**:
- 确认DALI服务器正在运行: `curl http://localhost:8000/health`
- 检查防火墙设置
- 如果在Docker中运行,使用 `host.docker.internal` 而非 `localhost`

### 问题: "Dataset not found"
**解决**:
- 先调用 `list_datasets` 查看可用数据集
- 确保在创建Pipeline之前先创建/导入Dataset

### 问题: "GPU not available"
**解决**:
- DALI会自动降级到CPU模式
- 检查服务器日志确认设备模式
- 如需GPU加速,确保nvidia-docker正确配置

## 示例:完整N8N工作流JSON

```json
{
  "name": "DALI Data Preparation Agent",
  "nodes": [
    {
      "parameters": {
        "path": "dali-chat",
        "responseMode": "lastNode"
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook"
    },
    {
      "parameters": {
        "systemPrompt": "{{ $('Prompt').item.json.prompt }}",
        "tools": ["create_dataset", "import_local_dataset", "create_pipeline", "run_pipeline"]
      },
      "name": "AI Agent",
      "type": "@n8n/n8n-nodes-langchain.agent"
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "AI Agent"}]]
    }
  }
}
```

## 相关资源

- [DALI HTTP API文档](http://localhost:8000/docs) - FastAPI自动生成的交互式文档
- [N8N AI Agent文档](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/)
- [DALI官方文档](https://docs.nvidia.com/deeplearning/dali/)

## 支持

遇到问题? 检查:
1. DALI服务器日志: `/workspaces/dali-tutorial/mcp/scripts/dali_http_server.py`
2. N8N执行日志: Workflow → Executions
3. API健康状态: `curl http://localhost:8000/health`
