# HTTP API 契约(v1)

服务地址:`http://127.0.0.1:8000`(开发机),接口文档 `/docs`。所有响应 `application/json`,流式接口为 `text/event-stream`。

## 1. 对话(核心)

`POST /api/chat`

请求体:

```json
{
  "message": "出差住宿标准是多少？",
  "session_id": "default",
  "user_id": "default",
  "mode": "auto",
  "stream": true
}
```

- `mode`: `auto`(意图路由) | `fast`(混合检索快路径) | `quality`(改写+HyDE+重排+自检) | `agent`(工具循环) | `agentic`(迭代检索)
- `stream=false` 时直接返回聚合 JSON:`{answer, citations, tools, retrieval, usage, meta, self_check, error}`

### SSE 事件(`event: <name>` + `data: <json>`)

| event | data | 说明 |
|---|---|---|
| `meta` | `request_id, mode, path, plan{intent,confidence,path,reason}, llm{backend,model,degraded}, kb_chunks, memory{turns,summary,estimated_tokens}` | 请求开始,前端据此渲染状态徽章 |
| `retrieval` | `{trace: {query, mode, total_ms, stages:[{name, ms, info, hits:[{chunk_id,doc_id,title,text,score,channel,rank,detail}]}], candidates}}` | 检索链路;阶段名:`rewrite/vector/bm25/hyde/fuse/rerank`。agentic 模式下 `trace.stages` 为多轮检索,信息在 `info.query/info.sufficient/info.missing` |
| `delta` | `{text}` | 回答增量,追加渲染 |
| `replace` | `{text, reason}` | 自检修订:用新文本**替换**已渲染的回答 |
| `citations` | `{items:[{index,doc_id,title,score,channel,text}]}` | 引用来源,`index` 与回答里的 `[n]` 对应 |
| `tools` | `{items:[{name,arguments,output,is_error,ms}]}` | Agent 工具调用记录(仅 agent/fast 路径) |
| `self_check` | `{verdict, reason}` | `supported / partial / unsupported` |
| `memory` | `{compressed_messages?, topic_changed?}` | 记忆维护事件(可选) |
| `usage` | `{prompt_tokens, completion_tokens, total_tokens, cost_usd, llm_calls, by_purpose, elapsed_ms}` | 用量与耗时 |
| `done` | `{answer, request_id, path, mode, self_check, elapsed_ms, trace}` | 结束,`trace` 为完整链路记录 |
| `error` | `{message}` | 错误 |

示例(截断):

```
event: meta
data: {"request_id":"40ecdd8ff44f","mode":"fast","path":"rag","plan":{"intent":"kb_query","confidence":0.86,"path":"rag","reason":"知识库问答"},"llm":{"backend":"openai","model":"deepseek/deepseek-v4.1-flash","degraded":false},"kb_chunks":75,"memory":{"turns":0,"summary":"","estimated_tokens":0}}

event: delta
data: {"text":"报销打款每月有两个批次：**5 日与 20 日**集中打款…"}
```

## 2. 系统信息

| 接口 | 返回 |
|---|---|
| `GET /api/meta` | `{app, llm{backend,model}, embedding{backend,model,dim}, vector_store, rerank{enabled,model}, intent{enabled,threshold,backend}, prompts{answer,rewrite,self_check}, kb{documents,chunks,embedding,vector_store,bm25,rerank}, offline_demo, modes[]}` |
| `GET /api/health` | `{status, version, uptime_s, kb_chunks, llm_backend}` |
| `GET /api/demo/questions` | `{questions:[{text,type,note}]}` 演示问题(取自评测集) |
| `GET /api/prompts` | `{versions{...}, templates_dir}` |
| `GET /metrics` | Prometheus 文本 |

## 3. 知识库

| 接口 | 说明 |
|---|---|
| `GET /api/kb/documents` | `{total, chunks, categories{名称:数量}, documents:[{doc_id,title,category,chunks,chars}]}` |
| `GET /api/kb/documents/{doc_id}` | `{doc_id,title,category,chunks:[{chunk_id,index,text}]}` |
| `POST /api/kb/ingest` | JSON `{title,category,text,strategy,rebuild}`,返回 `{documents,chunks,total_chunks,strategy}` |
| `POST /api/kb/upload` | multipart:`file`(.md/.txt,UTF-8) + `title` + `category` + `strategy` |
| `POST /api/kb/reindex?strategy=recursive` | 全量重建 `data/raw` 索引 |

## 4. 记忆

| 接口 | 说明 |
|---|---|
| `GET /api/memory/{user_id}/{session_id}` | `{user_id,session_id,window_turns,turns,summary,estimated_tokens,preferences,topic_changes}` |
| `DELETE /api/memory/{user_id}` | 删除该用户长期偏好 |

## 5. 前端约定

- 静态资源由 `GET /` 返回 `web/index.html`,`GET /static/*` 返回 `web/` 下的文件;
- **不允许引用任何 CDN / 外网资源**(演示环境可能断网);
- `POST /api/chat` 需用 `fetch` + `ReadableStream` 解析 SSE(逐行解析 `event:` / `data:`);
- `citations[].index` 与回答文本里的 `[n]` 对应,点击可定位到证据片段;
- 模式名、后端名、阈值等都通过 `/api/meta` 获取,不要写死。
