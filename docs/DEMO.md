# 功能导览与本地运行

> 这不是照读稿，而是**功能清单 + 本地怎么跑起来看**。
> 想验证系统是否工作，按下面顺序点一遍即可，全程不需要 API Key。

## 0. 启动

```bash
cd knowledgeflow-agent
./start_server.sh start                       # 起服务(首次会自动装依赖/建索引)
curl -s http://127.0.0.1:8000/api/health      # {"status":"ok",...}
curl -s http://127.0.0.1:8000/api/meta        # 当前生效的技术栈与开关
```

- 浏览器打开 `http://127.0.0.1:8000`
- 顶部徽章显示当前后端：真实模型 / 离线演示
- 首次运行会下载 embedding 模型（约 100MB）；无网络时自动降级为哈希向量

## 1. 界面分区

| 区域 | 内容 |
| --- | --- |
| 顶部栏 | LLM / Embedding 后端、知识库规模、运行时长、新会话 |
| 模式切换 | auto / fast / quality / agent / agentic，按钮由 `/api/meta` 的 `modes` 字段驱动 |
| 对话区 | 流式回答，`[n]` 编号引用可点 |
| 右侧「检索链路」 | 请求摘要、用量与耗时、各阶段时间轴、引用证据、Agent 工具卡片、完整 trace JSON |
| 知识库页 | 文档列表、分块预览、上传、文本入库、全量重建 |
| 关于页 | 技术栈、请求处理链路、五种模式说明、运行指标、设计取舍 |

## 2. 按顺序验证五条路径

### ① 关键词问题 → 引用与检索链路

问：「报销打款是每月几号」

- 回答流式输出；`[1]` 点击后右侧高亮证据卡片（原文片段 + 分数）；
- 时间轴显示 vector / bm25 / fuse 各多少 ms、候选多少；
- 底部 usage：token、LLM 调用次数、成本。

### ② 口语化问题 → 语义召回

问：「我出差住酒店的钱怎么找公司要回来」

- 这句话里没有「报销」字面，BM25 命中不了，向量检索能召回；
- 评测集里专门放了 16 条 paraphrase：纯关键词 Top-1 31.3%，混合 + 改写 81.3%。

### ③ 多跳问题 → Agentic 迭代检索

问：「出差住宿超标了怎么审批，钱谁出」

- 第一轮只查到标准，judge 判断「缺审批流程」，第二轮带反馈改写再查；
- 时间轴出现「第 1 轮 / 第 2 轮检索」与每轮的 `sufficient` / `missing`。

### ④ 库外问题 → 幻觉自检

问：「公司股价多少」

- 预期是拒答 + 建议，而不是编造数字；
- 这是独立 judge + 证据核查的结果；自检结论分 `supported` / `partial` / `unsupported` 三档。

### ⑤ Agent 工具循环

切到 agent 模式，问：「现在几点，另外算一下 (1200+300)*0.8」

- 右侧出现工具调用卡片（`get_datetime` / `calculator`），入参、输出、耗时可见；
- calculator 走 AST 白名单求值，不执行任意代码。

## 3. 用数据确认效果

```bash
python3 -m json.tool eval/report-retrieval.json # 六组检索消融 + 三组分块对比
cat eval/report-intent.json                     # 意图分类
cat eval/report-e2e.json | python -m json.tool  # 端到端(需 Key 才能重跑)
.venv/bin/python -m pytest -q                   # 103 个离线测试
```

主要结论见 [EVALUATION.md](EVALUATION.md)：

1. 检索 Top-1：BM25 62.5% → 混合 85% → 加查询改写 90%（paraphrase 类 31.3% → 81.3%）；
2. 意图分类 96.2%（规则基线 59.9%），规则分类器是模型缺失时的降级路径；
3. 端到端库外拒答率 100%（quality 模式），引用命中率 @3 97.5%。

完整链路可直接跑脚本复现：`make eval`（约 6 分钟，端到端部分需要 API Key）。

## 4. 三种"不做"的判断

评测数据不支持的事，默认就不开，每个"不用"都记在 [DECISIONS.md](DECISIONS.md)：

- **HyDE**：中文短查询上收益不稳定，还多约 2s → 默认关闭，保留开关；
- **Cross-Encoder 重排**：模型 1.1GB、每查询 +3.3s，本语料 Top-1 反而 -2.5pt → 默认关闭；
- **向量数据库**：当前 75 块的检索 p50 17ms，这个规模上 ANN 属于过度设计 → 保留 `BaseVectorStore` 接口。

原则是：先量化，再决定，默认值由数据来定而不是跟风。

## 5. 命令速查

```bash
./start_server.sh start|stop|restart|status
make seed                       # 重建索引
make eval                       # 检索消融 + 端到端评测
make test                       # 103 个离线测试
python -m app.cli.main --mode agent      # 终端里走 Agent 循环
python -m app.cli.main -q "报销打款是每月几号" --mode quality --no-stream
curl -N -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"报销打款是每月几号","mode":"fast","stream":true}'
```

## 6. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 顶栏显示「离线演示」 | 没配 `LLM_API_KEY`，走确定性 Mock；检索、引用、工具链路仍然真实 |
| 服务起不来 | `./start_server.sh status` → `tail -50 logs/server.log`；或直接用 CLI |
| 检索很慢 | `quality` 模式含 1 次改写 LLM 调用；切 `fast` 可降到毫秒级 |
| 意图分类走规则 | 未装 torch 或模型缺失，自动降级（准确率 59.9%），功能不受影响 |
| 重排没生效 | 默认关闭；`RERANK_ENABLED=true` 才加载 Cross-Encoder |
| 端口占用 | `PORT=8010 ./start_server.sh restart` |

## 7. 录制演示视频（可选）

```bash
uv pip install playwright imageio-ffmpeg
.venv/bin/python -m playwright install chromium     # 只装一次
make demo-video                                      # 或 python scripts/record_demo.py
```

脚本会真实操作页面（点示例问题、点引用、切模式、进知识库/关于页），并注入
「光标 + 点击涟漪 + 字幕条 + 片头卡」，输出：

- `docs/assets/demo.mp4`：H.264，适合发微信 / 贴到 README；
- `docs/assets/demo.gif`：README 里自动播放的压缩版；
- `docs/assets/demo-poster.png`：封面图。

成片 65 秒；录制墙钟耗时约 70~90 秒（取决于模型延迟），期间会真实消耗若干次问答的 token。
