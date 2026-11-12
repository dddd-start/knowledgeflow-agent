"""测试公共 fixture。

原则:所有测试离线、确定性、不碰真实模型/网络 ——
用 MockLLM + 哈希向量 + 临时目录,保证 CI 里 `pytest` 秒级跑完。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.llm.mock import MockLLM  # noqa: E402
from app.nlu.intent import RuleClassifier  # noqa: E402
from app.nlu.router import IntentRouter  # noqa: E402
from app.rag.embeddings import Embedder  # noqa: E402
from app.rag.pipeline import RetrievalPipeline  # noqa: E402
from app.rag.rerank import Reranker  # noqa: E402
from app.rag.schemas import Document  # noqa: E402
from app.rag.vectorstore import NumpyVectorStore  # noqa: E402


@pytest.fixture()
def cfg(tmp_path: Path) -> Settings:
    """测试配置:哈希向量 + mock 模型 + 临时目录。"""
    return Settings(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        raw_dir=tmp_path / "raw",
        web_dir=tmp_path / "web",
        llm_backend="mock",
        llm_api_key="",
        embedding_backend="hash",
        hash_dim=64,
        vector_backend="numpy",
        rerank_enabled=False,
        hyde_enabled=True,  # 测试覆盖完整优化链路(默认生产配置为 False)
        intent_backend="rule",
        self_check_enabled=True,
        warmup=False,
        top_k=3,
        candidate_k=10,
        max_agent_rounds=2,
        agentic_max_rounds=2,
        memory_window_turns=2,
        summary_threshold_tokens=60,
    )


@pytest.fixture()
def docs() -> list[Document]:
    return [
        Document(
            doc_id="doc-财务制度-费用报销流程",
            title="费用报销流程",
            category="财务制度",
            text=(
                "# 费用报销流程\n\n## 报销时限\n"
                "费用发生后 30 天内提交报销，跨年费用最迟 12 月 25 日前提交。\n\n"
                "## 审批层级\n"
                "2000 元以下由直属主管审批；2000 元至 10000 元加签部门负责人；"
                "10000 元以上加签分管 VP。\n\n"
                "## 打款\n每月 5 日与 20 日两个批次集中打款，遇节假日顺延。\n"
            ),
        ),
        Document(
            doc_id="doc-财务制度-差旅标准",
            title="差旅标准与出差申请",
            category="财务制度",
            text=(
                "# 差旅标准\n\n## 住宿标准\n"
                "一线城市每晚上限 600 元，新一线城市 450 元，其他城市 350 元。\n\n"
                "## 出差申请\n"
                "出差前在 OA 提交申请，注明目的地、时间与预算，由部门负责人审批。\n"
            ),
        ),
        Document(
            doc_id="doc-人事制度-考勤管理制度",
            title="考勤管理制度",
            category="人事制度",
            text=(
                "# 考勤管理制度\n\n## 打卡\n"
                "每个工作日 9:00 前打卡，18:00 后打下班卡，忘记打卡可申请补卡。\n\n"
                "## 补卡\n每月最多补卡 3 次，超过 3 次按流程违规记录。\n"
            ),
        ),
        Document(
            doc_id="doc-IT服务-VPN与远程接入",
            title="VPN 与远程接入",
            category="IT服务",
            text=(
                "# VPN 与远程接入\n\n## 申请方式\n"
                "在 IT 服务台提交 VPN 申请，由直属主管审批，1 个工作日内开通。\n\n"
                "## 使用规范\n"
                "VPN 仅用于办公，禁止在公共网吧使用，会话 8 小时无操作自动断开。\n"
            ),
        ),
    ]


@pytest.fixture()
def embedder() -> Embedder:
    return Embedder(backend="hash", hash_dim=64)


@pytest.fixture()
def store(cfg: Settings) -> NumpyVectorStore:
    return NumpyVectorStore(cfg.index_dir)


@pytest.fixture()
def pipeline(cfg: Settings, embedder: Embedder, store: NumpyVectorStore, docs) -> RetrievalPipeline:
    pipe = RetrievalPipeline(
        MockLLM(),
        embedder=embedder,
        store=store,
        reranker=Reranker(backend="none"),
        cfg=cfg,
    )
    pipe.ingest(docs, strategy="recursive")
    return pipe


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def router(cfg: Settings) -> IntentRouter:
    return IntentRouter(classifier=RuleClassifier(), cfg=cfg)


@pytest.fixture()
def service(cfg: Settings, pipeline: RetrievalPipeline, mock_llm: MockLLM):
    from app.services import ChatService

    return ChatService(
        cfg=cfg,
        llm=mock_llm,
        pipeline=pipeline,
        router=IntentRouter(classifier=RuleClassifier(), cfg=cfg),
    )


@pytest.fixture()
def eval_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "eval"
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_queries.json").write_text(
        json.dumps(
            [
                {
                    "query": "报销打款是每月几号",
                    "gold_doc": "doc-财务制度-费用报销流程",
                    "type": "keyword",
                },
                {
                    "query": "出差住宿上限多少",
                    "gold_doc": "doc-财务制度-差旅标准",
                    "type": "paraphrase",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
