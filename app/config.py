"""全局配置 —— 唯一配置入口。

设计原则:
1. 代码里不出现硬编码的模型名 / 端口 / 阈值,全部收敛到这里,便于测试注入;
2. `auto` 策略:没有 API Key、没装 ML 依赖、缺向量库时自动降级,
   保证 `git clone` 之后零配置即可运行(面试演示的关键);
3. `describe()` 输出当前实际生效的技术栈,前端顶栏、/api/meta、评测报告都用它,
   避免"文档写一套、跑的是另一套"。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """读取项目根 .env(已存在则覆盖),不引入额外依赖。"""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # setdefault:已存在的环境变量优先,方便测试/CI/临时覆盖(如 RERANK_ENABLED=false)
            os.environ.setdefault(key, value)


_load_dotenv()


def _s(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _i(name: str, default: int) -> int:
    try:
        return int(_s(name, str(default)))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(_s(name, str(default)))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = _s(name, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ------------------------------------------------------------------ 应用
    app_name: str = field(default_factory=lambda: _s("APP_NAME", "KnowledgeFlow"))
    app_env: str = field(default_factory=lambda: _s("APP_ENV", "dev"))
    # 启动时预热模型/分词器,避免首个请求承担加载耗时(测试环境关闭)
    warmup: bool = field(default_factory=lambda: _b("WARMUP", True))
    version: str = field(default_factory=lambda: _s("APP_VERSION", "1.0.0"))
    log_level: str = field(default_factory=lambda: _s("LOG_LEVEL", "INFO"))

    # ------------------------------------------------------------------ 路径
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _s("DATA_DIR", "data"))
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _s("INDEX_DIR", "data/index"))
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _s("RAW_DIR", "data/raw"))
    web_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "web")

    # ------------------------------------------------------------------ LLM
    llm_backend: str = field(default_factory=lambda: _s("LLM_BACKEND", "auto"))
    llm_base_url: str = field(
        default_factory=lambda: _s("LLM_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _s("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _s("LLM_MODEL", "deepseek/deepseek-v4.1-flash"))
    llm_model_pro: str = field(
        default_factory=lambda: _s("LLM_MODEL_PRO", "deepseek/deepseek-v4.1-flash")
    )
    # 辅助任务(查询改写 / HyDE / 摘要)用更快的模型,降低端到端延迟
    llm_model_fast: str = field(
        default_factory=lambda: _s("LLM_MODEL_FAST", "deepseek/deepseek-v4-flash-fast")
    )
    # 可选代理(如本机 sing-box / 公司网关);留空则直连,不做环境变量推断
    llm_proxy: str = field(default_factory=lambda: _s("LLM_PROXY"))
    llm_timeout: int = field(default_factory=lambda: _i("LLM_TIMEOUT", 90))
    llm_max_retries: int = field(default_factory=lambda: _i("LLM_MAX_RETRIES", 2))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.3))
    # 成本估算单价(每 1K token,美元),仅用于展示;按实际网关价改
    cost_per_1k_prompt: float = field(default_factory=lambda: _f("COST_PER_1K_PROMPT", 0.00014))
    cost_per_1k_completion: float = field(
        default_factory=lambda: _f("COST_PER_1K_COMPLETION", 0.00028)
    )

    # ------------------------------------------------------------------ Embedding
    embedding_backend: str = field(default_factory=lambda: _s("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(
        default_factory=lambda: _s("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    hash_dim: int = field(default_factory=lambda: _i("HASH_DIM", 512))
    hf_endpoint: str = field(default_factory=lambda: _s("HF_ENDPOINT"))

    # ------------------------------------------------------------------ 重排
    # 默认关闭:重排模型 1.1GB + 每次多 ~300ms,收益见 docs/EVALUATION.md(消融结论)
    rerank_enabled: bool = field(default_factory=lambda: _b("RERANK_ENABLED", False))
    rerank_backend: str = field(default_factory=lambda: _s("RERANK_BACKEND", "auto"))
    rerank_model: str = field(default_factory=lambda: _s("RERANK_MODEL", "BAAI/bge-reranker-base"))
    rerank_top_n: int = field(default_factory=lambda: _i("RERANK_TOP_N", 10))

    # ------------------------------------------------------------------ 向量库
    vector_backend: str = field(default_factory=lambda: _s("VECTOR_BACKEND", "auto"))
    collection_name: str = field(default_factory=lambda: _s("COLLECTION_NAME", "kf_docs"))

    # ------------------------------------------------------------------ 检索
    top_k: int = field(default_factory=lambda: _i("TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _i("CANDIDATE_K", 20))
    rrf_k: int = field(default_factory=lambda: _i("RRF_K", 60))
    rewrite_enabled: bool = field(default_factory=lambda: _b("REWRITE_ENABLED", True))
    # HyDE 在本项目语料上没有带来 Top-1 提升,反而多 ~1s 延迟 -> 默认关闭(见 docs/EVALUATION.md)
    hyde_enabled: bool = field(default_factory=lambda: _b("HYDE_ENABLED", False))

    # ------------------------------------------------------------------ Agent / NLU
    max_agent_rounds: int = field(default_factory=lambda: _i("MAX_AGENT_ROUNDS", 3))
    tool_timeout_s: float = field(default_factory=lambda: _f("TOOL_TIMEOUT_S", 8.0))
    intent_enabled: bool = field(default_factory=lambda: _b("INTENT_ENABLED", True))
    intent_backend: str = field(default_factory=lambda: _s("INTENT_BACKEND", "auto"))
    intent_model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _s("INTENT_MODEL_DIR", "data/intent_model")
    )
    intent_threshold: float = field(default_factory=lambda: _f("INTENT_THRESHOLD", 0.6))
    self_check_enabled: bool = field(default_factory=lambda: _b("SELF_CHECK_ENABLED", True))
    agentic_max_rounds: int = field(default_factory=lambda: _i("AGENTIC_MAX_ROUNDS", 3))

    # ------------------------------------------------------------------ 记忆
    memory_window_turns: int = field(default_factory=lambda: _i("MEMORY_WINDOW_TURNS", 5))
    summary_threshold_tokens: int = field(
        default_factory=lambda: _i("SUMMARY_THRESHOLD_TOKENS", 2000)
    )

    # ------------------------------------------------------------------ 服务
    host: str = field(default_factory=lambda: _s("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _i("PORT", 8000))

    # ------------------------------------------------------------------ 派生
    def __post_init__(self) -> None:
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        for p in (self.data_dir, self.index_dir, self.raw_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ 生效后端解析(带降级)
    def resolved_llm_backend(self) -> str:
        """auto: 有 key 且是 openai 兼容 -> openai;否则 mock(离线演示)。"""
        backend = self.llm_backend.lower()
        if backend != "auto":
            return backend
        return "openai" if self.llm_api_key else "mock"

    def resolved_embedding_backend(self) -> str:
        """auto: 装了 sentence-transformers 且模型可加载 -> st;否则 hash。"""
        backend = self.embedding_backend.lower()
        if backend != "auto":
            return backend
        try:
            import sentence_transformers  # noqa: F401

            return "st"
        except Exception:  # noqa: BLE001
            return "hash"

    def resolved_vector_backend(self) -> str:
        backend = self.vector_backend.lower()
        if backend != "auto":
            return backend
        try:
            import chromadb  # noqa: F401

            return "chroma"
        except Exception:  # noqa: BLE001
            return "numpy"

    def resolved_rerank_backend(self) -> str:
        if not self.rerank_enabled:
            return "none"
        backend = self.rerank_backend.lower()
        if backend != "auto":
            return backend
        try:
            import sentence_transformers  # noqa: F401

            return "st"
        except Exception:  # noqa: BLE001
            return "none"

    @property
    def offline_demo(self) -> bool:
        """是否处于完全没有外部依赖的演示模式。"""
        return self.resolved_llm_backend() == "mock"

    def describe(self) -> dict:
        return {
            "app": f"{self.app_name} v{self.version}",
            "llm": {
                "backend": self.resolved_llm_backend(),
                "model": self.llm_model if self.resolved_llm_backend() == "openai" else "mock",
            },
            "embedding": {
                "backend": self.resolved_embedding_backend(),
                "model": self.embedding_model
                if self.resolved_embedding_backend() == "st"
                else f"hash-{self.hash_dim}",
            },
            "vector_store": self.resolved_vector_backend(),
            "rerank": {
                "enabled": self.resolved_rerank_backend() != "none",
                "model": self.rerank_model,
            },
            "intent": {
                "enabled": self.intent_enabled,
                "threshold": self.intent_threshold,
            },
        }

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000 * self.cost_per_1k_prompt
            + completion_tokens / 1000 * self.cost_per_1k_completion
        )


settings = Settings()
