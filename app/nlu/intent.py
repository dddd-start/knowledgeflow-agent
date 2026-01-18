"""意图分类器 —— 微调 TinyBERT 为主,规则分类器兜底。

两类实现:
- `TinyBertClassifier`:用自建的 6 类意图数据集微调 TinyBERT_4L_zh(见
  scripts/train_intent.py),CPU 约 2 分钟,模型 ~50MB。评测准确率见 docs/EVALUATION.md;
- `RuleClassifier`:关键词 + 槽位规则,零依赖、可解释;模型没训练/未安装 torch 时自动启用,
  保证离线演示与 CI 都能跑通意图路由。

为什么要意图路由?
    知识库问答里 30%~40% 的请求其实是"现在几点""算一下""你好"这类简单意图,
    走一次 LLM 生成既慢又花钱。先分类:简单意图 0 次 LLM 直达工具,
    知识库问题走 1 次 LLM 的 RAG 快路径,只有复杂/不确定的才进 Agent 多轮循环。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings, settings

logger = logging.getLogger(__name__)

INTENTS = ["kb_query", "weather", "calculator", "datetime", "small_talk", "complex_agent"]

_RULES: list[tuple[str, re.Pattern, float]] = [
    ("weather", re.compile(r"天气|气温|下雨|温度|冷不冷|热不热|几度"), 0.95),
    ("calculator", re.compile(r"计算|算一下|等于多少|[\d.()]\s*[+\-*/×÷]\s*[\d.(]"), 0.93),
    ("datetime", re.compile(r"几点|现在时间|今天几号|星期几|周几|日期"), 0.92),
    ("small_talk", re.compile(r"^(你好|您好|hi|hello|嗨|在吗|谢谢|再见|早上好|晚上好)"), 0.90),
]
_KB_WORDS = re.compile(
    r"报销|差旅|发票|考勤|请假|年假|加班|入职|转正|培训|绩效|采购|预算|合同|"
    r"账号|权限|VPN|密码|邮箱|打印机|会议室|资产|发版|上线|部署|回滚|SLA|"
    r"数据|安全|合规|审计|应急|客服|产品|版本|定价|流程|制度|规定|标准"
)
_COMPLEX = re.compile(r"对比|分析|为什么|怎么优化|方案|分别|并且|同时|规划|评估|排查")


class RuleClassifier:
    backend = "rule"

    def predict(self, text: str) -> tuple[str, float]:
        text = (text or "").strip()
        for intent, pattern, confidence in _RULES:
            if pattern.search(text):
                return intent, confidence
        if _COMPLEX.search(text) and len(text) > 12:
            return "complex_agent", 0.78
        if _KB_WORDS.search(text):
            return "kb_query", 0.86
        # 兜底给低置信度 -> 路由层会回退到 Agent
        return "kb_query", 0.55


class TinyBertClassifier:
    backend = "tinybert"

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self._model: Any = None
        self._tokenizer: Any = None
        self._labels: list[str] = []
        self._lock = False

    @staticmethod
    def is_available(model_dir: Path) -> bool:
        weights = model_dir / "model.safetensors"
        pytorch = model_dir / "pytorch_model.bin"
        return (weights.exists() or pytorch.exists()) and (model_dir / "labels.json").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import BertForSequenceClassification, BertTokenizer

        # 本地 model_dir 里有 vocab.txt;用 slow tokenizer 兼容性最好
        self._tokenizer = BertTokenizer.from_pretrained(str(self.model_dir))
        self._model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
        self._model.eval()
        self._labels = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))[
            "labels"
        ]

    def predict(self, text: str) -> tuple[str, float]:
        import torch

        self._ensure_loaded()
        enc = self._tokenizer(
            text, truncation=True, max_length=32, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        index = int(probs.argmax().item())
        return self._labels[index], float(probs[index].item())


class IntentClassifier:
    """统一门面:优先微调模型,不可用时自动降级到规则。"""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self._impl: Any = None
        self._init_impl()

    def _init_impl(self) -> None:
        backend = self.cfg.intent_backend.lower()
        model_dir = Path(self.cfg.intent_model_dir)
        if backend in {"auto", "tinybert"} and TinyBertClassifier.is_available(model_dir):
            self._impl = TinyBertClassifier(model_dir)
            logger.info("意图分类器: TinyBERT (%s)", model_dir)
            return
        if backend == "tinybert":
            logger.warning("未找到微调模型 %s,降级为规则分类器", model_dir)
        self._impl = RuleClassifier()
        logger.info("意图分类器: 规则")

    @property
    def backend(self) -> str:
        return getattr(self._impl, "backend", "rule")

    def predict(self, text: str) -> tuple[str, float]:
        try:
            return self._impl.predict(text)
        except Exception as exc:  # noqa: BLE001  模型加载失败也要能服务
            logger.warning("意图分类失败,降级规则: %s", exc)
            self._impl = RuleClassifier()
            return self._impl.predict(text)


_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def reset_classifier() -> None:
    global _classifier
    _classifier = None


def load_labels() -> list[dict]:
    path = PROJECT_ROOT / "data" / "intents" / "labels.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
