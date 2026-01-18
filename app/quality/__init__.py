"""回答质量:幻觉自检(独立 judge + 证据核查)。"""

from app.quality.self_correction import REFUSAL, CheckResult, SelfChecker

__all__ = ["REFUSAL", "CheckResult", "SelfChecker"]
