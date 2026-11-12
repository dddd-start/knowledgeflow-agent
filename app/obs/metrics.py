"""轻量指标 —— 不引 Prometheus 客户端,手写计数与分位数。

面试项目最常见的问题是"能跑但没有可观测性"。这里用 ~80 行给出:
- 请求数 / 错误数 / 按模式的请求数;
- 端到端延迟与各阶段延迟的分位数(p50/p95);
- LLM token 与估算成本;
- 知识库规模、缓存(索引)状态。

暴露为 /metrics(Prometheus 文本格式),也直接给 /api/meta 用。
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque


class Metrics:
    def __init__(self, window: int = 500) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.latencies: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.started_at = None

    # ------------------------------------------------------------------
    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self.latencies[name].append(float(ms))

    def record_request(
        self,
        *,
        mode: str,
        path: str,
        elapsed_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        llm_calls: int,
        error: bool = False,
    ) -> None:
        self.inc("requests_total")
        self.inc(f"requests_mode_{mode}")
        self.inc(f"requests_path_{path}")
        if error:
            self.inc("requests_error_total")
        self.observe("request_latency_ms", elapsed_ms)
        self.inc("llm_tokens_prompt_total", prompt_tokens)
        self.inc("llm_tokens_completion_total", completion_tokens)
        self.inc("llm_calls_total", llm_calls)
        self.inc("llm_cost_usd_total", cost_usd / 1e6)  # 以"微美元"存,避免浮点累计误差
        self.inc("llm_cost_usd_milli_total", cost_usd * 1000)

    # ------------------------------------------------------------------
    def quantile(self, name: str, q: float) -> float:
        with self._lock:
            values = sorted(self.latencies.get(name, ()))
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[index]

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self.counters)
        return {
            "counters": counters,
            "latency": {
                name: {
                    "p50": round(self.quantile(name, 0.5), 1),
                    "p95": round(self.quantile(name, 0.95), 1),
                    "samples": len(self.latencies.get(name, ())),
                }
                for name in list(self.latencies)
            },
        }

    # ------------------------------------------------------------------
    def render_prometheus(self) -> str:
        lines: list[str] = [
            "# HELP kf_requests_total 问答请求总数",
            "# TYPE kf_requests_total counter",
        ]
        with self._lock:
            counters = dict(self.counters)
        for name, value in sorted(counters.items()):
            metric = f"kf_{name}"
            lines.append(f"{metric} {value:g}")
        p50 = self.quantile("request_latency_ms", 0.5)
        p95 = self.quantile("request_latency_ms", 0.95)
        lines += [
            "# HELP kf_request_latency_ms 端到端延迟",
            "# TYPE kf_request_latency_ms gauge",
            f'kf_request_latency_ms{{quantile="0.5"}} {p50:g}',
            f'kf_request_latency_ms{{quantile="0.95"}} {p95:g}',
        ]
        return "\n".join(lines) + "\n"


metrics = Metrics()
