"""OpenAI 兼容网关客户端(httpx 手写,不依赖 openai SDK)。

为什么手写而不是用 SDK:
- 只要 30 行 HTTP,换来零版本锁定;DeepSeek / one-api / Ollama / vLLM 全部兼容;
- 可以精确控制超时、重试、SSE 解析与 usage 采集(成本核算依赖 usage);
- `reasoning_content` 这类厂商扩展字段也能安全忽略。

可靠性:
- 429/5xx/网络错误/超时 -> 指数退避重试(默认 2 次);
- SSE 解析兼容 `data:` 前缀、空行、`[DONE]`、非 JSON 脏数据;
- 流式中断时自动回退为非流式,保证上层永远拿到完整答案。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

import httpx

from app.llm.base import ChatResult, LLMError, StreamEvent, ToolCall, Usage

logger = logging.getLogger(__name__)

_RETRY_STATUS = {403, 408, 409, 425, 429, 500, 502, 503, 504}

# 这些用途输出短、调用频繁,交给快模型;回答生成/自检用主模型保证质量
FAST_PURPOSES = {"rewrite", "hyde", "summarize", "preferences", "agentic_query", "judge"}


class OpenAICompatLLM:
    backend = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        model_pro: str | None = None,
        model_fast: str | None = None,
        timeout: int = 90,
        max_retries: int = 2,
        temperature: float = 0.3,
        proxy: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.model_pro = model_pro or model
        self.model_fast = model_fast or model
        self.temperature = temperature
        self.max_retries = max_retries
        # trust_env=False:环境变量里的 NO_PROXY(如 [::1])会让 httpx 解析代理时报错,
        # 代理统一走 LLM_PROXY 显式配置,行为可预期。
        self._client = httpx.Client(
            trust_env=False,
            proxy=proxy or None,
            timeout=httpx.Timeout(connect=10.0, read=float(timeout), write=30.0, pool=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    # ------------------------------------------------------------------ 基础
    def _payload(
        self,
        messages: list[dict],
        *,
        system: str | None,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
        model: str | None,
        thinking: bool,
        stream: bool,
        purpose: str = "answer",
    ) -> dict:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        # 辅助任务的"快模型"路由:改写/HyDE/摘要这类短输出、高频调用不值得用大模型
        if model:
            used_model = model
        elif thinking:
            used_model = self.model_pro
        elif purpose in FAST_PURPOSES:
            used_model = self.model_fast
        else:
            used_model = self.model
        payload: dict = {
            "model": used_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2**attempt, 8))
            try:
                resp = self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("LLM 请求异常(第 %d 次): %s", attempt + 1, exc)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_error = LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                logger.warning("LLM 限流/服务端错误 %s,重试中", resp.status_code)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"LLM 调用失败 HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError(f"LLM 返回非 JSON: {resp.text[:200]}") from exc
        raise LLMError(f"LLM 调用重试耗尽: {last_error}")

    # ------------------------------------------------------------------ 非流式
    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        model: str | None = None,
        thinking: bool = False,
        purpose: str = "answer",
    ) -> ChatResult:
        started = time.perf_counter()
        payload = self._payload(
            messages,
            system=system,
            tools=tools,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            model=model,
            thinking=thinking,
            stream=False,
            purpose=purpose,
        )
        data = self._post(payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(id=call.get("id", ""), name=fn.get("name", ""), arguments=args or {})
            )
        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )
        result = ChatResult(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", payload["model"]),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=choice.get("finish_reason") or "stop",
        )
        logger.debug(
            "LLM[%s] %s in=%d out=%d %dms",
            purpose,
            result.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            result.latency_ms,
        )
        return result

    def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        thinking: bool = False,
        purpose: str = "answer",
    ) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            purpose=purpose,
        ).text.strip()

    # ------------------------------------------------------------------ 流式
    def stream(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        purpose: str = "answer",
    ) -> Iterator[StreamEvent]:
        payload = self._payload(
            messages,
            system=system,
            tools=None,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            model=None,
            thinking=False,
            stream=True,
            purpose=purpose,
        )
        url = f"{self.base_url}/chat/completions"
        usage = Usage()
        try:
            with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read()[:300]
                    raise LLMError(f"流式调用失败 HTTP {resp.status_code}: {body!r}")
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        usage = Usage(
                            int(u.get("prompt_tokens", 0)),
                            int(u.get("completion_tokens", 0)),
                            int(u.get("total_tokens", 0)),
                        )
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content") or ""
                        if delta:
                            yield StreamEvent(delta=delta)
        except LLMError:
            logger.warning("流式失败,回退非流式")
            result = self.chat(messages, system=system, max_tokens=max_tokens, purpose=purpose)
            yield StreamEvent(delta=result.text, usage=result.usage, done=True)
            return
        except httpx.HTTPError as exc:
            logger.warning("流式网络异常(%s),回退非流式", exc)
            result = self.chat(messages, system=system, max_tokens=max_tokens, purpose=purpose)
            yield StreamEvent(delta=result.text, usage=result.usage, done=True)
            return
        yield StreamEvent(usage=usage, done=True)

    def close(self) -> None:
        self._client.close()
