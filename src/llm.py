"""LLM Provider 抽象层 —— 支持在 Claude 和 OpenAI 兼容 API 之间切换。

环境变量配置：
  LLM_PROVIDER   : anthropic | openai  （默认 anthropic）
  LLM_MODEL      : 模型 ID（默认按 provider 给合理值）
  LLM_API_KEY    : API 密钥（anthropic 也可用 ANTHROPIC_API_KEY）
  LLM_BASE_URL   : OpenAI 兼容厂商的自定义 endpoint，可选

常见 OpenAI 兼容厂商：
  DeepSeek   : LLM_BASE_URL=https://api.deepseek.com        LLM_MODEL=deepseek-chat
  Kimi       : LLM_BASE_URL=https://api.moonshot.cn/v1      LLM_MODEL=moonshot-v1-8k
  Qwen       : LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
               LLM_MODEL=qwen-turbo
  智谱 GLM    : LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
               LLM_MODEL=glm-4-flash
  OpenAI     : LLM_BASE_URL 不设，LLM_MODEL=gpt-4o-mini
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    """最小化接口：给定 system + user，返回文本输出，并打印缓存统计。"""

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str: ...


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
        )
        log.info(
            "[anthropic] cache create=%d read=%d input=%d output=%d",
            response.usage.cache_creation_input_tokens or 0,
            response.usage.cache_read_input_tokens or 0,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAICompatClient:
    """适用于 OpenAI、DeepSeek、Kimi、Qwen、智谱 GLM、SiliconFlow 等所有 OpenAI 兼容 API。"""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.model = model
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        if usage:
            log.info(
                "[openai-compat] input=%d output=%d total=%d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return response.choices[0].message.content or ""


def get_client() -> LLMClient:
    """按环境变量装配 LLM 客户端。"""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()

    if provider == "anthropic":
        model = os.getenv("LLM_MODEL", "claude-haiku-4-5")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        log.info("LLM provider=anthropic model=%s", model)
        return AnthropicClient(model=model, api_key=api_key)

    if provider == "openai":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires LLM_API_KEY or OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        log.info("LLM provider=openai model=%s base_url=%s", model, base_url or "<default>")
        return OpenAICompatClient(model=model, api_key=api_key, base_url=base_url)

    raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')")
