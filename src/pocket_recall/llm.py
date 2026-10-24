"""Model-switchable LLM layer: one prompt in, text and token usage out.

pocket-recall's memory itself calls no model. A model is used only by the optional fact
extraction and by the evaluation harness, and both go through this layer, so any provider works:

    claude-opus-5-5              Anthropic (the default when there is no provider prefix)
    anthropic:claude-sonnet-5-5  Anthropic, explicitly
    openai:gpt-5                 any OpenAI-compatible Chat Completions endpoint:
    deepseek:deepseek-chat         openai, deepseek, qwen, moonshot, openrouter, ollama are
    ollama:qwen2.5:7b              built in; any other name works once <NAME>_BASE_URL is set
    echo                         offline stand-in, no network and no cost

An OpenAI-compatible provider reads <NAME>_BASE_URL and <NAME>_API_KEY from the environment
(OPENAI_API_KEY, DEEPSEEK_BASE_URL, ...). Anthropic reads ANTHROPIC_API_KEY and, for a relay,
ANTHROPIC_BASE_URL. Keys are only sent to the endpoint they belong to and never logged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# OpenAI-compatible endpoints known by name; <NAME>_BASE_URL overrides any of them.
OPENAI_COMPATIBLE = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
ANTHROPIC = ("anthropic", "claude")


@dataclass
class Completion:
    text: str
    input_tokens: int  # all prompt tokens: uncached + cache writes + cache reads
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    model: str  # the model that actually answered (differs from the requested one after a fallback)
    stop_reason: str | None


class LLM(Protocol):
    name: str
    endpoint: str  # host the requests go to, recorded with results

    def complete(self, prompt: str, *, max_tokens: int) -> Completion: ...


def parse_model(spec: str) -> tuple[str, str]:
    """"provider:model" -> (provider, model); no prefix means Anthropic. Only the first colon
    splits, so "ollama:qwen2.5:7b" is provider "ollama", model "qwen2.5:7b"."""
    provider, sep, model = spec.partition(":")
    return (provider.lower(), model) if sep else ("anthropic", spec)


class AnthropicLLM:
    def __init__(self, model: str, effort: str | None = "medium", fallbacks: bool = True, **request: object):
        """effort=None leaves the model's default (for models without the effort setting);
        `request` adds parameters to each call, e.g. thinking={"type": "disabled"}."""
        try:
            import anthropic
        except ImportError:
            raise SystemExit("Anthropic models need the SDK: pip install 'pocket-recall[facts]'") from None
        self.name = model
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.request = request
        self.client = anthropic.Anthropic(max_retries=6)
        self.endpoint = self.client.base_url.host

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt}], **self.request)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
            response = self.client.beta.messages.create(**kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        # Some relays turn on prompt caching themselves, which moves most of the prompt
        # out of input_tokens; add the cache fields back to get the real prompt size.
        cache_write = u.cache_creation_input_tokens or 0
        cache_read = u.cache_read_input_tokens or 0
        return Completion(
            text=text,
            input_tokens=u.input_tokens + cache_write + cache_read,
            output_tokens=u.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            model=response.model,
            stop_reason=response.stop_reason,
        )


def served_as_requested(spec: str, served: str) -> bool:
    """Whether the model that answered is the one asked for; providers often answer with a dated
    version ("gpt-4o" -> "gpt-4o-2024-08-06"), which counts as the same model."""
    return served.startswith(parse_model(spec)[1]) or spec == served


ClaudeLLM = AnthropicLLM  # the name the harness used before providers existed


class OpenAICompatibleLLM:
    """Any endpoint that speaks the OpenAI Chat Completions format."""

    RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, provider: str, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 temperature: float | None = 0.0, timeout: float = 180.0, max_retries: int = 5, transport=None,
                 **request: object):
        """`request` adds fields to each request body, e.g. reasoning_effort="low"."""
        import httpx  # always installed: the MCP SDK depends on it

        env = provider.upper().replace("-", "_")
        url = base_url or os.environ.get(f"{env}_BASE_URL") or OPENAI_COMPATIBLE.get(provider)
        if not url:
            raise SystemExit(f"Unknown provider {provider!r}: set {env}_BASE_URL to its OpenAI-compatible endpoint.")
        self.name = f"{provider}:{model}"
        self.provider = provider
        self.model = model
        self.url = url.rstrip("/") + "/chat/completions"
        self.endpoint = httpx.URL(self.url).host
        self._key = api_key if api_key is not None else os.environ.get(f"{env}_API_KEY", "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.request = request
        self._token_param = "max_tokens"  # newer reasoning models want max_completion_tokens instead
        self.client = httpx.Client(timeout=timeout, transport=transport)

    def _payload(self, prompt: str, max_tokens: int) -> dict:
        payload: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                         self._token_param: max_tokens, **self.request}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        for attempt in range(self.max_retries + 1):
            r = self.client.post(self.url, json=self._payload(prompt, max_tokens), headers=headers)
            if r.status_code == 400 and "max_completion_tokens" in r.text and self._token_param == "max_tokens":
                # reasoning models reject max_tokens and a fixed temperature; switch once and retry
                self._token_param, self.temperature = "max_completion_tokens", None
                continue
            if r.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
            break
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return Completion(
            text=(choice.get("message") or {}).get("content") or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_write_tokens=0,
            cache_read_tokens=cached,
            model=data.get("model") or self.model,
            stop_reason=choice.get("finish_reason"),
        )


class EchoLLM:
    """Offline fake: answers with the tail of the prompt and judges everything 'no'."""

    name = "echo"
    endpoint = "offline"

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        approx = len(prompt) // 4
        text = "no" if prompt.rstrip().endswith("Answer yes or no only.") else prompt[-200:]
        return Completion(text=text, input_tokens=approx, output_tokens=len(text) // 4, cache_write_tokens=0,
                          cache_read_tokens=0, model="echo", stop_reason="end_turn")


def make_llm(spec: str, *, effort: str | None = "medium", fallbacks: bool = True, **request: object) -> LLM:
    """An LLM for a model spec such as "claude-opus-5-5", "openai:gpt-5" or "echo".
    effort and fallbacks are Anthropic settings; other providers ignore them."""
    if spec == "echo":
        return EchoLLM()
    provider, model = parse_model(spec)
    if provider in ANTHROPIC:
        return AnthropicLLM(model, effort=effort, fallbacks=fallbacks, **request)
    return OpenAICompatibleLLM(provider, model, **request)
