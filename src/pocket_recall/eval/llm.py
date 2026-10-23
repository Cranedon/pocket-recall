"""Thin LLM layer: one call in, text and token usage out.

`ClaudeLLM` talks to the Anthropic API; `EchoLLM` is an offline stand-in so the
pipeline can be smoke-tested without credentials or cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"


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

    def complete(self, prompt: str, *, max_tokens: int) -> Completion: ...


class ClaudeLLM:
    def __init__(self, model: str, effort: str = "medium", fallbacks: bool = True):
        self.name = model
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = anthropic.Anthropic(max_retries=6)

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if self.fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        response = self.client.beta.messages.create(**kwargs)
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


class EchoLLM:
    """Offline fake: answers with the tail of the prompt and judges everything 'no'."""

    name = "echo"

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        approx = len(prompt) // 4
        text = "no" if prompt.rstrip().endswith("Answer yes or no only.") else prompt[-200:]
        return Completion(
            text=text,
            input_tokens=approx,
            output_tokens=len(text) // 4,
            cache_write_tokens=0,
            cache_read_tokens=0,
            model="echo",
            stop_reason="end_turn",
        )
