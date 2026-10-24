"""The evaluation harness uses the shared model layer; see pocket_recall.llm."""

from ..llm import FALLBACK_BETA, LLM, AnthropicLLM, ClaudeLLM, Completion, EchoLLM, OpenAICompatibleLLM, make_llm

__all__ = ["FALLBACK_BETA", "LLM", "AnthropicLLM", "ClaudeLLM", "Completion", "EchoLLM", "OpenAICompatibleLLM",
           "make_llm"]
