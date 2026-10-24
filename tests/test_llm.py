import json

import httpx
import pytest

from pocket_recall.eval.run import cost
from pocket_recall.facts import Extractor, FactStore
from pocket_recall.llm import EchoLLM, OpenAICompatibleLLM, make_llm, parse_model, served_as_requested


def reply(text="hi", model="m-2026", **usage):
    return httpx.Response(200, json={
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, **usage},
    })


def test_parse_model_splits_on_first_colon_and_defaults_to_anthropic():
    assert parse_model("claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    assert parse_model("OpenAI:gpt-5") == ("openai", "gpt-5")
    assert parse_model("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")


def test_openai_compatible_request_and_completion(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    seen = []

    def handler(request):
        seen.append(request)
        return reply("answer", model="deepseek-chat", prompt_tokens_details={"cached_tokens": 5})

    llm = OpenAICompatibleLLM("deepseek", "deepseek-chat", transport=httpx.MockTransport(handler))
    c = llm.complete("question?", max_tokens=50)

    req = seen[0]
    assert str(req.url) == "https://api.deepseek.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body == {"model": "deepseek-chat", "messages": [{"role": "user", "content": "question?"}],
                    "max_tokens": 50, "temperature": 0.0}
    assert (c.text, c.input_tokens, c.output_tokens, c.cache_read_tokens) == ("answer", 12, 3, 5)
    assert (c.model, c.stop_reason, llm.name, llm.endpoint) == ("deepseek-chat", "stop", "deepseek:deepseek-chat",
                                                                "api.deepseek.com")


def test_custom_provider_needs_base_url(monkeypatch):
    monkeypatch.delenv("MYLAB_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="MYLAB_BASE_URL"):
        OpenAICompatibleLLM("mylab", "m")
    monkeypatch.setenv("MYLAB_BASE_URL", "http://10.0.0.5:8000/v1/")
    monkeypatch.delenv("MYLAB_API_KEY", raising=False)
    seen = []
    llm = OpenAICompatibleLLM("mylab", "m", transport=httpx.MockTransport(lambda r: seen.append(r) or reply()))
    llm.complete("x", max_tokens=5)
    assert str(seen[0].url) == "http://10.0.0.5:8000/v1/chat/completions"
    assert "authorization" not in seen[0].headers  # local servers need no key


def test_retries_transient_errors_then_raises_on_client_errors(monkeypatch):
    monkeypatch.setattr("pocket_recall.llm.time.sleep", lambda s: None)
    statuses = iter([429, 503, 200])
    llm = OpenAICompatibleLLM("openai", "gpt-x", api_key="k", transport=httpx.MockTransport(
        lambda r: reply("ok") if (s := next(statuses)) == 200 else httpx.Response(s, text="busy")))
    assert llm.complete("x", max_tokens=5).text == "ok"

    llm = OpenAICompatibleLLM("openai", "gpt-x", api_key="k",
                              transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        llm.complete("x", max_tokens=5)


def test_switches_to_max_completion_tokens_when_the_model_asks():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, text='{"error": "use max_completion_tokens instead"}')
        return reply()

    llm = OpenAICompatibleLLM("openai", "o-reasoner", api_key="k", transport=httpx.MockTransport(handler))
    llm.complete("x", max_tokens=7)
    llm.complete("y", max_tokens=7)
    assert bodies[1]["max_completion_tokens"] == 7 and "temperature" not in bodies[1]
    assert len(bodies) == 3  # the switch sticks for later calls


def test_make_llm_routes_by_provider():
    assert isinstance(make_llm("echo"), EchoLLM)
    llm = make_llm("ollama:qwen2.5:7b")
    assert isinstance(llm, OpenAICompatibleLLM) and llm.model == "qwen2.5:7b" and llm.endpoint == "localhost"


def test_served_model_and_pricing_across_providers():
    assert served_as_requested("openai:gpt-4o", "gpt-4o-2024-08-06")
    assert served_as_requested("claude-opus-5-5", "claude-opus-5-5")
    assert not served_as_requested("claude-opus-5-5", "claude-sonnet-5-5")
    assert cost("anthropic:claude-opus-5-5", 1_000_000, 0, 0, 0) == cost("claude-opus-5-5", 1_000_000, 0, 0, 0) == 4.0
    assert cost("deepseek:deepseek-chat", 1000, 0, 0, 10) is None
    assert cost("echo", 1000, 0, 0, 10) == 0.0


def test_fact_extractor_runs_on_any_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    raw = json.dumps([{"m": 1, "fact": "The user owns a kayak."}])
    store = FactStore(tmp_path / "facts.db")
    ex = Extractor(store, "ollama:qwen2.5:7b")
    ex.llm.client = httpx.Client(transport=httpx.MockTransport(lambda r: reply(raw, model="qwen2.5:7b")))
    facts = ex.extract("s1", "2026-01-02", ["my paddle snapped on the lake trip"])
    assert [f.text for f in facts] == ["The user owns a kayak."]
    assert store.usage()[1:] == (12, 3)
