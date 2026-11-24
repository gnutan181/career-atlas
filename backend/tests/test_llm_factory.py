"""Unit tests for app.utils.llm_factory Gemini key rotation + retry logic."""
import pytest

from app.utils import llm_factory


class _FakeChat:
    """Stand-in for ChatGoogleGenerativeAI; records the api key it was built with."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_rotating_model_cycles_through_keys(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "google_api_keys", ["a", "b", "c"])
    monkeypatch.setattr(llm_factory, "ChatGoogleGenerativeAI", _FakeChat)

    keys = [llm_factory._get_rotating_gemini_model("m", 0.2, attempt).kwargs["google_api_key"] for attempt in range(4)]
    assert keys == ["a", "b", "c", "a"]  # attempt % len(keys)


def test_rotating_model_raises_without_keys(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "google_api_keys", [])
    with pytest.raises(ValueError, match="No GOOGLE_API_KEY"):
        llm_factory._get_rotating_gemini_model("m", 0.2, 0)


def test_invoke_gemini_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Flaky:
        def invoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("rate limited")
            return "ok"

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Flaky())
    monkeypatch.setattr(llm_factory.time, "sleep", lambda s: None)  # no real backoff

    assert llm_factory.invoke_gemini("prompt", max_retries=5) == "ok"
    assert calls["n"] == 3


def test_invoke_gemini_raises_last_exception_after_exhausting_retries(monkeypatch):
    class _Broken:
        def invoke(self, prompt):
            raise RuntimeError("always fails")

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Broken())
    monkeypatch.setattr(llm_factory.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="always fails"):
        llm_factory.invoke_gemini("prompt", max_retries=3)


async def test_ainvoke_gemini_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Flaky:
        async def ainvoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return "async-ok"

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Flaky())

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(llm_factory.asyncio, "sleep", _no_sleep)

    assert await llm_factory.ainvoke_gemini("prompt", max_retries=3) == "async-ok"
    assert calls["n"] == 2


class _FakeRunnableSequence:
    def __init__(self, steps):
        self.steps = steps

    def invoke(self, input_val):
        res = input_val
        for step in self.steps:
            if hasattr(step, "invoke"):
                res = step.invoke(res)
            else:
                res = step(res)
        return res

    async def ainvoke(self, input_val):
        res = input_val
        for step in self.steps:
            if hasattr(step, "ainvoke"):
                res = await step.ainvoke(res)
            elif hasattr(step, "invoke"):
                res = step.invoke(res)
            else:
                res = step(res)
        return res

class _FakeRunnablePrompt:
    def __init__(self, invoke_res=None):
        self.invoke_res = invoke_res

    def invoke(self, *args, **kwargs):
        return self.invoke_res

    async def ainvoke(self, *args, **kwargs):
        return self.invoke_res

    def __or__(self, other):
        return _FakeRunnableSequence([self, other])

    def __getattr__(self, name):
        if name == "|":
            return True
        raise AttributeError(name)

def test_invoke_gemini_runnable_with_schema_retries(monkeypatch):
    calls = {"n": 0}

    class _FlakyChain:
        def invoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("rate limited with schema")
            return {"structured": "ok"}

    class _Model:
        def with_structured_output(self, schema):
            assert schema == {"type": "object"}
            return _FlakyChain()

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Model())
    monkeypatch.setattr(llm_factory.time, "sleep", lambda s: None)

    prompt = _FakeRunnablePrompt("dummy prompt")

    res = llm_factory.invoke_gemini(
        prompt=prompt,
        schema={"type": "object"},
        max_retries=5
    )
    assert res == {"structured": "ok"}
    assert calls["n"] == 3


async def test_ainvoke_gemini_runnable_with_schema_retries(monkeypatch):
    calls = {"n": 0}

    class _FlakyChain:
        async def ainvoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient async error")
            return {"async_structured": "ok"}

    class _Model:
        def with_structured_output(self, schema):
            assert schema == {"type": "object"}
            return _FlakyChain()

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Model())

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(llm_factory.asyncio, "sleep", _no_sleep)

    prompt = _FakeRunnablePrompt("dummy prompt")

    res = await llm_factory.ainvoke_gemini(
        prompt=prompt,
        schema={"type": "object"},
        max_retries=3
    )
    assert res == {"async_structured": "ok"}
    assert calls["n"] == 2

async def test_ainvoke_gemini_raises_last_exception_after_exhausting_retries(monkeypatch):
    class _Broken:
        async def ainvoke(self, prompt):
            raise RuntimeError("async always fails")

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Broken())

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(llm_factory.asyncio, "sleep", _no_sleep)

    with pytest.raises(RuntimeError, match="async always fails"):
        await llm_factory.ainvoke_gemini("prompt", max_retries=3)

def test_build_groq_structured_chain(monkeypatch):
    calls = []

    class _MockStructuredModel:
        def __init__(self, schema):
            self.schema = schema

    class _MockModel:
        def __init__(self, model_name, temperature):
            self.model_name = model_name
            self.temperature = temperature

        def with_structured_output(self, schema):
            calls.append(("with_structured_output", schema))
            return _MockStructuredModel(schema)

    def _mock_get_groq_model(model_name, temperature):
        calls.append(("get_groq_model", model_name, temperature))
        return _MockModel(model_name, temperature)

    monkeypatch.setattr(llm_factory, "get_groq_model", _mock_get_groq_model)

    class _MockPrompt:
        def __or__(self, other):
            calls.append(("__or__", other))
            return "chain"

    prompt = _MockPrompt()
    schema = {"type": "object"}

    chain = llm_factory.build_groq_structured_chain(prompt, schema, temperature=0.5, model_name="test-model")

    assert chain == "chain"
    assert len(calls) == 3
    assert calls[0] == ("get_groq_model", "test-model", 0.5)
    assert calls[1] == ("with_structured_output", schema)
    assert calls[2][0] == "__or__"
    assert isinstance(calls[2][1], _MockStructuredModel)


def test_build_groq_structured_chain_uses_json_schema_for_gpt_oss(monkeypatch):
    calls = []

    class _MockStructuredModel:
        pass

    class _MockModel:
        def with_structured_output(self, schema, **kwargs):
            calls.append((schema, kwargs))
            return _MockStructuredModel()

    monkeypatch.setattr(llm_factory, "get_groq_model", lambda *args: _MockModel())
    monkeypatch.setattr(llm_factory.settings, "groq_model", "openai/gpt-oss-120b")

    class _MockPrompt:
        def __or__(self, other):
            return other

    llm_factory.build_groq_structured_chain(_MockPrompt(), {"type": "object"})

    assert calls == [
        ({"type": "object"}, {"method": "json_schema", "strict": False})
    ]


# ── LLM_PROVIDER / Azure AI Foundry ──────────────────────────────────────────

class _FakeAzureChat:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeAzureChat.instances.append(self)

    def with_structured_output(self, schema, **_kwargs):
        from langchain_core.runnables import RunnableLambda

        self.schema = schema
        return RunnableLambda(lambda _value: f"azure-structured:{schema.__name__}")

    def invoke(self, prompt):
        return f"azure:{prompt}"

    async def ainvoke(self, prompt):
        return f"azure-async:{prompt}"


@pytest.fixture
def azure_settings(monkeypatch):
    llm_factory.get_azure_model.cache_clear()
    _FakeAzureChat.instances = []
    monkeypatch.setattr(llm_factory, "ChatOpenAI", _FakeAzureChat)
    monkeypatch.setattr(llm_factory.settings, "llm_provider", "azure")
    monkeypatch.setattr(llm_factory.settings, "azure_ai_endpoint", "https://res.services.ai.azure.com/openai/v1/")
    monkeypatch.setattr(llm_factory.settings, "azure_ai_api_key", "test-key")
    monkeypatch.setattr(llm_factory.settings, "azure_ai_deployment", "my-deployment")
    monkeypatch.setattr(llm_factory.settings, "azure_ai_timeout_seconds", 30.0)
    yield
    llm_factory.get_azure_model.cache_clear()


def test_azure_provider_builds_foundry_client_from_settings(azure_settings):
    model = llm_factory.get_azure_model(0.1)
    assert model.kwargs == {
        "model": "my-deployment",
        "base_url": "https://res.services.ai.azure.com/openai/v1",
        "api_key": "test-key",
        "temperature": 0.1,
        "timeout": 30.0,
        "max_retries": 1,
    }


def test_azure_provider_routes_invoke_and_structured_chains(azure_settings, monkeypatch):
    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: pytest.fail("Gemini used"))
    monkeypatch.setattr(llm_factory, "get_groq_model", lambda *a, **k: pytest.fail("Groq used"))
    assert llm_factory.invoke_gemini("hi") == "azure:hi"

    import asyncio
    assert asyncio.run(llm_factory.ainvoke_gemini("hi")) == "azure-async:hi"

    from langchain_core.prompts import PromptTemplate
    chain = llm_factory.build_groq_structured_chain(PromptTemplate.from_template("x {a}"), dict)
    assert chain.invoke({"a": 1}) == "azure-structured:dict"


def test_azure_provider_requires_configuration(azure_settings, monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "azure_ai_api_key", "")
    with pytest.raises(ValueError, match="AZURE_AI_API_KEY"):
        llm_factory.invoke_gemini("hi")


def test_default_provider_is_unchanged(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "llm_provider", "default")
    monkeypatch.setattr(llm_factory, "get_azure_model", lambda *a, **k: pytest.fail("Azure used"))

    class _Gemini:
        def invoke(self, prompt):
            return "gemini"

    monkeypatch.setattr(llm_factory, "_get_rotating_gemini_model", lambda *a, **k: _Gemini())
    assert llm_factory.invoke_gemini("p") == "gemini"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "llm_provider", "openai-direct")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm_factory.invoke_gemini("p")
