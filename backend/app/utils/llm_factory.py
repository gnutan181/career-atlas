import asyncio
import logging
import time
from functools import lru_cache
from typing import Any

from app.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)

# GPT-OSS supports Groq's JSON Schema response format.  It is more reliable
# than LangChain's default function-calling structured output for these models:
# the latter forces ``tool_choice=required`` for what is only a schema parse.
_JSON_SCHEMA_MODELS = frozenset({
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
})


def _get_rotating_gemini_model(model_name: str, temperature: float, attempt: int):
    keys = settings.google_api_keys
    if not keys:
        raise ValueError("No GOOGLE_API_KEY available for Gemini.")
    key = keys[attempt % len(keys)]
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=temperature,
        max_retries=0,
        timeout=60.0
    )


def invoke_gemini(prompt: Any, model_name: str = "gemini-3.5-flash", temperature: float = 0.2, schema: Any = None, max_retries: int = 5):
    last_exception = None
    for attempt in range(max_retries):
        try:
            model = _get_rotating_gemini_model(
                model_name, temperature, attempt)
            chain = model.with_structured_output(schema) if schema else model

            # If prompt is a Runnable (like ChatPromptTemplate), we pipe it
            if hasattr(prompt, "invoke") and hasattr(prompt, "|"):
                return (prompt | chain).invoke({})
            else:
                return chain.invoke(prompt)
        except Exception as e:
            last_exception = e
            logger.warning(f"Gemini API Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exception


async def ainvoke_gemini(prompt: Any, model_name: str = "gemini-3.5-flash", temperature: float = 0.2, schema: Any = None, max_retries: int = 5):
    last_exception = None
    for attempt in range(max_retries):
        try:
            model = _get_rotating_gemini_model(
                model_name, temperature, attempt)
            chain = model.with_structured_output(schema) if schema else model

            if hasattr(prompt, "ainvoke") and hasattr(prompt, "|"):
                return await (prompt | chain).ainvoke({})
            else:
                return await chain.ainvoke(prompt)
        except Exception as e:
            last_exception = e
            logger.warning(f"Gemini API Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_exception


@lru_cache()
def get_groq_model(model_name: str | None = None, temperature: float = 0.2):
    """
    Returns a configured Groq model instance. Best for fast, responsive generation.
    """
    return ChatGroq(
        model=model_name or settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=temperature
    )


def build_groq_structured_chain(prompt: Any, schema: Any, temperature: float = 0.2, model_name: str | None = None):
    """Compose a `prompt | groq.with_structured_output(...)` chain.

    Wraps the recurring pattern of grabbing a Groq model, binding a structured
    output schema, and piping a prompt template into it.  GPT-OSS models use
    JSON Schema mode so structured output does not masquerade as a forced tool
    call; other supported models retain function-calling mode for compatibility.
    """
    selected_model = model_name or settings.groq_model
    structured_kwargs = (
        {"method": "json_schema", "strict": True}
        if selected_model in _JSON_SCHEMA_MODELS
        else {}
    )
    model = get_groq_model(model_name, temperature)
    return prompt | model.with_structured_output(schema, **structured_kwargs)


@lru_cache()
def get_huggingface_model(repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Returns a configured HuggingFace model instance. Good for auxiliary subtasks.
    """
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.2,
        huggingfacehub_api_token=settings.huggingface_api_key
    )
    return ChatHuggingFace(llm=llm)
