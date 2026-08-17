from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from src.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    RuntimeConfig,
)


GENERATION_MODEL = DEFAULT_OPENROUTER_MODEL


def get_generation_llm(runtime_config: RuntimeConfig | None = None):
    config = runtime_config or RuntimeConfig.from_env()
    provider = config.provider_key

    if provider == "groq":
        return ChatGroq(
            model=config.generation_model or DEFAULT_GROQ_MODEL,
            api_key=config.groq_api_key,
            temperature=config.generation_temperature,
        )

    return ChatOpenAI(
        model=config.generation_model or GENERATION_MODEL,
        api_key=config.openrouter_api_key or OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=config.generation_temperature,
    )
