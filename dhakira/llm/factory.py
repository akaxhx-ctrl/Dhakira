"""LLM provider factory."""

from __future__ import annotations

from dhakira.config import LLMConfig
from dhakira.llm.base import BaseLLM


def create_llm(config: LLMConfig | None = None) -> BaseLLM:
    """Create an LLM provider instance based on configuration.

    Args:
        config: LLM configuration. Defaults to OpenAI gpt-4.1-nano.

    Returns:
        A BaseLLM implementation.

    Raises:
        ValueError: If the provider is not supported.
    """
    config = config or LLMConfig()

    if config.provider == "openai":
        from dhakira.llm.openai_ import OpenAILLM
        return OpenAILLM(config)
    elif config.provider == "anthropic":
        from dhakira.llm.anthropic_ import AnthropicLLM
        return AnthropicLLM(config)
    elif config.provider == "bedrock":
        from dhakira.llm.bedrock_ import BedrockLLM
        return BedrockLLM(config)
    elif config.provider == "ollama":
        from dhakira.llm.ollama_ import OllamaLLM
        return OllamaLLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}. Supported: openai, anthropic, bedrock, ollama")
