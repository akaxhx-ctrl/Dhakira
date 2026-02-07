"""Abstract LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0

    def _track_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from an API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The generated text response.
        """

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system: str | None = None,
    ) -> dict:
        """Generate a structured (JSON) response from the LLM.

        Args:
            prompt: The user prompt.
            schema: JSON schema for the expected response.
            system: Optional system prompt.

        Returns:
            Parsed JSON response as a dict.
        """
