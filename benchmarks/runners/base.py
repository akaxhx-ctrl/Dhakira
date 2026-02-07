"""Abstract base runner for benchmark systems."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmarks.config import BenchmarkConfig
from benchmarks.dataset import Conversation
from benchmarks.metrics import BenchmarkResult


class BaseRunner(ABC):
    """Abstract benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the memory system."""

    @abstractmethod
    async def add_conversations(self, conversations: list[Conversation]) -> None:
        """Add all conversations to the memory system."""

    @abstractmethod
    async def search(self, query: str, user_id: str) -> list[str]:
        """Search for memories. Returns list of result texts."""

    @abstractmethod
    def get_result(self) -> BenchmarkResult:
        """Get the collected benchmark result."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up resources."""
