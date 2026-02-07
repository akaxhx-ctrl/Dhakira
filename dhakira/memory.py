"""Main Memory class — synchronous public API for Dhakira."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from dhakira.async_memory import AsyncMemory
from dhakira.config import DhakiraConfig
from dhakira.models import MemoryResult

logger = logging.getLogger(__name__)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for sync API."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class Memory:
    """Synchronous Dhakira memory interface.

    Wraps AsyncMemory to provide a sync API. For async usage,
    use AsyncMemory directly.

    Usage:
        memory = Memory()
        memory.add(
            messages=[{"role": "user", "content": "اسمي أحمد"}],
            user_id="user_123",
        )
        results = memory.search(query="ما اسمه؟", user_id="user_123")
    """

    def __init__(self, config: dict[str, Any] | DhakiraConfig | None = None):
        if isinstance(config, dict):
            config = DhakiraConfig(**config)
        self._config = config or DhakiraConfig()
        self._async_memory = AsyncMemory(self._config)

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        loop = _get_or_create_event_loop()
        if loop.is_running():
            # If we're inside an existing event loop (e.g., Jupyter),
            # create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[str]:
        """Add memories from a conversation.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            user_id: User identifier (sets scope="user").
            session_id: Session identifier (sets scope="session").
            agent_id: Agent identifier (sets scope="agent").
            metadata: Additional metadata for extracted memories.

        Returns:
            List of created memory IDs.
        """
        return self._run(
            self._async_memory.add(
                messages=messages,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                metadata=metadata,
            )
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        """Search memories (zero LLM calls).

        Args:
            query: Search query in Arabic.
            user_id: Filter by user scope.
            session_id: Filter by session scope.
            agent_id: Filter by agent scope.
            limit: Maximum number of results.

        Returns:
            List of MemoryResult objects.
        """
        return self._run(
            self._async_memory.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                limit=limit,
            )
        )

    def get_all(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryResult]:
        """Get all memories for a scope.

        Args:
            user_id: Filter by user scope.
            session_id: Filter by session scope.
            agent_id: Filter by agent scope.

        Returns:
            List of MemoryResult objects.
        """
        return self._run(
            self._async_memory.get_all(
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
            )
        )

    def update(self, memory_id: str, text: str) -> None:
        """Update a memory's text.

        Args:
            memory_id: The memory ID to update.
            text: New text for the memory.
        """
        self._run(self._async_memory.update(memory_id=memory_id, text=text))

    def delete(self, memory_id: str) -> None:
        """Soft-delete a memory.

        Args:
            memory_id: The memory ID to delete.
        """
        self._run(self._async_memory.delete(memory_id=memory_id))
