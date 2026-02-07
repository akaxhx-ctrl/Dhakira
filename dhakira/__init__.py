"""Dhakira (ذاكرة) — Arabic-first agent memory system.

Reduce Arabic memory costs by 60-80% through Arabic-specific optimizations
at every layer: preprocessing, extraction, embedding, and retrieval.

Usage:
    from dhakira import Memory, AsyncMemory

    # Sync API
    memory = Memory()
    memory.add(
        messages=[{"role": "user", "content": "اسمي أحمد"}],
        user_id="user_123",
    )
    results = memory.search(query="ما اسمه؟", user_id="user_123")

    # Async API
    memory = AsyncMemory()
    await memory.add(messages=[...], user_id="user_123")
    results = await memory.search(query="...", user_id="user_123")
"""

from dhakira.async_memory import AsyncMemory
from dhakira.memory import Memory

__version__ = "0.1.0"
__all__ = ["Memory", "AsyncMemory"]
