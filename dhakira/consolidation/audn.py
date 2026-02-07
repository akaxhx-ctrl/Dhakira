"""AUDN cycle: Add/Update/Delete/Noop memory consolidation."""

from __future__ import annotations

import logging

from dhakira.config import ConsolidationConfig
from dhakira.consolidation.prompts import AUDN_PROMPT, AUDN_SYSTEM
from dhakira.llm.base import BaseLLM
from dhakira.models import AUDNAction, AUDNDecision, Fact, SearchResult
from dhakira.storage.base import VectorStore

logger = logging.getLogger(__name__)


class AUDNCycle:
    """Add/Update/Delete/Noop memory consolidation cycle.

    Decides whether a new fact should be added as a new memory, merged
    with an existing one, or if an existing memory should be invalidated.

    Cost optimization: If max similarity < threshold (default 0.5),
    the fact is clearly novel and we skip the LLM call entirely (~40-60%
    of new facts).
    """

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: VectorStore,
        config: ConsolidationConfig | None = None,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.config = config or ConsolidationConfig()

    async def process(
        self,
        fact: Fact,
        embedding: list[float],
        scope: str = "user",
        scope_id: str = "",
    ) -> AUDNDecision:
        """Process a fact through the AUDN cycle.

        Args:
            fact: The new fact to process.
            embedding: Embedding vector for the fact.
            scope: Memory scope (user/session/agent).
            scope_id: Scope identifier.

        Returns:
            AUDNDecision with the determined action.
        """
        # Search for similar existing memories
        filters = {"scope": scope, "scope_id": scope_id}
        similar = await self.vector_store.search(
            embedding=embedding,
            limit=self.config.top_k_similar,
            filters=filters,
        )

        # If no similar memories or max similarity below threshold → ADD (skip LLM)
        if not similar:
            return AUDNDecision(
                action=AUDNAction.ADD,
                reason="No similar memories found",
            )

        max_similarity = max(r.score for r in similar)
        if max_similarity < self.config.similarity_threshold:
            return AUDNDecision(
                action=AUDNAction.ADD,
                reason=f"Max similarity {max_similarity:.3f} below threshold {self.config.similarity_threshold}",
            )

        # Similar memories found → ask LLM to decide
        return await self._llm_decide(fact, similar)

    async def _llm_decide(self, fact: Fact, similar: list[SearchResult]) -> AUDNDecision:
        """Use LLM to decide the AUDN action."""
        # Format existing memories for the prompt
        memories_text = "\n".join(
            f"- ID: {r.record.id} | Text: {r.record.text} | Similarity: {r.score:.3f}"
            for r in similar
        )

        prompt = AUDN_PROMPT.format(
            new_fact=fact.text,
            existing_memories=memories_text,
        )

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema={"type": "object"},
                system=AUDN_SYSTEM,
            )
        except Exception as e:
            logger.error("AUDN LLM decision failed: %s. Defaulting to ADD.", e)
            return AUDNDecision(action=AUDNAction.ADD, reason=f"LLM error: {e}")

        return self._parse_decision(result)

    def _parse_decision(self, result: dict) -> AUDNDecision:
        """Parse LLM response into AUDNDecision."""
        action_str = result.get("action", "ADD").upper()
        try:
            action = AUDNAction(action_str)
        except ValueError:
            action = AUDNAction.ADD

        return AUDNDecision(
            action=action,
            target_id=result.get("target_id"),
            merged_text=result.get("merged_text"),
            reason=result.get("reason", ""),
        )
