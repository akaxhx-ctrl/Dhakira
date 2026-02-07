"""AUDN decision prompts."""

AUDN_SYSTEM = """You are a memory consolidation system. Given a new fact and similar existing memories, decide the correct action.

Actions:
- ADD: The new fact is genuinely new information not captured by existing memories.
- UPDATE: The new fact augments or refines an existing memory. Provide merged text.
- DELETE: The new fact contradicts an existing memory (the old one is now wrong).
- NOOP: The new fact is already captured by existing memories. No action needed.

Return valid JSON only."""

AUDN_PROMPT = """New fact:
{new_fact}

Similar existing memories:
{existing_memories}

Decide the action for this new fact. If UPDATE, provide the merged text that combines both pieces of information. If DELETE, specify which existing memory ID should be marked as outdated.

Return JSON:
{{
  "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "target_id": "ID of existing memory to update/delete (null if ADD/NOOP)",
  "merged_text": "merged text in Arabic (only for UPDATE, null otherwise)",
  "reason": "brief explanation"
}}"""
