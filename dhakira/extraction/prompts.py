"""Bilingual extraction prompts.

English instructions + Arabic content = cheaper tokenization.
English instructions are ~2-3x cheaper to tokenize than Arabic.
"""

FACT_EXTRACTION_SYSTEM = """You are a memory extraction system. Your job is to extract key facts, preferences, and events from Arabic conversations that would be useful to remember for future interactions.

Rules:
- Extract only meaningful, memorable facts (not greetings or small talk)
- Each fact should be a self-contained piece of information in Arabic
- Assign a category: fact, preference, event, or procedure
- Assign a confidence score (0-1) based on how clearly stated the information is
- Return valid JSON only"""

FACT_EXTRACTION_PROMPT = """Extract key facts, preferences, and events from this Arabic conversation.
For each fact, provide:
- "text": The fact in Arabic (concise, self-contained)
- "category": One of "fact", "preference", "event", "procedure"
- "confidence": Float 0-1 based on how clearly stated it is

Return JSON format: {{"facts": [{{"text": "...", "category": "...", "confidence": 0.0}}]}}

Only extract facts that would be useful to remember for future conversations.
Do NOT extract greetings, small talk, or trivial information.

Conversation:
{content}"""

ENTITY_EXTRACTION_SYSTEM = """You are an entity and relationship extraction system for Arabic text. Extract named entities and their relationships as knowledge graph triplets.

Rules:
- Extract entities with their type: person, place, org, concept, event
- Extract relationships between entities as (source, relation, target) triplets
- Entity names should be in Arabic
- Relationship labels should be in Arabic
- Return valid JSON only"""

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this Arabic text.

For entities, provide:
- "name": Entity name in Arabic
- "type": One of "person", "place", "org", "concept", "event"
- "summary": Brief description in Arabic (optional)

For relationships, provide:
- "source": Source entity name
- "target": Target entity name
- "relation": Relationship label in Arabic

Return JSON format:
{{
  "entities": [{{"name": "...", "type": "...", "summary": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Text:
{content}

Previously extracted facts for context:
{facts}"""
