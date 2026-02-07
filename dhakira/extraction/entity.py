"""Entity and relationship extraction for graph memory."""

from __future__ import annotations

import logging

from dhakira.arabic.normalizer import ArabicNormalizer
from dhakira.extraction.prompts import ENTITY_EXTRACTION_PROMPT, ENTITY_EXTRACTION_SYSTEM
from dhakira.llm.base import BaseLLM
from dhakira.models import Entity, EntityType, Fact, Relationship

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relationships from text for graph memory.

    Uses the nano LLM to identify entities and their relationships,
    producing knowledge graph triplets.
    """

    def __init__(self, llm: BaseLLM, normalizer: ArabicNormalizer | None = None):
        self.llm = llm
        self.normalizer = normalizer or ArabicNormalizer()

    async def extract(
        self,
        text: str,
        facts: list[Fact] | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from text.

        Args:
            text: Arabic text to extract from.
            facts: Previously extracted facts for additional context.

        Returns:
            Tuple of (entities, relationships).
        """
        normalized = self.normalizer.normalize(text)
        facts_text = "\n".join(f"- {f.text}" for f in (facts or []))

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            content=normalized,
            facts=facts_text or "None",
        )

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema={"type": "object"},
                system=ENTITY_EXTRACTION_SYSTEM,
            )
        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return [], []

        entities = self._parse_entities(result)
        relationships = self._parse_relationships(result, entities)

        return entities, relationships

    def _parse_entities(self, result: dict) -> list[Entity]:
        """Parse LLM response into Entity objects."""
        entities = []
        raw_entities = result.get("entities", [])

        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue

            name = raw.get("name", "").strip()
            if not name:
                continue

            type_str = raw.get("type", "concept")
            try:
                entity_type = EntityType(type_str)
            except ValueError:
                entity_type = EntityType.CONCEPT

            entity = Entity(
                name=name,
                name_normalized=self.normalizer.normalize_for_embedding(name),
                entity_type=entity_type,
                summary=raw.get("summary"),
            )
            entities.append(entity)

        return entities

    def _parse_relationships(
        self,
        result: dict,
        entities: list[Entity],
    ) -> list[Relationship]:
        """Parse LLM response into Relationship objects."""
        relationships = []
        raw_rels = result.get("relationships", [])

        # Build name → entity ID mapping
        name_to_id: dict[str, str] = {}
        for entity in entities:
            name_to_id[entity.name] = entity.id
            name_to_id[entity.name_normalized] = entity.id

        for raw in raw_rels:
            if not isinstance(raw, dict):
                continue

            source_name = raw.get("source", "").strip()
            target_name = raw.get("target", "").strip()
            relation = raw.get("relation", "").strip()

            if not (source_name and target_name and relation):
                continue

            source_id = name_to_id.get(source_name)
            target_id = name_to_id.get(target_name)

            if not source_id or not target_id:
                # Create entities for unmatched names
                if not source_id:
                    source_entity = Entity(
                        name=source_name,
                        name_normalized=self.normalizer.normalize_for_embedding(source_name),
                    )
                    entities.append(source_entity)
                    name_to_id[source_name] = source_entity.id
                    source_id = source_entity.id

                if not target_id:
                    target_entity = Entity(
                        name=target_name,
                        name_normalized=self.normalizer.normalize_for_embedding(target_name),
                    )
                    entities.append(target_entity)
                    name_to_id[target_name] = target_entity.id
                    target_id = target_entity.id

            relationships.append(Relationship(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
            ))

        return relationships
