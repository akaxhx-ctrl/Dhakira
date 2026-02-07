"""Data models for Dhakira memory system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Dialect(str, Enum):
    MSA = "MSA"
    GULF = "Gulf"
    EGYPTIAN = "Egyptian"
    LEVANTINE = "Levantine"
    MAGHREBI = "Maghrebi"
    UNKNOWN = "Unknown"


class FactCategory(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    EVENT = "event"
    PROCEDURE = "procedure"


class EntityType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "org"
    CONCEPT = "concept"
    EVENT = "event"


class AUDNAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class DialectResult(BaseModel):
    dialect: Dialect
    confidence: float = Field(ge=0.0, le=1.0)


class Chunk(BaseModel):
    text: str
    start_char: int
    end_char: int
    token_count: int | None = None


class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    metadata: dict | None = None


class Fact(BaseModel):
    text: str
    category: FactCategory = FactCategory.FACT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: str | None = None


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    name_normalized: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    summary: str | None = None
    metadata: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    relation: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    is_valid: bool = True
    metadata: dict = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    text_original: str = ""
    embedding: list[float] = Field(default_factory=list)
    category: FactCategory = FactCategory.FACT
    scope: str = "user"  # user | session | agent
    scope_id: str = ""
    dialect: Dialect | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_deleted: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_message_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    record: MemoryRecord
    score: float = 0.0
    source: str = "vector"  # vector | bm25 | graph


class MemoryResult(BaseModel):
    id: str
    text: str
    score: float
    category: FactCategory
    dialect: Dialect | None = None
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class Subgraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class AUDNDecision(BaseModel):
    action: AUDNAction
    target_id: str | None = None  # ID of memory to update/delete
    merged_text: str | None = None  # Merged text for UPDATE actions
    reason: str = ""
