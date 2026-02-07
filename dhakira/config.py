"""Configuration models for Dhakira."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArabicConfig(BaseModel):
    remove_diacritics: bool = True
    preserve_alif_variants: bool = False
    normalize_taa_marbuta: bool = True
    normalize_yaa: bool = True
    remove_tatweel: bool = True
    normalize_numerals: bool = True
    normalize_punctuation: bool = True
    detect_dialect: bool = True
    normalize_dialect: bool = False
    dialect_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-da"


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4.1-nano"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingsConfig(BaseModel):
    provider: str = "huggingface"
    model: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
    dim: int = 128
    device: str = "cpu"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    path: str | None = None  # None = in-memory
    collection_name: str = "dhakira_memories"
    host: str = "localhost"
    port: int = 6333


class GraphStoreConfig(BaseModel):
    provider: str = "networkx"
    path: str | None = None  # Persistence path for NetworkX pickle
    uri: str | None = None  # For Neo4j
    username: str | None = None
    password: str | None = None


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    top_k: int = 10


class BM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


class RetrievalConfig(BaseModel):
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    rrf_k: int = 60  # RRF constant
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    graph_weight: float = 1.0


class CacheConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.95
    max_size: int = 1000
    ttl_seconds: int = 3600


class ConsolidationConfig(BaseModel):
    similarity_threshold: float = 0.5  # Below this, always ADD (skip LLM)
    top_k_similar: int = 5


class ChunkerConfig(BaseModel):
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_ratio: float = 0.1


class DhakiraConfig(BaseModel):
    arabic: ArabicConfig = Field(default_factory=ArabicConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
