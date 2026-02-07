# Dhakira (ذاكرة) — Arabic-First Agent Memory System

Arabic text costs **2-5x more tokens** than English in major LLMs. Dhakira is an open-source, Arabic-first memory system that reduces Arabic memory costs by **60-80%** through optimizations at every layer.

## Features

- **Arabic-first**: Dialect-aware normalization (MSA, Gulf, Egyptian, Levantine, Maghrebi) with ~18% token reduction
- **Cost-efficient**: Zero-LLM retrieval, bilingual prompts, nano models, local embeddings
- **Graph memory**: Entity and relationship extraction with NetworkX (default) or Neo4j
- **Hybrid search**: Vector + BM25 + graph search with RRF fusion and cross-encoder reranking
- **CPU-only**: All local models run on CPU (~600MB total download)
- **Dual use**: Works as both chatbot/assistant memory and autonomous agent memory

## Quick Start

```bash
pip install dhakira
```

```python
from dhakira import Memory

memory = Memory()

# Add memories from a conversation
memory.add(
    messages=[
        {"role": "user", "content": "اسمي حسام وأحب القهوة العربية"},
        {"role": "assistant", "content": "أهلا حسام! القهوة العربية خيار رائع"},
    ],
    user_id="user_123",
)

# Search (zero LLM calls!)
results = memory.search(query="ما هي المشروبات المفضلة؟", user_id="user_123")
for r in results:
    print(f"[{r.score:.3f}] {r.text}")

# Agent memory
memory.add(
    messages=[{"role": "assistant", "content": "أفضل طريقة لتحليل البيانات العربية هي..."}],
    agent_id="data_analyst",
)

# CRUD
all_memories = memory.get_all(user_id="user_123")
memory.update(memory_id="...", text="يفضل القهوة التركية")
memory.delete(memory_id="...")
```

### Async API

```python
from dhakira import AsyncMemory

memory = AsyncMemory()
await memory.add(messages=[...], user_id="user_123")
results = await memory.search(query="...", user_id="user_123")
```

## Configuration

```python
memory = Memory(config={
    "llm": {"provider": "openai", "model": "gpt-4.1-nano"},
    "embeddings": {
        "provider": "huggingface",
        "model": "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
        "dim": 128,
    },
    "vector_store": {"provider": "qdrant", "path": "./dhakira_data"},
    "graph_store": {"provider": "networkx", "path": "./dhakira_graph.pkl"},
    "arabic": {
        "remove_diacritics": True,
        "detect_dialect": True,
        "normalize_dialect": False,
    },
})
```

### LLM Providers

| Provider | Config | Notes |
|----------|--------|-------|
| OpenAI (default) | `{"provider": "openai", "model": "gpt-4.1-nano"}` | Cheapest capable model |
| AWS Bedrock | `{"provider": "bedrock", "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"}` | Requires `pip install dhakira[bedrock]` |
| Anthropic | `{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}` | Requires `pip install dhakira[anthropic]` |
| Ollama | `{"provider": "ollama", "model": "llama3.2"}` | Fully local |

### Storage Backends

| Backend | Config | Notes |
|---------|--------|-------|
| Qdrant (default) | `{"provider": "qdrant"}` | In-memory or persistent |
| NetworkX (default graph) | `{"provider": "networkx"}` | Pickle persistence |

## Architecture

```
User/Agent Message (Arabic)
       │
       ▼
┌──────────────────────────┐
│  Arabic Preprocessor      │  ← ~18% token reduction
│  (dialect detection +     │
│   normalization)          │
└──────────┬───────────────┘
     ┌─────┴──────┐
     ▼             ▼
 [WRITE PATH]   [READ PATH]         ← Zero LLM calls on read
     │             │
     ▼             ▼
 LLM Extract    Embed Query
 (nano model)      │
     │           ┌─┴──────────────┐
     ▼           │  Parallel Search │
 AUDN Cycle      │  ├─ Vector       │
 (Add/Update/    │  ├─ BM25         │
  Delete/Noop)   │  └─ Graph        │
     │           └────────┬────────┘
     ▼                    ▼
 Store to         RRF Fusion + Rerank
 Vector + Graph        │
                       ▼
                 Return Memories
```

**Key principle**: Invest intelligence at *write time* (LLM calls), make *read path* zero-LLM.

## Benchmark: Dhakira vs Mem0

Real API benchmark on **12 Arabic conversations** across 4 dialects (MSA, Egyptian, Gulf, Levantine) with **14 search queries** and ground truth evaluation. Both systems use the same LLM (`gpt-4.1-nano`). Dhakira's reranker and semantic cache were **disabled** for fairness.

### Retrieval Quality

| Metric | Dhakira | Mem0 | Delta |
|--------|---------|------|-------|
| **Precision** | **0.093** | 0.014 | **6.6x better** |
| **Recall** | 0.571 | 0.869 | Mem0 higher |
| **F1 Score** | **0.160** | 0.028 | **5.7x better** |
| **MRR** | 0.386 | 0.678 | Mem0 higher |

Mem0 returns high recall but extremely low precision (0.014) — it finds relevant memories but drowns them in noise. Dhakira's **5.7x higher F1** means it returns more useful, focused context for agents and chatbots.

### Search Latency

| Metric | Dhakira | Mem0 |
|--------|---------|------|
| Avg add | 12,321ms | 14,358ms |
| **Avg search** | **21ms** | **416ms** |

Dhakira searches are **20x faster** thanks to local hybrid retrieval (vector + BM25 + graph) vs Mem0's API round-trips.

### Cost

| Component | Dhakira | Mem0 |
|-----------|---------|------|
| Embeddings | **$0** (local) | OpenAI API (paid) |
| LLM calls | 55 | Mem0-managed |
| LLM tokens | 39,197 | Not tracked |

Dhakira uses free local embeddings (`Arabic-Triplet-Matryoshka-V2`). At scale, embedding costs dominate — Dhakira's $0 embedding cost becomes a significant advantage.

### LLM Provider Comparison (Dhakira only)

| Metric | GPT-4.1-nano | Claude Sonnet 4.5 (Bedrock) |
|--------|-------------|----------------------------|
| Precision | 0.093 | 0.086 |
| Recall | 0.571 | 0.536 |
| F1 | 0.160 | 0.148 |
| Cost | **$0.008** | $0.515 |
| Avg add | **12,321ms** | 30,270ms |

Both LLMs produce similar Arabic quality. GPT-4.1-nano is **65x cheaper** and **2.5x faster** — the recommended default for Dhakira's extraction pipeline.

> Run the benchmark yourself: `python benchmarks/run_benchmark.py --mode real-api --provider openai`

## Cost Savings

| Optimization | Savings |
|-------------|---------|
| Arabic normalization | ~18% fewer tokens |
| Bilingual prompts | ~20-30% fewer prompt tokens |
| Nano model extraction | ~95% cheaper vs GPT-4 |
| Local embeddings (GATE 135M) | 100% savings vs API |
| Matryoshka 128d embeddings | 6x storage reduction |
| Zero-LLM retrieval | 100% retrieval savings |
| AUDN threshold skip | ~40-60% fewer LLM calls |
| **Combined** | **~70-85% cost reduction** |

## Default Models (All CPU-Compatible)

| Component | Model | Size |
|-----------|-------|------|
| Dialect detection | `CAMeL-Lab/bert-base-arabic-camelbert-da` | ~160M |
| Embeddings | `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` | 135M |
| Reranker | `BAAI/bge-reranker-v2-m3` | ~278M |
| Extraction LLM | `gpt-4.1-nano` (API) | - |

Total local download: ~600MB.

## Getting Started

### Install

```bash
pip install dhakira
```

With optional providers:

```bash
pip install dhakira[bedrock]     # AWS Bedrock (boto3)
pip install dhakira[anthropic]   # Anthropic API
pip install dhakira[all]         # All optional providers
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run the Benchmark

```bash
pip install dhakira[benchmark]

# Token-counting mode (no API keys needed)
python benchmarks/run_benchmark.py --mode token-counting

# Real API mode
python benchmarks/run_benchmark.py --mode real-api --provider openai
python benchmarks/run_benchmark.py --mode real-api --provider bedrock
```

## Development

```bash
git clone https://github.com/hesham-haroun/dhakira.git
cd dhakira
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmark]"
cp .env.example .env  # then add your API keys
pytest
```

## License

Apache 2.0
