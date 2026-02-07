"""Benchmark configuration and CLI argument parsing."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BenchmarkMode(Enum):
    TOKEN_COUNTING = "token-counting"
    REAL_API = "real-api"


@dataclass
class BenchmarkConfig:
    mode: BenchmarkMode = BenchmarkMode.TOKEN_COUNTING
    llm_model: str = "gpt-4.1-nano"
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    output_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))
    enable_zep: bool = False
    user_id: str = "benchmark_user"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> BenchmarkConfig:
    default_provider = os.getenv("PRIMARY_LLM_PROVIDER", "openai")

    parser = argparse.ArgumentParser(description="Dhakira benchmark: compare against Mem0 and Zep")
    parser.add_argument(
        "--mode",
        choices=["token-counting", "real-api"],
        default="token-counting",
        help="Benchmark mode (default: token-counting)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "bedrock", "anthropic"],
        default=default_provider,
        help=f"LLM provider for Dhakira (default: {default_provider})",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model (default: auto-detected from provider)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for Mem0 (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for reports (default: benchmarks/results)",
    )
    parser.add_argument(
        "--enable-zep",
        action="store_true",
        help="Enable Zep runner (requires ZEP_API_KEY)",
    )

    args = parser.parse_args()

    # Auto-detect model from provider if not specified
    if args.llm_model is None:
        model_defaults = {
            "openai": "gpt-4.1-nano",
            "bedrock": os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "anthropic": "claude-sonnet-4-5-20250929",
        }
        llm_model = model_defaults.get(args.provider, "gpt-4.1-nano")
    else:
        llm_model = args.llm_model

    return BenchmarkConfig(
        mode=BenchmarkMode(args.mode),
        llm_model=llm_model,
        llm_provider=args.provider,
        embedding_model=args.embedding_model,
        output_dir=Path(args.output_dir),
        enable_zep=args.enable_zep,
    )
