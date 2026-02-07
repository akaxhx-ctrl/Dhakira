"""GraphStore factory."""

from __future__ import annotations

from dhakira.config import GraphStoreConfig
from dhakira.storage.base import GraphStore


def create_graph_store(config: GraphStoreConfig | None = None) -> GraphStore:
    """Create a graph store instance based on configuration.

    Args:
        config: Graph store configuration. Defaults to NetworkX in-memory.

    Returns:
        A GraphStore implementation.

    Raises:
        ValueError: If the provider is not supported.
    """
    config = config or GraphStoreConfig()

    if config.provider == "networkx":
        from dhakira.storage.graph.networkx_ import NetworkXGraphStore
        return NetworkXGraphStore(config)
    else:
        raise ValueError(
            f"Unsupported graph store provider: {config.provider}. Supported: networkx"
        )
