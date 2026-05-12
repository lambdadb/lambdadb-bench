"""Adapter lookup."""

from __future__ import annotations

from ldbbench.adapters.base import VectorDBAdapter
from ldbbench.adapters.dryrun import DRYRUN_ADAPTERS
from ldbbench.adapters.lambdadb import LambdaDBAdapter
from ldbbench.adapters.pinecone import PineconeAdapter
from ldbbench.adapters.qdrant import QdrantAdapter
from ldbbench.config import ConfigError

REAL_ADAPTERS = {
    "lambdadb": LambdaDBAdapter(),
    "pinecone": PineconeAdapter(),
    "qdrant": QdrantAdapter(),
}

def get_adapter(vendor: str, *, dry_run: bool = False) -> VectorDBAdapter:
    if not dry_run and vendor in REAL_ADAPTERS:
        return REAL_ADAPTERS[vendor]
    try:
        return DRYRUN_ADAPTERS[vendor]
    except KeyError as exc:
        known = ", ".join(sorted(set(DRYRUN_ADAPTERS) | set(REAL_ADAPTERS)))
        message = f"unsupported target vendor {vendor!r}; known: {known}"
        raise ConfigError(message) from exc
