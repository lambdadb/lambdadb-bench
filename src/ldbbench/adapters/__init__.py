"""Database adapter registry."""

from ldbbench.adapters.base import (
    AdapterCapabilities,
    CheckResult,
    PrepareResult,
    QueryMatch,
    QueryResult,
    UpsertResult,
    VectorDBAdapter,
    VectorRecord,
)
from ldbbench.adapters.lambdadb import LambdaDBAdapter
from ldbbench.adapters.registry import get_adapter

__all__ = [
    "AdapterCapabilities",
    "CheckResult",
    "LambdaDBAdapter",
    "PrepareResult",
    "QueryMatch",
    "QueryResult",
    "UpsertResult",
    "VectorDBAdapter",
    "VectorRecord",
    "get_adapter",
]
