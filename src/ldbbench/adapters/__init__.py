"""Database adapter registry."""

from ldbbench.adapters.base import AdapterCapabilities, CheckResult, VectorDBAdapter
from ldbbench.adapters.registry import get_adapter

__all__ = [
    "AdapterCapabilities",
    "CheckResult",
    "VectorDBAdapter",
    "get_adapter",
]

