"""Adapter lookup."""

from __future__ import annotations

from ldbbench.adapters.base import VectorDBAdapter
from ldbbench.adapters.dryrun import DRYRUN_ADAPTERS
from ldbbench.config import ConfigError


def get_adapter(vendor: str) -> VectorDBAdapter:
    try:
        return DRYRUN_ADAPTERS[vendor]
    except KeyError as exc:
        known = ", ".join(sorted(DRYRUN_ADAPTERS))
        message = f"unsupported target vendor {vendor!r}; known: {known}"
        raise ConfigError(message) from exc
