"""Portable logical filter translation for adapter-native query APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ldbbench.config import ConfigError

PORTABLE_FILTER_KEYS = {"field", "operator", "value"}


def is_portable_filter(value: Mapping[str, Any]) -> bool:
    return PORTABLE_FILTER_KEYS.issubset(value.keys())


def lambdadb_filter(value: Mapping[str, Any]) -> dict[str, Any]:
    if not is_portable_filter(value):
        return dict(value)
    field, operator, filter_value = _portable_parts(value)
    if operator == "eq":
        query = f"{field}:{_format_lambdadb_scalar(filter_value)}"
        return {"queryString": {"query": query}}
    raise ConfigError(f"unsupported LambdaDB filter operator {operator!r}")


def qdrant_filter(value: Mapping[str, Any]) -> dict[str, Any]:
    if not is_portable_filter(value):
        return dict(value)
    field, operator, filter_value = _portable_parts(value)
    if operator == "eq":
        return {"must": [{"key": field, "match": {"value": filter_value}}]}
    raise ConfigError(f"unsupported Qdrant filter operator {operator!r}")


def pinecone_filter(value: Mapping[str, Any]) -> dict[str, Any]:
    if not is_portable_filter(value):
        return dict(value)
    field, operator, filter_value = _portable_parts(value)
    if operator == "eq":
        return {field: {"$eq": filter_value}}
    raise ConfigError(f"unsupported Pinecone filter operator {operator!r}")


def _portable_parts(value: Mapping[str, Any]) -> tuple[str, str, Any]:
    field = value.get("field")
    operator = value.get("operator")
    if not isinstance(field, str) or not field:
        raise ConfigError("portable filter field must be a non-empty string")
    if not isinstance(operator, str) or not operator:
        raise ConfigError("portable filter operator must be a non-empty string")
    return field, operator, value.get("value")


def _format_lambdadb_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    if any(char.isspace() for char in escaped) or ":" in escaped or "/" in escaped:
        return f'"{escaped}"'
    return escaped
