"""Real LambdaDB adapter using the official Python SDK."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock, get_ident
from typing import Any

from ldbbench.adapters.base import (
    AdapterCapabilities,
    CheckResult,
    PrepareResult,
    QueryMatch,
    QueryResult,
    UpsertResult,
    VectorRecord,
)
from ldbbench.config import ConfigError, TargetConfig

DEFAULT_VECTOR_FIELD = "vector"
DEFAULT_DELETE_WAIT_TIMEOUT_SECONDS = 60.0
DEFAULT_DELETE_WAIT_POLL_SECONDS = 1.0
DEFAULT_CREATE_WAIT_TIMEOUT_SECONDS = 300.0
DEFAULT_CREATE_WAIT_POLL_SECONDS = 1.0
ACTIVE_COLLECTION_STATUS = "ACTIVE"
SUPPORTED_METRIC_MAP = {
    "cosine": "cosine",
    "dot": "dot_product",
    "dot_product": "dot_product",
    "euclidean": "euclidean",
    "max_inner_product": "max_inner_product",
}

LAMBDADB_CAPABILITIES = AdapterCapabilities(
    supported_write_modes=frozenset({"upsert", "bulk_upsert"}),
    supported_query_consistency=frozenset({"eventual", "strong"}),
    supports_read_after_write_strong=True,
    vendor_consistency_options={"consistent_read": True},
)


@dataclass(frozen=True)
class LambdaDBTargetSettings:
    base_url: str | None
    project_name: str | None
    api_key_env: str | None
    collection_name: str
    vector_field: str
    index_configs: dict[str, Any]
    timeout_ms: int | None
    delete_wait_timeout_seconds: float
    delete_wait_poll_seconds: float
    create_wait_timeout_seconds: float
    create_wait_poll_seconds: float


class LambdaDBAdapter:
    vendor = "lambdadb"
    capabilities = LAMBDADB_CAPABILITIES

    def __init__(
        self,
        *,
        client_factory: Callable[..., Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._environ = os.environ if environ is None else environ
        self._clients: dict[tuple[object, ...], Any] = {}
        self._client_lock = Lock()

    def check(self, target: TargetConfig) -> CheckResult:
        try:
            settings = _settings_from_target(target)
        except ConfigError as exc:
            return CheckResult(ok=False, message=str(exc))

        details = {
            "vendor": target.vendor,
            "target": target.name,
            "collection_name": settings.collection_name,
            "vector_field": settings.vector_field,
            "api_key_env": settings.api_key_env,
            "api_key_present": bool(
                settings.api_key_env and self._environ.get(settings.api_key_env)
            ),
        }
        return CheckResult(
            ok=True,
            message="LambdaDB target metadata is valid",
            details=details,
        )

    def prepare(
        self,
        target: TargetConfig,
        *,
        dimensions: int | None = None,
        metric: str | None = None,
    ) -> PrepareResult:
        settings = _settings_from_target(target)
        client = self._client(settings)

        if target.prepare_mode == "existing":
            response = client.collections.get(
                collection_name=settings.collection_name,
            )
            ready_response = self._wait_until_active(
                client,
                settings,
                initial_response=response,
            )
            return PrepareResult(
                ok=True,
                message="LambdaDB collection exists",
                details={
                    "collection_name": settings.collection_name,
                    "response": ready_response,
                },
            )

        if target.prepare_mode == "recreate":
            self._delete_if_present(client, settings)

        index_configs = _index_configs(settings, dimensions=dimensions, metric=metric)
        response = client.collections.create(
            collection_name=settings.collection_name,
            index_configs=index_configs,
        )
        ready_response = self._wait_until_active(client, settings)
        return PrepareResult(
            ok=True,
            message="LambdaDB collection created",
            details={
                "collection_name": settings.collection_name,
                "index_configs": index_configs,
                "response": response,
                "ready_response": ready_response,
            },
        )

    def upsert_batch(
        self,
        target: TargetConfig,
        records: Sequence[Mapping[str, Any] | VectorRecord],
    ) -> UpsertResult:
        settings = _settings_from_target(target)
        docs = [
            _record_to_doc(record, vector_field=settings.vector_field)
            for record in records
        ]
        if not docs:
            return UpsertResult(count=0)

        response = self._client(settings).collections.docs.upsert(
            collection_name=settings.collection_name,
            docs=docs,
        )
        return UpsertResult(count=len(docs), raw_response=response)

    def query(
        self,
        target: TargetConfig,
        *,
        vector: Sequence[float],
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        filter_query: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        settings = _settings_from_target(target)
        knn: dict[str, Any] = {
            "field": settings.vector_field,
            "queryVector": list(vector),
            "k": top_k,
        }
        if filter_query is not None:
            knn["filter"] = dict(filter_query)

        response = self._client(settings).collections.query(
            collection_name=settings.collection_name,
            query={"knn": knn},
            size=top_k,
            consistent_read=_consistent_read(consistency),
            include_vectors=include_vectors,
        )
        return QueryResult(
            matches=_query_matches(response),
            raw_response=response,
        )

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: Sequence[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        settings = _settings_from_target(target)
        response = self._client(settings).collections.docs.fetch(
            collection_name=settings.collection_name,
            ids=list(ids),
            consistent_read=_consistent_read(consistency),
            include_vectors=include_vectors,
        )
        return _documents(response)

    def _client(self, settings: LambdaDBTargetSettings) -> Any:
        api_key = _api_key(settings, self._environ)
        cache_key = (
            get_ident(),
            settings.base_url,
            settings.project_name,
            api_key,
            settings.timeout_ms,
        )
        with self._client_lock:
            client = self._clients.get(cache_key)
            if client is not None:
                return client
            client = self._new_client(settings, api_key=api_key)
            self._clients[cache_key] = client
            return client

    def _new_client(
        self,
        settings: LambdaDBTargetSettings,
        *,
        api_key: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "project_api_key": api_key,
        }
        if settings.timeout_ms is not None:
            kwargs["timeout_ms"] = settings.timeout_ms
        kwargs["base_url"] = settings.base_url
        kwargs["project_name"] = settings.project_name
        return self._client_factory(**kwargs)

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> Any:
        from lambdadb import LambdaDB

        return LambdaDB(**kwargs)

    def _delete_if_present(
        self,
        client: Any,
        settings: LambdaDBTargetSettings,
    ) -> None:
        try:
            client.collections.delete(collection_name=settings.collection_name)
        except Exception as exc:
            if _is_not_found_error(exc):
                return
            raise
        self._wait_until_deleted(client, settings)

    @staticmethod
    def _wait_until_deleted(client: Any, settings: LambdaDBTargetSettings) -> None:
        deadline = time.monotonic() + settings.delete_wait_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            try:
                client.collections.get(collection_name=settings.collection_name)
            except Exception as exc:
                if _is_not_found_error(exc):
                    return
                last_error = exc
            time.sleep(settings.delete_wait_poll_seconds)
        if last_error is not None:
            raise ConfigError(
                "timed out waiting for LambdaDB collection "
                f"{settings.collection_name!r} deletion; last status check failed: "
                f"{last_error}"
            )
        raise ConfigError(
            "timed out waiting for LambdaDB collection "
            f"{settings.collection_name!r} deletion"
        )

    @staticmethod
    def _wait_until_active(
        client: Any,
        settings: LambdaDBTargetSettings,
        *,
        initial_response: Any | None = None,
    ) -> Any:
        deadline = time.monotonic() + settings.create_wait_timeout_seconds
        response = initial_response
        last_status = _collection_status(response)
        last_error: Exception | None = None

        while time.monotonic() <= deadline:
            if response is None:
                try:
                    response = client.collections.get(
                        collection_name=settings.collection_name,
                    )
                    last_error = None
                except Exception as exc:
                    last_error = exc
                    if not _is_not_found_error(exc):
                        raise
            last_status = _collection_status(response)
            if _is_active_collection_status(last_status):
                return response
            time.sleep(settings.create_wait_poll_seconds)
            response = None

        if last_error is not None:
            raise ConfigError(
                "timed out waiting for LambdaDB collection "
                f"{settings.collection_name!r} to become ACTIVE; "
                f"last status check failed: {last_error}"
            )
        raise ConfigError(
            "timed out waiting for LambdaDB collection "
            f"{settings.collection_name!r} to become ACTIVE; "
            f"last status was {_display_status(last_status)}"
        )


def _settings_from_target(target: TargetConfig) -> LambdaDBTargetSettings:
    if target.vendor != "lambdadb":
        raise ConfigError(
            f"LambdaDB adapter cannot use target vendor {target.vendor!r}"
        )
    if not target.collection_name:
        raise ConfigError("target.collection_name or target.collection is required")
    if not target.endpoint:
        raise ConfigError("target.endpoint is required")
    if not target.project_name:
        raise ConfigError("target.project_name is required")

    timeout_ms = target.raw.get("timeout_ms")
    if timeout_ms is not None and (
        not isinstance(timeout_ms, int) or timeout_ms <= 0
    ):
        raise ConfigError("target.timeout_ms must be a positive integer")

    delete_wait_timeout_seconds = _optional_positive_float(
        target.raw,
        "delete_wait_timeout_seconds",
        DEFAULT_DELETE_WAIT_TIMEOUT_SECONDS,
    )
    delete_wait_poll_seconds = _optional_positive_float(
        target.raw,
        "delete_wait_poll_seconds",
        DEFAULT_DELETE_WAIT_POLL_SECONDS,
    )
    create_wait_timeout_seconds = _optional_positive_float(
        target.raw,
        "create_wait_timeout_seconds",
        DEFAULT_CREATE_WAIT_TIMEOUT_SECONDS,
    )
    create_wait_poll_seconds = _optional_positive_float(
        target.raw,
        "create_wait_poll_seconds",
        DEFAULT_CREATE_WAIT_POLL_SECONDS,
    )

    return LambdaDBTargetSettings(
        base_url=target.endpoint,
        project_name=target.project_name,
        api_key_env=target.api_key_env,
        collection_name=target.collection_name,
        vector_field=target.vector_field or DEFAULT_VECTOR_FIELD,
        index_configs=target.index_configs,
        timeout_ms=timeout_ms,
        delete_wait_timeout_seconds=delete_wait_timeout_seconds,
        delete_wait_poll_seconds=delete_wait_poll_seconds,
        create_wait_timeout_seconds=create_wait_timeout_seconds,
        create_wait_poll_seconds=create_wait_poll_seconds,
    )


def _optional_positive_float(
    raw: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = raw.get(key, default)
    if not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"target.{key} must be a positive number")
    return float(value)


def _is_not_found_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "ResourceNotFoundError":
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code == 404


def _collection_status(response: Any) -> str | None:
    for key in ("collection_status", "collectionStatus", "status", "state"):
        value = _field_value(response, key)
        if value is not None:
            return _status_text(value)
    return None


def _field_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested_key in ("collection", "data"):
            nested = value.get(nested_key)
            nested_value = _field_value(nested, key)
            if nested_value is not None:
                return nested_value
        return None
    direct_value = getattr(value, key, None)
    if direct_value is not None:
        return direct_value
    for nested_key in ("collection", "data"):
        nested_value = _field_value(getattr(value, nested_key, None), key)
        if nested_value is not None:
            return nested_value
    return None


def _is_active_collection_status(status: str | None) -> bool:
    if status is None:
        return True
    return status.upper() == ACTIVE_COLLECTION_STATUS


def _status_text(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _display_status(status: str | None) -> str:
    return "unknown" if status is None else repr(status)


def _api_key(
    settings: LambdaDBTargetSettings,
    environ: Mapping[str, str],
) -> str:
    if not settings.api_key_env:
        raise ConfigError("target.api_key_env is required")
    api_key = environ.get(settings.api_key_env)
    if not api_key:
        raise ConfigError(f"environment variable {settings.api_key_env} is not set")
    return api_key


def _index_configs(
    settings: LambdaDBTargetSettings,
    *,
    dimensions: int | None,
    metric: str | None,
) -> dict[str, Any]:
    if settings.index_configs:
        return settings.index_configs
    if dimensions is None:
        raise ConfigError(
            "target.index_configs or dataset dimensions are required for create mode"
        )

    similarity = SUPPORTED_METRIC_MAP.get(metric or "cosine")
    if similarity is None:
        raise ConfigError(
            "LambdaDB vector similarity must be one of "
            f"{sorted(SUPPORTED_METRIC_MAP)}"
        )
    return {
        settings.vector_field: {
            "type": "vector",
            "dimensions": dimensions,
            "similarity": similarity,
        }
    }


def _record_to_doc(
    record: Mapping[str, Any] | VectorRecord,
    *,
    vector_field: str,
) -> dict[str, Any]:
    if isinstance(record, VectorRecord):
        return {
            "id": record.id,
            vector_field: _vector_values(record.vector),
            "metadata": dict(record.metadata),
        }

    record_id = record.get("id")
    vector = record.get("vector")
    metadata = record.get("metadata", {})
    if not isinstance(record_id, str) or not record_id:
        raise ConfigError("record id must be a non-empty string")
    if not isinstance(vector, list):
        raise ConfigError("record vector must be a list")
    if not isinstance(metadata, Mapping):
        raise ConfigError("record metadata must be a mapping")
    return {
        "id": record_id,
        vector_field: _vector_values(vector),
        "metadata": dict(metadata),
    }


def _vector_values(vector: Sequence[float]) -> Sequence[float]:
    return vector if isinstance(vector, list) else list(vector)


def _consistent_read(consistency: str) -> bool:
    if consistency == "eventual":
        return False
    if consistency == "strong":
        return True
    raise ConfigError("query consistency must be 'eventual' or 'strong'")


def _query_matches(response: Any) -> list[QueryMatch]:
    matches: list[QueryMatch] = []
    for item in _results(response):
        doc = _document_from_result(item)
        matches.append(
            QueryMatch(
                id=str(doc.get("id", "")),
                score=_score_from_result(item),
                document=doc,
            )
        )
    return matches


def _results(response: Any) -> Sequence[Any]:
    if isinstance(response, Mapping):
        value = response.get("docs") or response.get("results") or []
        return value if isinstance(value, Sequence) else []
    value = getattr(response, "results", None)
    if value is None:
        value = getattr(response, "docs", None)
    return value if isinstance(value, Sequence) else []


def _documents(response: Any) -> list[Mapping[str, Any]]:
    docs = _results(response)
    if docs:
        return [_document_from_result(item) for item in docs]
    if isinstance(response, Mapping):
        value = response.get("documents") or response.get("docs") or []
        return [dict(item) for item in value if isinstance(item, Mapping)]
    value = getattr(response, "documents", None)
    if isinstance(value, Sequence):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _document_from_result(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        doc = result.get("doc", result)
        return dict(doc) if isinstance(doc, Mapping) else {}
    doc = getattr(result, "doc", None)
    return dict(doc) if isinstance(doc, Mapping) else {}


def _score_from_result(result: Any) -> float | None:
    if isinstance(result, Mapping):
        score = result.get("score")
    else:
        score = getattr(result, "score", None)
    if isinstance(score, int | float):
        return float(score)
    return None
