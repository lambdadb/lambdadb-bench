"""Real Pinecone adapter using the official Python SDK."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

DEFAULT_CLOUD = "aws"
DEFAULT_NAMESPACE = ""
DEFAULT_CREATE_TIMEOUT_SECONDS = 600
DEFAULT_DELETE_TIMEOUT_SECONDS = 600
SUPPORTED_METRIC_MAP = {
    "cosine": "cosine",
    "dot": "dotproduct",
    "dot_product": "dotproduct",
    "euclidean": "euclidean",
}

PINECONE_CAPABILITIES = AdapterCapabilities(
    supported_write_modes=frozenset({"upsert"}),
    supported_query_consistency=frozenset({"eventual"}),
    supports_read_after_write_strong=False,
    vendor_consistency_options={"data_freshness_model": "eventual"},
)


@dataclass(frozen=True)
class PineconeTargetSettings:
    api_key_env: str | None
    index_name: str
    index_host: str | None
    cloud: str
    region: str | None
    namespace: str
    pool_threads: int | None
    connection_pool_maxsize: int | None
    timeout: float | None
    create_timeout_seconds: int
    delete_timeout_seconds: int
    spec: dict[str, Any]
    tags: dict[str, str] | None


class PineconeAdapter:
    vendor = "pinecone"
    capabilities = PINECONE_CAPABILITIES

    def __init__(
        self,
        *,
        client_factory: Callable[..., Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._environ = os.environ if environ is None else environ

    def check(self, target: TargetConfig) -> CheckResult:
        try:
            settings = _settings_from_target(target)
        except ConfigError as exc:
            return CheckResult(ok=False, message=str(exc))

        return CheckResult(
            ok=True,
            message="Pinecone target metadata is valid",
            details={
                "vendor": target.vendor,
                "target": target.name,
                "index_name": settings.index_name,
                "index_host_present": bool(settings.index_host),
                "api_key_env": settings.api_key_env,
                "api_key_present": bool(
                    settings.api_key_env and self._environ.get(settings.api_key_env)
                ),
                "cloud": settings.cloud,
                "region": settings.region,
                "namespace": settings.namespace,
            },
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
            if not client.has_index(settings.index_name):
                raise ConfigError(
                    f"Pinecone index {settings.index_name!r} does not exist"
                )
            response = client.describe_index(settings.index_name)
            return PrepareResult(
                ok=True,
                message="Pinecone index exists",
                details={
                    "index_name": settings.index_name,
                    "response": response,
                },
            )

        if target.prepare_mode == "recreate" and client.has_index(settings.index_name):
            client.delete_index(
                name=settings.index_name,
                timeout=settings.delete_timeout_seconds,
            )

        response = client.create_index(
            name=settings.index_name,
            dimension=_dimensions(dimensions),
            metric=_metric(metric),
            spec=_serverless_spec(settings),
            timeout=settings.create_timeout_seconds,
            deletion_protection="disabled",
            vector_type="dense",
            tags=settings.tags,
        )
        return PrepareResult(
            ok=True,
            message="Pinecone index prepared",
            details={
                "index_name": settings.index_name,
                "cloud": settings.cloud,
                "region": settings.region,
                "response": response,
            },
        )

    def upsert_batch(
        self,
        target: TargetConfig,
        records: Sequence[Mapping[str, Any] | VectorRecord],
    ) -> UpsertResult:
        settings = _settings_from_target(target)
        vectors = [_record_to_vector(record) for record in records]
        if not vectors:
            return UpsertResult(count=0)

        response = self._index(settings).upsert(
            vectors=vectors,
            namespace=settings.namespace,
        )
        return UpsertResult(count=len(vectors), raw_response=response)

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
        if consistency != "eventual":
            raise ConfigError("Pinecone adapter supports only eventual consistency")

        settings = _settings_from_target(target)
        kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": top_k,
            "namespace": settings.namespace,
            "include_values": include_vectors,
            "include_metadata": True,
        }
        if filter_query is not None:
            kwargs["filter"] = dict(filter_query)
        response = self._index(settings).query(**kwargs)
        return QueryResult(matches=_query_matches(response), raw_response=response)

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: Sequence[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        if consistency != "eventual":
            raise ConfigError("Pinecone adapter supports only eventual consistency")

        settings = _settings_from_target(target)
        response = self._index(settings).fetch(
            ids=list(ids),
            namespace=settings.namespace,
        )
        return _documents(response, include_vectors=include_vectors)

    def _client(self, settings: PineconeTargetSettings) -> Any:
        kwargs: dict[str, Any] = {"api_key": _api_key(settings, self._environ)}
        if settings.timeout is not None:
            kwargs["timeout"] = settings.timeout
        return self._client_factory(**kwargs)

    def _index(self, settings: PineconeTargetSettings) -> Any:
        kwargs: dict[str, Any] = {}
        if settings.pool_threads is not None:
            kwargs["pool_threads"] = settings.pool_threads
        if settings.connection_pool_maxsize is not None:
            kwargs["connection_pool_maxsize"] = settings.connection_pool_maxsize
        client = self._client(settings)
        if settings.index_host:
            return client.Index(host=settings.index_host, **kwargs)
        return client.Index(name=settings.index_name, **kwargs)

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> Any:
        from pinecone import Pinecone

        return Pinecone(**kwargs)


def _settings_from_target(target: TargetConfig) -> PineconeTargetSettings:
    if target.vendor != "pinecone":
        raise ConfigError(
            f"Pinecone adapter cannot use target vendor {target.vendor!r}"
        )
    if not target.collection_name:
        raise ConfigError("target.collection_name or target.collection is required")

    cloud = target.raw.get("cloud", DEFAULT_CLOUD)
    if not isinstance(cloud, str) or not cloud:
        raise ConfigError("target.cloud must be a non-empty string")

    namespace = target.raw.get("namespace", DEFAULT_NAMESPACE)
    if not isinstance(namespace, str):
        raise ConfigError("target.namespace must be a string")

    pool_threads = _optional_positive_int(target.raw, "pool_threads")
    connection_pool_maxsize = _optional_positive_int(
        target.raw,
        "connection_pool_maxsize",
    )
    timeout = _optional_positive_float(target.raw, "timeout")
    create_timeout_seconds = _optional_positive_int(
        target.raw,
        "create_timeout_seconds",
        default=DEFAULT_CREATE_TIMEOUT_SECONDS,
    )
    delete_timeout_seconds = _optional_positive_int(
        target.raw,
        "delete_timeout_seconds",
        default=DEFAULT_DELETE_TIMEOUT_SECONDS,
    )
    spec = _optional_mapping(target.raw, "spec")
    tags = _optional_str_mapping(target.raw, "tags")

    return PineconeTargetSettings(
        api_key_env=target.api_key_env,
        index_name=target.collection_name,
        index_host=target.endpoint,
        cloud=cloud,
        region=target.region,
        namespace=namespace,
        pool_threads=pool_threads,
        connection_pool_maxsize=connection_pool_maxsize,
        timeout=timeout,
        create_timeout_seconds=create_timeout_seconds,
        delete_timeout_seconds=delete_timeout_seconds,
        spec=spec,
        tags=tags,
    )


def _api_key(
    settings: PineconeTargetSettings,
    environ: Mapping[str, str],
) -> str:
    if not settings.api_key_env:
        raise ConfigError("target.api_key_env is required")
    api_key = environ.get(settings.api_key_env)
    if not api_key:
        raise ConfigError(f"environment variable {settings.api_key_env} is not set")
    return api_key


def _serverless_spec(settings: PineconeTargetSettings) -> Any:
    spec: dict[str, Any] = dict(settings.spec)
    spec.setdefault("cloud", settings.cloud)
    if "region" not in spec:
        if not settings.region:
            raise ConfigError("target.region or target.spec.region is required")
        spec["region"] = settings.region

    from pinecone import ServerlessSpec

    return ServerlessSpec(**spec)


def _dimensions(dimensions: int | None) -> int:
    if dimensions is None:
        raise ConfigError("dataset dimensions are required for Pinecone create mode")
    return dimensions


def _metric(metric: str | None) -> str:
    value = SUPPORTED_METRIC_MAP.get(metric or "cosine")
    if value is None:
        raise ConfigError(
            "Pinecone vector metric must be one of "
            f"{sorted(SUPPORTED_METRIC_MAP)}"
        )
    return value


def _record_to_vector(record: Mapping[str, Any] | VectorRecord) -> dict[str, Any]:
    if isinstance(record, VectorRecord):
        return {
            "id": record.id,
            "values": list(record.vector),
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
        "values": list(vector),
        "metadata": dict(metadata),
    }


def _query_matches(response: Any) -> list[QueryMatch]:
    return [
        QueryMatch(
            id=str(_match_id(match)),
            score=_score(match),
            document=_document_from_match(match),
        )
        for match in _matches(response)
    ]


def _matches(response: Any) -> Sequence[Any]:
    if isinstance(response, Mapping):
        value = response.get("matches") or []
        return value if isinstance(value, Sequence) else []
    value = getattr(response, "matches", None)
    return value if isinstance(value, Sequence) else []


def _documents(response: Any, *, include_vectors: bool) -> list[Mapping[str, Any]]:
    vectors = _vectors(response)
    return [
        _document_from_vector(vector_id, vector, include_vectors=include_vectors)
        for vector_id, vector in vectors.items()
    ]


def _vectors(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        value = response.get("vectors") or {}
        return value if isinstance(value, Mapping) else {}
    value = getattr(response, "vectors", None)
    return value if isinstance(value, Mapping) else {}


def _document_from_match(match: Any) -> Mapping[str, Any]:
    metadata = _metadata(match)
    document = dict(metadata)
    document["id"] = str(_match_id(match))
    values = _values(match)
    if values is not None:
        document["vector"] = values
    return document


def _document_from_vector(
    vector_id: str,
    vector: Any,
    *,
    include_vectors: bool,
) -> Mapping[str, Any]:
    metadata = _metadata(vector)
    document = dict(metadata)
    document["id"] = str(vector_id)
    values = _values(vector)
    if include_vectors and values is not None:
        document["vector"] = values
    return document


def _match_id(match: Any) -> object:
    if isinstance(match, Mapping):
        return match.get("id", "")
    return getattr(match, "id", "")


def _score(match: Any) -> float | None:
    if isinstance(match, Mapping):
        value = match.get("score")
    else:
        value = getattr(match, "score", None)
    if isinstance(value, int | float):
        return float(value)
    return None


def _metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        value = item.get("metadata") or {}
    else:
        value = getattr(item, "metadata", None) or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _values(item: Any) -> object | None:
    if isinstance(item, Mapping):
        return item.get("values")
    return getattr(item, "values", None)


def _optional_positive_int(
    raw: Mapping[str, Any],
    key: str,
    default: int | None = None,
) -> int | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"target.{key} must be a positive integer")
    return value


def _optional_positive_float(
    raw: Mapping[str, Any],
    key: str,
) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"target.{key} must be a positive number")
    return float(value)


def _optional_mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"target.{key} must be a mapping")
    return dict(value)


def _optional_str_mapping(
    raw: Mapping[str, Any],
    key: str,
) -> dict[str, str] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(
        isinstance(item_key, str) and isinstance(item_value, str)
        for item_key, item_value in value.items()
    ):
        raise ConfigError(f"target.{key} must be a string mapping")
    return dict(value)
