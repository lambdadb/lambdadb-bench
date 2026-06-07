"""Real Qdrant adapter using the official Python client."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
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
from ldbbench.adapters.filters import qdrant_filter
from ldbbench.config import ConfigError, TargetConfig

QDRANT_CAPABILITIES = AdapterCapabilities(
    supported_write_modes=frozenset({"upsert"}),
    supported_query_consistency=frozenset({"eventual"}),
    supports_read_after_write_strong=False,
    supports_query_filter=True,
    supports_query_partition_filter=False,
    vendor_consistency_options={
        "read_consistency": ["all", "majority", "quorum"],
        "write_ordering": ["weak", "medium", "strong"],
        "default_protocol": "grpc",
    },
)

DISTANCE_MAP = {
    "cosine": "COSINE",
    "dot": "DOT",
    "dot_product": "DOT",
    "euclidean": "EUCLID",
}
SOURCE_ID_PAYLOAD_KEY = "__ldbbench_source_id"


@dataclass(frozen=True)
class QdrantTargetSettings:
    url: str
    api_key_env: str | None
    collection_name: str
    vector_name: str | None
    timeout: int | None
    prefer_grpc: bool


class QdrantAdapter:
    vendor = "qdrant"
    capabilities = QDRANT_CAPABILITIES

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

        return CheckResult(
            ok=True,
            message="Qdrant target metadata is valid",
            details={
                "vendor": target.vendor,
                "target": target.name,
                "collection_name": settings.collection_name,
                "vector_name": settings.vector_name,
                "api_key_env": settings.api_key_env,
                "api_key_present": bool(
                    settings.api_key_env and self._environ.get(settings.api_key_env)
                ),
                "prefer_grpc": settings.prefer_grpc,
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
            if not client.collection_exists(collection_name=settings.collection_name):
                raise ConfigError(
                    f"Qdrant collection {settings.collection_name!r} does not exist"
                )
            response = client.get_collection(collection_name=settings.collection_name)
            return PrepareResult(
                ok=True,
                message="Qdrant collection exists",
                details={
                    "collection_name": settings.collection_name,
                    "response": response,
                },
            )

        vectors_config = _vectors_config(
            settings,
            dimensions=dimensions,
            metric=metric,
        )

        if target.prepare_mode == "recreate":
            response = client.recreate_collection(
                collection_name=settings.collection_name,
                vectors_config=vectors_config,
            )
        else:
            response = client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=vectors_config,
            )

        return PrepareResult(
            ok=True,
            message="Qdrant collection prepared",
            details={
                "collection_name": settings.collection_name,
                "vectors_config": vectors_config,
                "response": response,
            },
        )

    def upsert_batch(
        self,
        target: TargetConfig,
        records: Sequence[Mapping[str, Any] | VectorRecord],
        *,
        write_mode: str = "upsert",
    ) -> UpsertResult:
        if write_mode != "upsert":
            raise ConfigError(
                f"Qdrant adapter does not support write_mode {write_mode!r}"
            )
        settings = _settings_from_target(target)
        points = [
            _record_to_point(record, vector_name=settings.vector_name)
            for record in records
        ]
        if not points:
            return UpsertResult(count=0)

        response = self._client(settings).upsert(
            collection_name=settings.collection_name,
            points=points,
            wait=True,
        )
        return UpsertResult(count=len(points), raw_response=response)

    def query(
        self,
        target: TargetConfig,
        *,
        vector: Sequence[float],
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        filter_query: Mapping[str, Any] | None = None,
        partition_filter: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        if consistency != "eventual":
            raise ConfigError("Qdrant adapter supports only eventual consistency")
        if partition_filter is not None:
            raise ConfigError("Qdrant adapter does not support partition filters")

        settings = _settings_from_target(target)
        response = self._client(settings).query_points(
            collection_name=settings.collection_name,
            query=_vector_values(vector),
            using=settings.vector_name,
            query_filter=qdrant_filter(filter_query) if filter_query else None,
            limit=top_k,
            with_payload=True,
            with_vectors=include_vectors,
        )
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
            raise ConfigError("Qdrant adapter supports only eventual consistency")

        settings = _settings_from_target(target)
        response = self._client(settings).retrieve(
            collection_name=settings.collection_name,
            ids=[_qdrant_point_id(item) for item in ids],
            with_payload=True,
            with_vectors=include_vectors,
        )
        return [_document_from_point(point) for point in response]

    def _client(self, settings: QdrantTargetSettings) -> Any:
        api_key = _api_key(settings, self._environ)
        cache_key = (
            settings.url,
            api_key,
            settings.timeout,
            settings.prefer_grpc,
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
        settings: QdrantTargetSettings,
        *,
        api_key: str | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "url": settings.url,
            "api_key": api_key,
            "prefer_grpc": settings.prefer_grpc,
        }
        if settings.timeout is not None:
            kwargs["timeout"] = settings.timeout
        return self._client_factory(**kwargs)

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(**kwargs)


def _settings_from_target(target: TargetConfig) -> QdrantTargetSettings:
    if target.vendor != "qdrant":
        raise ConfigError(f"Qdrant adapter cannot use target vendor {target.vendor!r}")
    if not target.endpoint:
        raise ConfigError("target.endpoint is required")
    if not target.collection_name:
        raise ConfigError("target.collection_name or target.collection is required")

    timeout = target.raw.get("timeout")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise ConfigError("target.timeout must be a positive integer")

    prefer_grpc = target.raw.get("prefer_grpc", True)
    if not isinstance(prefer_grpc, bool):
        raise ConfigError("target.prefer_grpc must be a boolean")

    return QdrantTargetSettings(
        url=target.endpoint,
        api_key_env=target.api_key_env,
        collection_name=target.collection_name,
        vector_name=target.vector_field,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
    )


def _api_key(settings: QdrantTargetSettings, environ: Mapping[str, str]) -> str | None:
    if not settings.api_key_env:
        return None
    api_key = environ.get(settings.api_key_env)
    if not api_key:
        raise ConfigError(f"environment variable {settings.api_key_env} is not set")
    return api_key


def _vectors_config(
    settings: QdrantTargetSettings,
    *,
    dimensions: int | None,
    metric: str | None,
) -> Any:
    if dimensions is None:
        raise ConfigError("dataset dimensions are required for Qdrant create mode")

    from qdrant_client import models

    distance_name = DISTANCE_MAP.get(metric or "cosine")
    if distance_name is None:
        raise ConfigError(
            "Qdrant vector distance must be one of " f"{sorted(DISTANCE_MAP)}"
        )
    params = models.VectorParams(
        size=dimensions,
        distance=getattr(models.Distance, distance_name),
    )
    if settings.vector_name:
        return {settings.vector_name: params}
    return params


def _record_to_point(
    record: Mapping[str, Any] | VectorRecord,
    *,
    vector_name: str | None,
) -> Any:
    from qdrant_client import models

    if isinstance(record, VectorRecord):
        record_id = record.id
        vector = list(record.vector)
        payload = dict(record.metadata)
    else:
        record_id = record.get("id")
        vector = record.get("vector")
        metadata = record.get("metadata", {})
        if not isinstance(record_id, str) or not record_id:
            raise ConfigError("record id must be a non-empty string")
        if not isinstance(vector, list):
            raise ConfigError("record vector must be a list")
        if not isinstance(metadata, Mapping):
            raise ConfigError("record metadata must be a mapping")
        payload = dict(metadata)

    payload[SOURCE_ID_PAYLOAD_KEY] = record_id
    vector_values = _vector_values(vector)
    point_vector: Any = {vector_name: vector_values} if vector_name else vector_values
    return models.PointStruct(
        id=_qdrant_point_id(record_id),
        vector=point_vector,
        payload=payload,
    )


def _vector_values(vector: Sequence[float]) -> Sequence[float]:
    return vector if isinstance(vector, list) else list(vector)


def _query_matches(response: Any) -> list[QueryMatch]:
    return [
        QueryMatch(
            id=_document_id(point),
            score=_score(point),
            document=_document_from_point(point),
        )
        for point in _points(response)
    ]


def _points(response: Any) -> Sequence[Any]:
    if isinstance(response, Mapping):
        value = response.get("points") or response.get("result") or []
        return value if isinstance(value, Sequence) else []
    value = getattr(response, "points", None)
    if value is None:
        value = getattr(response, "result", None)
    return value if isinstance(value, Sequence) else []


def _document_from_point(point: Any) -> Mapping[str, Any]:
    payload = _payload(point)
    source_id = payload.pop(SOURCE_ID_PAYLOAD_KEY, None)
    document = dict(payload)
    document["id"] = str(source_id if source_id is not None else _point_id(point))
    vector = _vector(point)
    if vector is not None:
        document["vector"] = vector
    return document


def _document_id(point: Any) -> str:
    payload = _payload(point)
    source_id = payload.get(SOURCE_ID_PAYLOAD_KEY)
    return str(source_id if source_id is not None else _point_id(point))


def _qdrant_point_id(record_id: str) -> str:
    try:
        return str(uuid.UUID(record_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lambdadb-bench:qdrant:{record_id}"))


def _point_id(point: Any) -> object:
    if isinstance(point, Mapping):
        return point.get("id", "")
    return getattr(point, "id", "")


def _score(point: Any) -> float | None:
    if isinstance(point, Mapping):
        value = point.get("score")
    else:
        value = getattr(point, "score", None)
    if isinstance(value, int | float):
        return float(value)
    return None


def _payload(point: Any) -> dict[str, Any]:
    if isinstance(point, Mapping):
        value = point.get("payload") or {}
    else:
        value = getattr(point, "payload", None) or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _vector(point: Any) -> object | None:
    if isinstance(point, Mapping):
        return point.get("vector")
    return getattr(point, "vector", None)
