from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from lambdadb import LambdaDB

from ldbbench.adapters.base import CollectionStats
from ldbbench.adapters.lambdadb import LambdaDBAdapter
from ldbbench.config import ConfigError, TargetConfig

METADATA_INDEX_CONFIG = {
    "metadata": {
        "type": "object",
        "objectIndexConfigs": {
            "filter_bucket_2": {"type": "keyword"},
            "filter_bucket_10": {"type": "keyword"},
            "filter_bucket_20": {"type": "keyword"},
            "filter_bucket_100": {"type": "keyword"},
            "filter_bucket_1000": {"type": "keyword"},
        },
    },
}


class FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeDocs:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.bulk_upserts: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self.upserts.append(kwargs)
        return {"ok": True}

    def bulk_upsert_docs(self, **kwargs: Any) -> dict[str, Any]:
        self.bulk_upserts.append(kwargs)
        return {"ok": True}

    def fetch(self, **kwargs: Any) -> dict[str, Any]:
        self.fetches.append(kwargs)
        return {"docs": [{"doc": {"id": item}} for item in kwargs["ids"]]}

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.deletes.append(kwargs)
        return {"ok": True}


class FakeCollections:
    def __init__(
        self,
        *,
        missing_after_gets: int | None = None,
        statuses: list[str | None] | None = None,
        sdk_response_objects: bool = False,
    ) -> None:
        self.docs = FakeDocs()
        self.creates: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.missing_after_gets = missing_after_gets
        self.statuses = list(statuses or [])
        self.last_status: str | None = None
        self.sdk_response_objects = sdk_response_objects

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.creates.append(kwargs)
        self.missing_after_gets = None
        return {"created": kwargs["collection_name"]}

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.gets.append(kwargs)
        if (
            self.missing_after_gets is not None
            and len(self.gets) >= self.missing_after_gets
        ):
            exc = RuntimeError("not found")
            exc.status_code = 404
            raise exc
        response = {"name": kwargs["collection_name"]}
        if self.statuses:
            status = self.statuses.pop(0)
            self.last_status = status
            if status is not None:
                response = self._collection_response(kwargs["collection_name"], status)
        elif self.last_status is not None:
            response = self._collection_response(
                kwargs["collection_name"],
                self.last_status,
            )
        return response

    def _collection_response(self, collection_name: str, status: Any) -> Any:
        if self.sdk_response_objects:
            return SimpleNamespace(
                collection=SimpleNamespace(
                    collection_name=collection_name,
                    collection_status=status,
                )
            )
        return {
            "collection": {
                "collection_name": collection_name,
                "collection_status": status,
            }
        }

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.deletes.append(kwargs)
        return {"deleted": kwargs["collection_name"]}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        return {
            "took": 3,
            "docs": [
                {"doc": {"id": "a"}, "score": 0.91},
                {"doc": {"id": "b"}, "score": 0.42},
            ]
        }


class FakeClient:
    def __init__(
        self,
        *,
        missing_after_gets: int | None = None,
        statuses: list[str | None] | None = None,
        sdk_response_objects: bool = False,
    ) -> None:
        self.collections = FakeCollections(
            missing_after_gets=missing_after_gets,
            statuses=statuses,
            sdk_response_objects=sdk_response_objects,
        )

    def collection(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            docs=self.collections.docs,
        )


def make_target(**overrides: Any) -> TargetConfig:
    data = {
        "vendor": "lambdadb",
        "name": "lambda-ci",
        "endpoint": "https://api.example.test",
        "project_name": "demo",
        "api_key_env": "LAMBDADB_API_KEY",
        "collection_name": "smoke",
        "vector_field": "dense",
        "prepare": {"mode": "existing"},
    }
    data.update(overrides)
    return TargetConfig.from_mapping(data)


def make_adapter(client: FakeClient) -> LambdaDBAdapter:
    return LambdaDBAdapter(
        client_factory=lambda **_kwargs: client,
        environ={"LAMBDADB_API_KEY": "secret"},
    )


def test_check_validates_lambdadb_metadata_without_requiring_api_key() -> None:
    adapter = LambdaDBAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    result = adapter.check(make_target())

    assert result.ok
    assert result.details["collection_name"] == "smoke"
    assert result.details["api_key_present"] is False


def test_delete_batch_forwards_document_ids() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.delete_batch(make_target(), ["a", "b"])

    assert result.count == 2
    assert client.collections.docs.deletes == [{"ids": ["a", "b"]}]


def test_check_reports_missing_project_name() -> None:
    adapter = LambdaDBAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    result = adapter.check(make_target(project_name=None))

    assert not result.ok
    assert "project_name" in result.message


def test_prepare_create_builds_vector_index_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(prepare={"mode": "create"})

    result = adapter.prepare(target, dimensions=1024, metric="dot")

    assert result.ok
    assert client.collections.creates == [
        {
            "collection_name": "smoke",
            "index_configs": {
                "dense": {
                    "type": "vector",
                    "dimensions": 1024,
                    "similarity": "dot_product",
                },
                **METADATA_INDEX_CONFIG,
            },
        }
    ]
    assert client.collections.gets == [{"collection_name": "smoke"}]


def test_prepare_create_builds_euclidean_vector_index_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(prepare={"mode": "create"})

    adapter.prepare(target, dimensions=1024, metric="euclidean")

    assert client.collections.creates[0]["index_configs"]["dense"]["similarity"] == (
        "euclidean"
    )


def test_prepare_create_passes_partition_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        partition_config={
            "field_name": "metadata.url",
            "data_type": "keyword",
            "num_partitions": 16,
        },
    )

    result = adapter.prepare(target, dimensions=1024, metric="cosine")

    assert result.ok
    assert client.collections.creates == [
        {
            "collection_name": "smoke",
            "index_configs": {
                "dense": {
                    "type": "vector",
                    "dimensions": 1024,
                    "similarity": "cosine",
                },
                **METADATA_INDEX_CONFIG,
            },
            "partition_config": {
                "field_name": "metadata.url",
                "data_type": "keyword",
                "num_partitions": 16,
            },
        }
    ]


def test_prepare_create_rejects_invalid_partition_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        partition_config={"field_name": "url", "data_type": "text"},
    )

    with pytest.raises(ConfigError, match="partition_config"):
        adapter.prepare(target, dimensions=1024, metric="cosine")


def test_prepare_create_preserves_metadata_object_text_index_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        index_configs={
            "dense": {"type": "vector", "dimensions": 3},
            "metadata": {
                "type": "object",
                "objectIndexConfigs": {
                    "text": {"type": "text", "analyzers": ["english"]},
                    "url": {"type": "keyword"},
                },
            },
        },
    )

    result = adapter.prepare(target)

    assert result.ok
    assert client.collections.creates == [
        {
            "collection_name": "smoke",
            "index_configs": {
                "dense": {"type": "vector", "dimensions": 3},
                "metadata": {
                    "type": "object",
                    "objectIndexConfigs": {
                        "text": {"type": "text", "analyzers": ["english"]},
                        "url": {"type": "keyword"},
                        "filter_bucket_2": {"type": "keyword"},
                        "filter_bucket_10": {"type": "keyword"},
                        "filter_bucket_20": {"type": "keyword"},
                        "filter_bucket_100": {"type": "keyword"},
                        "filter_bucket_1000": {"type": "keyword"},
                    },
                },
            },
        }
    ]


def test_prepare_create_rejects_non_object_metadata_index_config() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        index_configs={
            "dense": {"type": "vector", "dimensions": 3},
            "metadata": {"type": "keyword"},
        },
    )

    with pytest.raises(ConfigError, match="metadata.type"):
        adapter.prepare(target)


def test_prepare_create_waits_until_collection_is_active() -> None:
    client = FakeClient(statuses=["CREATING", "ACTIVE"])
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        create_wait_timeout_seconds=1,
        create_wait_poll_seconds=0.001,
    )

    result = adapter.prepare(target, dimensions=1024, metric="cosine")

    assert result.ok
    assert client.collections.gets == [
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
    ]
    assert result.details["ready_response"]["collection"]["collection_status"] == (
        "ACTIVE"
    )


def test_prepare_create_reads_sdk_nested_status_enum_value() -> None:
    client = FakeClient(
        statuses=[FakeStatus("CREATING"), FakeStatus("ACTIVE")],
        sdk_response_objects=True,
    )
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        create_wait_timeout_seconds=1,
        create_wait_poll_seconds=0.001,
    )

    result = adapter.prepare(target, dimensions=1024, metric="cosine")

    assert result.ok
    assert len(client.collections.gets) == 2
    assert result.details["ready_response"].collection.collection_status.value == (
        "ACTIVE"
    )


def test_prepare_create_times_out_waiting_for_active_collection() -> None:
    client = FakeClient(statuses=["CREATING", "CREATING", "CREATING"])
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        create_wait_timeout_seconds=0.002,
        create_wait_poll_seconds=0.001,
    )

    with pytest.raises(ConfigError, match="ACTIVE"):
        adapter.prepare(target, dimensions=1024, metric="cosine")


def test_prepare_existing_checks_collection() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.prepare(make_target())

    assert result.ok
    assert client.collections.gets == [{"collection_name": "smoke"}]


def test_prepare_existing_waits_if_collection_is_creating() -> None:
    client = FakeClient(statuses=["CREATING", "ACTIVE"])
    adapter = make_adapter(client)
    target = make_target(
        create_wait_timeout_seconds=1,
        create_wait_poll_seconds=0.001,
    )

    result = adapter.prepare(target)

    assert result.ok
    assert client.collections.gets == [
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
    ]


def test_adapter_reuses_client_for_same_target() -> None:
    clients: list[FakeClient] = []

    def factory(**_kwargs: Any) -> FakeClient:
        client = FakeClient()
        clients.append(client)
        return client

    adapter = LambdaDBAdapter(
        client_factory=factory,
        environ={"LAMBDADB_API_KEY": "secret"},
    )
    target = make_target()

    adapter.prepare(target)
    adapter.upsert_batch(target, [{"id": "a", "vector": [0.1], "metadata": {}}])
    adapter.fetch(target, ids=["a"], consistency="eventual")

    assert len(clients) == 1


def test_adapter_uses_separate_clients_per_worker_thread() -> None:
    clients: list[FakeClient] = []

    def factory(**_kwargs: Any) -> FakeClient:
        client = FakeClient()
        clients.append(client)
        return client

    adapter = LambdaDBAdapter(
        client_factory=factory,
        environ={"LAMBDADB_API_KEY": "secret"},
    )
    target = make_target()
    barrier = Barrier(2)

    def upsert() -> None:
        barrier.wait(timeout=1)
        adapter.upsert_batch(
            target,
            [{"id": "a", "vector": [0.1], "metadata": {}}],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(upsert) for _ in range(2)]
        for future in futures:
            future.result()

    assert len(clients) == 2


def test_prepare_recreate_waits_for_delete_then_active_collection() -> None:
    client = FakeClient(
        missing_after_gets=2,
        statuses=[
            FakeStatus("DELETING"),
            FakeStatus("CREATING"),
            FakeStatus("ACTIVE"),
        ],
        sdk_response_objects=True,
    )
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "recreate"},
        index_configs={"dense": {"type": "vector", "dimensions": 3}},
        delete_wait_timeout_seconds=1,
        delete_wait_poll_seconds=0.001,
        create_wait_timeout_seconds=1,
        create_wait_poll_seconds=0.001,
    )

    result = adapter.prepare(target)

    assert result.ok
    assert client.collections.deletes == [{"collection_name": "smoke"}]
    assert client.collections.gets == [
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
    ]
    assert client.collections.creates == [
        {
            "collection_name": "smoke",
            "index_configs": {
                "dense": {"type": "vector", "dimensions": 3},
                **METADATA_INDEX_CONFIG,
            },
        }
    ]
    assert result.details["ready_response"].collection.collection_status.value == (
        "ACTIVE"
    )


def test_prepare_recreate_times_out_waiting_for_delete() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "recreate"},
        delete_wait_timeout_seconds=0.002,
        delete_wait_poll_seconds=0.001,
    )

    with pytest.raises(ConfigError, match="timed out"):
        adapter.prepare(target, dimensions=3, metric="cosine")

    assert client.collections.deletes == [{"collection_name": "smoke"}]
    assert client.collections.creates == []


def test_upsert_batch_maps_normalized_records_to_documents() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.upsert_batch(
        make_target(),
        [
            {"id": "a", "vector": [0.1, 0.2], "metadata": {"text": "alpha"}},
            {"id": "b", "vector": [0.3, 0.4], "metadata": {}},
        ],
    )

    assert result.count == 2
    assert client.collections.docs.upserts == [
        {
            "docs": [
                {"id": "a", "dense": [0.1, 0.2], "metadata": {"text": "alpha"}},
                {"id": "b", "dense": [0.3, 0.4], "metadata": {}},
            ],
        }
    ]
    assert client.collections.docs.bulk_upserts == []


def test_upsert_batch_uses_bulk_upsert_docs_for_bulk_write_mode() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.upsert_batch(
        make_target(),
        [
            {"id": "a", "vector": [0.1, 0.2], "metadata": {"text": "alpha"}},
            {"id": "b", "vector": [0.3, 0.4], "metadata": {}},
        ],
        write_mode="bulk_upsert",
    )

    assert result.count == 2
    assert client.collections.docs.upserts == []
    assert client.collections.docs.bulk_upserts == [
        {
            "docs": [
                {"id": "a", "dense": [0.1, 0.2], "metadata": {"text": "alpha"}},
                {"id": "b", "dense": [0.3, 0.4], "metadata": {}},
            ],
        }
    ]


def test_upsert_batch_keeps_partition_field_inside_metadata() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        partition_config={
            "field_name": "metadata.url",
            "data_type": "keyword",
            "num_partitions": 16,
        },
    )

    adapter.upsert_batch(
        target,
        [
            {
                "id": "a",
                "vector": [0.1, 0.2],
                "metadata": {"text": "alpha", "url": "https://example.test/a"},
            },
        ],
    )

    assert client.collections.docs.upserts == [
        {
            "docs": [
                {
                    "id": "a",
                    "dense": [0.1, 0.2],
                    "metadata": {
                        "text": "alpha",
                        "url": "https://example.test/a",
                    },
                }
            ],
        }
    ]


def test_upsert_batch_keeps_filter_bucket_fields_inside_metadata() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    adapter.upsert_batch(
        make_target(),
        [
            {
                "id": "a",
                "vector": [0.1, 0.2],
                "metadata": {
                    "text": "alpha",
                    "filter_bucket_100": "42",
                },
            },
        ],
    )

    doc = client.collections.docs.upserts[0]["docs"][0]
    assert "filter_bucket_100" not in doc
    assert doc["metadata"]["filter_bucket_100"] == "42"


def test_query_maps_strong_consistency_to_consistent_read() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.query(
        make_target(),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="strong",
        filter_query={"queryString": {"query": "*:*"}},
    )

    assert [match.id for match in result.matches] == ["a", "b"]
    assert result.matches[0].score == 0.91
    assert client.collections.queries == [
        {
            "collection_name": "smoke",
            "query": {
                "knn": {
                    "field": "dense",
                    "queryVector": [0.1, 0.2],
                    "k": 2,
                    "filter": {"queryString": {"query": "*:*"}},
                }
            },
            "size": 2,
            "consistent_read": True,
            "include_vectors": False,
        }
    ]


def test_query_translates_portable_filter() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    adapter.query(
        make_target(),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="eventual",
        filter_query={
            "field": "filter_bucket_100",
            "operator": "eq",
            "value": "42",
        },
    )

    assert client.collections.queries[0]["query"]["knn"]["filter"] == {
        "queryString": {"query": "metadata.filter_bucket_100:42"}
    }


def test_full_text_query_uses_query_string_default_field() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.full_text_query(
        make_target(),
        query_text="alpha beta",
        field="metadata.text",
        top_k=2,
        consistency="eventual",
    )

    assert [match.id for match in result.matches] == ["a", "b"]
    assert client.collections.queries == [
        {
            "collection_name": "smoke",
            "query": {
                "queryString": {
                    "query": "alpha beta",
                    "defaultField": "metadata.text",
                }
            },
            "size": 2,
            "consistent_read": False,
            "include_vectors": False,
        }
    ]


def test_queries_report_server_took() -> None:
    adapter = make_adapter(FakeClient())

    vector_result = adapter.query(
        make_target(),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="eventual",
    )
    text_result = adapter.full_text_query(
        make_target(),
        query_text="alpha",
        field="metadata.text",
        top_k=2,
        consistency="eventual",
    )

    assert vector_result.server_took_ms == 3.0
    assert text_result.server_took_ms == 3.0


def test_collection_stats_treats_describe_without_status_as_ready() -> None:
    response = SimpleNamespace(
        collection=SimpleNamespace(collection_name="smoke", num_docs=42),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda **_kwargs: response),
    )

    stats = make_adapter(client).collection_stats(make_target())

    assert stats == CollectionStats(
        ready=True,
        num_docs=42,
        status=None,
        supports_data_updated_at=True,
    )


def test_collection_stats_is_not_ready_while_collection_is_creating() -> None:
    response = {"collection": {"collectionStatus": "CREATING", "numDocs": 0}}
    client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda **_kwargs: response),
    )

    stats = make_adapter(client).collection_stats(make_target())

    assert stats == CollectionStats(
        ready=False,
        num_docs=0,
        status="CREATING",
        supports_data_updated_at=True,
    )


def test_full_text_query_passes_partition_filter() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    adapter.full_text_query(
        make_target(),
        query_text="alpha beta",
        field="metadata.text",
        top_k=2,
        consistency="eventual",
        partition_filter={"field": "url", "in_": ["https://example.test/doc"]},
    )

    assert client.collections.queries == [
        {
            "collection_name": "smoke",
            "query": {
                "queryString": {
                    "query": "alpha beta",
                    "defaultField": "metadata.text",
                }
            },
            "size": 2,
            "consistent_read": False,
            "include_vectors": False,
            "partition_filter": {
                "field": "metadata.url",
                "in_": ["https://example.test/doc"],
            },
        }
    ]


def test_query_passes_partition_filter() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    adapter.query(
        make_target(),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="eventual",
        partition_filter={"field": "url", "in_": ["https://example.test/doc"]},
    )

    assert client.collections.queries == [
        {
            "collection_name": "smoke",
            "query": {
                "knn": {
                    "field": "dense",
                    "queryVector": [0.1, 0.2],
                    "k": 2,
                }
            },
            "size": 2,
            "consistent_read": False,
            "include_vectors": False,
            "partition_filter": {
                "field": "metadata.url",
                "in_": ["https://example.test/doc"],
            },
        }
    ]


def test_fetch_maps_eventual_consistency_to_consistent_read_false() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    docs = adapter.fetch(
        make_target(),
        ids=["a", "b"],
        consistency="eventual",
        include_vectors=True,
    )

    assert [doc["id"] for doc in docs] == ["a", "b"]
    assert client.collections.docs.fetches == [
        {
            "collection_name": "smoke",
            "ids": ["a", "b"],
            "consistent_read": False,
            "include_vectors": True,
        }
    ]


class DevelopServer:
    """Answers the real SDK with lambdadb develop response shapes."""

    upload_url = "https://bucket.example.test/bulk/1.json?X-Amz-Signature=sig"

    def __init__(self, *, num_docs: int = 0) -> None:
        self.num_docs = num_docs
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "PUT" and str(request.url) == self.upload_url:
            return httpx.Response(200)
        if path == "/projects/demo/collections" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "collection": {
                        "collectionName": "smoke",
                        "description": "",
                        "tags": {},
                        "defaultBranchName": "main",
                        "snapshotRetentionInDays": 7,
                        "createdAt": 1,
                    }
                },
            )
        if path == "/projects/demo/collections/smoke" and request.method == "GET":
            return httpx.Response(200, json={"collection": self._collection()})
        if path == "/projects/demo/collections/smoke/query":
            return httpx.Response(
                200,
                json={
                    "took": 7,
                    "total": 1,
                    "docs": [{"collection": "smoke", "doc": {"id": "a"}, "score": 0.9}],
                    "isDocsInline": True,
                },
            )
        if path == "/projects/demo/collections/smoke/docs/bulk-upsert":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "url": self.upload_url,
                        "type": "application/json",
                        "httpMethod": "PUT",
                        "objectKey": "bulk/1.json",
                        "sizeLimitBytes": 1_000_000,
                        "headers": {"If-None-Match": "*"},
                    },
                )
            return httpx.Response(
                202,
                json={"message": "Bulk upsert request is accepted"},
            )
        return httpx.Response(404, json={"message": f"unexpected {path}"})

    def _collection(self) -> dict[str, Any]:
        return {
            "projectName": "demo",
            "collectionName": "smoke",
            "indexConfigs": {
                "dense": {"type": "vector", "dimensions": 2, "similarity": "cosine"}
            },
            "numPartitions": 1,
            "numDocs": self.num_docs,
            "description": "",
            "tags": {},
            "defaultBranchName": "main",
            "snapshotRetentionInDays": 7,
            "createdAt": 1,
            "updatedAt": 2,
            "dataUpdatedAt": 1790121600000,
        }


def make_sdk_adapter(server: DevelopServer) -> LambdaDBAdapter:
    def client_factory(**kwargs: Any) -> LambdaDB:
        http_client = httpx.Client(transport=httpx.MockTransport(server))
        return LambdaDB(**kwargs, client=http_client)

    return LambdaDBAdapter(
        client_factory=client_factory,
        environ={"LAMBDADB_API_KEY": "secret"},
    )


def test_sdk_prepare_create_accepts_develop_created_response() -> None:
    server = DevelopServer()
    adapter = make_sdk_adapter(server)
    target = make_target(prepare={"mode": "create"}, create_wait_poll_seconds=0.001)

    result = adapter.prepare(target, dimensions=2, metric="cosine")

    assert result.ok
    assert [(request.method, request.url.path) for request in server.requests] == [
        ("POST", "/projects/demo/collections"),
        ("GET", "/projects/demo/collections/smoke"),
    ]


def test_sdk_collection_stats_reads_develop_num_docs() -> None:
    adapter = make_sdk_adapter(DevelopServer(num_docs=3))

    stats = adapter.collection_stats(make_target())

    assert stats == CollectionStats(
        ready=True,
        num_docs=3,
        status=None,
        data_updated_at=1790121600000,
        supports_data_updated_at=True,
    )


def test_sdk_bulk_upsert_sends_signed_upload_headers() -> None:
    server = DevelopServer()
    adapter = make_sdk_adapter(server)

    adapter.upsert_batch(
        make_target(),
        [{"id": "a", "vector": [0.1, 0.2], "metadata": {}}],
        write_mode="bulk_upsert",
    )

    uploads = [request for request in server.requests if request.method == "PUT"]
    assert len(uploads) == 1
    assert uploads[0].headers["If-None-Match"] == "*"


def test_sdk_query_reports_develop_server_took() -> None:
    adapter = make_sdk_adapter(DevelopServer())

    result = adapter.query(
        make_target(),
        vector=[0.1, 0.2],
        top_k=1,
        consistency="eventual",
    )

    assert [match.id for match in result.matches] == ["a"]
    assert result.server_took_ms == 7.0


def test_operations_require_api_key_env() -> None:
    adapter = LambdaDBAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    with pytest.raises(ConfigError, match="LAMBDADB_API_KEY"):
        adapter.prepare(make_target())


@pytest.mark.skipif(
    os.getenv("LAMBDADB_BENCH_RUN_INTEGRATION") != "1",
    reason="set LAMBDADB_BENCH_RUN_INTEGRATION=1 to run LambdaDB integration tests",
)
def test_lambdadb_integration_existing_collection_check() -> None:
    required = [
        "LAMBDADB_API_KEY",
        "LAMBDADB_ENDPOINT",
        "LAMBDADB_PROJECT_NAME",
        "LAMBDADB_COLLECTION_NAME",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing LambdaDB integration env vars: {', '.join(missing)}")

    adapter = LambdaDBAdapter()
    target = TargetConfig.from_mapping(
        {
            "vendor": "lambdadb",
            "name": "lambda-integration",
            "endpoint": os.environ["LAMBDADB_ENDPOINT"],
            "project_name": os.environ["LAMBDADB_PROJECT_NAME"],
            "api_key_env": "LAMBDADB_API_KEY",
            "collection_name": os.environ["LAMBDADB_COLLECTION_NAME"],
            "prepare": {"mode": "existing"},
        }
    )

    assert adapter.check(target).ok
    assert adapter.prepare(target).ok
