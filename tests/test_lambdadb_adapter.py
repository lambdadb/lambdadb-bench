from __future__ import annotations

import os
from typing import Any

import pytest

from ldbbench.adapters.lambdadb import LambdaDBAdapter
from ldbbench.config import ConfigError, TargetConfig


class FakeDocs:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self.upserts.append(kwargs)
        return {"ok": True}

    def fetch(self, **kwargs: Any) -> dict[str, Any]:
        self.fetches.append(kwargs)
        return {"docs": [{"doc": {"id": item}} for item in kwargs["ids"]]}


class FakeCollections:
    def __init__(self, *, missing_after_gets: int | None = None) -> None:
        self.docs = FakeDocs()
        self.creates: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.missing_after_gets = missing_after_gets

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.creates.append(kwargs)
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
        return {"name": kwargs["collection_name"]}

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.deletes.append(kwargs)
        return {"deleted": kwargs["collection_name"]}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        return {
            "docs": [
                {"doc": {"id": "a"}, "score": 0.91},
                {"doc": {"id": "b"}, "score": 0.42},
            ]
        }


class FakeClient:
    def __init__(self, *, missing_after_gets: int | None = None) -> None:
        self.collections = FakeCollections(missing_after_gets=missing_after_gets)


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
                }
            },
        }
    ]


def test_prepare_existing_checks_collection() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.prepare(make_target())

    assert result.ok
    assert client.collections.gets == [{"collection_name": "smoke"}]


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


def test_prepare_recreate_deletes_before_create() -> None:
    client = FakeClient(missing_after_gets=2)
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "recreate"},
        index_configs={"dense": {"type": "vector", "dimensions": 3}},
        delete_wait_timeout_seconds=1,
        delete_wait_poll_seconds=0.001,
    )

    result = adapter.prepare(target)

    assert result.ok
    assert client.collections.deletes == [{"collection_name": "smoke"}]
    assert client.collections.gets == [
        {"collection_name": "smoke"},
        {"collection_name": "smoke"},
    ]
    assert client.collections.creates == [
        {
            "collection_name": "smoke",
            "index_configs": {"dense": {"type": "vector", "dimensions": 3}},
        }
    ]


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
            "collection_name": "smoke",
            "docs": [
                {"id": "a", "dense": [0.1, 0.2], "metadata": {"text": "alpha"}},
                {"id": "b", "dense": [0.3, 0.4], "metadata": {}},
            ],
        }
    ]


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
