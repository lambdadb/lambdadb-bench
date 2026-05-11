from __future__ import annotations

import os
from typing import Any
from uuid import UUID

import pytest

from ldbbench.adapters.qdrant import (
    SOURCE_ID_PAYLOAD_KEY,
    QdrantAdapter,
    _qdrant_point_id,
)
from ldbbench.config import ConfigError, TargetConfig


class FakePoint:
    def __init__(
        self,
        *,
        point_id: str,
        payload: dict[str, Any] | None = None,
        vector: Any = None,
        score: float | None = None,
    ) -> None:
        self.id = point_id
        self.payload = payload
        self.vector = vector
        self.score = score


class FakeQueryResponse:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.collection_exists_calls: list[dict[str, Any]] = []
        self.get_collection_calls: list[dict[str, Any]] = []
        self.create_collection_calls: list[dict[str, Any]] = []
        self.recreate_collection_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_points_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []

    def collection_exists(self, **kwargs: Any) -> bool:
        self.collection_exists_calls.append(kwargs)
        return self.exists

    def get_collection(self, **kwargs: Any) -> dict[str, Any]:
        self.get_collection_calls.append(kwargs)
        return {"name": kwargs["collection_name"]}

    def create_collection(self, **kwargs: Any) -> bool:
        self.create_collection_calls.append(kwargs)
        return True

    def recreate_collection(self, **kwargs: Any) -> bool:
        self.recreate_collection_calls.append(kwargs)
        return True

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self.upsert_calls.append(kwargs)
        return {"status": "completed"}

    def query_points(self, **kwargs: Any) -> FakeQueryResponse:
        self.query_points_calls.append(kwargs)
        return FakeQueryResponse(
            [
                FakePoint(
                    point_id=_qdrant_point_id("a"),
                    payload={SOURCE_ID_PAYLOAD_KEY: "a", "text": "alpha"},
                    score=0.9,
                ),
                FakePoint(
                    point_id=_qdrant_point_id("b"),
                    payload={SOURCE_ID_PAYLOAD_KEY: "b", "text": "beta"},
                    score=0.8,
                ),
            ]
        )

    def retrieve(self, **kwargs: Any) -> list[FakePoint]:
        self.retrieve_calls.append(kwargs)
        original_ids = {
            _qdrant_point_id("a"): "a",
            _qdrant_point_id("b"): "b",
        }
        return [
            FakePoint(
                point_id=item,
                payload={
                    SOURCE_ID_PAYLOAD_KEY: original_ids[item],
                    "text": original_ids[item],
                },
                vector=[0.1],
            )
            for item in kwargs["ids"]
        ]


def make_target(**overrides: Any) -> TargetConfig:
    data = {
        "vendor": "qdrant",
        "name": "qdrant-ci",
        "endpoint": "https://example.qdrant.io",
        "api_key_env": "QDRANT_API_KEY",
        "collection_name": "smoke",
        "prepare": {"mode": "existing"},
    }
    data.update(overrides)
    return TargetConfig.from_mapping(data)


def make_adapter(client: FakeClient) -> QdrantAdapter:
    return QdrantAdapter(
        client_factory=lambda **_kwargs: client,
        environ={"QDRANT_API_KEY": "secret"},
    )


def test_check_validates_qdrant_metadata_without_requiring_api_key() -> None:
    adapter = QdrantAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    result = adapter.check(make_target())

    assert result.ok
    assert result.details["collection_name"] == "smoke"
    assert result.details["prefer_grpc"] is True
    assert result.details["api_key_present"] is False


def test_check_requires_endpoint_and_collection() -> None:
    adapter = QdrantAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    assert not adapter.check(make_target(endpoint=None)).ok
    assert not adapter.check(make_target(collection_name=None)).ok


def test_prepare_existing_checks_collection() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.prepare(make_target())

    assert result.ok
    assert client.collection_exists_calls == [{"collection_name": "smoke"}]
    assert client.get_collection_calls == [{"collection_name": "smoke"}]


def test_prepare_existing_fails_when_collection_is_missing() -> None:
    adapter = make_adapter(FakeClient(exists=False))

    with pytest.raises(ConfigError, match="does not exist"):
        adapter.prepare(make_target())


def test_prepare_create_uses_unnamed_vector_by_default() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(prepare={"mode": "create"})

    result = adapter.prepare(target, dimensions=1024, metric="cosine")

    assert result.ok
    call = client.create_collection_calls[0]
    assert call["collection_name"] == "smoke"
    assert call["vectors_config"].size == 1024
    assert call["vectors_config"].distance.value == "Cosine"


def test_prepare_recreate_uses_named_vector_when_configured() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "recreate"},
        vector_field="dense",
    )

    result = adapter.prepare(target, dimensions=128, metric="dot")

    assert result.ok
    call = client.recreate_collection_calls[0]
    assert call["collection_name"] == "smoke"
    assert call["vectors_config"]["dense"].size == 128
    assert call["vectors_config"]["dense"].distance.value == "Dot"


def test_upsert_batch_maps_records_to_qdrant_points() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.upsert_batch(
        make_target(vector_field="dense"),
        [
            {
                "id": "20231101.en_34151850_6",
                "vector": [0.1, 0.2],
                "metadata": {"text": "alpha"},
            },
            {"id": "b", "vector": [0.3, 0.4], "metadata": {}},
        ],
    )

    assert result.count == 2
    call = client.upsert_calls[0]
    assert call["collection_name"] == "smoke"
    assert call["wait"] is True
    UUID(call["points"][0].id)
    assert call["points"][0].id == _qdrant_point_id("20231101.en_34151850_6")
    assert call["points"][0].vector == {"dense": [0.1, 0.2]}
    assert call["points"][0].payload == {
        SOURCE_ID_PAYLOAD_KEY: "20231101.en_34151850_6",
        "text": "alpha",
    }


def test_query_uses_eventual_consistency_and_grpc_client() -> None:
    created_clients: list[dict[str, Any]] = []
    client = FakeClient()
    adapter = QdrantAdapter(
        client_factory=lambda **kwargs: created_clients.append(kwargs) or client,
        environ={"QDRANT_API_KEY": "secret"},
    )

    result = adapter.query(
        make_target(vector_field="dense", prefer_grpc=True),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="eventual",
        filter_query={"must": []},
    )

    assert [match.id for match in result.matches] == ["a", "b"]
    assert result.matches[0].document == {"id": "a", "text": "alpha"}
    assert created_clients == [
        {
            "url": "https://example.qdrant.io",
            "api_key": "secret",
            "prefer_grpc": True,
        }
    ]
    assert client.query_points_calls == [
        {
            "collection_name": "smoke",
            "query": [0.1, 0.2],
            "using": "dense",
            "query_filter": {"must": []},
            "limit": 2,
            "with_payload": True,
            "with_vectors": False,
        }
    ]


def test_query_rejects_strong_consistency() -> None:
    adapter = make_adapter(FakeClient())

    with pytest.raises(ConfigError, match="eventual"):
        adapter.query(
            make_target(),
            vector=[0.1],
            top_k=1,
            consistency="strong",
        )


def test_fetch_retrieves_points_by_id() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    docs = adapter.fetch(
        make_target(),
        ids=["a", "b"],
        consistency="eventual",
        include_vectors=True,
    )

    assert docs == [
        {"id": "a", "text": "a", "vector": [0.1]},
        {"id": "b", "text": "b", "vector": [0.1]},
    ]
    assert client.retrieve_calls == [
        {
            "collection_name": "smoke",
            "ids": [_qdrant_point_id("a"), _qdrant_point_id("b")],
            "with_payload": True,
            "with_vectors": True,
        }
    ]


def test_operations_require_api_key_when_api_key_env_is_set() -> None:
    adapter = QdrantAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    with pytest.raises(ConfigError, match="QDRANT_API_KEY"):
        adapter.prepare(make_target())


@pytest.mark.skipif(
    os.getenv("QDRANT_BENCH_RUN_INTEGRATION") != "1",
    reason="set QDRANT_BENCH_RUN_INTEGRATION=1 to run Qdrant integration tests",
)
def test_qdrant_integration_existing_collection_check() -> None:
    required = ["QDRANT_ENDPOINT", "QDRANT_COLLECTION_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing Qdrant integration env vars: {', '.join(missing)}")

    adapter = QdrantAdapter()
    target = TargetConfig.from_mapping(
        {
            "vendor": "qdrant",
            "name": "qdrant-integration",
            "endpoint": os.environ["QDRANT_ENDPOINT"],
            "api_key_env": "QDRANT_API_KEY"
            if os.getenv("QDRANT_API_KEY")
            else None,
            "collection_name": os.environ["QDRANT_COLLECTION_NAME"],
            "prepare": {"mode": "existing"},
        }
    )

    assert adapter.check(target).ok
    assert adapter.prepare(target).ok
