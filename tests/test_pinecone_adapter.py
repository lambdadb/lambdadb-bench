from __future__ import annotations

import os
from typing import Any

import pytest

from ldbbench.adapters.pinecone import PineconeAdapter
from ldbbench.config import ConfigError, TargetConfig


class FakeMatch:
    def __init__(
        self,
        *,
        match_id: str,
        metadata: dict[str, Any] | None = None,
        values: list[float] | None = None,
        score: float | None = None,
    ) -> None:
        self.id = match_id
        self.metadata = metadata or {}
        self.values = values
        self.score = score


class FakeVector:
    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        values: list[float] | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self.values = values


class FakeQueryResponse:
    def __init__(self, matches: list[FakeMatch]) -> None:
        self.matches = matches


class FakeFetchResponse:
    def __init__(self, vectors: dict[str, FakeVector]) -> None:
        self.vectors = vectors


class FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self.upserts.append(kwargs)
        return {"upserted_count": len(kwargs["vectors"])}

    def query(self, **kwargs: Any) -> FakeQueryResponse:
        self.queries.append(kwargs)
        return FakeQueryResponse(
            [
                FakeMatch(match_id="a", metadata={"text": "alpha"}, score=0.9),
                FakeMatch(match_id="b", metadata={"text": "beta"}, score=0.8),
            ]
        )

    def fetch(self, **kwargs: Any) -> FakeFetchResponse:
        self.fetches.append(kwargs)
        return FakeFetchResponse(
            {
                "a": FakeVector(metadata={"text": "alpha"}, values=[0.1]),
                "b": FakeVector(metadata={"text": "beta"}, values=[0.2]),
            }
        )


class FakeClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.index = FakeIndex()
        self.has_index_calls: list[str] = []
        self.describe_index_calls: list[str] = []
        self.create_index_calls: list[dict[str, Any]] = []
        self.delete_index_calls: list[dict[str, Any]] = []
        self.index_calls: list[dict[str, Any]] = []

    def has_index(self, name: str) -> bool:
        self.has_index_calls.append(name)
        return self.exists

    def describe_index(self, name: str) -> dict[str, Any]:
        self.describe_index_calls.append(name)
        return {"name": name, "host": "https://index-host.pinecone.io"}

    def create_index(self, **kwargs: Any) -> dict[str, Any]:
        self.create_index_calls.append(kwargs)
        return {"name": kwargs["name"]}

    def delete_index(self, **kwargs: Any) -> None:
        self.delete_index_calls.append(kwargs)

    def Index(self, **kwargs: Any) -> FakeIndex:  # noqa: N802
        self.index_calls.append(kwargs)
        return self.index


def make_target(**overrides: Any) -> TargetConfig:
    data = {
        "vendor": "pinecone",
        "name": "pinecone-ci",
        "api_key_env": "PINECONE_API_KEY",
        "collection_name": "smoke-index",
        "region": "us-east-1",
        "cloud": "aws",
        "namespace": "bench",
        "prepare": {"mode": "existing"},
    }
    data.update(overrides)
    return TargetConfig.from_mapping(data)


def make_adapter(client: FakeClient) -> PineconeAdapter:
    return PineconeAdapter(
        client_factory=lambda **_kwargs: client,
        environ={"PINECONE_API_KEY": "secret"},
    )


def test_check_validates_pinecone_metadata_without_requiring_api_key() -> None:
    adapter = PineconeAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    result = adapter.check(make_target())

    assert result.ok
    assert result.details["index_name"] == "smoke-index"
    assert result.details["index_host_present"] is False
    assert result.details["api_key_present"] is False
    assert result.details["namespace"] == "bench"


def test_check_requires_index_name() -> None:
    adapter = PineconeAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    result = adapter.check(make_target(collection_name=None))

    assert not result.ok
    assert "collection_name" in result.message


def test_prepare_existing_checks_index() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.prepare(make_target())

    assert result.ok
    assert client.has_index_calls == ["smoke-index"]
    assert client.describe_index_calls == ["smoke-index"]


def test_adapter_reuses_client_and_index_for_same_target() -> None:
    clients: list[FakeClient] = []
    client = FakeClient()

    def factory(**_kwargs: Any) -> FakeClient:
        clients.append(client)
        return client

    adapter = PineconeAdapter(
        client_factory=factory,
        environ={"PINECONE_API_KEY": "secret"},
    )
    target = make_target()

    adapter.prepare(target)
    adapter.upsert_batch(target, [{"id": "a", "vector": [0.1], "metadata": {}}])
    adapter.query(target, vector=[0.1], top_k=1, consistency="eventual")
    adapter.fetch(target, ids=["a"], consistency="eventual")

    assert len(clients) == 1
    assert client.index_calls == [{"name": "smoke-index"}]


def test_adapter_can_use_explicit_index_host_when_configured() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    adapter.upsert_batch(
        make_target(endpoint="https://index-host.pinecone.io"),
        [{"id": "a", "vector": [0.1], "metadata": {}}],
    )

    assert client.index_calls == [{"host": "https://index-host.pinecone.io"}]


def test_prepare_existing_fails_when_index_is_missing() -> None:
    adapter = make_adapter(FakeClient(exists=False))

    with pytest.raises(ConfigError, match="does not exist"):
        adapter.prepare(make_target())


def test_prepare_create_uses_serverless_spec() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "create"},
        tags={"env": "test"},
    )

    result = adapter.prepare(target, dimensions=1024, metric="dot")

    assert result.ok
    call = client.create_index_calls[0]
    assert call["name"] == "smoke-index"
    assert call["dimension"] == 1024
    assert call["metric"] == "dotproduct"
    assert call["spec"].cloud == "aws"
    assert call["spec"].region == "us-east-1"
    assert call["timeout"] == 600
    assert call["deletion_protection"] == "disabled"
    assert call["vector_type"] == "dense"
    assert call["tags"] == {"env": "test"}


def test_prepare_recreate_deletes_before_create() -> None:
    client = FakeClient()
    adapter = make_adapter(client)
    target = make_target(
        prepare={"mode": "recreate"},
        delete_timeout_seconds=10,
        create_timeout_seconds=20,
    )

    result = adapter.prepare(target, dimensions=128, metric="cosine")

    assert result.ok
    assert client.delete_index_calls == [{"name": "smoke-index", "timeout": 10}]
    assert client.create_index_calls[0]["timeout"] == 20


def test_upsert_batch_maps_normalized_records_to_vectors() -> None:
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
    assert client.index_calls == [{"name": "smoke-index"}]
    assert client.index.upserts == [
        {
            "vectors": [
                {"id": "a", "values": [0.1, 0.2], "metadata": {"text": "alpha"}},
                {"id": "b", "values": [0.3, 0.4], "metadata": {}},
            ],
            "namespace": "bench",
        }
    ]


def test_query_uses_eventual_consistency() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    result = adapter.query(
        make_target(pool_threads=10, connection_pool_maxsize=20),
        vector=[0.1, 0.2],
        top_k=2,
        consistency="eventual",
        include_vectors=True,
        filter_query={"lang": {"$eq": "en"}},
    )

    assert [match.id for match in result.matches] == ["a", "b"]
    assert result.matches[0].document == {"id": "a", "text": "alpha"}
    assert client.index_calls == [
        {
            "name": "smoke-index",
            "pool_threads": 10,
            "connection_pool_maxsize": 20,
        }
    ]
    assert client.index.queries == [
        {
            "vector": [0.1, 0.2],
            "top_k": 2,
            "namespace": "bench",
            "include_values": True,
            "include_metadata": True,
            "filter": {"lang": {"$eq": "en"}},
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


def test_fetch_returns_documents_by_id() -> None:
    client = FakeClient()
    adapter = make_adapter(client)

    docs = adapter.fetch(
        make_target(),
        ids=["a", "b"],
        consistency="eventual",
        include_vectors=True,
    )

    assert docs == [
        {"id": "a", "text": "alpha", "vector": [0.1]},
        {"id": "b", "text": "beta", "vector": [0.2]},
    ]
    assert client.index_calls == [{"name": "smoke-index"}]
    assert client.index.fetches == [{"ids": ["a", "b"], "namespace": "bench"}]


def test_operations_require_api_key_env() -> None:
    adapter = PineconeAdapter(
        client_factory=lambda **_kwargs: FakeClient(),
        environ={},
    )

    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        adapter.prepare(make_target())


@pytest.mark.skipif(
    os.getenv("PINECONE_BENCH_RUN_INTEGRATION") != "1",
    reason="set PINECONE_BENCH_RUN_INTEGRATION=1 to run Pinecone integration tests",
)
def test_pinecone_integration_existing_index_check() -> None:
    required = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing Pinecone integration env vars: {', '.join(missing)}")

    adapter = PineconeAdapter()
    target = TargetConfig.from_mapping(
        {
            "vendor": "pinecone",
            "name": "pinecone-integration",
            "api_key_env": "PINECONE_API_KEY",
            "collection_name": os.environ["PINECONE_INDEX_NAME"],
            "region": os.getenv("PINECONE_REGION", "us-east-1"),
            "prepare": {"mode": "existing"},
        }
    )

    assert adapter.check(target).ok
    assert adapter.prepare(target).ok
