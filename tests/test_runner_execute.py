from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any

import msgpack
import pytest

from ldbbench.adapters.base import (
    AdapterCapabilities,
    CheckResult,
    DeleteResult,
    PrepareResult,
    QueryMatch,
    QueryResult,
    UpsertResult,
    VectorRecord,
)
from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets.deletion import prepare_deletion_plan
from ldbbench.datasets.ground_truth import prepare_ground_truth
from ldbbench.datasets.prepare import optimize_dataset, prepare_dataset
from ldbbench.runner.execute import (
    _batches,
    _dataset_metric,
    _ground_truth_path,
    _record_size_bytes,
    _split_concurrency,
    execute_benchmark,
    latency_summary,
    load_ground_truth,
    parse_size_bytes,
    read_records,
    recall_at_k,
)


class FakeAdapter:
    vendor = "fake"
    capabilities = AdapterCapabilities(
        supported_write_modes=frozenset({"upsert", "bulk_upsert"}),
        supported_query_consistency=frozenset({"eventual"}),
        supports_query_filter=True,
        supports_query_partition_filter=True,
        supports_full_text_search=True,
        supports_delete_by_id=True,
    )

    def __init__(
        self,
        *,
        fail_load_batch: int = 0,
        fail_every: int = 0,
        load_delay_seconds: float = 0.0,
        query_delay_seconds: float = 0.0,
        stale_deleted_fetch: bool = False,
    ) -> None:
        self.prepared: dict[str, Any] | None = None
        self.upserted: list[list[VectorRecord]] = []
        self.write_modes: list[str] = []
        self.queries: list[list[float]] = []
        self.filter_queries: list[dict[str, Any] | None] = []
        self.partition_filters: list[dict[str, Any] | None] = []
        self.full_text_queries: list[dict[str, Any]] = []
        self.deleted_batches: list[list[str]] = []
        self.deleted_ids: set[str] = set()
        self.fail_load_batch = fail_load_batch
        self.fail_every = fail_every
        self.load_delay_seconds = load_delay_seconds
        self.query_delay_seconds = query_delay_seconds
        self.stale_deleted_fetch = stale_deleted_fetch
        self.query_calls = 0
        self.load_calls = 0
        self._lock = Lock()

    def check(self, target: TargetConfig) -> CheckResult:
        return CheckResult(ok=True, message="ok")

    def prepare(
        self,
        target: TargetConfig,
        *,
        dimensions: int | None = None,
        metric: str | None = None,
    ) -> PrepareResult:
        self.prepared = {
            "target": target.name,
            "dimensions": dimensions,
            "metric": metric,
        }
        return PrepareResult(ok=True, message="prepared")

    def upsert_batch(
        self,
        target: TargetConfig,
        records: list[VectorRecord],
        *,
        write_mode: str = "upsert",
    ) -> UpsertResult:
        if self.load_delay_seconds:
            time.sleep(self.load_delay_seconds)
        with self._lock:
            self.load_calls += 1
            load_call = self.load_calls
            self.upserted.append(list(records))
            self.write_modes.append(write_mode)
        if self.fail_load_batch and load_call == self.fail_load_batch:
            raise RuntimeError("planned load failure")
        return UpsertResult(count=len(records))

    def query(
        self,
        target: TargetConfig,
        *,
        vector: list[float],
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        filter_query: dict[str, Any] | None = None,
        partition_filter: dict[str, Any] | None = None,
    ) -> QueryResult:
        if self.query_delay_seconds:
            time.sleep(self.query_delay_seconds)
        with self._lock:
            self.query_calls += 1
            call_number = self.query_calls
            self.queries.append(vector)
            self.filter_queries.append(filter_query)
            self.partition_filters.append(partition_filter)
        if self.fail_every and call_number % self.fail_every == 0:
            raise RuntimeError("planned query failure")
        scored = []
        for batch in self.upserted:
            for record in batch:
                if not _matches_portable_filter(record, filter_query):
                    continue
                scored.append(
                    (
                        sum(q * r for q, r in zip(vector, record.vector, strict=True)),
                        record.id,
                    )
                )
        if scored:
            return QueryResult(
                matches=[
                    QueryMatch(id=record_id, score=score, document={"id": record_id})
                    for score, record_id in sorted(
                        scored,
                        key=lambda item: (-item[0], item[1]),
                    )[:top_k]
                ]
            )
        return QueryResult(
            matches=[
                QueryMatch(id="a", score=1.0, document={"id": "a"}),
                QueryMatch(id="c", score=0.9, document={"id": "c"}),
            ][:top_k]
        )

    def delete_batch(
        self,
        target: TargetConfig,
        ids: list[str],
    ) -> DeleteResult:
        self.deleted_batches.append(list(ids))
        deleted = set(ids)
        self.deleted_ids.update(deleted)
        self.upserted = [
            [record for record in batch if record.id not in deleted]
            for batch in self.upserted
        ]
        return DeleteResult(count=len(ids))

    def full_text_query(
        self,
        target: TargetConfig,
        *,
        query_text: str,
        field: str,
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        partition_filter: dict[str, Any] | None = None,
    ) -> QueryResult:
        if self.query_delay_seconds:
            time.sleep(self.query_delay_seconds)
        with self._lock:
            self.query_calls += 1
            call_number = self.query_calls
            self.full_text_queries.append(
                {
                    "query_text": query_text,
                    "field": field,
                    "top_k": top_k,
                    "consistency": consistency,
                    "include_vectors": include_vectors,
                    "partition_filter": partition_filter,
                }
            )
        if self.fail_every and call_number % self.fail_every == 0:
            raise RuntimeError("planned query failure")
        terms = {term.lower() for term in query_text.split()}
        matches: list[QueryMatch] = []
        for batch in self.upserted:
            for record in batch:
                text = record.metadata.get("text")
                if not isinstance(text, str):
                    continue
                if terms.intersection(text.lower().split()):
                    matches.append(
                        QueryMatch(
                            id=record.id,
                            score=1.0,
                            document={"id": record.id},
                        )
                    )
        return QueryResult(matches=matches[:top_k])

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: list[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            {"id": item}
            for item in ids
            if self.stale_deleted_fetch or item not in self.deleted_ids
        ]


def make_scenario(
    *,
    rows: int = 3,
    stages: list[dict[str, Any]] | None = None,
    batch_size: int = 2,
    load_concurrency: int | None = None,
    wait_until_query_visible: bool = False,
    partition_filter: bool = False,
    query_filter: bool = False,
    top_k: int = 2,
    write_mode: str = "upsert",
    sharded_records: bool = False,
    shard_count: int | None = None,
    max_batch_bytes: str | None = "1MB",
    workload: str = "standard",
    search_under_ingest: dict[str, Any] | None = None,
    full_text: bool = False,
    metric: str = "cosine",
    delete_order: str | None = None,
) -> ScenarioConfig:
    query: dict[str, Any] = {
        "top_k": top_k,
        "query_count": 1,
        "consistency": "eventual",
    }
    if stages is not None:
        query["stages"] = stages
    if partition_filter:
        query["partition_filter"] = {
            "field": "metadata.url",
            "metadata_field": "url",
        }
    if query_filter:
        query["filter"] = {
            "name": "synthetic_bucket_50pct",
            "field": "filter_bucket_2",
            "operator": "eq",
            "value_source": {
                "type": "eligible_record_buckets",
                "min_candidates": "top_k",
            },
            "expected_selectivity": 0.5,
        }
    if full_text:
        workload = "full_text_search"
        query["full_text"] = {
            "field": "metadata.text",
            "metadata_field": "text",
            "max_terms": 3,
        }
    mapping: dict[str, Any] = {
        "name": "runner-smoke",
        "workload": workload,
        "dataset": {
            "provider": "huggingface",
            "source": "demo/source",
            "rows": rows,
            "dimensions": 2,
            "id_field": "_id",
            "vector_field": "emb",
            "text_field": "text",
            "metric": metric,
        },
        "load": {
            "write_mode": write_mode,
            "batch_size": batch_size,
            "wait_until_query_visible": wait_until_query_visible,
            "query_visibility_timeout": "50ms",
            "query_visibility_poll_interval": "1ms",
        },
        "query": query,
    }
    if search_under_ingest is not None:
        mapping["search_under_ingest"] = search_under_ingest
    if delete_order is not None:
        mapping["delete"] = {
            "order": delete_order,
            "seed": 0,
            "checkpoints_pct": [50, 75],
            "batch_size": 1,
        }
    if max_batch_bytes is not None:
        mapping["load"]["max_batch_bytes"] = max_batch_bytes
    if load_concurrency is not None:
        mapping["load"]["concurrency"] = load_concurrency
    if sharded_records:
        mapping["load"]["sharded_records"] = True
    if shard_count is not None:
        mapping["load"]["shard_count"] = shard_count
    return ScenarioConfig.from_mapping(mapping)


def _matches_portable_filter(
    record: VectorRecord,
    filter_query: dict[str, Any] | None,
) -> bool:
    if filter_query is None:
        return True
    if not {"field", "operator", "value"}.issubset(filter_query):
        return True
    if filter_query["operator"] != "eq":
        return False
    return str(record.metadata.get(str(filter_query["field"]))) == str(
        filter_query["value"]
    )


def make_target() -> TargetConfig:
    return TargetConfig.from_mapping(
        {
            "vendor": "fake",
            "name": "fake-target",
            "endpoint": "memory://fake",
            "prepare": {"mode": "existing"},
        }
    )


def write_configs(tmp_path, scenario: ScenarioConfig, target: TargetConfig):
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    scenario_path.write_text(
        """
name: runner-smoke
dataset:
  provider: huggingface
  source: demo/source
  rows: 3
  dimensions: 2
  id_field: _id
  vector_field: emb
  text_field: text
  metric: cosine
load:
  write_mode: upsert
  batch_size: 2
query:
  top_k: 2
  query_count: 1
  consistency: eventual
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: fake
name: fake-target
endpoint: memory://fake
prepare:
  mode: existing
""",
        encoding="utf-8",
    )
    return scenario_path, target_path


def prepare_fixture_dataset(
    tmp_path,
    scenario: ScenarioConfig,
    *,
    limit: int = 3,
    query_count: int = 1,
):
    rows = [
        {"_id": "query", "emb": [1.0, 0.0], "text": "query"},
        {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
        {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
        {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
    ]
    return prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=limit,
        query_count=query_count,
        source_rows=rows,
    )


def test_execute_benchmark_writes_events_and_summary(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]
    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.prepared == {
        "target": "fake-target",
        "dimensions": 2,
        "metric": "cosine",
    }
    assert [len(batch) for batch in adapter.upserted] == [2, 1]
    assert adapter.write_modes == ["upsert", "upsert"]
    assert len(adapter.queries) == 1
    assert [event["records"] for event in ingest_events] == [2, 1]
    assert [event["write_mode"] for event in ingest_events] == ["upsert", "upsert"]
    assert query_events[0]["query_id"] == "query"
    assert query_events[0]["matches"] == ["a", "c"]
    assert query_events[0]["recall_at_k"] == 1.0
    assert result.summary["load"]["records"] == 3
    assert result.summary["load"]["records_read"] == 3
    assert result.summary["load"]["write_mode"] == "upsert"
    assert result.summary["load"]["record_source"]["format"] == "msgpack"
    assert result.summary["load"]["batching_duration_seconds"] >= 0
    assert result.summary["load"]["batching_records_per_second"] >= 0
    assert result.summary["load"]["upsert_attempt_duration_seconds"] >= 0
    assert result.summary["load"]["attempt_latency_ms"]["max"] is not None
    assert result.summary["query"]["queries"] == 1
    assert result.summary["query"]["recall_at_k"] == 1.0
    assert result.summary_path.exists()


def test_execute_benchmark_forwards_bulk_write_mode(tmp_path) -> None:
    scenario = make_scenario(write_mode="bulk_upsert")
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, query_count=0)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.write_modes == ["bulk_upsert", "bulk_upsert"]
    assert [event["write_mode"] for event in ingest_events] == [
        "bulk_upsert",
        "bulk_upsert",
    ]
    assert result.summary["load"]["write_mode"] == "bulk_upsert"


def test_execute_benchmark_backfills_filter_buckets_for_legacy_load_cache(
    tmp_path,
) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, query_count=1)
    stripped_records = []
    with dataset.records_msgpack_path.open("rb") as file:
        for record in msgpack.Unpacker(file, raw=False):
            record["metadata"] = {
                key: value
                for key, value in record["metadata"].items()
                if not key.startswith("filter_bucket_")
            }
            stripped_records.append(record)
    packer = msgpack.Packer(use_bin_type=True, use_single_float=True)
    with dataset.records_msgpack_path.open("wb") as file:
        for record in stripped_records:
            file.write(packer.pack(record))
    adapter = FakeAdapter()

    execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    metadata = adapter.upserted[0][0].metadata
    assert metadata["filter_bucket_2"]
    assert metadata["filter_bucket_10"]
    assert metadata["filter_bucket_100"]
    assert metadata["filter_bucket_1000"]


def test_execute_benchmark_loads_sharded_records(tmp_path) -> None:
    scenario = make_scenario(
        batch_size=1,
        sharded_records=True,
        shard_count=2,
        max_batch_bytes=None,
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    optimize_dataset(dataset_dir=dataset.output_dir, shards=2)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [batch[0].id for batch in adapter.upserted] == ["a", "b", "c"]
    assert [event["batch_index"] for event in ingest_events] == [1, 2, 3]
    assert result.summary["load"]["sharded_records"] is True
    assert result.summary["load"]["shard_count"] == 2
    assert result.summary["load"]["effective_shard_count"] == 2
    assert result.summary["load"]["manifest_shard_count"] == 2
    assert result.summary["load"]["worker_shards"] == [2]
    assert result.summary["load"]["records"] == 3
    assert result.summary["load"]["records_read"] == 3
    assert result.summary["load"]["record_source"]["sharded"] is True
    assert result.summary["load"]["record_source"]["effective_shards"] == 2
    assert result.summary["load"]["record_source"]["manifest_shards"] == 2


def test_execute_benchmark_rejects_sharded_load_with_max_batch_bytes(
    tmp_path,
) -> None:
    scenario = make_scenario(
        sharded_records=True,
        shard_count=2,
        max_batch_bytes="1MB",
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    optimize_dataset(dataset_dir=dataset.output_dir, shards=2)

    with pytest.raises(ConfigError, match="max_batch_bytes"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
            load_only=True,
        )


def test_execute_benchmark_applies_partition_filter_and_skips_recall(tmp_path) -> None:
    scenario = make_scenario(partition_filter=True)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=3,
        query_count=1,
        source_rows=[
            {"_id": "query", "emb": [1.0, 0.0], "text": "query", "url": "q-url"},
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha", "url": "a-url"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "beta", "url": "b-url"},
            {"_id": "c", "emb": [0.8, 0.2], "text": "gamma", "url": "c-url"},
        ],
    )
    prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.partition_filters == [{"field": "metadata.url", "in_": ["q-url"]}]
    assert query_events[0]["partition_filter"] == {
        "field": "metadata.url",
        "in_": ["q-url"],
    }
    assert query_events[0]["recall_at_k"] is None
    assert query_events[0]["recall_skip_reason"] == "partition_filtered"
    assert result.summary["query"]["recall_at_k"] is None
    assert result.summary["query"]["recall_samples"] == 0
    assert result.summary["query"]["partition_filter_applied"] is True
    assert result.summary["query"]["recall_skip_reason"] == "partition_filtered"


def test_execute_benchmark_ignores_default_vector_ground_truth_for_full_text(
    tmp_path,
) -> None:
    scenario = make_scenario(full_text=True, top_k=2)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=3,
        query_count=1,
        source_rows=[
            {
                "_id": "query",
                "emb": [1.0, 0.0],
                "text": "Alpha beta gamma delta",
            },
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha reference"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "unrelated text"},
            {"_id": "c", "emb": [0.8, 0.2], "text": "beta reference"},
        ],
    )
    prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=1)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.full_text_queries == [
        {
            "query_text": "Alpha beta gamma",
            "field": "metadata.text",
            "top_k": 2,
            "consistency": "eventual",
            "include_vectors": False,
            "partition_filter": None,
        }
    ]
    assert adapter.queries == []
    assert query_events[0]["full_text"] == {
        "field": "metadata.text",
        "metadata_field": "text",
        "query": "Alpha beta gamma",
    }
    assert query_events[0]["recall_at_k"] is None
    assert query_events[0]["recall_skip_reason"] == "full_text_no_ground_truth"
    assert query_events[0]["returned_count"] == 2
    assert result.summary["query"]["full_text_applied"] is True
    assert result.summary["query"]["recall_at_k"] is None
    assert result.summary["query"]["recall_samples"] == 0
    assert result.summary["query"]["recall_skip_reason"] == "full_text_no_ground_truth"
    assert result.summary["query"]["returned_count"]["p50"] == 2.0
    assert result.summary["query"]["empty_result_rate"] == 0.0


def test_execute_benchmark_runs_full_text_query_with_partition_filter(
    tmp_path,
) -> None:
    scenario = make_scenario(full_text=True, partition_filter=True, top_k=2)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=3,
        query_count=1,
        source_rows=[
            {
                "_id": "query",
                "emb": [1.0, 0.0],
                "text": "Alpha beta gamma delta",
                "url": "q-url",
            },
            {
                "_id": "a",
                "emb": [1.0, 0.0],
                "text": "alpha reference",
                "url": "a-url",
            },
            {
                "_id": "b",
                "emb": [0.0, 1.0],
                "text": "unrelated text",
                "url": "b-url",
            },
            {
                "_id": "c",
                "emb": [0.8, 0.2],
                "text": "beta reference",
                "url": "c-url",
            },
        ],
    )
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.full_text_queries[0]["partition_filter"] == {
        "field": "metadata.url",
        "in_": ["q-url"],
    }
    assert query_events[0]["partition_filter"] == {
        "field": "metadata.url",
        "in_": ["q-url"],
    }
    assert query_events[0]["recall_skip_reason"] == "full_text_no_ground_truth"
    assert result.summary["query"]["partition_filter_applied"] is True
    assert result.summary["query"]["full_text_applied"] is True
    assert result.summary["query"]["recall_skip_reason"] == (
        "full_text_no_ground_truth"
    )


def test_execute_benchmark_applies_logical_filter_from_ground_truth(tmp_path) -> None:
    scenario = make_scenario(query_filter=True, top_k=1)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(
        dataset_dir=dataset.output_dir,
        top_k=1,
        filter_name="synthetic_bucket_50pct",
        filter_field="filter_bucket_2",
        filter_value_source="eligible-record-buckets",
    )
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        ground_truth_path=truth.ground_truth_path,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert adapter.filter_queries == [query_events[0]["filter"]]
    assert query_events[0]["filter_name"] == "synthetic_bucket_50pct"
    assert query_events[0]["filter_selectivity"] == 0.5
    assert query_events[0]["candidate_count"] >= 1
    assert query_events[0]["expected_count"] == 1
    assert query_events[0]["returned_count"] == 1
    assert query_events[0]["recall_at_k"] == 1.0
    assert result.summary["query"]["filter_applied"] is True
    assert result.summary["query"]["filter"]["field"] == "filter_bucket_2"
    assert result.summary["query"]["candidate_count"]["min"] >= 1
    assert result.summary["query"]["underfilled_result_rate"] == 0.0


def test_execute_benchmark_fails_logical_filter_without_filtered_truth(
    tmp_path,
) -> None:
    scenario = make_scenario(query_filter=True, top_k=1)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=1)

    with pytest.raises(ConfigError, match="filtered ground truth"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_execute_benchmark_fails_partition_filter_when_query_metadata_missing(
    tmp_path,
) -> None:
    scenario = make_scenario(partition_filter=True)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)

    with pytest.raises(ConfigError, match="metadata field 'url'"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_execute_benchmark_runs_search_under_ingest_upload_and_ask(tmp_path) -> None:
    scenario = make_scenario(
        rows=1,
        workload="search_under_ingest",
        search_under_ingest={
            "pattern": "upload_and_ask",
            "probe_source": "queries",
            "document_group_field": "url",
            "max_probe_documents": 1,
            "min_chunks_per_document": 1,
            "max_chunks_per_document": 10,
            "probe_queries_per_document": 1,
            "probe_concurrency": 1,
            "top_k": 2,
            "consistency": "eventual",
            "poll_until_visible": True,
            "visibility_timeout": "50ms",
            "visibility_poll_interval": "1ms",
        },
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=1,
        query_count=3,
        source_rows=[
            {"_id": "q1", "emb": [1.0, 0.0], "text": "query one", "url": "doc-1"},
            {"_id": "q2", "emb": [0.9, 0.1], "text": "query two", "url": "doc-1"},
            {"_id": "q3", "emb": [0.0, 1.0], "text": "other", "url": "doc-2"},
            {"_id": "base", "emb": [0.1, 0.9], "text": "base", "url": "base"},
        ],
    )
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        query_only=True,
    )

    search_events = [
        json.loads(line)
        for line in result.search_under_ingest_events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert result.summary["status"] == "completed"
    assert result.summary["workload"] == "search_under_ingest"
    assert result.summary["load"]["status"] == "skipped"
    assert result.summary["query"]["skip_reason"] == "search_under_ingest"
    assert [record.id for record in adapter.upserted[0]] == ["q1", "q2"]
    assert adapter.write_modes == ["upsert"]
    assert search_events[0]["document_group_value"] == "doc-1"
    assert search_events[0]["probe_chunk_ids"] == ["q1", "q2"]
    assert search_events[0]["exact_chunk_hit_at_k"] is True
    assert search_events[0]["same_document_hit_at_k"] is True
    assert search_events[0]["same_document_recall_at_k"] == 1.0
    assert search_events[0]["visible"] is True
    assert result.summary["search_under_ingest"]["probe_documents"] == 1
    assert result.summary["search_under_ingest"]["probe_chunks"] == 2
    assert (
        result.summary["search_under_ingest"][
            "read_after_write_same_document_hit_rate_at_k"
        ]
        == 1.0
    )
    assert result.query_events_path.read_text(encoding="utf-8") == ""


def test_execute_benchmark_runs_parallel_search_under_ingest(tmp_path) -> None:
    scenario = make_scenario(
        rows=4,
        batch_size=2,
        load_concurrency=2,
        workload="search_under_ingest",
        search_under_ingest={
            "pattern": "parallel_upsert_query",
            "duration": "50ms",
            "ingest_concurrency": 2,
            "query_concurrency": 2,
            "top_k": 2,
            "consistency": "eventual",
        },
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=4,
        query_count=2,
        source_rows=[
            {"_id": "q1", "emb": [1.0, 0.0], "text": "query one"},
            {"_id": "q2", "emb": [0.0, 1.0], "text": "query two"},
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
            {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
            {"_id": "d", "emb": [0.2, 0.8], "text": "delta"},
        ],
    )
    adapter = FakeAdapter(load_delay_seconds=0.005, query_delay_seconds=0.001)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]
    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.summary["status"] == "completed"
    assert result.summary["workload"] == "search_under_ingest"
    assert result.summary["search_under_ingest"]["pattern"] == "parallel_upsert_query"
    assert result.summary["load"]["records"] == 4
    assert result.summary["load"]["concurrency"] == 2
    assert result.summary["load"]["processes"] == 1
    assert result.summary["load"]["worker_threads_per_process"] == [2]
    assert result.summary["query"]["mode"] == "parallel_under_ingest"
    assert result.summary["query"]["processes"] == 1
    assert result.summary["query"]["stages"][0]["worker_threads_per_process"] == [2]
    assert result.summary["query"]["queries"] > 0
    assert result.summary["search_under_ingest"]["ingest_processes"] == 1
    assert result.summary["search_under_ingest"]["query_processes"] == 1
    assert result.summary["search_under_ingest"]["queries"] == len(query_events)
    assert result.summary["search_under_ingest"]["records"] == 4
    assert len(ingest_events) == 2
    assert {event["query_stage_index"] for event in query_events} == {1}
    assert result.search_under_ingest_events_path.read_text(encoding="utf-8") == ""


def test_parallel_search_under_ingest_rejects_sharded_load(tmp_path) -> None:
    scenario = make_scenario(
        workload="search_under_ingest",
        sharded_records=True,
        search_under_ingest={
            "pattern": "parallel_upsert_query",
            "duration": "50ms",
            "ingest_concurrency": 1,
            "query_concurrency": 1,
            "top_k": 2,
            "consistency": "eventual",
        },
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    optimize_dataset(dataset_dir=dataset.output_dir, shards=2)

    with pytest.raises(ConfigError, match="sharded_records"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_parallel_search_under_ingest_requires_queries(tmp_path) -> None:
    scenario = make_scenario(
        workload="search_under_ingest",
        search_under_ingest={
            "pattern": "parallel_upsert_query",
            "duration": "50ms",
            "ingest_concurrency": 1,
            "query_concurrency": 1,
            "top_k": 2,
            "consistency": "eventual",
        },
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, query_count=0)

    with pytest.raises(ConfigError, match="at least one query record"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_parallel_search_under_ingest_drops_queued_batches_after_deadline(
    tmp_path,
) -> None:
    scenario = make_scenario(
        rows=20,
        batch_size=1,
        load_concurrency=1,
        workload="search_under_ingest",
        search_under_ingest={
            "pattern": "parallel_upsert_query",
            "duration": "1ms",
            "ingest_concurrency": 1,
            "query_concurrency": 1,
            "top_k": 2,
            "consistency": "eventual",
        },
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=20,
        query_count=1,
        source_rows=[
            {"_id": f"q{i}", "emb": [1.0, 0.0], "text": f"query {i}"}
            for i in range(1, 2)
        ]
        + [
            {"_id": f"r{i}", "emb": [1.0, 0.0], "text": f"record {i}"}
            for i in range(1, 21)
        ],
    )
    adapter = FakeAdapter(load_delay_seconds=0.02, query_delay_seconds=0.02)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    assert result.summary["load"]["records"] < 20
    assert result.summary["load"]["skipped_batches"] > 0
    assert result.summary["load"]["skipped_records"] > 0


def test_read_records_uses_msgpack_estimated_size(tmp_path) -> None:
    dataset = prepare_fixture_dataset(tmp_path, make_scenario())

    records = list(read_records(dataset.records_msgpack_path))

    assert [record.id for record in records] == ["a", "b", "c"]
    assert records[0].vector == [1.0, 0.0]
    assert records[0].estimated_size_bytes == _record_size_bytes(records[0])


def test_load_stage_splits_batches_by_byte_limit(tmp_path) -> None:
    scenario = ScenarioConfig.from_mapping(
        {
            "name": "runner-smoke",
            "dataset": {
                "provider": "huggingface",
                "source": "demo/source",
                "rows": 4,
                "dimensions": 2,
            },
            "load": {"write_mode": "upsert", "batch_size": 10, "max_batch_bytes": 95},
            "query": {"top_k": 2, "query_count": 1, "consistency": "eventual"},
        }
    )
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=4,
        query_count=0,
        source_rows=[
            {"id": "a", "emb": [1.0, 0.0], "text": "alpha"},
            {"id": "b", "emb": [1.0, 0.0], "text": "beta"},
            {"id": "c", "emb": [1.0, 0.0], "text": "gamma"},
            {"id": "d", "emb": [1.0, 0.0], "text": "delta"},
        ],
    )
    adapter = FakeAdapter()

    execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    assert [len(batch) for batch in adapter.upserted] == [1, 1, 1, 1]


def test_execute_benchmark_loads_batches_concurrently(tmp_path) -> None:
    scenario = make_scenario(load_concurrency=2)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, limit=4, query_count=0)
    adapter = FakeAdapter(load_delay_seconds=0.001)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.summary["load"]["concurrency"] == 2
    assert result.summary["load"]["processes"] == 1
    assert result.summary["load"]["worker_threads_per_process"] == [2]
    assert result.summary["load"]["records"] == 4
    assert result.summary["load"]["records_read"] == 4
    assert result.summary["load"]["batches"] == 2
    assert result.summary["load"]["attempts"] == 2
    assert result.summary["load"]["errors"] == 0
    assert len(adapter.upserted) == 2
    assert sorted(event["batch_index"] for event in ingest_events) == [1, 2]
    assert sorted(event["records"] for event in ingest_events) == [2, 2]


def test_execute_benchmark_can_limit_records_and_queries(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        max_records=1,
        max_queries=1,
    )

    assert result.summary["load"]["records"] == 1
    assert result.summary["query"]["queries"] == 1
    assert result.summary["query"]["mode"] == "one_pass"


def test_execute_benchmark_runs_duration_query_stages(tmp_path) -> None:
    scenario = make_scenario(stages=[{"concurrency": 2, "duration": "50ms"}])
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, limit=2, query_count=2)
    adapter = FakeAdapter(query_delay_seconds=0.001)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.summary["query"]["mode"] == "staged"
    assert result.summary["query"]["queries"] > 2
    assert result.summary["query"]["errors"] == 0
    assert result.summary["query"]["stages"][0]["concurrency"] == 2
    assert result.summary["query"]["stages"][0]["processes"] == 1
    assert result.summary["query"]["stages"][0]["worker_threads_per_process"] == [2]
    assert result.summary["query"]["stages"][0]["queries"] == len(query_events)
    assert {event["query_stage_index"] for event in query_events} == {1}
    assert {event["worker_index"] for event in query_events} == {1, 2}


def test_execute_benchmark_caps_staged_queries_by_max_requests(tmp_path) -> None:
    scenario = make_scenario(stages=[{"concurrency": 2, "max_requests": 3}])
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, limit=2, query_count=2)
    adapter = FakeAdapter(query_delay_seconds=0.001)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]
    stage = result.summary["query"]["stages"][0]

    assert result.summary["query"]["mode"] == "staged"
    assert result.summary["query"]["attempts"] == 3
    assert result.summary["query"]["queries"] == 3
    assert len(query_events) == 3
    assert stage["concurrency"] == 2
    assert stage["max_requests"] == 3
    assert stage["configured_duration_seconds"] is None
    assert stage["attempts"] == 3
    assert sorted(event["query_index"] for event in query_events) == [1, 2, 3]


def test_process_concurrency_split_preserves_total_concurrency() -> None:
    assert _split_concurrency(64, 4) == [16, 16, 16, 16]
    assert _split_concurrency(10, 3) == [4, 3, 3]
    assert _split_concurrency(16, 8) == [2] * 8


def test_execute_benchmark_records_query_errors(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario, limit=2, query_count=2)
    adapter = FakeAdapter(fail_every=2)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        max_queries=2,
    )

    query_events = [
        json.loads(line)
        for line in result.query_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.summary["status"] == "completed_with_errors"
    assert result.summary["query"]["queries"] == 1
    assert result.summary["query"]["errors"] == 1
    assert result.summary["query"]["error_rate"] == 0.5
    assert [event["status"] for event in query_events] == ["ok", "error"]
    assert query_events[1]["error_type"] == "RuntimeError"


def test_execute_benchmark_waits_until_query_visible(tmp_path) -> None:
    scenario = make_scenario(wait_until_query_visible=True)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        max_queries=1,
    )

    assert result.summary["status"] == "completed"
    assert result.summary["load"]["visibility"]["status"] == "visible"
    assert result.summary["load"]["visibility"]["samples"] == 3
    assert result.summary["load"]["visibility"]["visible"] == 3


def test_execute_benchmark_load_only_skips_queries(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    truth.manifest_path.unlink()
    adapter = FakeAdapter()

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )

    assert result.summary["status"] == "completed"
    assert result.summary["query"]["mode"] == "skipped"
    assert result.summary["query"]["skip_reason"] == "load_only"
    assert result.summary["query"]["queries"] == 0
    assert result.query_events_path.read_text(encoding="utf-8") == ""


def test_execute_benchmark_query_only_skips_load(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    adapter = FakeAdapter()
    adapter.upserted.append(
        [
            VectorRecord(id="a", vector=[1.0, 0.0]),
            VectorRecord(id="b", vector=[0.0, 1.0]),
        ]
    )

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        max_queries=1,
        query_only=True,
    )

    assert result.summary["status"] == "completed"
    assert result.summary["load"]["status"] == "skipped"
    assert result.summary["load"]["skip_reason"] == "query_only"
    assert result.summary["query"]["queries"] == 1
    assert result.ingest_events_path.read_text(encoding="utf-8") == ""


def test_execute_benchmark_query_only_requires_existing_prepare_mode(tmp_path) -> None:
    scenario = make_scenario()
    target = TargetConfig.from_mapping(
        {
            "vendor": "fake",
            "name": "fake-target",
            "endpoint": "memory://fake",
            "prepare": {"mode": "create"},
        }
    )
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)

    with pytest.raises(ConfigError, match="query-only"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
            query_only=True,
        )


def test_execute_benchmark_delete_only_writes_checkpoint_state(tmp_path) -> None:
    scenario = make_scenario(delete_order="sequential")
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    plan = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    adapter = FakeAdapter()
    adapter.upserted.append(
        [
            VectorRecord(id="a", vector=[1.0, 0.0]),
            VectorRecord(id="b", vector=[0.0, 1.0]),
            VectorRecord(id="c", vector=[0.8, 0.2]),
        ]
    )

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "delete-50",
        dataset_dir=dataset.output_dir,
        delete_only=True,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=50,
        allow_destructive=True,
    )

    state = json.loads(result.deletion_state_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in result.delete_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert adapter.deleted_batches == [["a"]]
    assert events[0]["documents"] == 1
    assert state["status"] == "completed"
    assert state["deletion"]["deleted_count"] == 1
    assert state["deletion"]["remaining_count"] == 2
    assert state["deletion"]["visibility"]["status"] == "not_visible"
    assert result.summary["load"]["skip_reason"] == "delete_only"
    assert result.summary["query"]["skip_reason"] == "delete_only"


def test_delete_only_resumes_to_later_checkpoint_and_query_validates_state(
    tmp_path,
) -> None:
    scenario = make_scenario(delete_order="sequential", top_k=1)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    plan = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    ground_truth = prepare_ground_truth(
        dataset_dir=dataset.output_dir,
        top_k=1,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=75,
    )
    adapter = FakeAdapter()
    adapter.upserted.append(
        [
            VectorRecord(id="a", vector=[1.0, 0.0]),
            VectorRecord(id="b", vector=[0.0, 1.0]),
            VectorRecord(id="c", vector=[0.8, 0.2]),
        ]
    )

    first = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "delete-50",
        dataset_dir=dataset.output_dir,
        delete_only=True,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=50,
        allow_destructive=True,
    )
    second = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "delete-75",
        dataset_dir=dataset.output_dir,
        delete_only=True,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=75,
        deletion_state_path=first.deletion_state_path,
        allow_destructive=True,
    )
    query = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "query-75",
        dataset_dir=dataset.output_dir,
        ground_truth_path=ground_truth.ground_truth_path,
        max_queries=1,
        query_only=True,
        deletion_state_path=second.deletion_state_path,
    )

    assert adapter.deleted_batches == [["a"], ["b"]]
    assert second.summary["deletion"]["newly_deleted_count"] == 1
    assert second.summary["deletion"]["deleted_count"] == 2
    assert query.summary["query"]["recall_at_k"] == 1.0
    assert query.summary["deletion"]["status"] == "validated"


def test_delete_only_requires_destructive_opt_in(tmp_path) -> None:
    scenario = make_scenario(delete_order="sequential")
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    plan = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )

    with pytest.raises(ConfigError, match="allow-destructive"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "delete",
            dataset_dir=dataset.output_dir,
            delete_only=True,
            deletion_plan_path=plan.plan_path,
            delete_checkpoint_pct=50,
        )


def test_query_only_delete_scenario_requires_deletion_state(tmp_path) -> None:
    scenario = make_scenario(delete_order="sequential")
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)

    with pytest.raises(ConfigError, match="requires --deletion-state"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "query",
            dataset_dir=dataset.output_dir,
            query_only=True,
        )


def test_delete_only_fails_state_when_deleted_ids_remain_visible(tmp_path) -> None:
    scenario = make_scenario(delete_order="sequential")
    scenario.delete["visibility_timeout"] = "1ms"
    scenario.delete["visibility_poll_interval"] = "1ms"
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    plan = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=FakeAdapter(stale_deleted_fetch=True),
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "delete",
        dataset_dir=dataset.output_dir,
        delete_only=True,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=50,
        allow_destructive=True,
    )

    state = json.loads(result.deletion_state_path.read_text(encoding="utf-8"))
    assert result.summary["status"] == "failed"
    assert result.summary["deletion"]["status"] == "failed"
    assert state["status"] == "failed"
    assert state["deletion"]["visibility"]["status"] == "timeout"


def test_execute_benchmark_writes_partial_summary_on_load_error(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    adapter = FakeAdapter(fail_load_batch=2)

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
    )

    ingest_events = [
        json.loads(line)
        for line in result.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.summary["status"] == "failed"
    assert result.summary["load"]["records"] == 2
    assert result.summary["load"]["errors"] == 1
    assert result.summary["load"]["error_rate"] == 0.5
    assert result.summary["query"]["mode"] == "skipped"
    assert result.summary["query"]["skip_reason"] == "load_failed"
    assert result.query_events_path.read_text(encoding="utf-8") == ""
    assert [event["status"] for event in ingest_events] == ["ok", "error"]
    assert ingest_events[1]["error_type"] == "RuntimeError"


def test_execute_benchmark_can_resume_load_from_checkpoint(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    output_dir = tmp_path / "result"

    failed = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=FakeAdapter(fail_load_batch=2),
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=output_dir,
        dataset_dir=dataset.output_dir,
        load_only=True,
    )
    first_checkpoint = json.loads(
        failed.load_checkpoint_path.read_text(encoding="utf-8")
    )

    resumed_adapter = FakeAdapter()
    resumed = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=resumed_adapter,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=output_dir,
        dataset_dir=dataset.output_dir,
        load_only=True,
        resume_load=True,
    )
    final_checkpoint = json.loads(
        resumed.load_checkpoint_path.read_text(encoding="utf-8")
    )
    ingest_events = [
        json.loads(line)
        for line in resumed.ingest_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert first_checkpoint["status"] == "failed"
    assert first_checkpoint["highest_contiguous_successful_batch_index"] == 1
    assert [len(batch) for batch in resumed_adapter.upserted] == [1]
    assert resumed.summary["status"] == "completed"
    assert resumed.summary["load"]["records"] == 1
    assert resumed.summary["load"]["skipped_records"] == 2
    assert resumed.summary["load"]["skipped_batches"] == 1
    assert resumed.summary["load"]["checkpoint"]["resume_enabled"] is True
    assert resumed.summary["load"]["checkpoint"]["resumed_from_batch_index"] == 1
    assert final_checkpoint["status"] == "completed"
    assert final_checkpoint["highest_contiguous_successful_batch_index"] == 2
    assert [event["status"] for event in ingest_events] == ["ok", "error", "ok"]
    assert [event["batch_index"] for event in ingest_events] == [1, 2, 2]


def test_load_checkpoint_uses_contiguous_watermark_for_concurrent_loads(
    tmp_path,
) -> None:
    class OutOfOrderFailureAdapter(FakeAdapter):
        def upsert_batch(
            self,
            target: TargetConfig,
            records: list[VectorRecord],
            *,
            write_mode: str = "upsert",
        ) -> UpsertResult:
            if records[0].id == "b":
                time.sleep(0.02)
                raise RuntimeError("planned out-of-order failure")
            return super().upsert_batch(target, records, write_mode=write_mode)

    scenario = make_scenario(rows=4, batch_size=1, load_concurrency=2)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=4,
        query_count=0,
        source_rows=[
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
            {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
            {"_id": "d", "emb": [0.2, 0.8], "text": "delta"},
        ],
    )

    result = execute_benchmark(
        scenario=scenario,
        target=target,
        adapter=OutOfOrderFailureAdapter(),
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=tmp_path / "result",
        dataset_dir=dataset.output_dir,
        load_only=True,
    )
    checkpoint = json.loads(result.load_checkpoint_path.read_text(encoding="utf-8"))

    assert result.summary["status"] == "failed"
    assert result.summary["load"]["errors"] == 1
    assert checkpoint["highest_contiguous_successful_batch_index"] == 1
    assert set(checkpoint["successful_batch_indexes"]).issuperset({1, 3})


def test_execute_benchmark_resume_load_requires_checkpoint(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)

    with pytest.raises(ConfigError, match="load checkpoint"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
            load_only=True,
            resume_load=True,
        )


def test_execute_benchmark_requires_large_run_opt_in(tmp_path) -> None:
    scenario = make_scenario(rows=1_000_000)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, make_scenario())

    with pytest.raises(ConfigError, match="large real runs"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_execute_benchmark_rejects_missing_explicit_ground_truth(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)

    with pytest.raises(ConfigError, match="ground truth file"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
            ground_truth_path=tmp_path / "missing-ground-truth.jsonl",
        )


def test_execute_benchmark_rejects_missing_ground_truth_manifest(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    truth.manifest_path.unlink()

    with pytest.raises(ConfigError, match="ground truth manifest .* does not exist"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        (
            "ground_truth",
            "metric",
            "euclidean",
            "manifest metric 'euclidean'.*'cosine'",
        ),
        ("ground_truth", "top_k", 1, "manifest top_k 1.*scenario top_k 2"),
        (
            "artifacts",
            "ground_truth",
            "wrong-ground-truth.jsonl",
            "manifest artifact .* does not match",
        ),
    ],
)
def test_execute_benchmark_rejects_ground_truth_manifest_mismatch(
    tmp_path,
    section: str,
    key: str,
    value: Any,
    error: str,
) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    manifest = json.loads(truth.manifest_path.read_text(encoding="utf-8"))
    manifest[section][key] = value
    truth.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConfigError, match=error):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_execute_benchmark_rejects_ground_truth_checksum_mismatch(tmp_path) -> None:
    scenario = make_scenario()
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(dataset_dir=dataset.output_dir, top_k=2)
    with truth.ground_truth_path.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(ConfigError, match="ground truth checksum mismatch"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
        )


def test_execute_benchmark_rejects_filtered_manifest_mismatch(tmp_path) -> None:
    scenario = make_scenario(query_filter=True, top_k=1)
    target = make_target()
    scenario_path, target_path = write_configs(tmp_path, scenario, target)
    dataset = prepare_fixture_dataset(tmp_path, scenario)
    truth = prepare_ground_truth(
        dataset_dir=dataset.output_dir,
        top_k=1,
        filter_name="synthetic_bucket_50pct",
        filter_field="filter_bucket_2",
        filter_value_source="eligible-record-buckets",
    )
    manifest = json.loads(truth.manifest_path.read_text(encoding="utf-8"))
    manifest["ground_truth"]["filter"]["name"] = "wrong-filter"
    truth.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConfigError, match="manifest filter .* does not match"):
        execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=FakeAdapter(),
            scenario_path=scenario_path,
            target_path=target_path,
            output_dir=tmp_path / "result",
            dataset_dir=dataset.output_dir,
            ground_truth_path=truth.ground_truth_path,
        )


def test_ground_truth_path_uses_metric_specific_artifact(tmp_path) -> None:
    legacy = tmp_path / "ground_truth.jsonl"
    preferred = tmp_path / "ground_truth.euclidean.jsonl"
    legacy.touch()

    assert _ground_truth_path(tmp_path, None, metric="euclidean") == preferred


def test_ground_truth_path_uses_explicit_legacy_artifact(tmp_path) -> None:
    legacy = tmp_path / "ground_truth.jsonl"
    legacy.touch()

    assert _ground_truth_path(tmp_path, legacy, metric="euclidean") == legacy


def test_dataset_metric_prefers_scenario_over_manifest() -> None:
    assert _dataset_metric(
        make_scenario(metric="euclidean"),
        {"dataset": {"metric": "cosine"}},
    ) == "euclidean"


def test_recall_at_k() -> None:
    assert recall_at_k(actual=["a", "x"], expected=["a", "b"], k=2) == 0.5
    assert recall_at_k(actual=["a"], expected=None, k=2) is None


def test_recall_at_k_accepts_alternate_k_boundary_candidate() -> None:
    assert (
        recall_at_k(
            actual=["strict", "boundary-b"],
            expected=["strict", "boundary-a"],
            k=2,
            recall_k=2,
            strict=["strict"],
            boundary=["boundary-a", "boundary-b"],
        )
        == 1.0
    )


def test_recall_at_k_does_not_replace_missing_strict_with_boundary() -> None:
    assert (
        recall_at_k(
            actual=["boundary-a", "boundary-b"],
            expected=["strict", "boundary-a"],
            k=2,
            recall_k=2,
            strict=["strict"],
            boundary=["boundary-a", "boundary-b"],
        )
        == 0.5
    )


def test_recall_at_k_rejects_candidate_outside_boundary() -> None:
    assert (
        recall_at_k(
            actual=["strict", "outside"],
            expected=["strict", "boundary-a"],
            k=2,
            recall_k=2,
            strict=["strict"],
            boundary=["boundary-a", "boundary-b"],
        )
        == 0.5
    )


def test_load_ground_truth_keeps_legacy_recall_behavior(tmp_path) -> None:
    path = tmp_path / "ground_truth.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "query",
                "matches": [{"id": "a"}, {"id": "b"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entry = load_ground_truth(path)["query"]

    assert entry.recall_k is None
    assert entry.strict_ids is None
    assert entry.boundary_ids is None
    assert recall_at_k(actual=["a", "x"], expected=entry.ids, k=2) == 0.5


def test_load_ground_truth_reads_tie_aware_recall_groups(tmp_path) -> None:
    path = tmp_path / "ground_truth.euclidean.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "query",
                "matches": [{"id": "strict"}, {"id": "boundary-a"}],
                "recall_groups": {
                    "k": 2,
                    "strict_ids": ["strict"],
                    "boundary_ids": ["boundary-a", "boundary-b"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entry = load_ground_truth(path)["query"]

    assert entry.recall_k == 2
    assert entry.strict_ids == ["strict"]
    assert entry.boundary_ids == ["boundary-a", "boundary-b"]
    assert (
        recall_at_k(
            actual=["strict", "boundary-b"],
            expected=entry.ids,
            k=2,
            recall_k=entry.recall_k,
            strict=entry.strict_ids,
            boundary=entry.boundary_ids,
        )
        == 1.0
    )


def test_latency_summary_empty_and_percentiles() -> None:
    assert latency_summary([])["p50"] is None
    summary = latency_summary([1.0, 2.0, 3.0])
    assert summary["min"] == 1.0
    assert summary["p50"] == 2.0
    assert summary["max"] == 3.0


def test_parse_size_bytes() -> None:
    assert parse_size_bytes("2MB") == 2_000_000
    assert parse_size_bytes("2MiB") == 2 * 1024 * 1024
    assert parse_size_bytes("512") == 512


def test_batches_allows_single_record_larger_than_byte_limit() -> None:
    records = [VectorRecord(id="large", vector=[1.0] * 128)]

    batches = list(_batches(records, batch_size=10, max_batch_bytes=10))

    assert batches == [records]


def test_record_size_bytes_uses_precomputed_estimate() -> None:
    record = VectorRecord(
        id="large",
        vector=[1.0] * 128,
        estimated_size_bytes=123,
    )

    assert _record_size_bytes(record) == 123
