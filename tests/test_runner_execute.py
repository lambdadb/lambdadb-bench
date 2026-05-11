from __future__ import annotations

import json
from typing import Any

import pytest

from ldbbench.adapters.base import (
    AdapterCapabilities,
    CheckResult,
    PrepareResult,
    QueryMatch,
    QueryResult,
    UpsertResult,
    VectorRecord,
)
from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets.ground_truth import prepare_ground_truth
from ldbbench.datasets.prepare import prepare_dataset
from ldbbench.runner.execute import (
    execute_benchmark,
    latency_summary,
    recall_at_k,
)


class FakeAdapter:
    vendor = "fake"
    capabilities = AdapterCapabilities(
        supported_write_modes=frozenset({"upsert"}),
        supported_query_consistency=frozenset({"eventual"}),
    )

    def __init__(self) -> None:
        self.prepared: dict[str, Any] | None = None
        self.upserted: list[list[VectorRecord]] = []
        self.queries: list[list[float]] = []

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
    ) -> UpsertResult:
        self.upserted.append(list(records))
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
    ) -> QueryResult:
        self.queries.append(vector)
        return QueryResult(
            matches=[
                QueryMatch(id="a", score=1.0, document={"id": "a"}),
                QueryMatch(id="c", score=0.9, document={"id": "c"}),
            ][:top_k]
        )

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: list[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        return [{"id": item} for item in ids]


def make_scenario(*, rows: int = 3) -> ScenarioConfig:
    return ScenarioConfig.from_mapping(
        {
            "name": "runner-smoke",
            "dataset": {
                "provider": "huggingface",
                "source": "demo/source",
                "rows": rows,
                "dimensions": 2,
                "id_field": "_id",
                "vector_field": "emb",
                "text_field": "text",
                "metric": "cosine",
            },
            "load": {"write_mode": "upsert", "batch_size": 2},
            "query": {"top_k": 2, "query_count": 1, "consistency": "eventual"},
        }
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


def prepare_fixture_dataset(tmp_path, scenario: ScenarioConfig):
    rows = [
        {"_id": "query", "emb": [1.0, 0.0], "text": "query"},
        {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
        {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
        {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
    ]
    return prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path / "dataset",
        limit=3,
        query_count=1,
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
    assert len(adapter.queries) == 1
    assert [event["records"] for event in ingest_events] == [2, 1]
    assert query_events[0]["query_id"] == "query"
    assert query_events[0]["matches"] == ["a", "c"]
    assert query_events[0]["recall_at_k"] == 1.0
    assert result.summary["load"]["records"] == 3
    assert result.summary["query"]["queries"] == 1
    assert result.summary["query"]["recall_at_k"] == 1.0
    assert result.summary_path.exists()


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


def test_recall_at_k() -> None:
    assert recall_at_k(actual=["a", "x"], expected=["a", "b"], k=2) == 0.5
    assert recall_at_k(actual=["a"], expected=None, k=2) is None


def test_latency_summary_empty_and_percentiles() -> None:
    assert latency_summary([])["p50"] is None
    summary = latency_summary([1.0, 2.0, 3.0])
    assert summary["min"] == 1.0
    assert summary["p50"] == 2.0
    assert summary["max"] == 3.0
