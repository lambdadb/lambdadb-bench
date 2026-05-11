"""Sequential benchmark execution for prepared dataset artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ldbbench.adapters.base import VectorDBAdapter, VectorRecord
from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets.ground_truth import (
    GROUND_TRUTH_FILENAME,
    artifact_path,
    load_dataset_manifest,
    parse_vector_item,
)
from ldbbench.datasets.prepare import QUERIES_FILENAME, RECORDS_FILENAME
from ldbbench.manifest import initialize_run_artifacts
from ldbbench.runner.plan import build_run_plan

INGEST_EVENTS_FILENAME = "ingest_events.jsonl"
QUERY_EVENTS_FILENAME = "query_events.jsonl"
SUMMARY_FILENAME = "summary.json"
LARGE_RUN_ROW_THRESHOLD = 1_000_000


@dataclass(frozen=True)
class BenchmarkRunResult:
    output_dir: Path
    ingest_events_path: Path
    query_events_path: Path
    summary_path: Path
    summary: dict[str, Any]


def execute_benchmark(
    *,
    scenario: ScenarioConfig,
    target: TargetConfig,
    adapter: VectorDBAdapter,
    scenario_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    dataset_dir: str | Path,
    ground_truth_path: str | Path | None = None,
    max_records: int | None = None,
    max_queries: int | None = None,
    allow_destructive: bool = False,
    allow_large_run: bool = False,
) -> BenchmarkRunResult:
    """Execute a small or explicitly opted-in benchmark run sequentially."""

    _validate_limits(max_records=max_records, max_queries=max_queries)
    if _is_large_run(scenario, max_records=max_records) and not allow_large_run:
        raise ConfigError(
            "large real runs require --allow-large-run or a smaller --max-records"
        )

    plan = build_run_plan(
        scenario=scenario,
        target=target,
        capabilities=adapter.capabilities,
        allow_destructive=allow_destructive,
    )
    if not plan.can_run:
        raise ConfigError("run plan is unsupported: " + "; ".join(plan.unsupported))
    if plan.not_applicable:
        raise ConfigError(
            "run plan has not-applicable requirements: "
            + "; ".join(plan.not_applicable)
        )

    out = Path(output_dir)
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=out,
        adapter_capabilities=adapter.capabilities.as_dict(),
        dry_run_plan=plan.as_dict(),
    )

    dataset_path = Path(dataset_dir)
    dataset_manifest = load_dataset_manifest(dataset_path)
    records_path = artifact_path(
        dataset_path,
        dataset_manifest,
        "records",
        RECORDS_FILENAME,
    )
    queries_path = artifact_path(
        dataset_path,
        dataset_manifest,
        "queries",
        QUERIES_FILENAME,
    )
    truth_path = _ground_truth_path(dataset_path, ground_truth_path)
    if ground_truth_path is not None and not truth_path.exists():
        raise ConfigError(f"ground truth file {truth_path} does not exist")
    ground_truth = load_ground_truth(truth_path) if truth_path.exists() else {}

    ingest_events_path = out / INGEST_EVENTS_FILENAME
    query_events_path = out / QUERY_EVENTS_FILENAME
    summary_path = out / SUMMARY_FILENAME

    adapter.prepare(
        target,
        dimensions=_dataset_dimensions(scenario, dataset_manifest),
        metric=_dataset_metric(scenario, dataset_manifest),
    )

    ingest_summary = run_load_stage(
        adapter=adapter,
        target=target,
        records=read_records(records_path, limit=max_records),
        batch_size=_batch_size(scenario),
        events_path=ingest_events_path,
    )
    query_summary = run_query_stage(
        adapter=adapter,
        target=target,
        queries=read_records(queries_path, limit=max_queries),
        top_k=_top_k(scenario),
        consistency=str(scenario.query.get("consistency", "eventual")),
        include_vectors=bool(scenario.query.get("include_vectors", False)),
        ground_truth=ground_truth,
        events_path=query_events_path,
    )

    summary = {
        "status": "completed",
        "run_manifest": str(paths.run_manifest),
        "dataset_dir": str(dataset_path),
        "ground_truth": str(truth_path) if truth_path.exists() else None,
        "load": ingest_summary,
        "query": query_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return BenchmarkRunResult(
        output_dir=out,
        ingest_events_path=ingest_events_path,
        query_events_path=query_events_path,
        summary_path=summary_path,
        summary=summary,
    )


def run_load_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    records: Iterable[VectorRecord],
    batch_size: int,
    events_path: str | Path,
) -> dict[str, Any]:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)

    records_loaded = 0
    latencies: list[float] = []
    started = time.perf_counter()
    with events_output.open("w", encoding="utf-8") as file:
        for batch_index, batch in enumerate(_batches(records, batch_size), start=1):
            batch_started = time.perf_counter()
            result = adapter.upsert_batch(target, batch)
            latency_ms = _elapsed_ms(batch_started)
            count = result.count
            records_loaded += count
            latencies.append(latency_ms)
            file.write(
                json.dumps(
                    {
                        "stage": "load",
                        "batch_index": batch_index,
                        "records": count,
                        "latency_ms": latency_ms,
                        "status": "ok",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    duration_seconds = time.perf_counter() - started
    return {
        "records": records_loaded,
        "batches": len(latencies),
        "duration_seconds": duration_seconds,
        "records_per_second": _rate(records_loaded, duration_seconds),
        "latency_ms": latency_summary(latencies),
    }


def run_query_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    queries: Iterable[VectorRecord],
    top_k: int,
    consistency: str,
    include_vectors: bool,
    ground_truth: Mapping[str, list[str]],
    events_path: str | Path,
) -> dict[str, Any]:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)

    query_count = 0
    latencies: list[float] = []
    recalls: list[float] = []
    started = time.perf_counter()
    with events_output.open("w", encoding="utf-8") as file:
        for query_index, query in enumerate(queries, start=1):
            query_started = time.perf_counter()
            result = adapter.query(
                target,
                vector=query.vector,
                top_k=top_k,
                consistency=consistency,
                include_vectors=include_vectors,
            )
            latency_ms = _elapsed_ms(query_started)
            match_ids = [match.id for match in result.matches]
            recall = recall_at_k(
                actual=match_ids,
                expected=ground_truth.get(query.id),
                k=top_k,
            )
            if recall is not None:
                recalls.append(recall)
            latencies.append(latency_ms)
            query_count += 1
            file.write(
                json.dumps(
                    {
                        "stage": "query",
                        "query_index": query_index,
                        "query_id": query.id,
                        "matches": match_ids,
                        "latency_ms": latency_ms,
                        "recall_at_k": recall,
                        "status": "ok",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    duration_seconds = time.perf_counter() - started
    return {
        "queries": query_count,
        "duration_seconds": duration_seconds,
        "queries_per_second": _rate(query_count, duration_seconds),
        "latency_ms": latency_summary(latencies),
        "recall_at_k": _mean(recalls) if recalls else None,
        "recall_samples": len(recalls),
    }


def read_records(
    path: str | Path,
    *,
    limit: int | None = None,
) -> Iterable[VectorRecord]:
    with Path(path).open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            if limit is not None and index > limit:
                break
            if not line.strip():
                continue
            item = parse_vector_item(
                json.loads(line),
                path=Path(path),
                line_number=index,
            )
            yield VectorRecord(
                id=item.id,
                vector=item.vector,
                metadata=item.metadata,
            )


def load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            query_id = raw.get("query_id")
            matches = raw.get("matches")
            if not isinstance(query_id, str) or not isinstance(matches, list):
                raise ConfigError(f"{path}:{line_number} invalid ground truth row")
            truth[query_id] = [
                str(match["id"])
                for match in matches
                if isinstance(match, dict) and "id" in match
            ]
    return truth


def recall_at_k(
    *,
    actual: list[str],
    expected: list[str] | None,
    k: int,
) -> float | None:
    if not expected:
        return None
    expected_k = expected[:k]
    if not expected_k:
        return None
    actual_k = set(actual[:k])
    return len(actual_k.intersection(expected_k)) / len(expected_k)


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
    }


def percentile(ordered_values: list[float], percentile_value: int) -> float:
    if not ordered_values:
        raise ConfigError("cannot compute percentile for an empty list")
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (percentile_value / 100) * (len(ordered_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = rank - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def _batches(
    records: Iterable[VectorRecord],
    batch_size: int,
) -> Iterable[list[VectorRecord]]:
    batch: list[VectorRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batch_size(scenario: ScenarioConfig) -> int:
    batch_size = scenario.load.get("batch_size", 100)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigError("scenario.load.batch_size must be a positive integer")
    return batch_size


def _top_k(scenario: ScenarioConfig) -> int:
    top_k = scenario.query.get("top_k", 10)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigError("scenario.query.top_k must be a positive integer")
    return top_k


def _dataset_dimensions(
    scenario: ScenarioConfig,
    dataset_manifest: Mapping[str, Any],
) -> int | None:
    value = dataset_manifest.get("dataset", {}).get("dimensions")
    if isinstance(value, int):
        return value
    configured = scenario.dataset.get("dimensions")
    return configured if isinstance(configured, int) else None


def _dataset_metric(
    scenario: ScenarioConfig,
    dataset_manifest: Mapping[str, Any],
) -> str | None:
    value = dataset_manifest.get("dataset", {}).get("metric")
    if isinstance(value, str):
        return value
    configured = scenario.dataset.get("metric")
    return configured if isinstance(configured, str) else None


def _ground_truth_path(dataset_dir: Path, ground_truth_path: str | Path | None) -> Path:
    if ground_truth_path is not None:
        return Path(ground_truth_path)
    return dataset_dir / GROUND_TRUTH_FILENAME


def _validate_limits(*, max_records: int | None, max_queries: int | None) -> None:
    if max_records is not None and max_records <= 0:
        raise ConfigError("--max-records must be a positive integer")
    if max_queries is not None and max_queries <= 0:
        raise ConfigError("--max-queries must be a positive integer")


def _is_large_run(scenario: ScenarioConfig, *, max_records: int | None) -> bool:
    if max_records is not None and max_records < LARGE_RUN_ROW_THRESHOLD:
        return False
    rows = scenario.dataset.get("rows")
    return isinstance(rows, int) and rows >= LARGE_RUN_ROW_THRESHOLD


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _rate(count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return count / duration_seconds


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
