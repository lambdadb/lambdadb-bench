"""Sequential benchmark execution for prepared dataset artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
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
    queries = list(read_records(queries_path, limit=max_queries))
    query_summary = run_query_stage(
        adapter=adapter,
        target=target,
        queries=queries,
        top_k=_top_k(scenario),
        consistency=str(scenario.query.get("consistency", "eventual")),
        include_vectors=bool(scenario.query.get("include_vectors", False)),
        ground_truth=ground_truth,
        events_path=query_events_path,
        stages=None if max_queries is not None else _query_stages(scenario),
    )

    summary = {
        "status": "completed"
        if query_summary["errors"] == 0
        else "completed_with_errors",
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
    stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    query_list = list(queries)
    if not query_list:
        events_output.write_text("", encoding="utf-8")
        return _query_summary(
            mode="staged" if stages else "one_pass",
            started=time.perf_counter(),
            latencies=[],
            recalls=[],
            query_count=0,
            error_count=0,
            stage_summaries=[],
        )

    if stages:
        return run_staged_query_stage(
            adapter=adapter,
            target=target,
            queries=query_list,
            top_k=top_k,
            consistency=consistency,
            include_vectors=include_vectors,
            ground_truth=ground_truth,
            events_path=events_output,
            stages=stages,
        )

    state = QueryRunState()
    started = time.perf_counter()
    with events_output.open("w", encoding="utf-8") as file:
        for query_index, query in enumerate(query_list, start=1):
            event = execute_query_once(
                adapter=adapter,
                target=target,
                query=query,
                query_index=query_index,
                query_stage_index=None,
                worker_index=None,
                top_k=top_k,
                consistency=consistency,
                include_vectors=include_vectors,
                ground_truth=ground_truth,
            )
            state.record(event)
            _write_event(file, event)

    return _query_summary(
        mode="one_pass",
        started=started,
        latencies=state.latencies,
        recalls=state.recalls,
        query_count=state.queries,
        error_count=state.errors,
        stage_summaries=[],
    )


def run_staged_query_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    queries: list[VectorRecord],
    top_k: int,
    consistency: str,
    include_vectors: bool,
    ground_truth: Mapping[str, list[str]],
    events_path: Path,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    query_lock = Lock()
    write_lock = Lock()
    state = QueryRunState()
    next_query_index = 1
    started = time.perf_counter()
    stage_summaries: list[dict[str, Any]] = []

    with events_path.open("w", encoding="utf-8") as file:
        for stage_index, stage in enumerate(stages, start=1):
            concurrency = _stage_concurrency(stage, stage_index=stage_index)
            duration_seconds = parse_duration_seconds(
                _stage_duration(stage, stage_index=stage_index)
            )
            deadline = time.perf_counter() + duration_seconds
            stage_state = QueryRunState()
            stage_started = time.perf_counter()

            def next_query(
                stage_deadline: float = deadline,
            ) -> tuple[int, VectorRecord] | None:
                nonlocal next_query_index
                with query_lock:
                    if time.perf_counter() >= stage_deadline:
                        return None
                    query_index = next_query_index
                    query = queries[(query_index - 1) % len(queries)]
                    next_query_index += 1
                    return query_index, query

            def worker(
                worker_index: int,
                current_stage_index: int = stage_index,
                current_stage_state: QueryRunState = stage_state,
            ) -> None:
                while True:
                    item = next_query()
                    if item is None:
                        return
                    query_index, query = item
                    event = execute_query_once(
                        adapter=adapter,
                        target=target,
                        query=query,
                        query_index=query_index,
                        query_stage_index=current_stage_index,
                        worker_index=worker_index,
                        top_k=top_k,
                        consistency=consistency,
                        include_vectors=include_vectors,
                        ground_truth=ground_truth,
                    )
                    with write_lock:
                        state.record(event)
                        current_stage_state.record(event)
                        _write_event(file, event)

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(worker, index)
                    for index in range(1, concurrency + 1)
                ]
                for future in futures:
                    future.result()

            stage_summaries.append(
                _query_stage_summary(
                    stage_index=stage_index,
                    concurrency=concurrency,
                    configured_duration_seconds=duration_seconds,
                    elapsed_seconds=time.perf_counter() - stage_started,
                    state=stage_state,
                )
            )

    return _query_summary(
        mode="staged",
        started=started,
        latencies=state.latencies,
        recalls=state.recalls,
        query_count=state.queries,
        error_count=state.errors,
        stage_summaries=stage_summaries,
    )


@dataclass
class QueryRunState:
    queries: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)

    def record(self, event: Mapping[str, Any]) -> None:
        if event["status"] == "ok":
            self.queries += 1
            latency_ms = event.get("latency_ms")
            if isinstance(latency_ms, int | float):
                self.latencies.append(float(latency_ms))
            recall = event.get("recall_at_k")
            if isinstance(recall, int | float):
                self.recalls.append(float(recall))
        else:
            self.errors += 1


def execute_query_once(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    query: VectorRecord,
    query_index: int,
    query_stage_index: int | None,
    worker_index: int | None,
    top_k: int,
    consistency: str,
    include_vectors: bool,
    ground_truth: Mapping[str, list[str]],
) -> dict[str, Any]:
    query_started = time.perf_counter()
    base_event: dict[str, Any] = {
        "stage": "query",
        "query_index": query_index,
        "query_id": query.id,
    }
    if query_stage_index is not None:
        base_event["query_stage_index"] = query_stage_index
    if worker_index is not None:
        base_event["worker_index"] = worker_index

    try:
        result = adapter.query(
            target,
            vector=query.vector,
            top_k=top_k,
            consistency=consistency,
            include_vectors=include_vectors,
        )
    except Exception as exc:  # noqa: BLE001
        base_event.update(
            {
                "error_message": str(exc),
                "error_type": type(exc).__name__,
                "latency_ms": _elapsed_ms(query_started),
                "status": "error",
            }
        )
        return base_event

    match_ids = [match.id for match in result.matches]
    recall = recall_at_k(
        actual=match_ids,
        expected=ground_truth.get(query.id),
        k=top_k,
    )
    base_event.update(
        {
            "matches": match_ids,
            "latency_ms": _elapsed_ms(query_started),
            "recall_at_k": recall,
            "status": "ok",
        }
    )
    return base_event


def _write_event(file: Any, event: Mapping[str, Any]) -> None:
    file.write(json.dumps(dict(event), sort_keys=True) + "\n")
    file.flush()


def _query_summary(
    *,
    mode: str,
    started: float,
    latencies: list[float],
    recalls: list[float],
    query_count: int,
    error_count: int,
    stage_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    duration_seconds = time.perf_counter() - started
    attempts = query_count + error_count
    summary: dict[str, Any] = {
        "mode": mode,
        "queries": query_count,
        "errors": error_count,
        "attempts": attempts,
        "error_rate": _error_rate(error_count, attempts),
        "duration_seconds": duration_seconds,
        "queries_per_second": _rate(query_count, duration_seconds),
        "attempts_per_second": _rate(attempts, duration_seconds),
        "latency_ms": latency_summary(latencies),
        "recall_at_k": _mean(recalls) if recalls else None,
        "recall_samples": len(recalls),
    }
    if stage_summaries:
        summary["stages"] = stage_summaries
    return summary


def _query_stage_summary(
    *,
    stage_index: int,
    concurrency: int,
    configured_duration_seconds: float,
    elapsed_seconds: float,
    state: QueryRunState,
) -> dict[str, Any]:
    attempts = state.queries + state.errors
    return {
        "stage_index": stage_index,
        "concurrency": concurrency,
        "configured_duration_seconds": configured_duration_seconds,
        "duration_seconds": elapsed_seconds,
        "queries": state.queries,
        "errors": state.errors,
        "attempts": attempts,
        "error_rate": _error_rate(state.errors, attempts),
        "queries_per_second": _rate(state.queries, elapsed_seconds),
        "attempts_per_second": _rate(attempts, elapsed_seconds),
        "latency_ms": latency_summary(state.latencies),
        "recall_at_k": _mean(state.recalls) if state.recalls else None,
        "recall_samples": len(state.recalls),
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


def _query_stages(scenario: ScenarioConfig) -> list[dict[str, Any]] | None:
    stages = scenario.query.get("stages")
    if not stages:
        return None
    if not isinstance(stages, list):
        raise ConfigError("scenario.query.stages must be a list")
    return [
        dict(_as_stage_mapping(stage, stage_index=index))
        for index, stage in enumerate(stages, start=1)
    ]


def _as_stage_mapping(stage: Any, *, stage_index: int) -> Mapping[str, Any]:
    if not isinstance(stage, Mapping):
        raise ConfigError(f"scenario.query.stages[{stage_index}] must be a mapping")
    return stage


def _stage_concurrency(stage: Mapping[str, Any], *, stage_index: int) -> int:
    concurrency = stage.get("concurrency")
    if not isinstance(concurrency, int) or concurrency <= 0:
        raise ConfigError(
            f"scenario.query.stages[{stage_index}].concurrency "
            "must be a positive integer"
        )
    return concurrency


def _stage_duration(stage: Mapping[str, Any], *, stage_index: int) -> str:
    duration = stage.get("duration")
    if not isinstance(duration, str):
        raise ConfigError(
            f"scenario.query.stages[{stage_index}].duration must be a string"
        )
    return duration


def parse_duration_seconds(value: str) -> float:
    text = value.strip().lower()
    units = {
        "ms": 0.001,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
    }
    multiplier = 1.0
    for suffix, unit_multiplier in units.items():
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            multiplier = unit_multiplier
            break
    else:
        number = text
    try:
        seconds = float(number) * multiplier
    except ValueError as exc:
        raise ConfigError(f"invalid duration {value!r}") from exc
    if seconds <= 0:
        raise ConfigError(f"duration {value!r} must be positive")
    return seconds


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


def _error_rate(error_count: int, attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return error_count / attempts


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
