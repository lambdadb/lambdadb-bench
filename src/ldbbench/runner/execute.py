"""Sequential benchmark execution for prepared dataset artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from ldbbench.progress import ProgressCallback, ProgressTicker
from ldbbench.runner.plan import build_run_plan

INGEST_EVENTS_FILENAME = "ingest_events.jsonl"
QUERY_EVENTS_FILENAME = "query_events.jsonl"
SUMMARY_FILENAME = "summary.json"
LOAD_CHECKPOINT_FILENAME = "load_checkpoint.json"
LARGE_RUN_ROW_THRESHOLD = 1_000_000
LOAD_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkRunResult:
    output_dir: Path
    ingest_events_path: Path
    query_events_path: Path
    load_checkpoint_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class LoadStageResult:
    summary: dict[str, Any]
    visibility_samples: list[VectorRecord]


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
    load_only: bool = False,
    query_only: bool = False,
    resume_load: bool = False,
    allow_destructive: bool = False,
    allow_large_run: bool = False,
    progress: ProgressCallback | None = None,
) -> BenchmarkRunResult:
    """Execute a small or explicitly opted-in benchmark run sequentially."""

    if load_only and query_only:
        raise ConfigError("--load-only and --query-only cannot be used together")
    if query_only and target.prepare_mode != "existing":
        raise ConfigError("--query-only requires target prepare.mode: existing")
    if resume_load and query_only:
        raise ConfigError("--resume-load cannot be used with --query-only")
    if resume_load and target.prepare_mode != "existing":
        raise ConfigError("--resume-load requires target prepare.mode: existing")

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
    load_checkpoint_path = out / LOAD_CHECKPOINT_FILENAME
    summary_path = out / SUMMARY_FILENAME

    ticker = ProgressTicker(progress)
    ticker.emit(
        f"run: preparing target vendor={target.vendor} mode={target.prepare_mode}"
    )
    adapter.prepare(
        target,
        dimensions=_dataset_dimensions(scenario, dataset_manifest),
        metric=_dataset_metric(scenario, dataset_manifest),
    )
    ticker.emit("run: target prepared")

    if query_only:
        ticker.emit("run: skipping load query_only=true")
        ingest_events_path.write_text("", encoding="utf-8")
        ingest_summary = skipped_load_summary(reason="query_only")
    else:
        ticker.emit("run: starting load stage")
        load_result = run_load_stage(
            adapter=adapter,
            target=target,
            records=read_records(records_path, limit=max_records),
            batch_size=_batch_size(scenario),
            max_batch_bytes=_max_batch_bytes(scenario),
            concurrency=_load_concurrency(scenario),
            events_path=ingest_events_path,
            checkpoint_path=load_checkpoint_path,
            checkpoint_context=_load_checkpoint_context(
                records_path=records_path,
                dataset_manifest=dataset_manifest,
                scenario=scenario,
                target=target,
                batch_size=_batch_size(scenario),
                max_batch_bytes=_max_batch_bytes(scenario),
                max_records=max_records,
            ),
            resume_load=resume_load,
            progress=progress,
        )
        ingest_summary = load_result.summary
        ticker.emit(
            "run: load stage finished "
            f"status={ingest_summary['status']} records={ingest_summary['records']} "
            f"errors={ingest_summary['errors']}"
        )
        if ingest_summary["errors"] == 0 and _wait_until_query_visible(scenario):
            ticker.emit("run: waiting for query visibility")
            ingest_summary["visibility"] = wait_until_query_visible(
                adapter=adapter,
                target=target,
                records=load_result.visibility_samples,
                top_k=_top_k(scenario),
                consistency=str(scenario.query.get("consistency", "eventual")),
                timeout_seconds=_visibility_timeout_seconds(scenario),
                poll_interval_seconds=_visibility_poll_interval_seconds(scenario),
                progress=progress,
            )
            ticker.emit(
                "run: query visibility "
                f"status={ingest_summary['visibility']['status']} "
                f"visible={ingest_summary['visibility']['visible']}/"
                f"{ingest_summary['visibility']['samples']}"
            )

    skip_reason = _query_skip_reason(load_summary=ingest_summary, load_only=load_only)
    if skip_reason is not None:
        ticker.emit(f"run: skipping query reason={skip_reason}")
        query_events_path.write_text("", encoding="utf-8")
        query_summary = skipped_query_summary(reason=skip_reason)
    else:
        ticker.emit("run: loading query vectors")
        queries = list(read_records(queries_path, limit=max_queries))
        ticker.emit(f"run: starting query stage query_vectors={len(queries)}")
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
            progress=progress,
        )
        ticker.emit(
            "run: query stage finished "
            f"mode={query_summary['mode']} queries={query_summary['queries']} "
            f"errors={query_summary['errors']}"
        )

    summary = {
        "status": run_status(load_summary=ingest_summary, query_summary=query_summary),
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
    ticker.emit(f"run: wrote summary status={summary['status']}")

    return BenchmarkRunResult(
        output_dir=out,
        ingest_events_path=ingest_events_path,
        query_events_path=query_events_path,
        load_checkpoint_path=load_checkpoint_path,
        summary_path=summary_path,
        summary=summary,
    )


def run_load_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    records: Iterable[VectorRecord],
    batch_size: int,
    max_batch_bytes: int | None,
    concurrency: int,
    events_path: str | Path,
    checkpoint_path: str | Path | None = None,
    checkpoint_context: Mapping[str, Any] | None = None,
    resume_load: bool = False,
    visibility_sample_size: int = 10,
    progress: ProgressCallback | None = None,
) -> LoadStageResult:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_context_dict = dict(checkpoint_context or {})
    resumed_from_batch_index = _resume_batch_index(
        checkpoint_output,
        context=checkpoint_context_dict,
        resume_load=resume_load,
    )

    records_loaded = 0
    records_read = 0
    skipped_records = 0
    load_errors = 0
    latencies: list[float] = []
    attempt_latencies: list[float] = []
    upsert_attempt_duration_seconds = 0.0
    visibility_samples: list[VectorRecord] = []
    successful_batches = 0
    skipped_batches = 0
    attempted_batches = 0
    batching_duration_seconds = 0.0
    successful_batch_indexes: set[int] = set()
    highest_contiguous_successful_batch_index = resumed_from_batch_index
    started = time.perf_counter()
    ticker = ProgressTicker(progress)
    ticker.emit(
        f"load: starting batch_size={batch_size} concurrency={concurrency} "
        f"resume={resume_load}"
    )
    if resumed_from_batch_index:
        ticker.emit(f"load: resuming after batch={resumed_from_batch_index}")
    _write_load_checkpoint(
        checkpoint_output,
        context=checkpoint_context_dict,
        status="in_progress",
        resumed_from_batch_index=resumed_from_batch_index,
        highest_contiguous_successful_batch_index=(
            highest_contiguous_successful_batch_index
        ),
        successful_batch_indexes=successful_batch_indexes,
        attempted_batches=attempted_batches,
        successful_batches=successful_batches,
        skipped_batches=skipped_batches,
        skipped_records=skipped_records,
        records_loaded=records_loaded,
        records_read=records_read,
        batching_duration_seconds=batching_duration_seconds,
        upsert_attempt_duration_seconds=upsert_attempt_duration_seconds,
        errors=load_errors,
    )
    events_mode = "a" if resume_load else "w"
    with events_output.open(events_mode, encoding="utf-8") as file:
        batch_iter = enumerate(
            _batches(records, batch_size, max_batch_bytes=max_batch_bytes),
            start=1,
        )

        if concurrency == 1:
            while True:
                item = _next_load_batch(batch_iter)
                if item is None:
                    break
                batch_index, batch, batching_duration = item
                batching_duration_seconds += batching_duration
                records_read += len(batch)
                if batch_index <= resumed_from_batch_index:
                    skipped_batches += 1
                    skipped_records += len(batch)
                    continue
                event = execute_load_batch(
                    adapter=adapter,
                    target=target,
                    batch=batch,
                    batch_index=batch_index,
                )
                attempted_batches += 1
                _write_event(file, event)
                event_latency = float(event["latency_ms"])
                attempt_latencies.append(event_latency)
                upsert_attempt_duration_seconds += event_latency / 1000
                if event["status"] != "ok":
                    load_errors += 1
                    _write_load_checkpoint(
                        checkpoint_output,
                        context=checkpoint_context_dict,
                        status="failed",
                        resumed_from_batch_index=resumed_from_batch_index,
                        highest_contiguous_successful_batch_index=(
                            highest_contiguous_successful_batch_index
                        ),
                        successful_batch_indexes=successful_batch_indexes,
                        attempted_batches=attempted_batches,
                        successful_batches=successful_batches,
                        skipped_batches=skipped_batches,
                        skipped_records=skipped_records,
                        records_loaded=records_loaded,
                        records_read=records_read,
                        batching_duration_seconds=batching_duration_seconds,
                        upsert_attempt_duration_seconds=(
                            upsert_attempt_duration_seconds
                        ),
                        errors=load_errors,
                    )
                    break
                successful_batches += 1
                successful_batch_indexes.add(batch_index)
                highest_contiguous_successful_batch_index = (
                    _advance_contiguous_watermark(
                        highest_contiguous_successful_batch_index,
                        successful_batch_indexes,
                    )
                )
                records_loaded += int(event["records"])
                latencies.append(float(event["latency_ms"]))
                _write_load_checkpoint(
                    checkpoint_output,
                    context=checkpoint_context_dict,
                    status="in_progress",
                    resumed_from_batch_index=resumed_from_batch_index,
                    highest_contiguous_successful_batch_index=(
                        highest_contiguous_successful_batch_index
                    ),
                    successful_batch_indexes=successful_batch_indexes,
                    attempted_batches=attempted_batches,
                    successful_batches=successful_batches,
                        skipped_batches=skipped_batches,
                        skipped_records=skipped_records,
                        records_loaded=records_loaded,
                        records_read=records_read,
                        batching_duration_seconds=batching_duration_seconds,
                        upsert_attempt_duration_seconds=(
                            upsert_attempt_duration_seconds
                        ),
                        errors=load_errors,
                    )
                ticker.maybe(
                    "load: progress "
                    f"records={records_loaded} batches={successful_batches} "
                    f"errors={load_errors}"
                )
                _extend_visibility_samples(
                    visibility_samples,
                    batch,
                    visibility_sample_size=visibility_sample_size,
                )
        else:
            pending: dict[Future[tuple[dict[str, Any], list[VectorRecord]]], None] = {}
            accepting = True

            def submit_next(executor: ThreadPoolExecutor) -> bool:
                nonlocal attempted_batches
                nonlocal batching_duration_seconds, records_read
                nonlocal skipped_batches, skipped_records
                nonlocal upsert_attempt_duration_seconds
                if not accepting:
                    return False
                while True:
                    item = _next_load_batch(batch_iter)
                    if item is None:
                        return False
                    batch_index, batch, batching_duration = item
                    batching_duration_seconds += batching_duration
                    records_read += len(batch)
                    if batch_index <= resumed_from_batch_index:
                        skipped_batches += 1
                        skipped_records += len(batch)
                        continue
                    attempted_batches += 1
                    future = executor.submit(
                        _execute_load_batch_with_records,
                        adapter,
                        target,
                        batch,
                        batch_index,
                    )
                    pending[future] = None
                    return True

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                for _ in range(concurrency):
                    if not submit_next(executor):
                        break

                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        event, batch = future.result()
                        _write_event(file, event)
                        event_latency = float(event["latency_ms"])
                        attempt_latencies.append(event_latency)
                        upsert_attempt_duration_seconds += event_latency / 1000
                        if event["status"] != "ok":
                            load_errors += 1
                            accepting = False
                            _write_load_checkpoint(
                                checkpoint_output,
                                context=checkpoint_context_dict,
                                status="failed",
                                resumed_from_batch_index=resumed_from_batch_index,
                                highest_contiguous_successful_batch_index=(
                                    highest_contiguous_successful_batch_index
                                ),
                                successful_batch_indexes=successful_batch_indexes,
                                attempted_batches=attempted_batches,
                                successful_batches=successful_batches,
                                skipped_batches=skipped_batches,
                                skipped_records=skipped_records,
                                records_loaded=records_loaded,
                                records_read=records_read,
                                batching_duration_seconds=batching_duration_seconds,
                                upsert_attempt_duration_seconds=(
                                    upsert_attempt_duration_seconds
                                ),
                                errors=load_errors,
                            )
                            continue
                        successful_batches += 1
                        successful_batch_indexes.add(int(event["batch_index"]))
                        highest_contiguous_successful_batch_index = (
                            _advance_contiguous_watermark(
                                highest_contiguous_successful_batch_index,
                                successful_batch_indexes,
                            )
                        )
                        records_loaded += int(event["records"])
                        latencies.append(float(event["latency_ms"]))
                        _write_load_checkpoint(
                            checkpoint_output,
                            context=checkpoint_context_dict,
                            status="in_progress",
                            resumed_from_batch_index=resumed_from_batch_index,
                            highest_contiguous_successful_batch_index=(
                                highest_contiguous_successful_batch_index
                            ),
                            successful_batch_indexes=successful_batch_indexes,
                            attempted_batches=attempted_batches,
                            successful_batches=successful_batches,
                            skipped_batches=skipped_batches,
                            skipped_records=skipped_records,
                            records_loaded=records_loaded,
                            records_read=records_read,
                            batching_duration_seconds=batching_duration_seconds,
                            upsert_attempt_duration_seconds=(
                                upsert_attempt_duration_seconds
                            ),
                            errors=load_errors,
                        )
                        ticker.maybe(
                            "load: progress "
                            f"records={records_loaded} batches={successful_batches} "
                            f"errors={load_errors}"
                        )
                        _extend_visibility_samples(
                            visibility_samples,
                            batch,
                            visibility_sample_size=visibility_sample_size,
                        )
                    while load_errors == 0 and len(pending) < concurrency:
                        if not submit_next(executor):
                            break

    duration_seconds = time.perf_counter() - started
    final_status = "completed" if load_errors == 0 else "failed"
    _write_load_checkpoint(
        checkpoint_output,
        context=checkpoint_context_dict,
        status=final_status,
        resumed_from_batch_index=resumed_from_batch_index,
        highest_contiguous_successful_batch_index=(
            highest_contiguous_successful_batch_index
        ),
        successful_batch_indexes=successful_batch_indexes,
        attempted_batches=attempted_batches,
        successful_batches=successful_batches,
        skipped_batches=skipped_batches,
        skipped_records=skipped_records,
        records_loaded=records_loaded,
        records_read=records_read,
        batching_duration_seconds=batching_duration_seconds,
        upsert_attempt_duration_seconds=upsert_attempt_duration_seconds,
        errors=load_errors,
    )
    ticker.emit(
        "load: finished "
        f"records={records_loaded} batches={successful_batches} errors={load_errors}"
    )
    return LoadStageResult(
        summary={
            "status": final_status,
            "records": records_loaded,
            "records_read": records_read,
            "skipped_records": skipped_records,
            "batches": successful_batches,
            "skipped_batches": skipped_batches,
            "attempts": attempted_batches,
            "errors": load_errors,
            "error_rate": _error_rate(load_errors, attempted_batches),
            "concurrency": concurrency,
            "checkpoint": {
                "path": str(checkpoint_output) if checkpoint_output else None,
                "resume_enabled": resume_load,
                "resumed_from_batch_index": resumed_from_batch_index,
                "highest_contiguous_successful_batch_index": (
                    highest_contiguous_successful_batch_index
                ),
            },
            "duration_seconds": duration_seconds,
            "batching_duration_seconds": batching_duration_seconds,
            "upsert_attempt_duration_seconds": upsert_attempt_duration_seconds,
            "records_per_second": _rate(records_loaded, duration_seconds),
            "batching_records_per_second": _rate(
                records_read,
                batching_duration_seconds,
            ),
            "attempts_per_second": _rate(attempted_batches, duration_seconds),
            "latency_ms": latency_summary(latencies),
            "attempt_latency_ms": latency_summary(attempt_latencies),
        },
        visibility_samples=visibility_samples,
    )


def _execute_load_batch_with_records(
    adapter: VectorDBAdapter,
    target: TargetConfig,
    batch: list[VectorRecord],
    batch_index: int,
) -> tuple[dict[str, Any], list[VectorRecord]]:
    return (
        execute_load_batch(
            adapter=adapter,
            target=target,
            batch=batch,
            batch_index=batch_index,
        ),
        batch,
    )


def _next_load_batch(
    batch_iter: Iterator[tuple[int, list[VectorRecord]]],
) -> tuple[int, list[VectorRecord], float] | None:
    started = time.perf_counter()
    try:
        batch_index, batch = next(batch_iter)
    except StopIteration:
        return None
    return batch_index, batch, time.perf_counter() - started


def execute_load_batch(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    batch: list[VectorRecord],
    batch_index: int,
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    try:
        result = adapter.upsert_batch(target, batch)
    except Exception as exc:  # noqa: BLE001
        return {
            "stage": "load",
            "batch_index": batch_index,
            "records": len(batch),
            "latency_ms": _elapsed_ms(batch_started),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "stage": "load",
        "batch_index": batch_index,
        "records": result.count,
        "latency_ms": _elapsed_ms(batch_started),
        "status": "ok",
    }


def _extend_visibility_samples(
    visibility_samples: list[VectorRecord],
    batch: list[VectorRecord],
    *,
    visibility_sample_size: int,
) -> None:
    if len(visibility_samples) >= visibility_sample_size:
        return
    remaining = visibility_sample_size - len(visibility_samples)
    visibility_samples.extend(batch[:remaining])


def _resume_batch_index(
    checkpoint_path: Path | None,
    *,
    context: Mapping[str, Any],
    resume_load: bool,
) -> int:
    if not resume_load:
        return 0
    if checkpoint_path is None:
        raise ConfigError("--resume-load requires a load checkpoint path")
    if not checkpoint_path.exists():
        raise ConfigError(f"load checkpoint {checkpoint_path} does not exist")
    checkpoint = _read_load_checkpoint(checkpoint_path)
    checkpoint_context = checkpoint.get("context")
    if checkpoint_context != dict(context):
        raise ConfigError(
            "load checkpoint does not match this run's dataset, target, or "
            "load settings"
        )
    batch_index = checkpoint.get("highest_contiguous_successful_batch_index")
    if not isinstance(batch_index, int) or batch_index < 0:
        raise ConfigError(
            f"load checkpoint {checkpoint_path} has an invalid batch watermark"
        )
    return batch_index


def _read_load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read load checkpoint {checkpoint_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"could not parse load checkpoint {checkpoint_path}") from exc
    if not isinstance(checkpoint, dict):
        raise ConfigError(f"load checkpoint {checkpoint_path} must be a JSON object")
    if checkpoint.get("schema_version") != LOAD_CHECKPOINT_SCHEMA_VERSION:
        raise ConfigError(f"load checkpoint {checkpoint_path} has unsupported schema")
    return checkpoint


def _write_load_checkpoint(
    checkpoint_path: Path | None,
    *,
    context: Mapping[str, Any],
    status: str,
    resumed_from_batch_index: int,
    highest_contiguous_successful_batch_index: int,
    successful_batch_indexes: set[int],
    attempted_batches: int,
    successful_batches: int,
    skipped_batches: int,
    skipped_records: int,
    records_loaded: int,
    records_read: int,
    batching_duration_seconds: float,
    upsert_attempt_duration_seconds: float,
    errors: int,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LOAD_CHECKPOINT_SCHEMA_VERSION,
        "stage": "load",
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "context": dict(context),
        "resumed_from_batch_index": resumed_from_batch_index,
        "highest_contiguous_successful_batch_index": (
            highest_contiguous_successful_batch_index
        ),
        "successful_batch_indexes": sorted(successful_batch_indexes),
        "attempted_batches": attempted_batches,
        "successful_batches": successful_batches,
        "skipped_batches": skipped_batches,
        "skipped_records": skipped_records,
        "records_loaded": records_loaded,
        "records_read": records_read,
        "batching_duration_seconds": batching_duration_seconds,
        "upsert_attempt_duration_seconds": upsert_attempt_duration_seconds,
        "errors": errors,
    }
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _advance_contiguous_watermark(
    current_watermark: int,
    successful_batch_indexes: set[int],
) -> int:
    watermark = current_watermark
    while watermark + 1 in successful_batch_indexes:
        watermark += 1
    return watermark


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
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    query_list = list(queries)
    ticker = ProgressTicker(progress)
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
            progress=progress,
        )

    state = QueryRunState()
    started = time.perf_counter()
    ticker.emit(f"query: starting one_pass queries={len(query_list)}")
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
            ticker.maybe(
                "query: one_pass progress "
                f"attempts={state.queries + state.errors}/{len(query_list)} "
                f"errors={state.errors}"
            )

    ticker.emit(
        f"query: one_pass finished queries={state.queries} errors={state.errors}"
    )
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
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    query_lock = Lock()
    write_lock = Lock()
    state = QueryRunState()
    next_query_index = 1
    started = time.perf_counter()
    stage_summaries: list[dict[str, Any]] = []
    ticker = ProgressTicker(progress)

    with events_path.open("w", encoding="utf-8") as file:
        for stage_index, stage in enumerate(stages, start=1):
            concurrency = _stage_concurrency(stage, stage_index=stage_index)
            duration_seconds = parse_duration_seconds(
                _stage_duration(stage, stage_index=stage_index)
            )
            deadline = time.perf_counter() + duration_seconds
            stage_state = QueryRunState()
            stage_started = time.perf_counter()
            ticker.emit(
                "query: starting stage "
                f"stage={stage_index}/{len(stages)} concurrency={concurrency} "
                f"duration_seconds={duration_seconds}"
            )

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
                current_stage_started: float = stage_started,
                current_duration_seconds: float = duration_seconds,
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
                        elapsed = time.perf_counter() - current_stage_started
                        attempts = (
                            current_stage_state.queries + current_stage_state.errors
                        )
                        ticker.maybe(
                            "query: stage progress "
                            f"stage={current_stage_index} "
                            f"elapsed_seconds={elapsed:.1f}/"
                            f"{current_duration_seconds:.1f} "
                            f"attempts={attempts} "
                            f"errors={current_stage_state.errors}"
                        )

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
            ticker.emit(
                "query: stage finished "
                f"stage={stage_index} queries={stage_state.queries} "
                f"errors={stage_state.errors}"
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


def skipped_query_summary(*, reason: str) -> dict[str, Any]:
    summary = _query_summary(
        mode="skipped",
        started=time.perf_counter(),
        latencies=[],
        recalls=[],
        query_count=0,
        error_count=0,
        stage_summaries=[],
    )
    summary["skip_reason"] = reason
    return summary


def skipped_load_summary(*, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "records": 0,
        "records_read": 0,
        "skipped_records": 0,
        "batches": 0,
        "skipped_batches": 0,
        "attempts": 0,
        "errors": 0,
        "error_rate": 0.0,
        "duration_seconds": 0.0,
        "batching_duration_seconds": 0.0,
        "upsert_attempt_duration_seconds": 0.0,
        "records_per_second": 0.0,
        "batching_records_per_second": 0.0,
        "latency_ms": latency_summary([]),
        "attempt_latency_ms": latency_summary([]),
    }


def wait_until_query_visible(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    records: list[VectorRecord],
    top_k: int,
    consistency: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "enabled": True,
            "status": "skipped",
            "skip_reason": "no_loaded_records",
            "samples": 0,
            "visible": 0,
            "pending_ids": [],
            "attempts": 0,
            "duration_seconds": 0.0,
            "latency_ms": latency_summary([]),
        }

    pending = {record.id: record for record in records}
    attempts = 0
    latencies: list[float] = []
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_error: dict[str, str] | None = None
    ticker = ProgressTicker(progress)
    ticker.emit(
        f"visibility: waiting samples={len(records)} "
        f"timeout_seconds={timeout_seconds}"
    )

    while pending and time.perf_counter() <= deadline:
        for record_id, record in list(pending.items()):
            query_started = time.perf_counter()
            attempts += 1
            try:
                result = adapter.query(
                    target,
                    vector=record.vector,
                    top_k=top_k,
                    consistency=consistency,
                    include_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                continue
            latencies.append(_elapsed_ms(query_started))
            if record_id in {match.id for match in result.matches}:
                pending.pop(record_id, None)
            ticker.maybe(
                "visibility: progress "
                f"visible={len(records) - len(pending)}/{len(records)} "
                f"attempts={attempts}"
            )

        if pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_seconds, remaining))

    duration_seconds = time.perf_counter() - started
    status = "visible" if not pending else "timeout"
    ticker.emit(
        "visibility: finished "
        f"status={status} visible={len(records) - len(pending)}/{len(records)}"
    )
    summary: dict[str, Any] = {
        "enabled": True,
        "status": status,
        "samples": len(records),
        "visible": len(records) - len(pending),
        "pending_ids": sorted(pending),
        "attempts": attempts,
        "duration_seconds": duration_seconds,
        "latency_ms": latency_summary(latencies),
    }
    if last_error is not None:
        summary.update(last_error)
    return summary


def run_status(
    *,
    load_summary: Mapping[str, Any],
    query_summary: Mapping[str, Any],
) -> str:
    if load_summary.get("errors", 0):
        return "failed"
    visibility = load_summary.get("visibility")
    if isinstance(visibility, Mapping) and visibility.get("status") == "timeout":
        return "failed"
    if query_summary.get("errors", 0):
        return "completed_with_errors"
    return "completed"


def _query_skip_reason(
    *,
    load_summary: Mapping[str, Any],
    load_only: bool,
) -> str | None:
    if load_summary.get("errors", 0):
        return "load_failed"
    visibility = load_summary.get("visibility")
    if isinstance(visibility, Mapping) and visibility.get("status") == "timeout":
        return "visibility_timeout"
    if load_only:
        return "load_only"
    return None


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


def _load_checkpoint_context(
    *,
    records_path: Path,
    dataset_manifest: Mapping[str, Any],
    scenario: ScenarioConfig,
    target: TargetConfig,
    batch_size: int,
    max_batch_bytes: int | None,
    max_records: int | None,
) -> dict[str, Any]:
    artifacts = dataset_manifest.get("artifacts", {})
    artifact_checksums = artifacts if isinstance(artifacts, Mapping) else {}
    return {
        "scenario_name": scenario.name,
        "target_vendor": target.vendor,
        "target_name": target.name,
        "target_collection_name": target.collection_name,
        "records_path": str(records_path),
        "records_sha256": artifact_checksums.get("records_sha256"),
        "batch_size": batch_size,
        "max_batch_bytes": max_batch_bytes,
        "max_records": max_records,
    }


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
    *,
    max_batch_bytes: int | None = None,
) -> Iterable[list[VectorRecord]]:
    batch: list[VectorRecord] = []
    batch_bytes = 0
    for record in records:
        record_bytes = _record_size_bytes(record)
        if (
            batch
            and max_batch_bytes is not None
            and batch_bytes + record_bytes > max_batch_bytes
        ):
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(record)
        batch_bytes += record_bytes
        if len(batch) >= batch_size:
            yield batch
            batch = []
            batch_bytes = 0
    if batch:
        yield batch


def _record_size_bytes(record: VectorRecord) -> int:
    payload = {
        "id": record.id,
        "metadata": dict(record.metadata),
        "vector": list(record.vector),
    }
    return len(json.dumps(payload, sort_keys=True).encode("utf-8")) + 2


def _batch_size(scenario: ScenarioConfig) -> int:
    batch_size = scenario.load.get("batch_size", 100)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigError("scenario.load.batch_size must be a positive integer")
    return batch_size


def _load_concurrency(scenario: ScenarioConfig) -> int:
    concurrency = scenario.load.get("concurrency", 1)
    if not isinstance(concurrency, int) or concurrency <= 0:
        raise ConfigError("scenario.load.concurrency must be a positive integer")
    return concurrency


def _max_batch_bytes(scenario: ScenarioConfig) -> int | None:
    value = scenario.load.get("max_batch_bytes")
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ConfigError("scenario.load.max_batch_bytes must be positive")
        return value
    if isinstance(value, str):
        return parse_size_bytes(value)
    raise ConfigError("scenario.load.max_batch_bytes must be an integer or string")


def parse_size_bytes(value: str) -> int:
    text = value.strip().lower()
    units = {
        "kb": 1_000,
        "kib": 1024,
        "mb": 1_000_000,
        "mib": 1024 * 1024,
        "gb": 1_000_000_000,
        "gib": 1024 * 1024 * 1024,
        "b": 1,
    }
    multiplier = 1
    for suffix, unit_multiplier in sorted(
        units.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            multiplier = unit_multiplier
            break
    else:
        number = text
    try:
        size = float(number) * multiplier
    except ValueError as exc:
        raise ConfigError(f"invalid byte size {value!r}") from exc
    if size <= 0:
        raise ConfigError(f"byte size {value!r} must be positive")
    return int(size)


def _wait_until_query_visible(scenario: ScenarioConfig) -> bool:
    value = scenario.load.get("wait_until_query_visible", False)
    if not isinstance(value, bool):
        raise ConfigError("scenario.load.wait_until_query_visible must be a boolean")
    return value


def _visibility_timeout_seconds(scenario: ScenarioConfig) -> float:
    value = scenario.load.get("query_visibility_timeout", "60s")
    if not isinstance(value, str):
        raise ConfigError("scenario.load.query_visibility_timeout must be a string")
    return parse_duration_seconds(value)


def _visibility_poll_interval_seconds(scenario: ScenarioConfig) -> float:
    value = scenario.load.get("query_visibility_poll_interval", "1s")
    if not isinstance(value, str):
        raise ConfigError(
            "scenario.load.query_visibility_poll_interval must be a string"
        )
    return parse_duration_seconds(value)


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
