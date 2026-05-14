"""Sequential benchmark execution for prepared dataset artifacts."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

import msgpack

from ldbbench.adapters.base import VectorDBAdapter, VectorRecord
from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets.ground_truth import (
    GROUND_TRUTH_FILENAME,
    artifact_path,
    load_dataset_manifest,
)
from ldbbench.datasets.prepare import (
    QUERIES_FILENAME,
    QUERIES_MSGPACK_FILENAME,
    RECORDS_FILENAME,
    RECORDS_MSGPACK_FILENAME,
)
from ldbbench.manifest import initialize_run_artifacts
from ldbbench.progress import ProgressCallback, ProgressTicker
from ldbbench.runner.plan import build_run_plan

try:
    import orjson
except ImportError:  # pragma: no cover - kept for source-tree reuse without deps.
    orjson = None  # type: ignore[assignment]

INGEST_EVENTS_FILENAME = "ingest_events.jsonl"
QUERY_EVENTS_FILENAME = "query_events.jsonl"
SUMMARY_FILENAME = "summary.json"
LOAD_CHECKPOINT_FILENAME = "load_checkpoint.json"
LARGE_RUN_ROW_THRESHOLD = 1_000_000
LOAD_CHECKPOINT_SCHEMA_VERSION = 1
QUERY_EVENT_FLUSH_INTERVAL = 1000


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


@dataclass(frozen=True)
class PartitionFilterSpec:
    field: str
    metadata_field: str

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "metadata_field": self.metadata_field,
        }


@dataclass(frozen=True)
class RecordShard:
    path: Path
    sha256: str | None
    records: int
    first_record_index: int
    last_record_index: int
    batch_base: int = 0
    effective_records: int | None = None
    manifest_shard_count: int | None = None

    def as_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "records": self.records,
            "first_record_index": self.first_record_index,
            "last_record_index": self.last_record_index,
            "batch_base": self.batch_base,
            "effective_records": self.effective_records,
            "manifest_shard_count": self.manifest_shard_count,
        }


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
    records_path, records_sha256 = _preferred_artifact_path(
        dataset_path,
        dataset_manifest,
        preferred_key="records_msgpack",
        preferred_filename=RECORDS_MSGPACK_FILENAME,
        fallback_key="records",
        fallback_filename=RECORDS_FILENAME,
    )
    json_records_path = artifact_path(
        dataset_path,
        dataset_manifest,
        "records",
        RECORDS_FILENAME,
    )
    queries_path, _queries_sha256 = _preferred_artifact_path(
        dataset_path,
        dataset_manifest,
        preferred_key="queries_msgpack",
        preferred_filename=QUERIES_MSGPACK_FILENAME,
        fallback_key="queries",
        fallback_filename=QUERIES_FILENAME,
    )
    truth_path = _ground_truth_path(dataset_path, ground_truth_path)
    if ground_truth_path is not None and not truth_path.exists():
        raise ConfigError(f"ground truth file {truth_path} does not exist")
    ground_truth = load_ground_truth(truth_path) if truth_path.exists() else {}
    record_shards = None
    if not query_only:
        record_shards = _record_shards_for_load(
            scenario=scenario,
            dataset_dir=dataset_path,
            dataset_manifest=dataset_manifest,
            batch_size=_batch_size(scenario),
            max_batch_bytes=_max_batch_bytes(scenario),
            max_records=max_records,
        )

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
            record_source_path=records_path,
            record_shards=record_shards,
            write_mode=str(scenario.load.get("write_mode")),
            batch_size=_batch_size(scenario),
            max_batch_bytes=_max_batch_bytes(scenario),
            concurrency=_load_concurrency(scenario),
            processes=_load_processes(scenario),
            events_path=ingest_events_path,
            checkpoint_path=load_checkpoint_path,
            checkpoint_context=_load_checkpoint_context(
                records_path=records_path,
                records_sha256=records_sha256,
                json_records_path=json_records_path,
                dataset_manifest=dataset_manifest,
                scenario=scenario,
                target=target,
                write_mode=str(scenario.load.get("write_mode")),
                record_shards=record_shards,
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
            partition_filter_spec=_partition_filter_spec(scenario),
            events_path=query_events_path,
            stages=None if max_queries is not None else _query_stages(scenario),
            processes=_query_processes(scenario),
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
    record_source_path: str | Path | None = None,
    record_shards: Sequence[RecordShard] | None = None,
    write_mode: str,
    batch_size: int,
    max_batch_bytes: int | None,
    concurrency: int,
    processes: int = 1,
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
    if record_shards is not None:
        return _run_sharded_load_stage(
            adapter=adapter,
            target=target,
            record_shards=record_shards,
            write_mode=write_mode,
            batch_size=batch_size,
            concurrency=concurrency,
            processes=processes,
            events_output=events_output,
            checkpoint_output=checkpoint_output,
            checkpoint_context=checkpoint_context_dict,
            resume_load=resume_load,
            resumed_from_batch_index=resumed_from_batch_index,
            visibility_sample_size=visibility_sample_size,
            progress=progress,
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
    process_count = _effective_process_count(processes, concurrency)
    worker_threads = _split_concurrency(concurrency, process_count)
    ticker.emit(
        f"load: starting write_mode={write_mode} batch_size={batch_size} "
        f"concurrency={concurrency} "
        f"processes={process_count} worker_threads_per_process={worker_threads} "
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

        if process_count > 1:
            task_queue: Any = mp.Queue()
            result_queue: Any = mp.Queue()
            workers = [
                mp.Process(
                    target=_load_process_worker,
                    args=(
                        target.vendor,
                        target,
                        write_mode,
                        task_queue,
                        result_queue,
                        thread_count,
                    ),
                    name=f"ldbbench-load-{index}",
                )
                for index, thread_count in enumerate(worker_threads, start=1)
            ]
            for worker_process in workers:
                worker_process.start()

            accepting = True
            exhausted = False
            in_flight = 0

            def submit_next_process_batch() -> bool:
                nonlocal attempted_batches
                nonlocal batching_duration_seconds, records_read
                nonlocal skipped_batches, skipped_records
                nonlocal exhausted, in_flight
                if not accepting or exhausted:
                    return False
                while True:
                    item = _next_load_batch(batch_iter)
                    if item is None:
                        exhausted = True
                        return False
                    batch_index, batch, batching_duration = item
                    batching_duration_seconds += batching_duration
                    records_read += len(batch)
                    if batch_index <= resumed_from_batch_index:
                        skipped_batches += 1
                        skipped_records += len(batch)
                        continue
                    attempted_batches += 1
                    _extend_visibility_samples(
                        visibility_samples,
                        batch,
                        visibility_sample_size=visibility_sample_size,
                    )
                    task_queue.put((batch_index, batch))
                    in_flight += 1
                    return True

            try:
                while in_flight < concurrency and submit_next_process_batch():
                    pass

                while in_flight:
                    try:
                        event = result_queue.get(timeout=0.1)
                    except Empty:
                        for worker_process in workers:
                            if worker_process.exitcode not in (None, 0):
                                raise RuntimeError(
                                    "load worker process exited with "
                                    f"code {worker_process.exitcode}"
                                ) from None
                        continue
                    in_flight -= 1
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
                        upsert_attempt_duration_seconds=upsert_attempt_duration_seconds,
                        errors=load_errors,
                    )
                    ticker.maybe(
                        "load: progress "
                        f"records={records_loaded} batches={successful_batches} "
                        f"errors={load_errors}"
                    )
                    while (
                        load_errors == 0
                        and in_flight < concurrency
                        and submit_next_process_batch()
                    ):
                        pass
            finally:
                for _ in range(sum(worker_threads)):
                    task_queue.put(None)
                for worker_process in workers:
                    worker_process.join()
        elif concurrency == 1:
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
                    write_mode=write_mode,
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
                        write_mode,
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
            "write_mode": write_mode,
            "skipped_records": skipped_records,
            "batches": successful_batches,
            "skipped_batches": skipped_batches,
            "attempts": attempted_batches,
            "errors": load_errors,
            "error_rate": _error_rate(load_errors, attempted_batches),
            "concurrency": concurrency,
            "processes": process_count,
            "worker_threads_per_process": worker_threads,
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
            "record_source": _record_source_summary(record_source_path),
        },
        visibility_samples=visibility_samples,
    )


def _run_sharded_load_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    record_shards: Sequence[RecordShard],
    write_mode: str,
    batch_size: int,
    concurrency: int,
    processes: int,
    events_output: Path,
    checkpoint_output: Path | None,
    checkpoint_context: Mapping[str, Any],
    resume_load: bool,
    resumed_from_batch_index: int,
    visibility_sample_size: int,
    progress: ProgressCallback | None,
) -> LoadStageResult:
    manifest_shard_count = _manifest_shard_count(record_shards)
    records_loaded = 0
    records_read = 0
    skipped_records = 0
    load_errors = 0
    latencies: list[float] = []
    attempt_latencies: list[float] = []
    upsert_attempt_duration_seconds = 0.0
    batching_duration_seconds = 0.0
    visibility_samples: list[VectorRecord] = []
    successful_batches = 0
    skipped_batches = 0
    attempted_batches = 0
    successful_batch_indexes: set[int] = set()
    highest_contiguous_successful_batch_index = resumed_from_batch_index
    started = time.perf_counter()
    ticker = ProgressTicker(progress)
    process_count = min(
        _effective_process_count(processes, concurrency),
        len(record_shards),
    )
    process_count = max(1, process_count)
    worker_threads = _split_concurrency(concurrency, process_count)
    worker_shards = _split_shards(record_shards, process_count)
    ticker.emit(
        f"load: starting write_mode={write_mode} batch_size={batch_size} "
        f"concurrency={concurrency} processes={process_count} "
        f"worker_threads_per_process={worker_threads} sharded_records=true "
        f"effective_shards={len(record_shards)} "
        f"manifest_shards={manifest_shard_count} resume={resume_load}"
    )
    if resumed_from_batch_index:
        ticker.emit(f"load: resuming after batch={resumed_from_batch_index}")
    _write_load_checkpoint(
        checkpoint_output,
        context=checkpoint_context,
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
        if process_count == 1:
            state = {
                "records_loaded": 0,
                "records_read": 0,
                "skipped_records": 0,
                "load_errors": 0,
                "successful_batches": 0,
                "skipped_batches": 0,
                "attempted_batches": 0,
                "highest_contiguous_successful_batch_index": (
                    highest_contiguous_successful_batch_index
                ),
                "upsert_attempt_duration_seconds": 0.0,
                "batching_duration_seconds": 0.0,
            }
            done_message = _run_sharded_load_worker_local(
                adapter=adapter,
                target=target,
                write_mode=write_mode,
                record_shards=list(record_shards),
                batch_size=batch_size,
                concurrency=worker_threads[0],
                resumed_from_batch_index=resumed_from_batch_index,
                result_callback=lambda event: _handle_sharded_load_event(
                    event,
                    file=file,
                    checkpoint_output=checkpoint_output,
                    checkpoint_context=checkpoint_context,
                    resumed_from_batch_index=resumed_from_batch_index,
                    state=state,
                    successful_batch_indexes=successful_batch_indexes,
                    latencies=latencies,
                    attempt_latencies=attempt_latencies,
                    visibility_samples=visibility_samples,
                    visibility_sample_size=visibility_sample_size,
                    ticker=ticker,
                ),
            )
            state["records_read"] += int(done_message["records_read"])
            state["skipped_records"] += int(done_message["skipped_records"])
            state["skipped_batches"] += int(done_message["skipped_batches"])
            records_loaded = int(state["records_loaded"])
            records_read = int(state["records_read"])
            skipped_records = int(state["skipped_records"])
            load_errors = int(state["load_errors"])
            successful_batches = int(state["successful_batches"])
            skipped_batches = int(state["skipped_batches"])
            attempted_batches = int(state["attempted_batches"])
            highest_contiguous_successful_batch_index = int(
                state["highest_contiguous_successful_batch_index"]
            )
            upsert_attempt_duration_seconds = float(
                state["upsert_attempt_duration_seconds"]
            )
            batching_duration_seconds = float(state["batching_duration_seconds"])
        else:
            state = {
                "records_loaded": 0,
                "records_read": 0,
                "skipped_records": 0,
                "load_errors": 0,
                "successful_batches": 0,
                "skipped_batches": 0,
                "attempted_batches": 0,
                "highest_contiguous_successful_batch_index": (
                    highest_contiguous_successful_batch_index
                ),
                "upsert_attempt_duration_seconds": 0.0,
                "batching_duration_seconds": 0.0,
            }
            stop_event = mp.Event()
            result_queue: Any = mp.Queue()
            workers = [
                mp.Process(
                    target=_sharded_load_process_worker,
                    args=(
                        target.vendor,
                        target,
                        write_mode,
                        list(shards),
                        batch_size,
                        resumed_from_batch_index,
                        result_queue,
                        thread_count,
                        stop_event,
                        visibility_sample_size,
                    ),
                    name=f"ldbbench-sharded-load-{index}",
                )
                for index, (shards, thread_count) in enumerate(
                    zip(worker_shards, worker_threads, strict=True),
                    start=1,
                )
                if shards
            ]
            for worker_process in workers:
                worker_process.start()
            done_workers = 0
            try:
                while done_workers < len(workers):
                    try:
                        message = result_queue.get(timeout=0.1)
                    except Empty:
                        for worker_process in workers:
                            if worker_process.exitcode not in (None, 0):
                                raise RuntimeError(
                                    "sharded load worker process exited with "
                                    f"code {worker_process.exitcode}"
                                ) from None
                        continue
                    if message.get("_type") == "worker_done":
                        done_workers += 1
                        state["records_read"] += int(message["records_read"])
                        state["skipped_records"] += int(message["skipped_records"])
                        state["skipped_batches"] += int(message["skipped_batches"])
                        continue
                    _handle_sharded_load_event(
                        message,
                        file=file,
                        checkpoint_output=checkpoint_output,
                        checkpoint_context=checkpoint_context,
                        resumed_from_batch_index=resumed_from_batch_index,
                        state=state,
                        successful_batch_indexes=successful_batch_indexes,
                        latencies=latencies,
                        attempt_latencies=attempt_latencies,
                        visibility_samples=visibility_samples,
                        visibility_sample_size=visibility_sample_size,
                        ticker=ticker,
                    )
                    if state["load_errors"]:
                        stop_event.set()
            finally:
                stop_event.set()
                for worker_process in workers:
                    worker_process.join()
            records_loaded = int(state["records_loaded"])
            records_read = int(state["records_read"])
            skipped_records = int(state["skipped_records"])
            load_errors = int(state["load_errors"])
            successful_batches = int(state["successful_batches"])
            skipped_batches = int(state["skipped_batches"])
            attempted_batches = int(state["attempted_batches"])
            highest_contiguous_successful_batch_index = int(
                state["highest_contiguous_successful_batch_index"]
            )
            upsert_attempt_duration_seconds = float(
                state["upsert_attempt_duration_seconds"]
            )
            batching_duration_seconds = float(state["batching_duration_seconds"])

    duration_seconds = time.perf_counter() - started
    final_status = "completed" if load_errors == 0 else "failed"
    _write_load_checkpoint(
        checkpoint_output,
        context=checkpoint_context,
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
            "write_mode": write_mode,
            "skipped_records": skipped_records,
            "batches": successful_batches,
            "skipped_batches": skipped_batches,
            "attempts": attempted_batches,
            "errors": load_errors,
            "error_rate": _error_rate(load_errors, attempted_batches),
            "concurrency": concurrency,
            "processes": process_count,
            "worker_threads_per_process": worker_threads,
            "sharded_records": True,
            "shard_count": len(record_shards),
            "effective_shard_count": len(record_shards),
            "manifest_shard_count": manifest_shard_count,
            "worker_shards": [len(shards) for shards in worker_shards],
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
            "record_source": {
                "format": "msgpack",
                "path": None,
                "sharded": True,
                "shards": len(record_shards),
                "effective_shards": len(record_shards),
                "manifest_shards": manifest_shard_count,
            },
        },
        visibility_samples=visibility_samples,
    )


def _handle_sharded_load_event(
    event: dict[str, Any],
    *,
    file: Any,
    checkpoint_output: Path | None,
    checkpoint_context: Mapping[str, Any],
    resumed_from_batch_index: int,
    state: dict[str, Any],
    successful_batch_indexes: set[int],
    latencies: list[float],
    attempt_latencies: list[float],
    visibility_samples: list[VectorRecord],
    visibility_sample_size: int,
    ticker: ProgressTicker,
) -> None:
    samples = event.pop("_visibility_samples", [])
    batching_duration_ms = float(event.pop("_batching_duration_ms", 0.0))
    _write_event(file, event)
    event_latency = float(event["latency_ms"])
    attempt_latencies.append(event_latency)
    state["attempted_batches"] += 1
    state["upsert_attempt_duration_seconds"] += event_latency / 1000
    state["batching_duration_seconds"] += batching_duration_ms / 1000
    if event["status"] != "ok":
        state["load_errors"] += 1
        _write_load_checkpoint(
            checkpoint_output,
            context=checkpoint_context,
            status="failed",
            resumed_from_batch_index=resumed_from_batch_index,
            highest_contiguous_successful_batch_index=int(
                state["highest_contiguous_successful_batch_index"]
            ),
            successful_batch_indexes=successful_batch_indexes,
            attempted_batches=int(state["attempted_batches"]),
            successful_batches=int(state["successful_batches"]),
            skipped_batches=int(state["skipped_batches"]),
            skipped_records=int(state["skipped_records"]),
            records_loaded=int(state["records_loaded"]),
            records_read=int(state["records_read"]),
            batching_duration_seconds=float(state["batching_duration_seconds"]),
            upsert_attempt_duration_seconds=float(
                state["upsert_attempt_duration_seconds"]
            ),
            errors=int(state["load_errors"]),
        )
        return
    state["successful_batches"] += 1
    successful_batch_indexes.add(int(event["batch_index"]))
    state["highest_contiguous_successful_batch_index"] = (
        _advance_contiguous_watermark(
            int(state["highest_contiguous_successful_batch_index"]),
            successful_batch_indexes,
        )
    )
    state["records_loaded"] += int(event["records"])
    latencies.append(event_latency)
    if samples:
        _extend_visibility_samples(
            visibility_samples,
            samples,
            visibility_sample_size=visibility_sample_size,
        )
    _write_load_checkpoint(
        checkpoint_output,
        context=checkpoint_context,
        status="in_progress",
        resumed_from_batch_index=resumed_from_batch_index,
        highest_contiguous_successful_batch_index=int(
            state["highest_contiguous_successful_batch_index"]
        ),
        successful_batch_indexes=successful_batch_indexes,
        attempted_batches=int(state["attempted_batches"]),
        successful_batches=int(state["successful_batches"]),
        skipped_batches=int(state["skipped_batches"]),
        skipped_records=int(state["skipped_records"]),
        records_loaded=int(state["records_loaded"]),
        records_read=int(state["records_read"]),
        batching_duration_seconds=float(state["batching_duration_seconds"]),
        upsert_attempt_duration_seconds=float(state["upsert_attempt_duration_seconds"]),
        errors=int(state["load_errors"]),
    )
    ticker.maybe(
        "load: progress "
        f"records={state['records_loaded']} "
        f"batches={state['successful_batches']} errors={state['load_errors']}"
    )


def _run_sharded_load_worker_local(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    write_mode: str,
    record_shards: list[RecordShard],
    batch_size: int,
    concurrency: int,
    resumed_from_batch_index: int,
    result_callback: Any,
) -> dict[str, Any]:
    task_queue: Queue[Any] = Queue(maxsize=max(1, concurrency * 2))
    stop_event = Event()
    result_lock = Lock()
    stats = {
        "records_read": 0,
        "skipped_records": 0,
        "skipped_batches": 0,
    }

    def reader() -> None:
        try:
            for shard in record_shards:
                if stop_event.is_set():
                    break
                for batch_index, batch, batching_duration in _iter_shard_batches(
                    shard,
                    batch_size=batch_size,
                ):
                    stats["records_read"] += len(batch)
                    if batch_index <= resumed_from_batch_index:
                        stats["skipped_batches"] += 1
                        stats["skipped_records"] += len(batch)
                        continue
                    task_queue.put((batch_index, batch, batching_duration))
                    if stop_event.is_set():
                        break
        finally:
            for _ in range(concurrency):
                task_queue.put(None)

    def writer() -> None:
        while True:
            item = task_queue.get()
            if item is None:
                return
            batch_index, batch, batching_duration = item
            if stop_event.is_set():
                continue
            event = execute_load_batch(
                adapter=adapter,
                target=target,
                batch=batch,
                batch_index=batch_index,
                write_mode=write_mode,
            )
            event["_batching_duration_ms"] = batching_duration * 1000
            if event["status"] == "ok":
                event["_visibility_samples"] = batch
            with result_lock:
                result_callback(event)
            if event["status"] != "ok":
                stop_event.set()

    reader_thread = Thread(target=reader, name="ldbbench-sharded-reader-local")
    reader_thread.start()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(writer) for _ in range(concurrency)]
        for future in futures:
            future.result()
    reader_thread.join()
    return {
        "_type": "worker_done",
        **stats,
    }


def _sharded_load_process_worker(
    vendor: str,
    target: TargetConfig,
    write_mode: str,
    record_shards: list[RecordShard],
    batch_size: int,
    resumed_from_batch_index: int,
    result_queue: Any,
    concurrency: int,
    stop_event: Any,
    visibility_sample_size: int,
) -> None:
    task_queue: Queue[Any] = Queue(maxsize=max(1, concurrency * 2))
    stats = {
        "records_read": 0,
        "skipped_records": 0,
        "skipped_batches": 0,
    }

    def reader() -> None:
        try:
            for shard in record_shards:
                if stop_event.is_set():
                    break
                for batch_index, batch, batching_duration in _iter_shard_batches(
                    shard,
                    batch_size=batch_size,
                ):
                    stats["records_read"] += len(batch)
                    if batch_index <= resumed_from_batch_index:
                        stats["skipped_batches"] += 1
                        stats["skipped_records"] += len(batch)
                        continue
                    task_queue.put((batch_index, batch, batching_duration))
                    if stop_event.is_set():
                        break
        finally:
            for _ in range(concurrency):
                task_queue.put(None)

    def writer() -> None:
        adapter = _worker_adapter(vendor)
        samples_sent = 0
        while True:
            item = task_queue.get()
            if item is None:
                return
            batch_index, batch, batching_duration = item
            if stop_event.is_set():
                continue
            event = execute_load_batch(
                adapter=adapter,
                target=target,
                batch=batch,
                batch_index=batch_index,
                write_mode=write_mode,
            )
            event["_batching_duration_ms"] = batching_duration * 1000
            if event["status"] == "ok" and samples_sent < visibility_sample_size:
                remaining = visibility_sample_size - samples_sent
                event["_visibility_samples"] = batch[:remaining]
                samples_sent += len(event["_visibility_samples"])
            result_queue.put(event)
            if event["status"] != "ok":
                stop_event.set()

    reader_thread = Thread(target=reader, name="ldbbench-sharded-reader")
    reader_thread.start()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(writer) for _ in range(concurrency)]
        for future in futures:
            future.result()
    reader_thread.join()
    result_queue.put({"_type": "worker_done", **stats})


def _iter_shard_batches(
    shard: RecordShard,
    *,
    batch_size: int,
) -> Iterator[tuple[int, list[VectorRecord], float]]:
    records = read_records(shard.path, limit=shard.effective_records)
    for local_batch_index, (batch, batching_duration) in enumerate(
        _timed_batches(records, batch_size),
        start=1,
    ):
        batch_index = shard.batch_base + local_batch_index
        yield batch_index, batch, batching_duration


def _timed_batches(
    records: Iterable[VectorRecord],
    batch_size: int,
) -> Iterator[tuple[list[VectorRecord], float]]:
    batch: list[VectorRecord] = []
    started = time.perf_counter()
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch, time.perf_counter() - started
            batch = []
            started = time.perf_counter()
    if batch:
        yield batch, time.perf_counter() - started


def _split_shards(
    shards: Sequence[RecordShard],
    process_count: int,
) -> list[list[RecordShard]]:
    assignments = [[] for _ in range(process_count)]
    for index, shard in enumerate(shards):
        assignments[index % process_count].append(shard)
    return assignments


def _execute_load_batch_with_records(
    adapter: VectorDBAdapter,
    target: TargetConfig,
    write_mode: str,
    batch: list[VectorRecord],
    batch_index: int,
) -> tuple[dict[str, Any], list[VectorRecord]]:
    return (
        execute_load_batch(
            adapter=adapter,
            target=target,
            batch=batch,
            batch_index=batch_index,
            write_mode=write_mode,
        ),
        batch,
    )


def _load_process_worker(
    vendor: str,
    target: TargetConfig,
    write_mode: str,
    task_queue: Any,
    result_queue: Any,
    concurrency: int,
) -> None:
    def worker() -> None:
        adapter = _worker_adapter(vendor)
        while True:
            item = task_queue.get()
            if item is None:
                return
            batch_index, batch = item
            result_queue.put(
                execute_load_batch(
                    adapter=adapter,
                    target=target,
                    batch=batch,
                    batch_index=batch_index,
                    write_mode=write_mode,
                )
            )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()


def _worker_adapter(vendor: str) -> VectorDBAdapter:
    from ldbbench.adapters import get_adapter

    return get_adapter(vendor)


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
    write_mode: str = "upsert",
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    try:
        result = adapter.upsert_batch(target, batch, write_mode=write_mode)
    except Exception as exc:  # noqa: BLE001
        return {
            "stage": "load",
            "batch_index": batch_index,
            "records": len(batch),
            "write_mode": write_mode,
            "latency_ms": _elapsed_ms(batch_started),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "stage": "load",
        "batch_index": batch_index,
        "records": result.count,
        "write_mode": write_mode,
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
    partition_filter_spec: PartitionFilterSpec | None = None,
    stages: list[dict[str, Any]] | None = None,
    processes: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    events_output = Path(events_path)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    query_list = list(queries)
    ticker = ProgressTicker(progress)
    _validate_partition_filter_query_records(query_list, partition_filter_spec)
    if not query_list:
        events_output.write_text("", encoding="utf-8")
        return _query_summary(
            mode="staged" if stages else "one_pass",
            started=time.perf_counter(),
            latencies=[],
            recalls=[],
            query_count=0,
            error_count=0,
            processes=1,
            stage_summaries=[],
            partition_filter_spec=partition_filter_spec,
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
            partition_filter_spec=partition_filter_spec,
            events_path=events_output,
            stages=stages,
            processes=processes,
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
                partition_filter_spec=partition_filter_spec,
            )
            state.record(event)
            _write_event(file, event, flush=False, sort_keys=False)
            ticker.maybe(
                "query: one_pass progress "
                f"attempts={state.queries + state.errors}/{len(query_list)} "
                f"errors={state.errors}"
            )
        file.flush()

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
        processes=1,
        stage_summaries=[],
        partition_filter_spec=partition_filter_spec,
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
    partition_filter_spec: PartitionFilterSpec | None,
    events_path: Path,
    stages: list[dict[str, Any]],
    processes: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    query_lock = Lock()
    state = QueryRunState()
    next_query_index = 1
    started = time.perf_counter()
    stage_summaries: list[dict[str, Any]] = []
    ticker = ProgressTicker(progress)

    with events_path.open("w", encoding="utf-8") as file:
        for stage_index, stage in enumerate(stages, start=1):
            concurrency = _stage_concurrency(stage, stage_index=stage_index)
            process_count = _effective_process_count(processes, concurrency)
            duration_seconds = parse_duration_seconds(
                _stage_duration(stage, stage_index=stage_index)
            )
            deadline = time.perf_counter() + duration_seconds
            stage_state = QueryRunState()
            stage_started = time.perf_counter()
            worker_threads = _split_concurrency(concurrency, process_count)
            ticker.emit(
                "query: starting stage "
                f"stage={stage_index}/{len(stages)} concurrency={concurrency} "
                f"processes={process_count} "
                f"worker_threads_per_process={worker_threads} "
                f"duration_seconds={duration_seconds}"
            )
            if process_count > 1:
                next_query_index = _run_staged_query_processes(
                    vendor=target.vendor,
                    target=target,
                    queries=queries,
                    top_k=top_k,
                    consistency=consistency,
                    include_vectors=include_vectors,
                    ground_truth=ground_truth,
                    partition_filter_spec=partition_filter_spec,
                    file=file,
                    stage_index=stage_index,
                    concurrency=concurrency,
                    process_count=process_count,
                    deadline=deadline,
                    state=state,
                    stage_state=stage_state,
                    next_query_index=next_query_index,
                    ticker=ticker,
                    stage_started=stage_started,
                    duration_seconds=duration_seconds,
                )
                stage_summaries.append(
                    _query_stage_summary(
                        stage_index=stage_index,
                        concurrency=concurrency,
                        processes=process_count,
                        worker_threads_per_process=worker_threads,
                        configured_duration_seconds=duration_seconds,
                        elapsed_seconds=time.perf_counter() - stage_started,
                        state=stage_state,
                        partition_filter_spec=partition_filter_spec,
                    )
                )
                ticker.emit(
                    "query: stage finished "
                    f"stage={stage_index} queries={stage_state.queries} "
                    f"errors={stage_state.errors}"
                )
                continue

            event_queue: Queue[Mapping[str, Any] | None] = Queue(
                maxsize=max(concurrency * 16, QUERY_EVENT_FLUSH_INTERVAL),
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
                current_event_queue: Queue[Mapping[str, Any] | None] = event_queue,
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
                        partition_filter_spec=partition_filter_spec,
                    )
                    current_event_queue.put(event)

            def write_events(
                current_event_queue: Queue[Mapping[str, Any] | None] = event_queue,
                current_stage_index: int = stage_index,
                current_stage_state: QueryRunState = stage_state,
                current_stage_started: float = stage_started,
                current_duration_seconds: float = duration_seconds,
            ) -> None:
                pending_flushes = 0
                while True:
                    event = current_event_queue.get()
                    try:
                        if event is None:
                            return
                        state.record(event)
                        current_stage_state.record(event)
                        _write_event(file, event, flush=False, sort_keys=False)
                        pending_flushes += 1
                        if pending_flushes >= QUERY_EVENT_FLUSH_INTERVAL:
                            file.flush()
                            pending_flushes = 0
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
                    finally:
                        current_event_queue.task_done()

            writer = Thread(target=write_events, name=f"query-events-{stage_index}")
            writer.start()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                try:
                    futures = [
                        executor.submit(worker, index)
                        for index in range(1, concurrency + 1)
                    ]
                    for future in futures:
                        future.result()
                finally:
                    event_queue.put(None)
                    event_queue.join()
                    writer.join()
                    file.flush()

            stage_summaries.append(
                _query_stage_summary(
                    stage_index=stage_index,
                    concurrency=concurrency,
                    processes=process_count,
                    worker_threads_per_process=worker_threads,
                    configured_duration_seconds=duration_seconds,
                    elapsed_seconds=time.perf_counter() - stage_started,
                    state=stage_state,
                    partition_filter_spec=partition_filter_spec,
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
        processes=_effective_process_count(processes, max(1, max(
            _stage_concurrency(stage, stage_index=index)
            for index, stage in enumerate(stages, start=1)
        ))),
        stage_summaries=stage_summaries,
        partition_filter_spec=partition_filter_spec,
    )


def _run_staged_query_processes(
    *,
    vendor: str,
    target: TargetConfig,
    queries: list[VectorRecord],
    top_k: int,
    consistency: str,
    include_vectors: bool,
    ground_truth: Mapping[str, list[str]],
    partition_filter_spec: PartitionFilterSpec | None,
    file: Any,
    stage_index: int,
    concurrency: int,
    process_count: int,
    deadline: float,
    state: QueryRunState,
    stage_state: QueryRunState,
    next_query_index: int,
    ticker: ProgressTicker,
    stage_started: float,
    duration_seconds: float,
) -> int:
    worker_threads = _split_concurrency(concurrency, process_count)
    task_queue: Any = mp.Queue()
    result_queue: Any = mp.Queue()
    workers = [
        mp.Process(
            target=_query_process_worker,
            args=(
                vendor,
                target,
                task_queue,
                result_queue,
                thread_count,
                worker_offset,
                stage_index,
                top_k,
                consistency,
                include_vectors,
                dict(ground_truth),
                partition_filter_spec,
            ),
            name=f"ldbbench-query-{stage_index}-{index}",
        )
        for index, (thread_count, worker_offset) in enumerate(
            _worker_offsets(worker_threads),
            start=1,
        )
    ]
    for worker_process in workers:
        worker_process.start()

    in_flight = 0
    pending_flushes = 0

    def submit_query() -> bool:
        nonlocal in_flight, next_query_index
        if time.perf_counter() >= deadline:
            return False
        query_index = next_query_index
        query = queries[(query_index - 1) % len(queries)]
        next_query_index += 1
        task_queue.put((query_index, query))
        in_flight += 1
        return True

    try:
        while in_flight < concurrency and submit_query():
            pass

        while in_flight:
            try:
                event = result_queue.get(timeout=0.1)
            except Empty:
                for worker_process in workers:
                    if worker_process.exitcode not in (None, 0):
                        raise RuntimeError(
                            "query worker process exited with "
                            f"code {worker_process.exitcode}"
                        ) from None
                continue
            in_flight -= 1
            state.record(event)
            stage_state.record(event)
            _write_event(file, event, flush=False, sort_keys=False)
            pending_flushes += 1
            if pending_flushes >= QUERY_EVENT_FLUSH_INTERVAL:
                file.flush()
                pending_flushes = 0
            elapsed = time.perf_counter() - stage_started
            attempts = stage_state.queries + stage_state.errors
            ticker.maybe(
                "query: stage progress "
                f"stage={stage_index} "
                f"elapsed_seconds={elapsed:.1f}/{duration_seconds:.1f} "
                f"attempts={attempts} "
                f"errors={stage_state.errors}"
            )
            while in_flight < concurrency and submit_query():
                pass
    finally:
        for _ in range(sum(worker_threads)):
            task_queue.put(None)
        for worker_process in workers:
            worker_process.join()
        file.flush()
    return next_query_index


def _query_process_worker(
    vendor: str,
    target: TargetConfig,
    task_queue: Any,
    result_queue: Any,
    concurrency: int,
    worker_offset: int,
    stage_index: int,
    top_k: int,
    consistency: str,
    include_vectors: bool,
    ground_truth: Mapping[str, list[str]],
    partition_filter_spec: PartitionFilterSpec | None,
) -> None:
    def worker(local_index: int) -> None:
        adapter = _worker_adapter(vendor)
        worker_index = worker_offset + local_index
        while True:
            item = task_queue.get()
            if item is None:
                return
            query_index, query = item
            result_queue.put(
                execute_query_once(
                    adapter=adapter,
                    target=target,
                    query=query,
                    query_index=query_index,
                    query_stage_index=stage_index,
                    worker_index=worker_index,
                    top_k=top_k,
                    consistency=consistency,
                    include_vectors=include_vectors,
                    ground_truth=ground_truth,
                    partition_filter_spec=partition_filter_spec,
                )
            )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker, index)
            for index in range(1, concurrency + 1)
        ]
        for future in futures:
            future.result()


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


def _query_partition_filter(
    query: VectorRecord,
    partition_filter_spec: PartitionFilterSpec | None,
) -> dict[str, Any] | None:
    if partition_filter_spec is None:
        return None
    value = query.metadata.get(partition_filter_spec.metadata_field)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            "query partition_filter metadata field "
            f"{partition_filter_spec.metadata_field!r} must be a non-empty string"
        )
    return {
        "field": partition_filter_spec.field,
        "in_": [value],
    }


def _validate_partition_filter_query_records(
    queries: Sequence[VectorRecord],
    partition_filter_spec: PartitionFilterSpec | None,
) -> None:
    if partition_filter_spec is None:
        return
    for query in queries:
        _query_partition_filter(query, partition_filter_spec)


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
    partition_filter_spec: PartitionFilterSpec | None = None,
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
    partition_filter = _query_partition_filter(query, partition_filter_spec)
    if partition_filter is not None:
        base_event["partition_filter"] = partition_filter

    try:
        result = adapter.query(
            target,
            vector=query.vector,
            top_k=top_k,
            consistency=consistency,
            include_vectors=include_vectors,
            partition_filter=partition_filter,
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
    recall = None
    recall_skip_reason = None
    if partition_filter_spec is not None:
        recall_skip_reason = "partition_filtered"
    else:
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
    if recall_skip_reason is not None:
        base_event["recall_skip_reason"] = recall_skip_reason
    return base_event


def _write_event(
    file: Any,
    event: Mapping[str, Any],
    *,
    flush: bool = True,
    sort_keys: bool = True,
) -> None:
    file.write(json.dumps(dict(event), sort_keys=sort_keys) + "\n")
    if flush:
        file.flush()


def _query_summary(
    *,
    mode: str,
    started: float,
    latencies: list[float],
    recalls: list[float],
    query_count: int,
    error_count: int,
    processes: int,
    stage_summaries: list[dict[str, Any]],
    partition_filter_spec: PartitionFilterSpec | None = None,
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
        "processes": processes,
    }
    if partition_filter_spec is not None:
        summary["partition_filter"] = partition_filter_spec.as_dict()
        summary["partition_filter_applied"] = True
        summary["recall_skip_reason"] = "partition_filtered"
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
        processes=1,
        stage_summaries=[],
        partition_filter_spec=None,
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
    processes: int,
    worker_threads_per_process: list[int],
    configured_duration_seconds: float,
    elapsed_seconds: float,
    state: QueryRunState,
    partition_filter_spec: PartitionFilterSpec | None,
) -> dict[str, Any]:
    attempts = state.queries + state.errors
    summary: dict[str, Any] = {
        "stage_index": stage_index,
        "concurrency": concurrency,
        "processes": processes,
        "worker_threads_per_process": worker_threads_per_process,
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
    if partition_filter_spec is not None:
        summary["partition_filter"] = partition_filter_spec.as_dict()
        summary["partition_filter_applied"] = True
        summary["recall_skip_reason"] = "partition_filtered"
    return summary


def read_records(
    path: str | Path,
    *,
    limit: int | None = None,
) -> Iterable[VectorRecord]:
    source_path = Path(path)
    if source_path.suffix == ".msgpack":
        yield from _read_msgpack_records(source_path, limit=limit)
        return
    with source_path.open("rb") as file:
        for index, line in enumerate(file, start=1):
            if limit is not None and index > limit:
                break
            if not line.strip():
                continue
            yield _parse_record_line(
                line,
                path=source_path,
                line_number=index,
            )


def _read_msgpack_records(
    path: Path,
    *,
    limit: int | None = None,
) -> Iterable[VectorRecord]:
    with path.open("rb") as file:
        unpacker = msgpack.Unpacker(file, raw=False)
        for index, raw in enumerate(unpacker, start=1):
            if limit is not None and index > limit:
                break
            yield _parse_msgpack_record(raw, path=path, record_number=index)


def load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    with Path(path).open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = _loads_json(line)
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


def _parse_record_line(line: bytes, *, path: Path, line_number: int) -> VectorRecord:
    raw = _loads_json(line)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}:{line_number} record row must be a JSON object")
    record_id = raw.get("id")
    vector = raw.get("vector")
    metadata = raw.get("metadata", {})
    if not isinstance(record_id, str) or not record_id:
        raise ConfigError(f"{path}:{line_number} missing non-empty id")
    if not isinstance(vector, list) or not vector:
        raise ConfigError(f"{path}:{line_number} missing vector list")
    if not isinstance(metadata, dict):
        raise ConfigError(f"{path}:{line_number} metadata must be a mapping")
    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata=metadata,
        estimated_size_bytes=len(line.rstrip(b"\r\n")) + 2,
    )


def _parse_msgpack_record(
    raw: Any,
    *,
    path: Path,
    record_number: int,
) -> VectorRecord:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}:{record_number} record row must be a mapping")
    record_id = raw.get("id")
    vector = raw.get("vector")
    metadata = raw.get("metadata", {})
    estimated_size_bytes = raw.get("estimated_size_bytes")
    if not isinstance(record_id, str) or not record_id:
        raise ConfigError(f"{path}:{record_number} missing non-empty id")
    if not isinstance(vector, list) or not vector:
        raise ConfigError(f"{path}:{record_number} missing vector list")
    if not isinstance(metadata, dict):
        raise ConfigError(f"{path}:{record_number} metadata must be a mapping")
    if estimated_size_bytes is not None and not isinstance(estimated_size_bytes, int):
        raise ConfigError(f"{path}:{record_number} estimated_size_bytes must be an int")
    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata=metadata,
        estimated_size_bytes=estimated_size_bytes,
    )


def _loads_json(line: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _load_checkpoint_context(
    *,
    records_path: Path,
    records_sha256: str | None = None,
    json_records_path: Path | None = None,
    dataset_manifest: Mapping[str, Any],
    scenario: ScenarioConfig,
    target: TargetConfig,
    write_mode: str,
    record_shards: Sequence[RecordShard] | None,
    batch_size: int,
    max_batch_bytes: int | None,
    max_records: int | None,
) -> dict[str, Any]:
    artifacts = dataset_manifest.get("artifacts", {})
    artifact_checksums = artifacts if isinstance(artifacts, Mapping) else {}
    checksum = records_sha256
    if checksum is None:
        checksum = artifact_checksums.get("records_sha256")
    context = {
        "scenario_name": scenario.name,
        "target_vendor": target.vendor,
        "target_name": target.name,
        "target_collection_name": target.collection_name,
        "record_source": "sharded_msgpack" if record_shards is not None else "single",
        "records_path": str(records_path),
        "records_sha256": checksum,
        "json_records_path": str(json_records_path) if json_records_path else None,
        "write_mode": write_mode,
        "batch_size": batch_size,
        "max_batch_bytes": max_batch_bytes,
        "max_records": max_records,
    }
    if record_shards is not None:
        context["records_shards"] = [
            shard.as_checkpoint_dict() for shard in record_shards
        ]
    return context


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
    if record.estimated_size_bytes is not None:
        return record.estimated_size_bytes
    payload = {
        "id": record.id,
        "metadata": dict(record.metadata),
        "vector": list(record.vector),
    }
    return len(json.dumps(payload, sort_keys=True).encode("utf-8")) + 2


def _record_source_summary(path: str | Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "format": None}
    source_path = Path(path)
    return {
        "path": str(source_path),
        "format": "msgpack" if source_path.suffix == ".msgpack" else "jsonl",
    }


def _record_shards_for_load(
    *,
    scenario: ScenarioConfig,
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    batch_size: int,
    max_batch_bytes: int | None,
    max_records: int | None,
) -> list[RecordShard] | None:
    if not bool(scenario.load.get("sharded_records", False)):
        return None
    if max_batch_bytes is not None:
        raise ConfigError(
            "scenario.load.sharded_records does not support max_batch_bytes yet"
        )
    artifacts = dataset_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ConfigError("dataset manifest artifacts must be a mapping")
    raw_shards = artifacts.get("records_shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ConfigError(
            "scenario.load.sharded_records requires records_shards in the "
            "dataset manifest; run `ldbbench dataset optimize --shards N` first"
        )
    configured_shard_count = scenario.load.get("shard_count")
    if configured_shard_count is not None and configured_shard_count != len(raw_shards):
        raise ConfigError(
            "scenario.load.shard_count does not match dataset records_shards "
            f"({configured_shard_count} != {len(raw_shards)})"
        )
    shards: list[RecordShard] = []
    next_batch_base = 0
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise ConfigError(f"records_shards[{index}] must be a mapping")
        shard = _record_shard_from_manifest(
            dataset_dir=dataset_dir,
            raw_shard=raw_shard,
            index=index,
            max_records=max_records,
            batch_base=next_batch_base,
            manifest_shard_count=len(raw_shards),
        )
        if shard.effective_records is None or shard.effective_records <= 0:
            continue
        shards.append(shard)
        next_batch_base += math.ceil(shard.effective_records / batch_size)
    if not shards:
        raise ConfigError("sharded load has no record shards to load")
    return shards


def _record_shard_from_manifest(
    *,
    dataset_dir: Path,
    raw_shard: Mapping[str, Any],
    index: int,
    max_records: int | None,
    batch_base: int,
    manifest_shard_count: int,
) -> RecordShard:
    path_value = raw_shard.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ConfigError(f"records_shards[{index}].path must be a string")
    path = Path(path_value)
    if not path.is_absolute() and not path.exists():
        candidate = dataset_dir / path.name
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise ConfigError(f"records_shards[{index}] file {path} does not exist")
    records = raw_shard.get("records")
    first_record_index = raw_shard.get("first_record_index")
    last_record_index = raw_shard.get("last_record_index")
    if not isinstance(records, int) or records < 0:
        raise ConfigError(f"records_shards[{index}].records must be an integer")
    if not isinstance(first_record_index, int) or first_record_index <= 0:
        raise ConfigError(
            f"records_shards[{index}].first_record_index must be a positive integer"
        )
    if not isinstance(last_record_index, int) or last_record_index < first_record_index:
        raise ConfigError(
            f"records_shards[{index}].last_record_index must be a valid integer"
        )
    effective_records = records
    if max_records is not None:
        if first_record_index > max_records:
            effective_records = 0
        else:
            effective_records = min(records, max_records - first_record_index + 1)
    if effective_records > records:
        raise ConfigError(f"records_shards[{index}] effective records are invalid")
    sha256 = raw_shard.get("sha256")
    return RecordShard(
        path=path,
        sha256=sha256 if isinstance(sha256, str) else None,
        records=records,
        first_record_index=first_record_index,
        last_record_index=last_record_index,
        batch_base=batch_base,
        effective_records=effective_records,
        manifest_shard_count=manifest_shard_count,
    )


def _manifest_shard_count(record_shards: Sequence[RecordShard]) -> int:
    for shard in record_shards:
        if shard.manifest_shard_count is not None:
            return shard.manifest_shard_count
    return len(record_shards)


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


def _load_processes(scenario: ScenarioConfig) -> int:
    processes = scenario.load.get("processes", 1)
    if not isinstance(processes, int) or processes <= 0:
        raise ConfigError("scenario.load.processes must be a positive integer")
    return processes


def _query_processes(scenario: ScenarioConfig) -> int:
    processes = scenario.query.get("processes", 1)
    if not isinstance(processes, int) or processes <= 0:
        raise ConfigError("scenario.query.processes must be a positive integer")
    return processes


def _effective_process_count(processes: int, concurrency: int) -> int:
    return max(1, min(processes, concurrency))


def _split_concurrency(concurrency: int, process_count: int) -> list[int]:
    base, remainder = divmod(concurrency, process_count)
    return [
        base + (1 if index < remainder else 0)
        for index in range(process_count)
    ]


def _worker_offsets(thread_counts: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    next_offset = 0
    for thread_count in thread_counts:
        offsets.append((thread_count, next_offset))
        next_offset += thread_count
    return offsets


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


def _partition_filter_spec(scenario: ScenarioConfig) -> PartitionFilterSpec | None:
    value = scenario.query.get("partition_filter")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("scenario.query.partition_filter must be a mapping")
    field = value.get("field")
    metadata_field = value.get("metadata_field")
    if not isinstance(field, str) or not field:
        raise ConfigError("scenario.query.partition_filter.field must be a string")
    if not isinstance(metadata_field, str) or not metadata_field:
        raise ConfigError(
            "scenario.query.partition_filter.metadata_field must be a string"
        )
    return PartitionFilterSpec(field=field, metadata_field=metadata_field)


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


def _preferred_artifact_path(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    *,
    preferred_key: str,
    preferred_filename: str,
    fallback_key: str,
    fallback_filename: str,
) -> tuple[Path, str | None]:
    artifacts = dataset_manifest.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        preferred = artifacts.get(preferred_key)
        if isinstance(preferred, str) and preferred and Path(preferred).exists():
            checksum = artifacts.get(f"{preferred_key}_sha256")
            return Path(preferred), checksum if isinstance(checksum, str) else None
        preferred_default = dataset_dir / preferred_filename
        if preferred_default.exists():
            checksum = artifacts.get(f"{preferred_key}_sha256")
            return preferred_default, checksum if isinstance(checksum, str) else None
    fallback = artifact_path(
        dataset_dir,
        dataset_manifest,
        fallback_key,
        fallback_filename,
    )
    checksum = artifacts.get(f"{fallback_key}_sha256") if isinstance(
        artifacts,
        Mapping,
    ) else None
    return fallback, checksum if isinstance(checksum, str) else None


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
