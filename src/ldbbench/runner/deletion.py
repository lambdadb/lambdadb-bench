"""Delete-only execution and checkpoint-state validation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ldbbench.__about__ import __version__
from ldbbench.adapters.base import VectorDBAdapter
from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets.deletion import (
    LoadedDeletionPlan,
    deletion_checkpoint_count,
)
from ldbbench.datasets.ground_truth import ground_truth_manifest_path
from ldbbench.manifest import sha256_file, sha256_mapping
from ldbbench.progress import ProgressCallback, ProgressTicker

DELETION_STATE_SCHEMA_VERSION = 2


def run_delete_stage(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    ids: Sequence[str],
    batch_size: int,
    events_path: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ConfigError("delete batch_size must be a positive integer")
    output = Path(events_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ticker = ProgressTicker(progress)
    ticker.emit(f"delete: starting documents={len(ids)} batch_size={batch_size}")
    started = time.perf_counter()
    latencies: list[float] = []
    documents = 0
    attempts = 0
    errors = 0
    with output.open("w", encoding="utf-8") as file:
        for batch_index, offset in enumerate(range(0, len(ids), batch_size), start=1):
            batch = list(ids[offset : offset + batch_size])
            attempts += 1
            batch_started = time.perf_counter()
            try:
                result = adapter.delete_batch(target, batch)
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - batch_started) * 1000
                event = {
                    "stage": "delete",
                    "batch_index": batch_index,
                    "documents": len(batch),
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                file.write(json.dumps(event, sort_keys=True) + "\n")
                file.flush()
                errors += 1
                break
            latency_ms = (time.perf_counter() - batch_started) * 1000
            event = {
                "stage": "delete",
                "batch_index": batch_index,
                "documents": result.count,
                "latency_ms": latency_ms,
                "status": "ok",
            }
            file.write(json.dumps(event, sort_keys=True) + "\n")
            file.flush()
            documents += result.count
            latencies.append(latency_ms)
            ticker.maybe(
                f"delete: progress documents={documents}/{len(ids)} batches={attempts}"
            )

    elapsed = time.perf_counter() - started
    status = "completed" if errors == 0 and documents == len(ids) else "failed"
    ticker.emit(
        f"delete: finished status={status} documents={documents} errors={errors}"
    )
    return {
        "status": status,
        "documents": documents,
        "requested_documents": len(ids),
        "batches": len(latencies),
        "attempts": attempts,
        "errors": errors,
        "error_rate": errors / attempts if attempts else 0.0,
        "duration_seconds": elapsed,
        "documents_per_second": documents / elapsed if elapsed > 0 else 0.0,
        "batch_size": batch_size,
        "latency_ms": _latency_summary(latencies),
    }


def wait_until_deleted(
    *,
    adapter: VectorDBAdapter,
    target: TargetConfig,
    ids: Sequence[str],
    prefix_count: int | None = None,
    consistency: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sample_size: int,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ConfigError("delete visibility timeout must be positive")
    if poll_interval_seconds <= 0:
        raise ConfigError("delete visibility poll interval must be positive")
    if sample_size <= 0:
        raise ConfigError("delete visibility sample size must be positive")

    samples = _sample_ids(
        ids,
        sample_size=sample_size,
        prefix_count=prefix_count,
    )
    if not samples:
        return {
            "status": "not_applicable",
            "samples": 0,
            "visible": 0,
            "attempts": 0,
            "duration_seconds": 0.0,
            "last_error": None,
        }

    ticker = ProgressTicker(progress)
    ticker.emit(
        f"delete_visibility: waiting samples={len(samples)} "
        f"timeout_seconds={timeout_seconds}"
    )
    started = time.perf_counter()
    deadline = started + timeout_seconds
    attempts = 0
    visible = len(samples)
    last_error: dict[str, str] | None = None
    while True:
        attempts += 1
        try:
            documents = adapter.fetch(
                target,
                ids=samples,
                consistency=consistency,
                include_vectors=False,
            )
            visible = len(documents)
            last_error = None
            if visible == 0:
                elapsed = time.perf_counter() - started
                ticker.emit(
                    "delete_visibility: finished status=not_visible "
                    f"attempts={attempts}"
                )
                return {
                    "status": "not_visible",
                    "samples": len(samples),
                    "visible": 0,
                    "attempts": attempts,
                    "duration_seconds": elapsed,
                    "last_error": None,
                }
        except Exception as exc:  # noqa: BLE001
            last_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        now = time.perf_counter()
        if now >= deadline:
            elapsed = now - started
            ticker.emit(
                "delete_visibility: finished status=timeout "
                f"visible={visible}/{len(samples)} attempts={attempts}"
            )
            return {
                "status": "timeout",
                "samples": len(samples),
                "visible": visible,
                "attempts": attempts,
                "duration_seconds": elapsed,
                "last_error": last_error,
            }
        ticker.maybe(
            f"delete_visibility: pending visible={visible}/{len(samples)}"
        )
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def write_deletion_state(
    *,
    output_path: str | Path,
    target: TargetConfig,
    plan: LoadedDeletionPlan,
    checkpoint_pct: int,
    previous_deleted_count: int,
    delete_summary: Mapping[str, Any],
) -> dict[str, Any]:
    newly_deleted = int(delete_summary.get("documents", 0))
    deleted_count = previous_deleted_count + newly_deleted
    requested_documents = int(delete_summary.get("requested_documents", 0))
    expected_deleted_count = previous_deleted_count + requested_documents
    visibility = delete_summary.get("visibility")
    visibility_status = (
        visibility.get("status") if isinstance(visibility, Mapping) else None
    )
    status = (
        "completed"
        if delete_summary.get("status") == "completed"
        and deleted_count == expected_deleted_count
        and visibility_status in {"not_visible", "not_applicable"}
        else "failed"
    )
    state: dict[str, Any] = {
        "schema_version": DELETION_STATE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {"name": "lambdadb-bench", "version": __version__},
        "status": status,
        "target": {
            "vendor": target.vendor,
            "name": target.name,
            "collection_name": target.collection_name,
            "identity_sha256": deletion_target_identity_sha256(target),
        },
        "dataset": {
            "records": str(plan.records_path),
            "records_sha256": plan.records_sha256,
            "total_records": plan.total_records,
        },
        "deletion": {
            "order": plan.order,
            "seed": plan.seed,
            "checkpoint_pct": checkpoint_pct,
            "deleted_count": deleted_count,
            "remaining_count": plan.total_records - deleted_count,
            "deletion_plan": str(plan.path),
            "deletion_plan_sha256": plan.plan_sha256,
            "visibility": visibility,
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state


def load_deletion_state(
    path: str | Path,
    *,
    target: TargetConfig,
    records_path: str | Path,
    plan: LoadedDeletionPlan | None = None,
) -> dict[str, Any]:
    state_path = Path(path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read deletion state {state_path}") from exc
    if not isinstance(state, dict):
        raise ConfigError(f"{state_path} must contain a JSON object")
    if state.get("schema_version") != DELETION_STATE_SCHEMA_VERSION:
        raise ConfigError(
            f"{state_path} schema_version must be {DELETION_STATE_SCHEMA_VERSION}"
        )
    if state.get("status") != "completed":
        raise ConfigError(f"{state_path} deletion state is not completed")

    state_target = state.get("target")
    dataset = state.get("dataset")
    deletion = state.get("deletion")
    if not all(isinstance(value, dict) for value in (state_target, dataset, deletion)):
        raise ConfigError(f"{state_path} has invalid deletion state metadata")
    expected_target = {
        "vendor": target.vendor,
        "name": target.name,
        "collection_name": target.collection_name,
        "identity_sha256": deletion_target_identity_sha256(target),
    }
    actual_target = {key: state_target.get(key) for key in expected_target}
    if actual_target != expected_target:
        raise ConfigError(
            f"{state_path} target {actual_target!r} does not match "
            f"selected target {expected_target!r}"
        )

    selected_records = Path(records_path)
    declared_records = dataset.get("records")
    declared_sha256 = dataset.get("records_sha256")
    if not isinstance(declared_records, str) or not isinstance(declared_sha256, str):
        raise ConfigError(f"{state_path} has invalid records artifact metadata")
    if Path(declared_records).resolve() != selected_records.resolve():
        raise ConfigError(
            f"{state_path} records artifact does not match selected dataset"
        )
    if sha256_file(selected_records) != declared_sha256:
        raise ConfigError(f"{state_path} records checksum does not match dataset")

    _validate_state_counts(state_path, dataset, deletion)
    if plan is not None:
        _validate_state_plan(state_path, dataset, deletion, plan)
    return state


def validate_ground_truth_deletion_state(
    ground_truth_path: str | Path,
    deletion_state: Mapping[str, Any] | None,
    *,
    queries_path: str | Path | None = None,
) -> None:
    manifest_path = ground_truth_manifest_path(ground_truth_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"could not read ground truth manifest {manifest_path}"
        ) from exc
    ground_truth = manifest.get("ground_truth")
    artifacts = manifest.get("artifacts")
    if not isinstance(ground_truth, dict) or not isinstance(artifacts, dict):
        raise ConfigError(f"{manifest_path} has invalid ground truth metadata")
    gt_deletion = ground_truth.get("deletion")
    if gt_deletion is None:
        if deletion_state is not None:
            raise ConfigError("--deletion-state requires deletion-aware ground truth")
        return
    if deletion_state is None:
        raise ConfigError("deletion-aware ground truth requires --deletion-state")
    state_deletion = deletion_state.get("deletion")
    if not isinstance(gt_deletion, dict) or not isinstance(state_deletion, Mapping):
        raise ConfigError("invalid deletion metadata in ground truth or state")
    keys = (
        "order",
        "seed",
        "checkpoint_pct",
        "deleted_count",
        "remaining_count",
    )
    gt_values = {key: gt_deletion.get(key) for key in keys}
    state_values = {key: state_deletion.get(key) for key in keys}
    if gt_values != state_values:
        raise ConfigError(
            f"ground truth deletion checkpoint {gt_values!r} does not match "
            f"deletion state {state_values!r}"
        )
    gt_plan_sha256 = artifacts.get("deletion_plan_sha256")
    state_plan_sha256 = state_deletion.get("deletion_plan_sha256")
    if gt_plan_sha256 != state_plan_sha256:
        raise ConfigError(
            "ground truth deletion plan checksum does not match deletion state"
        )
    state_dataset = deletion_state.get("dataset")
    if not isinstance(state_dataset, Mapping):
        raise ConfigError("deletion state is missing dataset metadata")
    gt_records_sha256 = artifacts.get("records_sha256")
    state_records_sha256 = state_dataset.get("records_sha256")
    if gt_records_sha256 != state_records_sha256:
        raise ConfigError(
            "ground truth records checksum does not match deletion state"
        )
    if queries_path is not None:
        gt_queries_sha256 = artifacts.get("queries_sha256")
        if not isinstance(gt_queries_sha256, str) or not gt_queries_sha256:
            raise ConfigError(
                "deletion-aware ground truth is missing queries checksum"
            )
        if sha256_file(queries_path) != gt_queries_sha256:
            raise ConfigError(
                "ground truth queries checksum does not match selected dataset"
            )


def validate_deletion_state_scenario(
    scenario: ScenarioConfig,
    deletion_state: Mapping[str, Any],
) -> None:
    if not scenario.delete:
        raise ConfigError("--deletion-state requires scenario.delete configuration")
    deletion = deletion_state.get("deletion")
    if not isinstance(deletion, Mapping):
        raise ConfigError("deletion state is missing deletion metadata")
    expected_order = scenario.delete.get("order")
    expected_seed = int(scenario.delete.get("seed", 0))
    actual_order = deletion.get("order")
    actual_seed = deletion.get("seed")
    if actual_order != expected_order or actual_seed != expected_seed:
        raise ConfigError(
            "deletion state does not match scenario.delete order/seed: "
            f"state=({actual_order}, {actual_seed}), "
            f"scenario=({expected_order}, {expected_seed})"
        )
    checkpoint_pct = deletion.get("checkpoint_pct")
    checkpoints = scenario.delete.get("checkpoints_pct", [])
    if checkpoint_pct not in checkpoints:
        raise ConfigError(
            f"deletion state checkpoint {checkpoint_pct!r} is not configured in "
            "scenario.delete.checkpoints_pct"
        )


def deletion_target_identity_sha256(target: TargetConfig) -> str:
    identity: dict[str, Any] = {
        "vendor": target.vendor,
        "name": target.name,
        "endpoint": target.endpoint,
        "collection_name": target.collection_name,
        "project_name": target.project_name,
        "region": target.region,
        "api_key_env": target.api_key_env,
    }
    if target.vendor == "pinecone":
        identity["namespace"] = target.raw.get("namespace", "")
        identity["cloud"] = target.raw.get("cloud", "aws")
    return sha256_mapping(identity)


def _validate_state_plan(
    state_path: Path,
    dataset: Mapping[str, Any],
    deletion: Mapping[str, Any],
    plan: LoadedDeletionPlan,
) -> None:
    expected = {
        "order": plan.order,
        "seed": plan.seed,
        "deletion_plan_sha256": plan.plan_sha256,
    }
    actual = {key: deletion.get(key) for key in expected}
    if actual != expected:
        raise ConfigError(
            f"{state_path} deletion plan {actual!r} does not match "
            f"selected plan {expected!r}"
        )
    if dataset.get("total_records") != plan.total_records:
        raise ConfigError(f"{state_path} total_records does not match deletion plan")


def _validate_state_counts(
    state_path: Path,
    dataset: Mapping[str, Any],
    deletion: Mapping[str, Any],
) -> None:
    total_records = dataset.get("total_records")
    checkpoint_pct = deletion.get("checkpoint_pct")
    deleted_count = deletion.get("deleted_count")
    remaining_count = deletion.get("remaining_count")
    visibility = deletion.get("visibility")
    if not isinstance(total_records, int) or total_records < 0:
        raise ConfigError(f"{state_path} total_records must be non-negative")
    if (
        not isinstance(checkpoint_pct, int)
        or checkpoint_pct <= 0
        or checkpoint_pct >= 100
    ):
        raise ConfigError(f"{state_path} checkpoint_pct must be between 1 and 99")
    if not isinstance(deleted_count, int) or deleted_count < 0:
        raise ConfigError(f"{state_path} deleted_count must be non-negative")
    if not isinstance(remaining_count, int) or remaining_count < 0:
        raise ConfigError(f"{state_path} remaining_count must be non-negative")
    if not isinstance(visibility, Mapping) or visibility.get("status") not in {
        "not_visible",
        "not_applicable",
    }:
        raise ConfigError(
            f"{state_path} does not contain completed delete visibility metadata"
        )

    expected_deleted_count = deletion_checkpoint_count(
        total_records,
        checkpoint_pct,
    )
    if deleted_count != expected_deleted_count:
        raise ConfigError(
            f"{state_path} deleted_count does not match its checkpoint"
        )
    if remaining_count != total_records - deleted_count:
        raise ConfigError(
            f"{state_path} remaining_count does not match total_records"
        )


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _sample_ids(
    ids: Sequence[str],
    *,
    sample_size: int,
    prefix_count: int | None,
) -> list[str]:
    count = len(ids) if prefix_count is None else prefix_count
    if count < 0 or count > len(ids):
        raise ConfigError("delete visibility prefix count is out of range")
    if count <= sample_size:
        return list(ids[:count])
    last_index = count - 1
    if sample_size == 1:
        return [ids[last_index]]
    indexes = {
        round(index * last_index / (sample_size - 1))
        for index in range(sample_size)
    }
    return [ids[index] for index in sorted(indexes)]


def _percentile(values: list[float], percentile: int) -> float:
    rank = (len(values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight
