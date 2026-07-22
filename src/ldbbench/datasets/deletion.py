"""Deterministic document-deletion plans for prepared datasets."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ldbbench.__about__ import __version__
from ldbbench.config import VALID_DELETE_ORDERS, ConfigError
from ldbbench.datasets.prepare import DATASET_MANIFEST_FILENAME, RECORDS_FILENAME
from ldbbench.manifest import sha256_file
from ldbbench.progress import ProgressCallback, ProgressTicker

DELETION_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DeletionPlanResult:
    plan_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedDeletionPlan:
    path: Path
    manifest_path: Path
    ids: list[str]
    order: str
    seed: int
    records_path: Path
    records_sha256: str
    plan_sha256: str

    @property
    def total_records(self) -> int:
        return len(self.ids)


def prepare_deletion_plan(
    *,
    dataset_dir: str | Path,
    order: str,
    seed: int = 0,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> DeletionPlanResult:
    _validate_order_and_seed(order, seed)
    directory = Path(dataset_dir)
    dataset_manifest = _load_dataset_manifest(directory)
    records_path = _artifact_path(
        directory,
        dataset_manifest,
        key="records",
        default_name=RECORDS_FILENAME,
    )
    plan_path = directory / deletion_plan_filename(order=order, seed=seed)
    manifest_path = deletion_plan_manifest_path(plan_path)
    expected_records = _dataset_record_count(dataset_manifest)
    ticker = ProgressTicker(progress)

    ids: list[str] = []
    if dry_run:
        ticker.emit(f"delete_plan: planning order={order} seed={seed}")
        status = "planned"
    else:
        ticker.emit(f"delete_plan: loading records order={order} seed={seed}")
        ids = _read_record_ids(records_path, progress=progress)
        if expected_records is not None and len(ids) != expected_records:
            raise ConfigError(
                f"{records_path} has {len(ids)} records but dataset manifest "
                f"declares {expected_records}"
            )
        if order == "random":
            random.Random(seed).shuffle(ids)
        _write_plan(plan_path, ids)
        ticker.emit(f"delete_plan: wrote records={len(ids)}")
        status = "prepared"

    total_records = (
        expected_records if dry_run and expected_records is not None else len(ids)
    )
    manifest: dict[str, Any] = {
        "schema_version": DELETION_PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {"name": "lambdadb-bench", "version": __version__},
        "status": status,
        "dry_run": dry_run,
        "deletion_plan": {
            "order": order,
            "seed": seed,
            "records": total_records,
        },
        "artifacts": {
            "records": str(records_path),
            "records_sha256": _sha256_if_exists(records_path),
            "deletion_plan": str(plan_path),
            "deletion_plan_sha256": _sha256_if_exists(plan_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DeletionPlanResult(
        plan_path=plan_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_deletion_plan(
    plan_path: str | Path,
    *,
    records_path: str | Path | None = None,
) -> LoadedDeletionPlan:
    path = Path(plan_path)
    manifest_path = deletion_plan_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"could not read deletion plan manifest {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ConfigError(f"{manifest_path} must contain a JSON object")
    if manifest.get("schema_version") != DELETION_PLAN_SCHEMA_VERSION:
        raise ConfigError(
            f"{manifest_path} schema_version must be {DELETION_PLAN_SCHEMA_VERSION}"
        )
    if manifest.get("status") != "prepared":
        raise ConfigError(f"{manifest_path} deletion plan is not prepared")

    details = manifest.get("deletion_plan")
    artifacts = manifest.get("artifacts")
    if not isinstance(details, dict) or not isinstance(artifacts, dict):
        raise ConfigError(f"{manifest_path} has invalid deletion plan metadata")
    order = details.get("order")
    seed = details.get("seed")
    if not isinstance(order, str) or not isinstance(seed, int):
        raise ConfigError(f"{manifest_path} has invalid order or seed")
    _validate_order_and_seed(order, seed)

    declared_plan = artifacts.get("deletion_plan")
    declared_plan_sha256 = artifacts.get("deletion_plan_sha256")
    declared_records = artifacts.get("records")
    declared_records_sha256 = artifacts.get("records_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (
            declared_plan,
            declared_plan_sha256,
            declared_records,
            declared_records_sha256,
        )
    ):
        raise ConfigError(f"{manifest_path} has incomplete artifact metadata")
    if Path(declared_plan).resolve() != path.resolve():
        raise ConfigError(
            f"{manifest_path} deletion plan artifact {declared_plan!r} "
            f"does not match selected file {str(path)!r}"
        )
    actual_plan_sha256 = sha256_file(path)
    if actual_plan_sha256 != declared_plan_sha256:
        raise ConfigError(f"{path} checksum does not match {manifest_path}")

    selected_records = (
        Path(records_path) if records_path is not None else Path(declared_records)
    )
    if selected_records.resolve() != Path(declared_records).resolve():
        raise ConfigError(
            f"{manifest_path} records artifact {declared_records!r} does not "
            f"match selected records {str(selected_records)!r}"
        )
    actual_records_sha256 = sha256_file(selected_records)
    if actual_records_sha256 != declared_records_sha256:
        raise ConfigError(
            f"{selected_records} checksum does not match deletion plan manifest"
        )

    ids = _read_plan_ids(path)
    declared_count = details.get("records")
    if not isinstance(declared_count, int) or declared_count != len(ids):
        raise ConfigError(
            f"{manifest_path} records count does not match deletion plan rows"
        )
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path} contains duplicate document IDs")

    return LoadedDeletionPlan(
        path=path,
        manifest_path=manifest_path,
        ids=ids,
        order=order,
        seed=seed,
        records_path=selected_records,
        records_sha256=actual_records_sha256,
        plan_sha256=actual_plan_sha256,
    )


def deletion_plan_filename(*, order: str, seed: int) -> str:
    _validate_order_and_seed(order, seed)
    if order == "random":
        return f"deletion_plan.random.seed-{seed}.jsonl"
    return "deletion_plan.sequential.jsonl"


def deletion_plan_manifest_path(plan_path: str | Path) -> Path:
    return Path(plan_path).with_suffix(".manifest.json")


def deletion_checkpoint_count(total_records: int, checkpoint_pct: int) -> int:
    if total_records < 0:
        raise ConfigError("total records must be non-negative")
    if (
        not isinstance(checkpoint_pct, int)
        or checkpoint_pct < 0
        or checkpoint_pct >= 100
    ):
        raise ConfigError("delete checkpoint percent must be between 0 and 99")
    return total_records * checkpoint_pct // 100


def deletion_artifact_suffix(plan: LoadedDeletionPlan, checkpoint_pct: int) -> str:
    deletion_checkpoint_count(plan.total_records, checkpoint_pct)
    if plan.order == "random":
        identity = f"random.seed-{plan.seed}"
    else:
        identity = "sequential"
    return f"deleted.{identity}.pct-{checkpoint_pct:03d}"


def _validate_order_and_seed(order: str, seed: int) -> None:
    if order not in VALID_DELETE_ORDERS:
        raise ConfigError(
            f"deletion order must be one of {sorted(VALID_DELETE_ORDERS)}"
        )
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError("deletion seed must be a non-negative integer")


def _read_record_ids(
    records_path: Path,
    *,
    progress: ProgressCallback | None,
) -> list[str]:
    ids: list[str] = []
    ticker = ProgressTicker(progress)
    with records_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ConfigError(f"{records_path}:{line_number} must be an object")
            record_id = raw.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ConfigError(f"{records_path}:{line_number} missing non-empty id")
            ids.append(record_id)
            ticker.maybe(f"delete_plan: loading records={len(ids)}")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{records_path} contains duplicate document IDs")
    return ids


def _write_plan(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record_id in ids:
            file.write(json.dumps({"id": record_id}, sort_keys=True) + "\n")


def _read_plan_ids(path: Path) -> list[str]:
    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        f"could not parse deletion plan {path}:{line_number}"
                    ) from exc
                record_id = raw.get("id") if isinstance(raw, dict) else None
                if not isinstance(record_id, str) or not record_id:
                    raise ConfigError(f"{path}:{line_number} missing non-empty id")
                ids.append(record_id)
    except OSError as exc:
        raise ConfigError(f"could not read deletion plan {path}") from exc
    return ids


def _load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / DATASET_MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read dataset manifest {path}") from exc
    if not isinstance(manifest, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return manifest


def _artifact_path(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    *,
    key: str,
    default_name: str,
) -> Path:
    artifacts = dataset_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ConfigError("dataset manifest artifacts must be a mapping")
    value = artifacts.get(key)
    path = (
        Path(value) if isinstance(value, str) and value else dataset_dir / default_name
    )
    if not path.is_absolute() and not path.exists():
        candidate = dataset_dir / path.name
        if candidate.exists():
            return candidate
    return path


def _dataset_record_count(dataset_manifest: Mapping[str, Any]) -> int | None:
    value = dataset_manifest.get("dataset", {}).get("written_rows")
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ConfigError("dataset manifest written_rows must be non-negative")
    return value


def _sha256_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None
