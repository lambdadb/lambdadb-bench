"""Dataset preparation skeleton."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack

from ldbbench.__about__ import __version__
from ldbbench.config import ConfigError, ScenarioConfig
from ldbbench.progress import ProgressCallback, ProgressTicker

RAW_RECORDS_FILENAME = "raw_records.jsonl"
RECORDS_FILENAME = "records.jsonl"
QUERIES_FILENAME = "queries.jsonl"
RECORDS_MSGPACK_FILENAME = "records.msgpack"
QUERIES_MSGPACK_FILENAME = "queries.msgpack"
RECORD_SHARD_FILENAME_TEMPLATE = "records-{index:05d}.msgpack"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
SUPPORTED_PROVIDERS = {"huggingface"}
DEFAULT_QUERY_COUNT = 1000

try:
    import orjson
except ImportError:  # pragma: no cover - msgpack optimization still works.
    orjson = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DatasetPrepareResult:
    output_dir: Path
    manifest_path: Path
    raw_records_path: Path
    records_path: Path
    queries_path: Path
    records_msgpack_path: Path
    queries_msgpack_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DatasetOptimizeResult:
    output_dir: Path
    manifest_path: Path
    records_msgpack_path: Path
    queries_msgpack_path: Path
    record_shard_paths: list[Path]
    manifest: dict[str, Any]


def prepare_dataset(
    *,
    scenario: ScenarioConfig,
    output_dir: str | Path,
    limit: int | None = None,
    dry_run: bool = False,
    query_count: int | None = None,
    source_rows: Iterable[Mapping[str, Any]] | None = None,
    progress: ProgressCallback | None = None,
) -> DatasetPrepareResult:
    """Prepare dataset cache artifacts for a scenario.

    The current phase writes a cache manifest and, when not in dry-run mode, a
    raw JSONL sample. Later phases will normalize records and add query splits.
    """

    provider = str(scenario.dataset.get("provider", "huggingface"))
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"dataset provider {provider!r} is not supported; "
            f"known: {sorted(SUPPORTED_PROVIDERS)}"
        )

    requested_rows = _requested_rows(scenario, limit)
    if requested_rows <= 0:
        raise ConfigError("dataset prepare row count must be a positive integer")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_records_path = out / RAW_RECORDS_FILENAME
    records_path = out / RECORDS_FILENAME
    queries_path = out / QUERIES_FILENAME
    records_msgpack_path = out / RECORDS_MSGPACK_FILENAME
    queries_msgpack_path = out / QUERIES_MSGPACK_FILENAME
    manifest_path = out / DATASET_MANIFEST_FILENAME
    requested_query_rows = _requested_query_rows(scenario, query_count)
    requested_source_rows = requested_rows + requested_query_rows

    written_rows = 0
    written_queries = 0
    written_source_rows = 0
    ticker = ProgressTicker(progress)
    if dry_run:
        ticker.emit("dataset_prepare: planning artifacts")
        status = "planned"
    else:
        rows = source_rows
        if rows is None:
            ticker.emit(
                "dataset_prepare: streaming source "
                f"provider={provider} requested_source_rows={requested_source_rows}"
            )
            rows = load_huggingface_rows(scenario=scenario, limit=requested_source_rows)
        ticker.emit(
            "dataset_prepare: writing artifacts "
            f"requested_queries={requested_query_rows} "
            f"requested_records={requested_rows}"
        )
        prepared = write_prepared_records(
            scenario=scenario,
            rows=rows,
            raw_output_path=raw_records_path,
            records_output_path=records_path,
            queries_output_path=queries_path,
            records_msgpack_output_path=records_msgpack_path,
            queries_msgpack_output_path=queries_msgpack_path,
            record_limit=requested_rows,
            query_count=requested_query_rows,
            progress=progress,
        )
        written_rows = prepared.records
        written_queries = prepared.queries
        written_source_rows = prepared.source_rows
        ticker.emit(
            "dataset_prepare: wrote artifacts "
            f"source_rows={written_source_rows} queries={written_queries} "
            f"records={written_rows}"
        )
        status = "prepared"

    manifest = build_dataset_manifest(
        scenario=scenario,
        output_dir=out,
        requested_rows=requested_rows,
        written_rows=written_rows,
        requested_source_rows=requested_source_rows,
        written_source_rows=written_source_rows,
        requested_query_rows=requested_query_rows,
        written_queries=written_queries,
        dry_run=dry_run,
        status=status,
        artifact_checksums=None
        if dry_run
        else {
            "raw_records_sha256": prepared.raw_records_sha256,
            "records_sha256": prepared.records_sha256,
            "queries_sha256": prepared.queries_sha256,
            "records_msgpack_sha256": prepared.records_msgpack_sha256,
            "queries_msgpack_sha256": prepared.queries_msgpack_sha256,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return DatasetPrepareResult(
        output_dir=out,
        manifest_path=manifest_path,
        raw_records_path=raw_records_path,
        records_path=records_path,
        queries_path=queries_path,
        records_msgpack_path=records_msgpack_path,
        queries_msgpack_path=queries_msgpack_path,
        manifest=manifest,
    )


@dataclass(frozen=True)
class PreparedCounts:
    source_rows: int
    records: int
    queries: int
    raw_records_sha256: str
    records_sha256: str
    queries_sha256: str
    records_msgpack_sha256: str
    queries_msgpack_sha256: str


def load_huggingface_rows(
    *,
    scenario: ScenarioConfig,
    limit: int,
) -> Iterator[Mapping[str, Any]]:
    """Stream rows from Hugging Face using deterministic shuffle when possible."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ConfigError(
            "dataset provider 'huggingface' requires the 'datasets' package"
        ) from exc

    dataset = scenario.dataset
    source = dataset.get("source")
    if not isinstance(source, str) or not source:
        raise ConfigError("scenario.dataset.source is required for Hugging Face")

    subset = dataset.get("subset")
    split = str(dataset.get("split", "train"))
    seed = int(dataset.get("seed", 0))
    buffer_size = int(dataset.get("shuffle_buffer_size", 10_000))

    config_name = subset if isinstance(subset, str) and subset else None
    iterable = load_dataset(source, config_name, split=split, streaming=True)
    if hasattr(iterable, "shuffle"):
        iterable = iterable.shuffle(seed=seed, buffer_size=buffer_size)
    return iter(iterable.take(limit) if hasattr(iterable, "take") else iterable)


def write_prepared_records(
    *,
    scenario: ScenarioConfig,
    rows: Iterable[Mapping[str, Any]],
    raw_output_path: str | Path,
    records_output_path: str | Path,
    queries_output_path: str | Path,
    records_msgpack_output_path: str | Path,
    queries_msgpack_output_path: str | Path,
    record_limit: int,
    query_count: int,
    progress: ProgressCallback | None = None,
) -> PreparedCounts:
    raw_output = Path(raw_output_path)
    records_output = Path(records_output_path)
    queries_output = Path(queries_output_path)
    records_msgpack_output = Path(records_msgpack_output_path)
    queries_msgpack_output = Path(queries_msgpack_output_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    records_output.parent.mkdir(parents=True, exist_ok=True)
    queries_output.parent.mkdir(parents=True, exist_ok=True)
    records_msgpack_output.parent.mkdir(parents=True, exist_ok=True)
    queries_msgpack_output.parent.mkdir(parents=True, exist_ok=True)

    id_field = str(scenario.dataset.get("id_field", "id"))
    vector_field = str(scenario.dataset.get("vector_field", "emb"))
    text_field = scenario.dataset.get("text_field")
    if text_field is not None and not isinstance(text_field, str):
        raise ConfigError("scenario.dataset.text_field must be a string")

    source_rows = 0
    record_rows = 0
    query_rows = 0
    raw_digest = hashlib.sha256()
    records_digest = hashlib.sha256()
    queries_digest = hashlib.sha256()
    records_msgpack_digest = hashlib.sha256()
    queries_msgpack_digest = hashlib.sha256()
    msgpack_packer = msgpack.Packer(use_bin_type=True, use_single_float=True)
    ticker = ProgressTicker(progress)
    with (
        raw_output.open("wb") as raw_file,
        records_output.open("wb") as records_file,
        queries_output.open("wb") as queries_file,
        records_msgpack_output.open("wb") as records_msgpack_file,
        queries_msgpack_output.open("wb") as queries_msgpack_file,
    ):
        for row in rows:
            if record_rows >= record_limit and query_rows >= query_count:
                break
            raw = dict(row)
            normalized = normalize_record(
                raw,
                ordinal=source_rows,
                id_field=id_field,
                vector_field=vector_field,
                text_field=text_field,
            )
            _write_json_line(raw_file, raw_digest, raw)
            if query_rows < query_count:
                query = {
                    "id": normalized["id"],
                    "vector": normalized["vector"],
                    "metadata": normalized["metadata"],
                }
                json_size = _write_json_line(queries_file, queries_digest, query)
                _write_msgpack_record(
                    queries_msgpack_file,
                    queries_msgpack_digest,
                    msgpack_packer,
                    query,
                    estimated_size_bytes=json_size,
                )
                query_rows += 1
            elif record_rows < record_limit:
                json_size = _write_json_line(records_file, records_digest, normalized)
                _write_msgpack_record(
                    records_msgpack_file,
                    records_msgpack_digest,
                    msgpack_packer,
                    normalized,
                    estimated_size_bytes=json_size,
                )
                record_rows += 1
            source_rows += 1
            ticker.maybe(
                "dataset_prepare: writing "
                f"source_rows={source_rows} queries={query_rows}/{query_count} "
                f"records={record_rows}/{record_limit}"
            )
    return PreparedCounts(
        source_rows=source_rows,
        records=record_rows,
        queries=query_rows,
        raw_records_sha256=raw_digest.hexdigest(),
        records_sha256=records_digest.hexdigest(),
        queries_sha256=queries_digest.hexdigest(),
        records_msgpack_sha256=records_msgpack_digest.hexdigest(),
        queries_msgpack_sha256=queries_msgpack_digest.hexdigest(),
    )


def optimize_dataset(
    *,
    dataset_dir: str | Path,
    shards: int | None = None,
    progress: ProgressCallback | None = None,
) -> DatasetOptimizeResult:
    if shards is not None and shards <= 0:
        raise ConfigError("dataset optimize shard count must be a positive integer")
    out = Path(dataset_dir)
    manifest_path = out / DATASET_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ConfigError(f"dataset manifest {manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ConfigError(f"dataset manifest {manifest_path} must be a JSON object")

    records_path = _manifest_artifact_path(
        out,
        manifest,
        "records",
        RECORDS_FILENAME,
    )
    queries_path = _manifest_artifact_path(
        out,
        manifest,
        "queries",
        QUERIES_FILENAME,
    )
    records_msgpack_path = out / RECORDS_MSGPACK_FILENAME
    queries_msgpack_path = out / QUERIES_MSGPACK_FILENAME

    records_digest, records_count = _write_msgpack_from_jsonl(
        records_path,
        records_msgpack_path,
        progress=progress,
        label="records",
    )
    queries_digest, queries_count = _write_msgpack_from_jsonl(
        queries_path,
        queries_msgpack_path,
        progress=progress,
        label="queries",
    )
    record_shards: list[dict[str, Any]] = []
    if shards is not None:
        record_shards = _write_record_msgpack_shards_from_jsonl(
            records_path,
            output_dir=out,
            shard_count=shards,
            total_records=_manifest_written_rows(manifest),
            progress=progress,
        )

    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ConfigError(f"{manifest_path}: artifacts must be a mapping")
    artifacts.update(
        {
            "records_msgpack": str(records_msgpack_path),
            "records_msgpack_sha256": records_digest,
            "queries_msgpack": str(queries_msgpack_path),
            "queries_msgpack_sha256": queries_digest,
        }
    )
    if shards is not None:
        artifacts["records_shards"] = record_shards
    manifest["optimized_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ticker = ProgressTicker(progress)
    ticker.emit(
        "dataset_optimize: wrote msgpack "
        f"records={records_count} queries={queries_count}"
    )
    if shards is not None:
        ticker.emit(
            "dataset_optimize: wrote record shards "
            f"shards={len(record_shards)} records={records_count}"
        )
    return DatasetOptimizeResult(
        output_dir=out,
        manifest_path=manifest_path,
        records_msgpack_path=records_msgpack_path,
        queries_msgpack_path=queries_msgpack_path,
        record_shard_paths=[Path(str(item["path"])) for item in record_shards],
        manifest=manifest,
    )


def _write_json_line(file: Any, digest: Any, value: Mapping[str, Any]) -> int:
    line = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    file.write(line)
    digest.update(line)
    return _estimated_json_size_bytes(line)


def _write_msgpack_record(
    file: Any,
    digest: Any,
    packer: msgpack.Packer,
    value: Mapping[str, Any],
    *,
    estimated_size_bytes: int,
) -> None:
    packed = packer.pack(
        {
            "id": value["id"],
            "vector": value["vector"],
            "metadata": value.get("metadata", {}),
            "estimated_size_bytes": estimated_size_bytes,
        }
    )
    file.write(packed)
    digest.update(packed)


def _write_msgpack_from_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    progress: ProgressCallback | None,
    label: str,
) -> tuple[str, int]:
    if not input_path.exists():
        raise ConfigError(f"dataset artifact {input_path} does not exist")
    digest = hashlib.sha256()
    packer = msgpack.Packer(use_bin_type=True, use_single_float=True)
    ticker = ProgressTicker(progress)
    count = 0
    with input_path.open("rb") as input_file, output_path.open("wb") as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = _loads_json(line)
            if not isinstance(record, dict):
                raise ConfigError(f"{input_path}:{line_number} row must be an object")
            _write_msgpack_record(
                output_file,
                digest,
                packer,
                record,
                estimated_size_bytes=_estimated_json_size_bytes(line),
            )
            count += 1
            ticker.maybe(f"dataset_optimize: {label} records={count}")
    return digest.hexdigest(), count


def _write_record_msgpack_shards_from_jsonl(
    input_path: Path,
    *,
    output_dir: Path,
    shard_count: int,
    total_records: int,
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise ConfigError(f"dataset artifact {input_path} does not exist")
    records_per_shard = max(1, (total_records + shard_count - 1) // shard_count)
    packer = msgpack.Packer(use_bin_type=True, use_single_float=True)
    ticker = ProgressTicker(progress)
    shard_records: list[dict[str, Any]] = []
    shard_file: Any | None = None
    shard_digest: Any | None = None
    shard_path: Path | None = None
    shard_index = -1
    shard_record_count = 0
    shard_first_record_index = 0
    total_written = 0

    def close_shard() -> None:
        nonlocal shard_file
        if shard_file is None or shard_digest is None or shard_path is None:
            return
        shard_file.close()
        shard_records.append(
            {
                "path": str(shard_path),
                "sha256": shard_digest.hexdigest(),
                "records": shard_record_count,
                "first_record_index": shard_first_record_index,
                "last_record_index": total_written,
            }
        )
        shard_file = None

    with input_path.open("rb") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            if shard_file is None or shard_record_count >= records_per_shard:
                close_shard()
                shard_index += 1
                shard_path = output_dir / RECORD_SHARD_FILENAME_TEMPLATE.format(
                    index=shard_index,
                )
                shard_digest = hashlib.sha256()
                shard_file = shard_path.open("wb")
                shard_record_count = 0
                shard_first_record_index = total_written + 1
            record = _loads_json(line)
            if not isinstance(record, dict):
                raise ConfigError(f"{input_path}:{line_number} row must be an object")
            packed = packer.pack(
                {
                    "id": record["id"],
                    "vector": record["vector"],
                    "metadata": record.get("metadata", {}),
                    "estimated_size_bytes": _estimated_json_size_bytes(line),
                }
            )
            shard_file.write(packed)
            shard_digest.update(packed)
            shard_record_count += 1
            total_written += 1
            ticker.maybe(
                "dataset_optimize: sharding "
                f"records={total_written}/{total_records} shards={shard_index + 1}"
            )
    close_shard()
    return shard_records


def _manifest_written_rows(manifest: Mapping[str, Any]) -> int:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ConfigError("dataset manifest dataset must be a mapping")
    rows = dataset.get("written_rows")
    if not isinstance(rows, int) or rows < 0:
        raise ConfigError("dataset manifest dataset.written_rows must be an integer")
    return rows


def _loads_json(line: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _estimated_json_size_bytes(line: bytes) -> int:
    return len(line.rstrip(b"\r\n")) + 2


def _manifest_artifact_path(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    key: str,
    default_filename: str,
) -> Path:
    artifacts = manifest.get("artifacts", {})
    artifact = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    path = (
        Path(artifact)
        if isinstance(artifact, str) and artifact
        else dataset_dir / default_filename
    )
    if not path.is_absolute() and not path.exists():
        candidate = dataset_dir / path.name
        if candidate.exists():
            return candidate
    return path


def normalize_record(
    row: Mapping[str, Any],
    *,
    ordinal: int,
    id_field: str,
    vector_field: str,
    text_field: str | None = None,
) -> dict[str, Any]:
    vector = row.get(vector_field)
    if not isinstance(vector, list) or not vector:
        raise ConfigError(f"row {ordinal} field {vector_field!r} must be a vector list")

    record_id = row.get(id_field)
    if record_id is None or record_id == "":
        record_id = f"row-{ordinal}"

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {id_field, vector_field}
    }
    if text_field is not None and text_field in row:
        metadata["text"] = row[text_field]

    return {
        "id": str(record_id),
        "vector": vector,
        "metadata": metadata,
    }


def build_dataset_manifest(
    *,
    scenario: ScenarioConfig,
    output_dir: Path,
    requested_rows: int,
    written_rows: int,
    requested_source_rows: int,
    written_source_rows: int,
    requested_query_rows: int,
    written_queries: int,
    dry_run: bool,
    status: str,
    artifact_checksums: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    raw_records_path = output_dir / RAW_RECORDS_FILENAME
    records_path = output_dir / RECORDS_FILENAME
    queries_path = output_dir / QUERIES_FILENAME
    records_msgpack_path = output_dir / RECORDS_MSGPACK_FILENAME
    queries_msgpack_path = output_dir / QUERIES_MSGPACK_FILENAME
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {
            "name": "lambdadb-bench",
            "version": __version__,
        },
        "status": status,
        "dry_run": dry_run,
        "scenario": {
            "name": scenario.name,
        },
        "dataset": {
            "provider": scenario.dataset.get("provider", "huggingface"),
            "source": scenario.dataset.get("source"),
            "subset": scenario.dataset.get("subset"),
            "split": scenario.dataset.get("split", "train"),
            "rows": scenario.dataset.get("rows"),
            "dimensions": scenario.dataset.get("dimensions"),
            "vector_field": scenario.dataset.get("vector_field"),
            "metric": scenario.dataset.get("metric", "cosine"),
            "id_field": scenario.dataset.get("id_field", "id"),
            "text_field": scenario.dataset.get("text_field"),
            "seed": scenario.dataset.get("seed"),
            "requested_source_rows": requested_source_rows,
            "written_source_rows": written_source_rows,
            "requested_rows": requested_rows,
            "written_rows": written_rows,
            "requested_query_rows": requested_query_rows,
            "written_query_rows": written_queries,
        },
        "artifacts": {
            "raw_records": str(raw_records_path),
            "raw_records_sha256": None,
            "records": str(records_path),
            "records_sha256": None,
            "records_msgpack": str(records_msgpack_path),
            "records_msgpack_sha256": None,
            "queries": str(queries_path),
            "queries_sha256": None,
            "queries_msgpack": str(queries_msgpack_path),
            "queries_msgpack_sha256": None,
        },
    }
    if artifact_checksums:
        manifest["artifacts"].update(dict(artifact_checksums))
    return manifest


def default_dataset_output_dir(scenario: ScenarioConfig) -> Path:
    return Path("data") / "datasets" / scenario.name


def _requested_rows(scenario: ScenarioConfig, limit: int | None) -> int:
    if limit is not None:
        return limit
    rows = scenario.dataset.get("rows")
    if not isinstance(rows, int):
        raise ConfigError("scenario.dataset.rows must be an integer")
    return rows


def _requested_query_rows(
    scenario: ScenarioConfig,
    query_count: int | None,
) -> int:
    if query_count is not None:
        if query_count < 0:
            raise ConfigError("dataset prepare query count must be non-negative")
        return query_count
    configured = scenario.query.get("query_count", DEFAULT_QUERY_COUNT)
    if not isinstance(configured, int):
        raise ConfigError("scenario.query.query_count must be an integer")
    if configured < 0:
        raise ConfigError("scenario.query.query_count must be non-negative")
    return configured
