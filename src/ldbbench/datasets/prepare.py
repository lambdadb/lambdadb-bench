"""Dataset preparation skeleton."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ldbbench.__about__ import __version__
from ldbbench.config import ConfigError, ScenarioConfig
from ldbbench.manifest import sha256_file
from ldbbench.progress import ProgressCallback, ProgressTicker

RAW_RECORDS_FILENAME = "raw_records.jsonl"
RECORDS_FILENAME = "records.jsonl"
QUERIES_FILENAME = "queries.jsonl"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
SUPPORTED_PROVIDERS = {"huggingface"}
DEFAULT_QUERY_COUNT = 1000


@dataclass(frozen=True)
class DatasetPrepareResult:
    output_dir: Path
    manifest_path: Path
    raw_records_path: Path
    records_path: Path
    queries_path: Path
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
        manifest=manifest,
    )


@dataclass(frozen=True)
class PreparedCounts:
    source_rows: int
    records: int
    queries: int


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
    record_limit: int,
    query_count: int,
    progress: ProgressCallback | None = None,
) -> PreparedCounts:
    raw_output = Path(raw_output_path)
    records_output = Path(records_output_path)
    queries_output = Path(queries_output_path)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    records_output.parent.mkdir(parents=True, exist_ok=True)
    queries_output.parent.mkdir(parents=True, exist_ok=True)

    id_field = str(scenario.dataset.get("id_field", "id"))
    vector_field = str(scenario.dataset.get("vector_field", "emb"))
    text_field = scenario.dataset.get("text_field")
    if text_field is not None and not isinstance(text_field, str):
        raise ConfigError("scenario.dataset.text_field must be a string")

    source_rows = 0
    record_rows = 0
    query_rows = 0
    ticker = ProgressTicker(progress)
    with (
        raw_output.open("w", encoding="utf-8") as raw_file,
        records_output.open("w", encoding="utf-8") as records_file,
        queries_output.open("w", encoding="utf-8") as queries_file,
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
            raw_file.write(json.dumps(raw, sort_keys=True) + "\n")
            if query_rows < query_count:
                query = {
                    "id": normalized["id"],
                    "vector": normalized["vector"],
                    "metadata": normalized["metadata"],
                }
                queries_file.write(json.dumps(query, sort_keys=True) + "\n")
                query_rows += 1
            elif record_rows < record_limit:
                records_file.write(json.dumps(normalized, sort_keys=True) + "\n")
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
    )


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
) -> dict[str, Any]:
    raw_records_path = output_dir / RAW_RECORDS_FILENAME
    records_path = output_dir / RECORDS_FILENAME
    queries_path = output_dir / QUERIES_FILENAME
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
            "queries": str(queries_path),
            "queries_sha256": None,
        },
    }
    if raw_records_path.exists():
        manifest["artifacts"]["raw_records_sha256"] = sha256_file(raw_records_path)
    if records_path.exists():
        manifest["artifacts"]["records_sha256"] = sha256_file(records_path)
    if queries_path.exists():
        manifest["artifacts"]["queries_sha256"] = sha256_file(queries_path)
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
