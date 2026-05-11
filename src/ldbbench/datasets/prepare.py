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

RAW_RECORDS_FILENAME = "raw_records.jsonl"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
SUPPORTED_PROVIDERS = {"huggingface"}


@dataclass(frozen=True)
class DatasetPrepareResult:
    output_dir: Path
    manifest_path: Path
    raw_records_path: Path
    manifest: dict[str, Any]


def prepare_dataset(
    *,
    scenario: ScenarioConfig,
    output_dir: str | Path,
    limit: int | None = None,
    dry_run: bool = False,
    source_rows: Iterable[Mapping[str, Any]] | None = None,
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
    manifest_path = out / DATASET_MANIFEST_FILENAME

    written_rows = 0
    if dry_run:
        status = "planned"
    else:
        rows = source_rows
        if rows is None:
            rows = load_huggingface_rows(scenario=scenario, limit=requested_rows)
        written_rows = write_raw_records(
            rows=rows,
            output_path=raw_records_path,
            limit=requested_rows,
        )
        status = "prepared"

    manifest = build_dataset_manifest(
        scenario=scenario,
        output_dir=out,
        requested_rows=requested_rows,
        written_rows=written_rows,
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
        manifest=manifest,
    )


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


def write_raw_records(
    *,
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    limit: int,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            if count >= limit:
                break
            file.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def build_dataset_manifest(
    *,
    scenario: ScenarioConfig,
    output_dir: Path,
    requested_rows: int,
    written_rows: int,
    dry_run: bool,
    status: str,
) -> dict[str, Any]:
    raw_records_path = output_dir / RAW_RECORDS_FILENAME
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
            "seed": scenario.dataset.get("seed"),
            "requested_rows": requested_rows,
            "written_rows": written_rows,
        },
        "artifacts": {
            "raw_records": str(raw_records_path),
            "raw_records_sha256": None,
        },
    }
    if raw_records_path.exists():
        manifest["artifacts"]["raw_records_sha256"] = sha256_file(raw_records_path)
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

