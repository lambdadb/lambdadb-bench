"""Exact ground truth generation for prepared dataset artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ldbbench.__about__ import __version__
from ldbbench.config import ConfigError
from ldbbench.datasets.prepare import (
    DATASET_MANIFEST_FILENAME,
    QUERIES_FILENAME,
    RECORDS_FILENAME,
)
from ldbbench.manifest import sha256_file

GROUND_TRUTH_FILENAME = "ground_truth.jsonl"
GROUND_TRUTH_MANIFEST_FILENAME = "ground_truth_manifest.json"
SUPPORTED_BACKENDS = {"exact"}
SUPPORTED_METRICS = {"cosine", "dot"}


@dataclass(frozen=True)
class GroundTruthResult:
    output_dir: Path
    ground_truth_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class VectorItem:
    id: str
    vector: list[float]
    metadata: dict[str, Any]
    norm: float


def prepare_ground_truth(
    *,
    dataset_dir: str | Path,
    top_k: int,
    metric: str | None = None,
    backend: str = "exact",
    limit_queries: int | None = None,
    dry_run: bool = False,
) -> GroundTruthResult:
    if top_k <= 0:
        raise ConfigError("ground truth top_k must be a positive integer")
    if limit_queries is not None and limit_queries < 0:
        raise ConfigError("ground truth query limit must be non-negative")
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"ground truth backend {backend!r} is not supported; "
            f"known: {sorted(SUPPORTED_BACKENDS)}"
        )

    out = Path(dataset_dir)
    dataset_manifest = load_dataset_manifest(out)
    selected_metric = metric or str(dataset_manifest["dataset"].get("metric", "cosine"))
    if selected_metric not in SUPPORTED_METRICS:
        raise ConfigError(
            f"ground truth metric {selected_metric!r} is not supported; "
            f"known: {sorted(SUPPORTED_METRICS)}"
        )

    records_path = artifact_path(out, dataset_manifest, "records", RECORDS_FILENAME)
    queries_path = artifact_path(out, dataset_manifest, "queries", QUERIES_FILENAME)
    ground_truth_path = out / GROUND_TRUTH_FILENAME
    manifest_path = out / GROUND_TRUTH_MANIFEST_FILENAME

    query_count = 0
    record_count = 0
    if dry_run:
        status = "planned"
    else:
        records = list(read_vector_items(records_path))
        queries = read_vector_items(queries_path)
        query_count = write_exact_ground_truth(
            records=records,
            queries=queries,
            output_path=ground_truth_path,
            top_k=top_k,
            metric=selected_metric,
            limit_queries=limit_queries,
        )
        record_count = len(records)
        status = "prepared"

    manifest = build_ground_truth_manifest(
        dataset_manifest=dataset_manifest,
        records_path=records_path,
        queries_path=queries_path,
        ground_truth_path=ground_truth_path,
        backend=backend,
        metric=selected_metric,
        top_k=top_k,
        limit_queries=limit_queries,
        dry_run=dry_run,
        status=status,
        record_count=record_count,
        query_count=query_count,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return GroundTruthResult(
        output_dir=out,
        ground_truth_path=ground_truth_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def write_exact_ground_truth(
    *,
    records: list[VectorItem],
    queries: Iterable[VectorItem],
    output_path: str | Path,
    top_k: int,
    metric: str,
    limit_queries: int | None = None,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for query in queries:
            if limit_queries is not None and count >= limit_queries:
                break
            matches = exact_top_k(
                query=query,
                records=records,
                top_k=top_k,
                metric=metric,
            )
            file.write(
                json.dumps(
                    {
                        "query_id": query.id,
                        "matches": matches,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return count


def exact_top_k(
    *,
    query: VectorItem,
    records: list[VectorItem],
    top_k: int,
    metric: str,
) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        if record.id == query.id:
            continue
        score = score_vectors(query=query, record=record, metric=metric)
        scored.append((score, record.id))

    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
    return [
        {
            "id": record_id,
            "rank": rank,
            "score": score,
        }
        for rank, (score, record_id) in enumerate(best, start=1)
    ]


def score_vectors(*, query: VectorItem, record: VectorItem, metric: str) -> float:
    if len(query.vector) != len(record.vector):
        raise ConfigError(
            f"vector dimension mismatch for query {query.id!r} "
            f"and record {record.id!r}"
        )
    dot = sum(q * r for q, r in zip(query.vector, record.vector, strict=True))
    if metric == "dot":
        return dot
    if metric == "cosine":
        denominator = query.norm * record.norm
        if denominator == 0:
            return 0.0
        return dot / denominator
    raise ConfigError(f"unsupported metric {metric!r}")


def read_vector_items(path: str | Path) -> Iterable[VectorItem]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            yield parse_vector_item(raw, path=Path(path), line_number=line_number)


def parse_vector_item(
    raw: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> VectorItem:
    record_id = raw.get("id")
    vector = raw.get("vector")
    metadata = raw.get("metadata", {})
    if not isinstance(record_id, str) or not record_id:
        raise ConfigError(f"{path}:{line_number} missing non-empty id")
    if not isinstance(vector, list) or not vector:
        raise ConfigError(f"{path}:{line_number} missing vector list")
    if not isinstance(metadata, dict):
        raise ConfigError(f"{path}:{line_number} metadata must be a mapping")
    values = []
    for index, value in enumerate(vector):
        if not isinstance(value, int | float):
            raise ConfigError(f"{path}:{line_number} vector[{index}] must be numeric")
        values.append(float(value))
    return VectorItem(
        id=record_id,
        vector=values,
        metadata=dict(metadata),
        norm=math.sqrt(sum(value * value for value in values)),
    )


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read dataset manifest {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ConfigError(f"{manifest_path} must contain a JSON object")
    return manifest


def artifact_path(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    key: str,
    default_name: str,
) -> Path:
    artifacts = dataset_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ConfigError("dataset manifest artifacts must be a mapping")
    value = artifacts.get(key)
    path = (
        Path(value)
        if isinstance(value, str) and value
        else dataset_dir / default_name
    )
    if not path.is_absolute() and not path.exists():
        candidate = dataset_dir / path.name
        if candidate.exists():
            return candidate
    return path


def build_ground_truth_manifest(
    *,
    dataset_manifest: Mapping[str, Any],
    records_path: Path,
    queries_path: Path,
    ground_truth_path: Path,
    backend: str,
    metric: str,
    top_k: int,
    limit_queries: int | None,
    dry_run: bool,
    status: str,
    record_count: int,
    query_count: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {
            "name": "lambdadb-bench",
            "version": __version__,
        },
        "status": status,
        "dry_run": dry_run,
        "dataset": {
            "source": dataset_manifest.get("dataset", {}).get("source"),
            "subset": dataset_manifest.get("dataset", {}).get("subset"),
            "records": record_count,
            "queries": query_count,
        },
        "ground_truth": {
            "backend": backend,
            "metric": metric,
            "top_k": top_k,
            "limit_queries": limit_queries,
        },
        "artifacts": {
            "records": str(records_path),
            "records_sha256": _sha256_if_exists(records_path),
            "queries": str(queries_path),
            "queries_sha256": _sha256_if_exists(queries_path),
            "ground_truth": str(ground_truth_path),
            "ground_truth_sha256": _sha256_if_exists(ground_truth_path),
        },
    }
    return manifest


def _sha256_if_exists(path: Path) -> str | None:
    if path.exists():
        return sha256_file(path)
    return None
