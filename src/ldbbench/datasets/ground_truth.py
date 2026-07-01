"""Ground truth generation for prepared dataset artifacts."""

from __future__ import annotations

import json
import math
import random
import re
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
    filter_bucket_metadata,
)
from ldbbench.manifest import sha256_file
from ldbbench.progress import ProgressCallback, ProgressTicker

GROUND_TRUTH_FILENAME = "ground_truth.jsonl"
GROUND_TRUTH_MANIFEST_FILENAME = "ground_truth_manifest.json"
SUPPORTED_BACKENDS = {"exact", "faiss"}
SUPPORTED_METRICS = {"cosine", "dot"}
DEFAULT_FAISS_BATCH_SIZE = 100


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


@dataclass(frozen=True)
class FilterSpec:
    name: str
    field: str
    operator: str
    value_source: str
    seed: int
    min_candidates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "operator": self.operator,
            "value_source": self.value_source,
            "seed": self.seed,
            "min_candidates": self.min_candidates,
        }


@dataclass(frozen=True)
class FilteredFaissBucket:
    filter_value: str
    record_ids: list[str]
    vectors: Any

    @property
    def candidate_count(self) -> int:
        return len(self.record_ids)


@dataclass(frozen=True)
class FilteredFaissQuery:
    ordinal: int
    filter_value: str
    query: VectorItem


def prepare_ground_truth(
    *,
    dataset_dir: str | Path,
    top_k: int,
    metric: str | None = None,
    backend: str = "exact",
    limit_queries: int | None = None,
    batch_size: int = DEFAULT_FAISS_BATCH_SIZE,
    filter_name: str | None = None,
    filter_field: str | None = None,
    filter_operator: str = "eq",
    filter_value_source: str | None = None,
    filter_seed: int = 0,
    filter_min_candidates: int | None = None,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> GroundTruthResult:
    if top_k <= 0:
        raise ConfigError("ground truth top_k must be a positive integer")
    if limit_queries is not None and limit_queries < 0:
        raise ConfigError("ground truth query limit must be non-negative")
    if batch_size <= 0:
        raise ConfigError("ground truth batch size must be a positive integer")
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"ground truth backend {backend!r} is not supported; "
            f"known: {sorted(SUPPORTED_BACKENDS)}"
        )
    filter_spec = _filter_spec(
        filter_name=filter_name,
        filter_field=filter_field,
        filter_operator=filter_operator,
        filter_value_source=filter_value_source,
        filter_seed=filter_seed,
        filter_min_candidates=filter_min_candidates or top_k,
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
    ground_truth_path = _ground_truth_output_path(out, filter_spec=filter_spec)
    manifest_path = _ground_truth_manifest_path(out, filter_spec=filter_spec)

    query_count = 0
    record_count = 0
    backend_details: dict[str, Any] = backend_manifest_details(
        backend=backend,
        metric=selected_metric,
        batch_size=batch_size,
    )
    ticker = ProgressTicker(progress)
    if dry_run:
        ticker.emit(
            f"ground_truth: planning backend={backend} metric={selected_metric}"
        )
        status = "planned"
    else:
        ticker.emit(
            f"ground_truth: starting backend={backend} metric={selected_metric}"
        )
        if backend == "exact":
            ticker.emit("ground_truth: loading records for exact search")
            records = list(read_vector_items(records_path))
            if filter_spec is not None:
                ensure_filter_bucket_metadata(records, dataset_manifest)
            ticker.emit(f"ground_truth: loaded records={len(records)}")
            queries = read_vector_items(queries_path)
            if filter_spec is None:
                query_count = write_exact_ground_truth(
                    records=records,
                    queries=queries,
                    output_path=ground_truth_path,
                    top_k=top_k,
                    metric=selected_metric,
                    limit_queries=limit_queries,
                    progress=progress,
                )
            else:
                result = write_filtered_exact_ground_truth(
                    records=records,
                    queries=queries,
                    output_path=ground_truth_path,
                    top_k=top_k,
                    metric=selected_metric,
                    filter_spec=filter_spec,
                    limit_queries=limit_queries,
                    progress=progress,
                )
                query_count = result["queries"]
                backend_details.update(result["backend_details"])
            record_count = len(records)
        elif backend == "faiss":
            if filter_spec is None:
                result = write_faiss_ground_truth(
                    records_path=records_path,
                    queries_path=queries_path,
                    output_path=ground_truth_path,
                    top_k=top_k,
                    metric=selected_metric,
                    limit_queries=limit_queries,
                    batch_size=batch_size,
                    dataset_manifest=dataset_manifest,
                    progress=progress,
                )
            else:
                result = write_filtered_faiss_ground_truth(
                    records_path=records_path,
                    queries_path=queries_path,
                    output_path=ground_truth_path,
                    top_k=top_k,
                    metric=selected_metric,
                    limit_queries=limit_queries,
                    batch_size=batch_size,
                    dataset_manifest=dataset_manifest,
                    filter_spec=filter_spec,
                    progress=progress,
                )
            query_count = result["queries"]
            record_count = result["records"]
            backend_details.update(result["backend_details"])
        ticker.emit(
            f"ground_truth: computed records={record_count} queries={query_count}"
        )
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
        backend_details=backend_details,
        dry_run=dry_run,
        status=status,
        record_count=record_count,
        query_count=query_count,
        filter_spec=filter_spec,
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
    progress: ProgressCallback | None = None,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    ticker = ProgressTicker(progress)
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
            ticker.maybe(f"ground_truth: exact queries={count}")
    return count


def write_filtered_exact_ground_truth(
    *,
    records: list[VectorItem],
    queries: Iterable[VectorItem],
    output_path: str | Path,
    top_k: int,
    metric: str,
    filter_spec: FilterSpec,
    limit_queries: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    buckets, eligible_values = _eligible_record_buckets(
        records,
        filter_spec=filter_spec,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    ticker = ProgressTicker(progress)
    with output.open("w", encoding="utf-8") as file:
        for query in queries:
            if limit_queries is not None and count >= limit_queries:
                break
            filter_value = _assigned_filter_value(eligible_values, count)
            candidates = buckets[filter_value]
            matches = exact_top_k(
                query=query,
                records=candidates,
                top_k=top_k,
                metric=metric,
            )
            write_filtered_ground_truth_row(
                file,
                query=query,
                filter_spec=filter_spec,
                filter_value=filter_value,
                candidate_count=len(candidates),
                matches=matches,
            )
            count += 1
            ticker.maybe(f"ground_truth: exact filtered queries={count}")
    return {
        "queries": count,
        "backend_details": {
            "candidate_count": _candidate_count_summary(
                buckets,
                eligible_values,
            ),
        },
    }


def write_faiss_ground_truth(
    *,
    records_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    top_k: int,
    metric: str,
    limit_queries: int | None,
    batch_size: int,
    dataset_manifest: Mapping[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    faiss, np = import_faiss_dependencies()
    dimensions = dataset_dimensions(dataset_manifest)
    expected_records = dataset_record_count(dataset_manifest)
    ticker = ProgressTicker(progress)
    ticker.emit(
        "ground_truth: loading records for faiss "
        f"expected_records={expected_records or 'unknown'} dimensions={dimensions}"
    )
    record_ids, record_vectors = read_faiss_records(
        records_path=records_path,
        dimensions=dimensions,
        expected_records=expected_records,
        np=np,
        progress=progress,
    )
    normalize = metric == "cosine"
    if normalize:
        ticker.emit("ground_truth: normalizing record vectors")
        faiss.normalize_L2(record_vectors)

    ticker.emit(f"ground_truth: building faiss index records={len(record_ids)}")
    index = faiss.IndexFlatIP(dimensions)
    index.add(record_vectors)
    ticker.emit("ground_truth: searching queries with faiss")

    query_count = write_faiss_query_results(
        faiss=faiss,
        np=np,
        index=index,
        record_ids=record_ids,
        queries_path=queries_path,
        output_path=output_path,
        top_k=top_k,
        metric=metric,
        normalize=normalize,
        limit_queries=limit_queries,
        batch_size=batch_size,
        dimensions=dimensions,
        progress=progress,
    )
    return {
        "records": len(record_ids),
        "queries": query_count,
        "backend_details": {
            "batch_size": batch_size,
            "index_type": "IndexFlatIP",
            "normalize_vectors": normalize,
        },
    }


def write_filtered_faiss_ground_truth(
    *,
    records_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    top_k: int,
    metric: str,
    limit_queries: int | None,
    batch_size: int,
    dataset_manifest: Mapping[str, Any],
    filter_spec: FilterSpec,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    faiss, np = import_faiss_dependencies()
    dimensions = dataset_dimensions(dataset_manifest)
    ticker = ProgressTicker(progress)
    ticker.emit(
        f"ground_truth: counting records for filtered faiss dimensions={dimensions}"
    )
    bucket_counts, eligible_values, record_count = count_filtered_record_buckets(
        records_path=records_path,
        dataset_manifest=dataset_manifest,
        dimensions=dimensions,
        filter_spec=filter_spec,
        progress=progress,
    )
    normalize = metric == "cosine"
    queries_by_value, ordered_query_count = read_filtered_faiss_queries(
        queries_path=queries_path,
        dimensions=dimensions,
        eligible_values=eligible_values,
        limit_queries=limit_queries,
    )
    needed_values = set(queries_by_value)
    result_rows: list[str | None] = [None] * ordered_query_count
    large_bucket_bytes = max(
        (
            bucket_counts[value] * dimensions * np.dtype(np.float32).itemsize
            for value in needed_values
        ),
        default=0,
    )
    build_one_bucket_at_a_time = large_bucket_bytes >= 8 * 1024 * 1024 * 1024

    if build_one_bucket_at_a_time:
        ticker.emit(
            "ground_truth: building filtered faiss indexes one bucket at a time "
            f"values={len(needed_values)}"
        )
        for filter_value in sorted(needed_values):
            bucket = read_filtered_faiss_buckets(
                records_path=records_path,
                dataset_manifest=dataset_manifest,
                dimensions=dimensions,
                filter_spec=filter_spec,
                bucket_counts=bucket_counts,
                needed_values={filter_value},
                np=np,
                progress=progress,
            )[filter_value]
            search_filtered_faiss_bucket(
                faiss=faiss,
                np=np,
                bucket=bucket,
                queries=queries_by_value[filter_value],
                result_rows=result_rows,
                top_k=top_k,
                metric=metric,
                normalize=normalize,
                batch_size=batch_size,
                filter_spec=filter_spec,
            )
            ticker.emit(
                "ground_truth: searched filtered faiss bucket "
                f"value={filter_value} candidates={bucket.candidate_count}"
            )
    else:
        ticker.emit(
            f"ground_truth: loading filtered faiss buckets values={len(needed_values)}"
        )
        buckets = read_filtered_faiss_buckets(
            records_path=records_path,
            dataset_manifest=dataset_manifest,
            dimensions=dimensions,
            filter_spec=filter_spec,
            bucket_counts=bucket_counts,
            needed_values=needed_values,
            np=np,
            progress=progress,
        )
        for filter_value in sorted(needed_values):
            candidate_count = buckets[filter_value].candidate_count
            search_filtered_faiss_bucket(
                faiss=faiss,
                np=np,
                bucket=buckets[filter_value],
                queries=queries_by_value[filter_value],
                result_rows=result_rows,
                top_k=top_k,
                metric=metric,
                normalize=normalize,
                batch_size=batch_size,
                filter_spec=filter_spec,
            )
            ticker.emit(
                "ground_truth: searched filtered faiss bucket "
                f"value={filter_value} candidates={candidate_count}"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for index, row in enumerate(result_rows, start=1):
            if row is None:
                raise ConfigError("filtered faiss ground truth query result missing")
            file.write(row)
            ticker.maybe(f"ground_truth: faiss filtered queries={index}")
    return {
        "records": record_count,
        "queries": ordered_query_count,
        "backend_details": {
            "batch_size": batch_size,
            "index_type": "IndexFlatIP",
            "normalize_vectors": normalize,
            "filtered_index_values": len(needed_values),
            "eligible_filter_values": len(eligible_values),
            "candidate_count": _candidate_count_summary_from_counts(
                bucket_counts,
                eligible_values,
            ),
        },
    }


def count_filtered_record_buckets(
    *,
    records_path: str | Path,
    dataset_manifest: Mapping[str, Any],
    dimensions: int,
    filter_spec: FilterSpec,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, int], list[str], int]:
    bucket_counts: dict[str, int] = {}
    ticker = ProgressTicker(progress)
    record_count = 0
    with Path(records_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            _validate_raw_vector_dimensions(
                raw,
                dimensions=dimensions,
                path=Path(records_path),
                line_number=line_number,
            )
            value = _raw_record_filter_value(
                raw,
                filter_spec=filter_spec,
                dataset_manifest=dataset_manifest,
                record_index=record_count,
                path=Path(records_path),
                line_number=line_number,
            )
            if value is not None:
                bucket_counts[value] = bucket_counts.get(value, 0) + 1
            record_count += 1
            ticker.maybe(
                f"ground_truth: counting filtered records records={record_count}"
            )
    eligible_values = [
        value
        for value, count in bucket_counts.items()
        if count >= filter_spec.min_candidates
    ]
    if not eligible_values:
        raise ConfigError(
            "filtered ground truth has no eligible record buckets for "
            f"{filter_spec.field!r} with min_candidates={filter_spec.min_candidates}"
        )
    eligible_values = sorted(eligible_values)
    random.Random(filter_spec.seed).shuffle(eligible_values)
    return bucket_counts, eligible_values, record_count


def read_filtered_faiss_queries(
    *,
    queries_path: str | Path,
    dimensions: int,
    eligible_values: list[str],
    limit_queries: int | None,
) -> tuple[dict[str, list[FilteredFaissQuery]], int]:
    queries_by_value: dict[str, list[FilteredFaissQuery]] = {}
    query_count = 0
    for query in read_vector_items(queries_path):
        if limit_queries is not None and query_count >= limit_queries:
            break
        if len(query.vector) != dimensions:
            raise ConfigError(
                f"vector dimension mismatch for query {query.id!r}: "
                f"expected {dimensions}, got {len(query.vector)}"
            )
        filter_value = _assigned_filter_value(eligible_values, query_count)
        queries_by_value.setdefault(filter_value, []).append(
            FilteredFaissQuery(
                ordinal=query_count,
                filter_value=filter_value,
                query=query,
            )
        )
        query_count += 1
    return queries_by_value, query_count


def read_filtered_faiss_buckets(
    *,
    records_path: str | Path,
    dataset_manifest: Mapping[str, Any],
    dimensions: int,
    filter_spec: FilterSpec,
    bucket_counts: Mapping[str, int],
    needed_values: set[str],
    np: Any,
    progress: ProgressCallback | None = None,
) -> dict[str, FilteredFaissBucket]:
    buckets = {
        value: FilteredFaissBucket(
            filter_value=value,
            record_ids=[],
            vectors=np.empty((bucket_counts[value], dimensions), dtype=np.float32),
        )
        for value in needed_values
    }
    positions = {value: 0 for value in needed_values}
    ticker = ProgressTicker(progress)
    record_count = 0
    with Path(records_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            value = _raw_record_filter_value(
                raw,
                filter_spec=filter_spec,
                dataset_manifest=dataset_manifest,
                record_index=record_count,
                path=Path(records_path),
                line_number=line_number,
            )
            if value in buckets:
                vector = _raw_vector(
                    raw,
                    dimensions=dimensions,
                    path=Path(records_path),
                    line_number=line_number,
                )
                bucket = buckets[value]
                position = positions[value]
                bucket.vectors[position] = vector
                bucket.record_ids.append(
                    _raw_record_id(raw, Path(records_path), line_number)
                )
                positions[value] = position + 1
            record_count += 1
            ticker.maybe(
                "ground_truth: loading filtered faiss records "
                f"records={record_count} values={len(needed_values)}"
            )
    for value, position in positions.items():
        expected = bucket_counts[value]
        if position != expected:
            raise ConfigError(
                f"filtered bucket {value!r} has {position} records but count pass "
                f"found {expected}"
            )
    return buckets


def search_filtered_faiss_bucket(
    *,
    faiss: Any,
    np: Any,
    bucket: FilteredFaissBucket,
    queries: list[FilteredFaissQuery],
    result_rows: list[str | None],
    top_k: int,
    metric: str,
    normalize: bool,
    batch_size: int,
    filter_spec: FilterSpec,
) -> None:
    vectors = bucket.vectors
    if normalize:
        faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    search_k = min(bucket.candidate_count, top_k + 1)
    for start in range(0, len(queries), batch_size):
        query_batch = queries[start : start + batch_size]
        query_vectors = np.asarray(
            [entry.query.vector for entry in query_batch],
            dtype=np.float32,
        )
        if normalize:
            faiss.normalize_L2(query_vectors)
        scores, indices = index.search(query_vectors, search_k)
        for entry, query_scores, query_indices in zip(
            query_batch,
            scores,
            indices,
            strict=True,
        ):
            matches = faiss_matches(
                query=entry.query,
                scores=query_scores,
                indices=query_indices,
                record_ids=bucket.record_ids,
                top_k=top_k,
                metric=metric,
            )
            result_rows[entry.ordinal] = filtered_ground_truth_json_line(
                query=entry.query,
                filter_spec=filter_spec,
                filter_value=entry.filter_value,
                candidate_count=bucket.candidate_count,
                matches=matches,
            )


def write_faiss_query_results(
    *,
    faiss: Any,
    np: Any,
    index: Any,
    record_ids: list[str],
    queries_path: str | Path,
    output_path: str | Path,
    top_k: int,
    metric: str,
    normalize: bool,
    limit_queries: int | None,
    batch_size: int,
    dimensions: int,
    progress: ProgressCallback | None = None,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    search_k = min(len(record_ids), top_k + 1)
    ticker = ProgressTicker(progress)
    with output.open("w", encoding="utf-8") as file:
        query_batch: list[VectorItem] = []
        for query in read_vector_items(queries_path):
            if limit_queries is not None and query_count >= limit_queries:
                break
            if len(query.vector) != dimensions:
                raise ConfigError(
                    f"vector dimension mismatch for query {query.id!r}: "
                    f"expected {dimensions}, got {len(query.vector)}"
                )
            query_batch.append(query)
            if len(query_batch) >= batch_size:
                query_count += write_faiss_batch(
                    faiss=faiss,
                    np=np,
                    index=index,
                    record_ids=record_ids,
                    queries=query_batch,
                    file=file,
                    top_k=top_k,
                    search_k=search_k,
                    metric=metric,
                    normalize=normalize,
                )
                query_batch = []
                ticker.maybe(f"ground_truth: faiss queries={query_count}")
        if query_batch and (limit_queries is None or query_count < limit_queries):
            if limit_queries is not None:
                query_batch = query_batch[: limit_queries - query_count]
            query_count += write_faiss_batch(
                faiss=faiss,
                np=np,
                index=index,
                record_ids=record_ids,
                queries=query_batch,
                file=file,
                top_k=top_k,
                search_k=search_k,
                metric=metric,
                normalize=normalize,
            )
            ticker.maybe(f"ground_truth: faiss queries={query_count}")
    return query_count


def write_faiss_batch(
    *,
    faiss: Any,
    np: Any,
    index: Any,
    record_ids: list[str],
    queries: list[VectorItem],
    file: Any,
    top_k: int,
    search_k: int,
    metric: str,
    normalize: bool,
) -> int:
    query_vectors = np.asarray([query.vector for query in queries], dtype=np.float32)
    if normalize:
        faiss.normalize_L2(query_vectors)
    scores, indices = index.search(query_vectors, search_k)
    for query, query_scores, query_indices in zip(
        queries,
        scores,
        indices,
        strict=True,
    ):
        matches = faiss_matches(
            query=query,
            scores=query_scores,
            indices=query_indices,
            record_ids=record_ids,
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
    return len(queries)


def faiss_matches(
    *,
    query: VectorItem,
    scores: Any,
    indices: Any,
    record_ids: list[str],
    top_k: int,
    metric: str,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, str]] = []
    for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
        if index < 0:
            continue
        record_id = record_ids[index]
        if record_id == query.id:
            continue
        candidates.append((faiss_score(score, metric=metric), record_id))
    best = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
    return [
        {
            "id": record_id,
            "rank": rank,
            "score": score,
        }
        for rank, (score, record_id) in enumerate(best, start=1)
    ]


def faiss_score(score: float, *, metric: str) -> float:
    if metric in {"cosine", "dot"}:
        return float(score)
    raise ConfigError(f"unsupported metric {metric!r}")


def read_faiss_records(
    *,
    records_path: str | Path,
    dimensions: int,
    expected_records: int | None,
    np: Any,
    progress: ProgressCallback | None = None,
) -> tuple[list[str], Any]:
    ids: list[str] = []
    if expected_records is not None:
        vectors = np.empty((expected_records, dimensions), dtype=np.float32)
    else:
        rows = []
        vectors = None
    ticker = ProgressTicker(progress)
    with Path(records_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            item = parse_vector_item(
                raw,
                path=Path(records_path),
                line_number=line_number,
            )
            if len(item.vector) != dimensions:
                raise ConfigError(
                    f"vector dimension mismatch for record {item.id!r}: "
                    f"expected {dimensions}, got {len(item.vector)}"
                )
            ids.append(item.id)
            if vectors is not None:
                if len(ids) > expected_records:
                    raise ConfigError(
                        f"{records_path} has more records than dataset manifest "
                        f"declares ({expected_records})"
                    )
                vectors[len(ids) - 1] = item.vector
            else:
                rows.append(item.vector)
            ticker.maybe(
                "ground_truth: loading records "
                f"records={len(ids)}"
                + (f"/{expected_records}" if expected_records is not None else "")
            )

    if vectors is not None:
        if len(ids) != expected_records:
            raise ConfigError(
                f"{records_path} has {len(ids)} records but dataset manifest "
                f"declares {expected_records}"
            )
        return ids, vectors
    return ids, np.asarray(rows, dtype=np.float32)


def import_faiss_dependencies() -> tuple[Any, Any]:
    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        raise ConfigError(
            "ground truth backend 'faiss' requires optional dependencies; "
            "install with `uv sync --extra groundtruth`"
        ) from exc
    return faiss, np


def dataset_dimensions(dataset_manifest: Mapping[str, Any]) -> int:
    dimensions = dataset_manifest.get("dataset", {}).get("dimensions")
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ConfigError("dataset manifest must include positive dataset.dimensions")
    return dimensions


def dataset_record_count(dataset_manifest: Mapping[str, Any]) -> int | None:
    value = dataset_manifest.get("dataset", {}).get("written_rows")
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ConfigError("dataset manifest dataset.written_rows must be non-negative")
    return value


def backend_manifest_details(
    *,
    backend: str,
    metric: str,
    batch_size: int,
) -> dict[str, Any]:
    if backend == "faiss":
        return {
            "batch_size": batch_size,
            "index_type": "IndexFlatIP",
            "normalize_vectors": metric == "cosine",
        }
    return {}


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


def write_filtered_ground_truth_row(
    file: Any,
    *,
    query: VectorItem,
    filter_spec: FilterSpec,
    filter_value: str,
    candidate_count: int,
    matches: list[dict[str, Any]],
) -> None:
    file.write(
        filtered_ground_truth_json_line(
            query=query,
            filter_spec=filter_spec,
            filter_value=filter_value,
            candidate_count=candidate_count,
            matches=matches,
        )
    )


def filtered_ground_truth_json_line(
    *,
    query: VectorItem,
    filter_spec: FilterSpec,
    filter_value: str,
    candidate_count: int,
    matches: list[dict[str, Any]],
) -> str:
    return (
        json.dumps(
            {
                "query_id": query.id,
                "filter": {
                    "field": filter_spec.field,
                    "operator": filter_spec.operator,
                    "value": filter_value,
                },
                "filter_name": filter_spec.name,
                "candidate_count": candidate_count,
                "expected_count": len(matches),
                "matches": matches,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _eligible_record_buckets(
    records: list[VectorItem],
    *,
    filter_spec: FilterSpec,
) -> tuple[dict[str, list[VectorItem]], list[str]]:
    buckets: dict[str, list[VectorItem]] = {}
    for record in records:
        value = record.metadata.get(filter_spec.field)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(record)
    eligible_values = [
        value
        for value, candidates in buckets.items()
        if len(candidates) >= filter_spec.min_candidates
    ]
    if not eligible_values:
        raise ConfigError(
            "filtered ground truth has no eligible record buckets for "
            f"{filter_spec.field!r} with min_candidates={filter_spec.min_candidates}"
        )
    eligible_values = sorted(eligible_values)
    random.Random(filter_spec.seed).shuffle(eligible_values)
    return buckets, eligible_values


def _candidate_count_summary(
    buckets: Mapping[str, list[VectorItem]],
    eligible_values: list[str],
) -> dict[str, Any]:
    return _candidate_count_summary_from_counts(
        {value: len(candidates) for value, candidates in buckets.items()},
        eligible_values,
    )


def _candidate_count_summary_from_counts(
    bucket_counts: Mapping[str, int],
    eligible_values: list[str],
) -> dict[str, Any]:
    counts = sorted(bucket_counts[value] for value in eligible_values)
    return {
        "eligible_values": len(eligible_values),
        "min": counts[0],
        "p50": _percentile(counts, 50),
        "p95": _percentile(counts, 95),
        "max": counts[-1],
    }


def _percentile(values: list[int], percentile_value: int) -> float:
    if len(values) == 1:
        return float(values[0])
    rank = (percentile_value / 100) * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def ensure_filter_bucket_metadata(
    records: list[VectorItem],
    dataset_manifest: Mapping[str, Any],
) -> None:
    query_offset = dataset_manifest.get("dataset", {}).get("written_query_rows", 0)
    if not isinstance(query_offset, int) or query_offset < 0:
        query_offset = 0
    for index, record in enumerate(records):
        missing = [
            field
            for field in (
                "filter_bucket_2",
                "filter_bucket_10",
                "filter_bucket_100",
                "filter_bucket_1000",
            )
            if field not in record.metadata
        ]
        if not missing:
            continue
        record.metadata.update(
            filter_bucket_metadata(
                record_id=record.id,
                ordinal=query_offset + index,
            )
        )


def _assigned_filter_value(eligible_values: list[str], query_index: int) -> str:
    return eligible_values[query_index % len(eligible_values)]


def _raw_record_filter_value(
    raw: Mapping[str, Any],
    *,
    filter_spec: FilterSpec,
    dataset_manifest: Mapping[str, Any],
    record_index: int,
    path: Path,
    line_number: int,
) -> str | None:
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConfigError(f"{path}:{line_number} metadata must be a mapping")
    value = metadata.get(filter_spec.field)
    if value is None and filter_spec.field.startswith("filter_bucket_"):
        value = filter_bucket_metadata(
            record_id=_raw_record_id(raw, path, line_number),
            ordinal=_filter_bucket_ordinal(dataset_manifest, record_index),
        ).get(filter_spec.field)
    if value is None:
        return None
    return str(value)


def _filter_bucket_ordinal(
    dataset_manifest: Mapping[str, Any],
    record_index: int,
) -> int:
    query_offset = dataset_manifest.get("dataset", {}).get("written_query_rows", 0)
    if not isinstance(query_offset, int) or query_offset < 0:
        query_offset = 0
    return query_offset + record_index


def _raw_record_id(raw: Mapping[str, Any], path: Path, line_number: int) -> str:
    record_id = raw.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ConfigError(f"{path}:{line_number} missing non-empty id")
    return record_id


def _validate_raw_vector_dimensions(
    raw: Mapping[str, Any],
    *,
    dimensions: int,
    path: Path,
    line_number: int,
) -> None:
    vector = raw.get("vector")
    if not isinstance(vector, list) or not vector:
        raise ConfigError(f"{path}:{line_number} missing vector list")
    if len(vector) != dimensions:
        record_id = _raw_record_id(raw, path, line_number)
        raise ConfigError(
            f"vector dimension mismatch for record {record_id!r}: "
            f"expected {dimensions}, got {len(vector)}"
        )


def _raw_vector(
    raw: Mapping[str, Any],
    *,
    dimensions: int,
    path: Path,
    line_number: int,
) -> list[float]:
    _validate_raw_vector_dimensions(
        raw,
        dimensions=dimensions,
        path=path,
        line_number=line_number,
    )
    vector = raw["vector"]
    values = []
    for index, value in enumerate(vector):
        if not isinstance(value, int | float):
            raise ConfigError(f"{path}:{line_number} vector[{index}] must be numeric")
        values.append(float(value))
    return values


def score_vectors(*, query: VectorItem, record: VectorItem, metric: str) -> float:
    if len(query.vector) != len(record.vector):
        raise ConfigError(
            f"vector dimension mismatch for query {query.id!r} and record {record.id!r}"
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
        Path(value) if isinstance(value, str) and value else dataset_dir / default_name
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
    backend_details: Mapping[str, Any],
    dry_run: bool,
    status: str,
    record_count: int,
    query_count: int,
    filter_spec: FilterSpec | None = None,
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
            **dict(backend_details),
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
    if filter_spec is not None:
        manifest["ground_truth"]["filter"] = filter_spec.as_dict()
    return manifest


def _sha256_if_exists(path: Path) -> str | None:
    if path.exists():
        return sha256_file(path)
    return None


def _filter_spec(
    *,
    filter_name: str | None,
    filter_field: str | None,
    filter_operator: str,
    filter_value_source: str | None,
    filter_seed: int,
    filter_min_candidates: int,
) -> FilterSpec | None:
    if filter_name is None and filter_field is None and filter_value_source is None:
        return None
    if not isinstance(filter_name, str) or not filter_name:
        raise ConfigError("filtered ground truth requires --filter-name")
    if not isinstance(filter_field, str) or not filter_field:
        raise ConfigError("filtered ground truth requires --filter-field")
    if filter_operator != "eq":
        raise ConfigError("filtered ground truth supports only --filter-operator eq")
    if filter_value_source != "eligible-record-buckets":
        raise ConfigError(
            "filtered ground truth supports only "
            "--filter-value-source eligible-record-buckets"
        )
    if filter_min_candidates <= 0:
        raise ConfigError("--filter-min-candidates must be positive")
    return FilterSpec(
        name=filter_name,
        field=filter_field,
        operator=filter_operator,
        value_source=filter_value_source,
        seed=filter_seed,
        min_candidates=filter_min_candidates,
    )


def _ground_truth_output_path(
    dataset_dir: Path,
    *,
    filter_spec: FilterSpec | None,
) -> Path:
    if filter_spec is None:
        return dataset_dir / GROUND_TRUTH_FILENAME
    return dataset_dir / f"ground_truth.filtered.{_safe_name(filter_spec.name)}.jsonl"


def _ground_truth_manifest_path(
    dataset_dir: Path,
    *,
    filter_spec: FilterSpec | None,
) -> Path:
    if filter_spec is None:
        return dataset_dir / GROUND_TRUTH_MANIFEST_FILENAME
    return (
        dataset_dir
        / f"ground_truth.filtered.{_safe_name(filter_spec.name)}.manifest.json"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "filter"
