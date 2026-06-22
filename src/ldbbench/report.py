"""Report generation for completed benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ldbbench.config import ConfigError

SUMMARY_FILENAME = "summary.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
SCENARIO_RESOLVED_FILENAME = "scenario.resolved.yaml"
LOAD_CSV_HEADERS = [
    "result_dir",
    "target",
    "status",
    "records",
    "records_per_second",
    "duration_seconds",
    "batches",
    "concurrency",
    "processes",
    "worker_threads_per_process",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "errors",
    "error_rate",
    "visibility",
]
QUERY_CSV_HEADERS = [
    "result_dir",
    "target",
    "stage_index",
    "concurrency",
    "processes",
    "worker_threads_per_process",
    "partition_filter_applied",
    "filter_applied",
    "filter_name",
    "filter_selectivity",
    "recall_skip_reason",
    "queries",
    "max_requests",
    "queries_per_second",
    "duration_seconds",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "recall_at_k",
    "candidate_count_p50",
    "expected_count_p50",
    "returned_count_p50",
    "underfilled_result_rate",
    "errors",
    "error_rate",
]


@dataclass(frozen=True)
class ReportResult:
    markdown_path: Path
    load_csv_path: Path
    query_csv_path: Path
    run_count: int


@dataclass(frozen=True)
class RunReport:
    path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    scenario: dict[str, Any]


def generate_report(
    result_dirs: list[str | Path],
    *,
    output_path: str | Path,
) -> ReportResult:
    """Generate Markdown and CSV summaries for one or more result dirs."""

    runs = [_load_run_report(Path(path)) for path in result_dirs]
    if not runs:
        raise ConfigError("report requires at least one result directory")

    markdown_path = Path(output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    load_csv_path = markdown_path.with_name(f"{markdown_path.stem}-load.csv")
    query_csv_path = markdown_path.with_name(f"{markdown_path.stem}-query-stages.csv")

    markdown_path.write_text(_render_markdown(runs), encoding="utf-8")
    _write_load_csv(load_csv_path, runs)
    _write_query_csv(query_csv_path, runs)

    return ReportResult(
        markdown_path=markdown_path,
        load_csv_path=load_csv_path,
        query_csv_path=query_csv_path,
        run_count=len(runs),
    )


def _load_run_report(path: Path) -> RunReport:
    summary_path = path / SUMMARY_FILENAME
    manifest_path = path / RUN_MANIFEST_FILENAME
    scenario_path = path / SCENARIO_RESOLVED_FILENAME
    if not manifest_path.exists():
        raise ConfigError(f"report input {path} is missing {RUN_MANIFEST_FILENAME}")

    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path) if summary_path.exists() else _dry_run_summary()
    scenario = _load_yaml(scenario_path) if scenario_path.exists() else {}
    if not isinstance(summary, dict):
        raise ConfigError(f"{summary_path} must contain a JSON object")
    if not isinstance(manifest, dict):
        raise ConfigError(f"{manifest_path} must contain a JSON object")
    if not isinstance(scenario, dict):
        raise ConfigError(f"{scenario_path} must contain a YAML mapping")
    return RunReport(path=path, summary=summary, manifest=manifest, scenario=scenario)


def _dry_run_summary() -> dict[str, Any]:
    return {
        "status": "dry_run",
        "load": {
            "status": "skipped",
            "skip_reason": "dry_run",
            "records": 0,
            "duration_seconds": 0.0,
            "records_per_second": 0.0,
            "batches": 0,
            "errors": 0,
            "error_rate": 0.0,
        },
        "query": {
            "mode": "skipped",
            "skip_reason": "dry_run",
            "queries": 0,
            "duration_seconds": 0.0,
            "queries_per_second": 0.0,
            "recall_at_k": None,
            "recall_samples": 0,
            "errors": 0,
            "error_rate": 0.0,
        },
    }


def _render_markdown(runs: list[RunReport]) -> str:
    lines: list[str] = [
        "# LambdaDB Benchmark Report",
        "",
        "## Workload Summary",
        "",
    ]
    lines.extend(_markdown_table(_workload_rows(runs)))
    lines.extend(
        [
            "",
            "## Target Configurations",
            "",
        ]
    )
    lines.extend(_markdown_table(_target_rows(runs)))
    lines.extend(
        [
            "",
            "## Data Loading Results",
            "",
        ]
    )
    lines.extend(_markdown_table(_load_rows(runs)))
    lines.extend(
        [
            "",
            "## Query Performance By Concurrency",
            "",
        ]
    )
    query_rows = _query_stage_rows(runs)
    lines.extend(
        _markdown_table(query_rows)
        if query_rows
        else ["No staged query results were found."]
    )
    lines.extend(
        [
            "",
            "## Search-Under-Ingest Results",
            "",
        ]
    )
    search_rows = _search_under_ingest_rows(runs)
    lines.extend(
        _markdown_table(search_rows)
        if search_rows
        else ["No search-under-ingest results were found."]
    )
    lines.extend(
        [
            "",
            "## Recall And Quality Gates",
            "",
        ]
    )
    lines.extend(_markdown_table(_quality_rows(runs)))
    warnings = _warning_lines(runs)
    lines.extend(
        [
            "",
            "## Notes, Warnings, And Limitations",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No warnings detected from the available summaries.")
    lines.extend(
        [
            "- Cost assumptions are not normalized yet; values are reported as N/A.",
            "",
            "## Generated Artifacts",
            "",
            "- Load CSV: see sibling `*-load.csv` file.",
            "- Query stages CSV: see sibling `*-query-stages.csv` file.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_load_csv(path: Path, runs: list[RunReport]) -> None:
    rows = _load_rows(runs)
    _write_csv(path, rows, headers=LOAD_CSV_HEADERS)


def _write_query_csv(path: Path, runs: list[RunReport]) -> None:
    rows = _query_stage_rows(runs)
    _write_csv(path, rows, headers=QUERY_CSV_HEADERS)


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    headers: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _workload_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        scenario = _mapping(run.manifest.get("scenario"))
        dataset = _mapping(scenario.get("dataset"))
        query = _mapping(scenario.get("query"))
        rows.append(
            {
                "result_dir": str(run.path),
                "scenario": _fmt(scenario.get("name")),
                "dataset_rows": _fmt(dataset.get("rows")),
                "dimensions": _fmt(dataset.get("dimensions")),
                "source": _fmt(dataset.get("source")),
                "top_k": _fmt(query.get("top_k")),
                "consistency": _fmt(query.get("consistency")),
            }
        )
    return rows


def _target_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        target = _mapping(run.manifest.get("target"))
        rows.append(
            {
                "result_dir": str(run.path),
                "target": _target_label(run),
                "vendor": _fmt(target.get("vendor")),
                "region": _fmt(target.get("region")),
                "endpoint": _fmt(target.get("endpoint")),
                "prepare_mode": _fmt(target.get("prepare_mode")),
                "partition_config": _fmt(target.get("partition_config")),
                "deployment_mode": _fmt(target.get("deployment_mode")),
                "config_notes": _fmt(target.get("user_declared_config")),
            }
        )
    return rows


def _load_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        load = _mapping(run.summary.get("load"))
        latency = _mapping(load.get("attempt_latency_ms") or load.get("latency_ms"))
        visibility = _mapping(load.get("visibility"))
        rows.append(
            {
                "result_dir": str(run.path),
                "target": _target_label(run),
                "status": _fmt(load.get("status")),
                "records": _fmt(load.get("records")),
                "records_per_second": _fmt_float(load.get("records_per_second")),
                "duration_seconds": _fmt_float(load.get("duration_seconds")),
                "batches": _fmt(load.get("batches")),
                "concurrency": _fmt(load.get("concurrency")),
                "processes": _fmt(load.get("processes")),
                "worker_threads_per_process": _fmt(
                    load.get("worker_threads_per_process"),
                ),
                "p50_ms": _fmt_float(latency.get("p50")),
                "p95_ms": _fmt_float(latency.get("p95")),
                "p99_ms": _fmt_float(latency.get("p99")),
                "errors": _fmt(load.get("errors")),
                "error_rate": _fmt_float(load.get("error_rate")),
                "visibility": _fmt(visibility.get("status")),
            }
        )
    return rows


def _query_stage_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        query = _mapping(run.summary.get("query"))
        stages = query.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict):
                    rows.append(_query_stage_row(run, stage))
        elif query.get("mode") not in {None, "skipped"}:
            rows.append(_query_stage_row(run, query))
    return rows


def _query_stage_row(run: RunReport, stage: dict[str, Any]) -> dict[str, str]:
    latency = _mapping(stage.get("latency_ms"))
    filter_config = _mapping(stage.get("filter"))
    candidate_count = _mapping(stage.get("candidate_count"))
    expected_count = _mapping(stage.get("expected_count"))
    returned_count = _mapping(stage.get("returned_count"))
    return {
        "result_dir": str(run.path),
        "target": _target_label(run),
        "stage_index": _fmt(stage.get("stage_index")),
        "concurrency": _fmt(stage.get("concurrency")),
        "processes": _fmt(stage.get("processes")),
        "worker_threads_per_process": _fmt(
            stage.get("worker_threads_per_process"),
        ),
        "partition_filter_applied": _fmt(stage.get("partition_filter_applied")),
        "filter_applied": _fmt(stage.get("filter_applied")),
        "filter_name": _fmt(filter_config.get("name")),
        "filter_selectivity": _fmt_float(filter_config.get("expected_selectivity")),
        "recall_skip_reason": _fmt(stage.get("recall_skip_reason")),
        "queries": _fmt(stage.get("queries")),
        "max_requests": _fmt(stage.get("max_requests")),
        "queries_per_second": _fmt_float(stage.get("queries_per_second")),
        "duration_seconds": _fmt_float(stage.get("duration_seconds")),
        "p50_ms": _fmt_float(latency.get("p50")),
        "p95_ms": _fmt_float(latency.get("p95")),
        "p99_ms": _fmt_float(latency.get("p99")),
        "recall_at_k": _fmt_float(stage.get("recall_at_k")),
        "candidate_count_p50": _fmt_float(candidate_count.get("p50")),
        "expected_count_p50": _fmt_float(expected_count.get("p50")),
        "returned_count_p50": _fmt_float(returned_count.get("p50")),
        "underfilled_result_rate": _fmt_float(stage.get("underfilled_result_rate")),
        "errors": _fmt(stage.get("errors")),
        "error_rate": _fmt_float(stage.get("error_rate")),
    }


def _search_under_ingest_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        search = _mapping(run.summary.get("search_under_ingest"))
        if not search or search.get("mode") == "skipped":
            continue
        write_latency = _mapping(search.get("write_latency_ms"))
        query_latency = _mapping(
            search.get("immediate_query_latency_ms") or search.get("query_latency_ms")
        )
        visible_latency = _mapping(search.get("time_to_visible_ms"))
        rows.append(
            {
                "result_dir": str(run.path),
                "target": _target_label(run),
                "pattern": _fmt(search.get("pattern")),
                "consistency": _fmt(search.get("consistency")),
                "records": _fmt(search.get("records")),
                "records_per_second": _fmt_float(search.get("records_per_second")),
                "queries": _fmt(search.get("queries")),
                "queries_per_second": _fmt_float(search.get("queries_per_second")),
                "ingest_concurrency": _fmt(search.get("ingest_concurrency")),
                "query_concurrency": _fmt(search.get("query_concurrency")),
                "probe_documents": _fmt(search.get("probe_documents")),
                "probe_chunks": _fmt(search.get("probe_chunks")),
                "same_document_hit_rate_at_k": _fmt_float(
                    search.get("read_after_write_same_document_hit_rate_at_k"),
                ),
                "exact_chunk_hit_rate_at_k": _fmt_float(
                    search.get("read_after_write_exact_chunk_hit_rate_at_k"),
                ),
                "same_document_recall_at_k": _fmt_float(
                    search.get("read_after_write_same_document_recall_at_k"),
                ),
                "write_p95_ms": _fmt_float(write_latency.get("p95")),
                "query_p95_ms": _fmt_float(query_latency.get("p95")),
                "visible_p95_ms": _fmt_float(visible_latency.get("p95")),
                "recall_at_k": _fmt_float(search.get("recall_at_k")),
                "errors": _fmt(search.get("errors")),
                "error_rate": _fmt_float(search.get("error_rate")),
            }
        )
    return rows


def _quality_rows(runs: list[RunReport]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in runs:
        query = _mapping(run.summary.get("query"))
        gate = _recall_gate(run)
        recall = query.get("recall_at_k")
        passed = "N/A"
        if isinstance(recall, int | float) and gate is not None:
            passed = "yes" if float(recall) >= gate else "no"
        rows.append(
            {
                "result_dir": str(run.path),
                "target": _target_label(run),
                "recall_at_k": _fmt_float(recall),
                "recall_samples": _fmt(query.get("recall_samples")),
                "recall_skip_reason": _fmt(query.get("recall_skip_reason")),
                "min_recall": _fmt_float(gate),
                "quality_gate": passed,
                "query_mode": _fmt(query.get("mode")),
            }
        )
    return rows


def _warning_lines(runs: list[RunReport]) -> list[str]:
    warnings: list[str] = []
    for run in runs:
        target = _target_label(run)
        status = run.summary.get("status")
        if status != "completed":
            warnings.append(f"{target}: run status is {_fmt(status)}.")
        load = _mapping(run.summary.get("load"))
        query = _mapping(run.summary.get("query"))
        search = _mapping(run.summary.get("search_under_ingest"))
        if load.get("status") == "skipped":
            warnings.append(
                f"{target}: load stage skipped ({_fmt(load.get('skip_reason'))})."
            )
        if query.get("mode") == "skipped":
            warnings.append(
                f"{target}: query stage skipped ({_fmt(query.get('skip_reason'))})."
            )
        if query.get("recall_skip_reason") == "partition_filtered":
            warnings.append(
                f"{target}: recall is N/A because partition-filtered search "
                "uses a restricted candidate set."
            )
        underfilled = query.get("underfilled_result_rate")
        if isinstance(underfilled, int | float) and underfilled > 0:
            warnings.append(
                f"{target}: filtered search underfilled "
                f"{_fmt_float(underfilled)} of successful query results."
            )
        for stage_name, stage in (("load", load), ("query", query)):
            errors = stage.get("errors")
            if isinstance(errors, int) and errors:
                warnings.append(f"{target}: {stage_name} recorded {errors} errors.")
        if search:
            errors = search.get("errors")
            if isinstance(errors, int) and errors:
                warnings.append(
                    f"{target}: search-under-ingest recorded {errors} errors."
                )
        gate = _recall_gate(run)
        recall = query.get("recall_at_k")
        if (
            isinstance(recall, int | float)
            and gate is not None
            and float(recall) < gate
        ):
            warnings.append(
                f"{target}: recall {_fmt_float(recall)} is below gate "
                f"{_fmt_float(gate)}."
            )
    return warnings


def _markdown_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["N/A"]
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = " | ".join(_escape_md(row[header]) for header in headers)
        lines.append(f"| {cells} |")
    return lines


def _recall_gate(run: RunReport) -> float | None:
    quality = _mapping(run.scenario.get("quality"))
    query = _mapping(run.manifest.get("scenario")).get("query")
    top_k = _mapping(query).get("top_k")
    if isinstance(top_k, int):
        value = quality.get(f"min_recall_at_{top_k}")
        if isinstance(value, int | float):
            return float(value)
    value = quality.get("min_recall")
    return float(value) if isinstance(value, int | float) else None


def _target_label(run: RunReport) -> str:
    target = _mapping(run.manifest.get("target"))
    return _fmt(target.get("report_label") or target.get("name") or run.path.name)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, int | float):
        return _fmt(value)
    return f"{float(value):.3f}"


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
