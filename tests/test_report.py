from __future__ import annotations

import csv
import json

import pytest

from ldbbench.config import ConfigError
from ldbbench.report import generate_report


def test_generate_report_writes_markdown_and_csv(tmp_path) -> None:
    qdrant = write_result(
        tmp_path,
        "qdrant",
        target_label="qdrant-cloud",
        vendor="qdrant",
        load_rps=952.68,
        query_qps=515.46,
    )
    lambdadb = write_result(
        tmp_path,
        "lambdadb",
        target_label="lambdadb-cloud",
        vendor="lambdadb",
        load_rps=1200.0,
        query_qps=700.0,
    )

    result = generate_report(
        [qdrant, lambdadb],
        output_path=tmp_path / "reports" / "cohere.md",
    )

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert result.run_count == 2
    assert "# LambdaDB Benchmark Report" in markdown
    assert "qdrant-cloud" in markdown
    assert "lambdadb-cloud" in markdown
    assert "Query Performance By Concurrency" in markdown
    assert result.load_csv_path.exists()
    assert result.query_csv_path.exists()

    load_rows = list(csv.DictReader(result.load_csv_path.open(encoding="utf-8")))
    assert [row["target"] for row in load_rows] == ["qdrant-cloud", "lambdadb-cloud"]
    assert load_rows[0]["records_per_second"] == "952.680"

    query_rows = list(csv.DictReader(result.query_csv_path.open(encoding="utf-8")))
    assert [row["concurrency"] for row in query_rows] == ["16", "16"]
    assert query_rows[1]["queries_per_second"] == "700.000"


def test_generate_report_rejects_missing_manifest(tmp_path) -> None:
    result_dir = tmp_path / "empty-result"
    result_dir.mkdir()

    with pytest.raises(ConfigError, match="run_manifest.json"):
        generate_report([result_dir], output_path=tmp_path / "report.md")


def test_generate_report_includes_dry_run_manifest_without_summary(tmp_path) -> None:
    result_dir = tmp_path / "dry-run"
    result_dir.mkdir()
    (result_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "scenario": {
                    "dataset": {"dimensions": 2, "rows": 10, "source": "demo"},
                    "name": "smoke",
                    "query": {"consistency": "eventual", "top_k": 10},
                },
                "target": {
                    "name": "qdrant",
                    "report_label": "qdrant-dry-run",
                    "vendor": "qdrant",
                },
            }
        ),
        encoding="utf-8",
    )

    result = generate_report([result_dir], output_path=tmp_path / "report.md")

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "qdrant-dry-run" in markdown
    assert "dry_run" in markdown


def write_result(
    tmp_path,
    name: str,
    *,
    target_label: str,
    vendor: str,
    load_rps: float,
    query_qps: float,
):
    result_dir = tmp_path / name
    result_dir.mkdir()
    summary = {
        "dataset_dir": "data/datasets/cohere-wikipedia-1m",
        "ground_truth": "data/datasets/cohere-wikipedia-1m/ground_truth.jsonl",
        "load": {
            "attempt_latency_ms": {
                "p50": 1307.9,
                "p95": 1957.6,
                "p99": 2252.0,
            },
            "batches": 4196,
            "concurrency": 8,
            "duration_seconds": 1049.7,
            "error_rate": 0.0,
            "errors": 0,
            "records": 1_000_000,
            "records_per_second": load_rps,
            "status": "completed",
            "visibility": {"status": "visible"},
        },
        "query": {
            "duration_seconds": 60.0,
            "error_rate": 0.0,
            "errors": 0,
            "mode": "staged",
            "queries": 30_000,
            "queries_per_second": query_qps,
            "recall_at_k": 0.9837,
            "recall_samples": 30_000,
            "stages": [
                {
                    "concurrency": 16,
                    "duration_seconds": 60.0,
                    "error_rate": 0.0,
                    "errors": 0,
                    "latency_ms": {
                        "p50": 29.4,
                        "p95": 51.5,
                        "p99": 69.2,
                    },
                    "queries": 30_000,
                    "queries_per_second": query_qps,
                    "recall_at_k": 0.9837,
                    "recall_samples": 30_000,
                    "stage_index": 1,
                }
            ],
        },
        "run_manifest": str(result_dir / "run_manifest.json"),
        "status": "completed",
    }
    manifest = {
        "run_id": name,
        "scenario": {
            "dataset": {
                "dimensions": 1024,
                "rows": 1_000_000,
                "source": "CohereLabs/wikipedia",
            },
            "name": "cohere-wikipedia-1m-vector",
            "query": {"consistency": "eventual", "top_k": 10},
        },
        "target": {
            "endpoint": "https://<redacted-host>",
            "name": target_label,
            "prepare_mode": "existing",
            "region": "us-east-1",
            "report_label": target_label,
            "vendor": vendor,
        },
    }
    scenario = {
        "quality": {
            "min_recall_at_10": 0.95,
        }
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (result_dir / "scenario.resolved.yaml").write_text(
        "quality:\n  min_recall_at_10: 0.95\n",
        encoding="utf-8",
    )
    assert scenario["quality"]["min_recall_at_10"] == 0.95
    return result_dir
