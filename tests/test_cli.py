from __future__ import annotations

import json

import pytest

from ldbbench.cli import main


def test_help_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: ldbbench" in captured.out


def test_doctor_reports_ok(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: ok" in captured.out


def test_version_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert captured.out.startswith("ldbbench ")


def test_config_validate_command(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-ci
prepare:
  mode: existing
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "config",
            "validate",
            "--scenario",
            str(scenario_path),
            "--target",
            str(target_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "scenario: smoke" in captured.out
    assert "target: qdrant-ci (qdrant)" in captured.out


def test_dataset_prepare_dry_run_command(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    output_dir = tmp_path / "dataset"
    scenario_path.write_text(
        """
name: smoke
dataset:
  provider: huggingface
  source: demo/source
  rows: 10
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "dataset",
            "prepare",
            "--scenario",
            str(scenario_path),
            "--out",
            str(output_dir),
            "--limit",
            "3",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "progress: dataset_prepare: planning artifacts" in captured.out
    assert "status: planned" in captured.out
    assert (output_dir / "dataset_manifest.json").exists()


def test_dataset_ground_truth_command(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    dataset_dir = tmp_path / "dataset"
    scenario_path.write_text(
        """
name: smoke
dataset:
  provider: huggingface
  source: demo/source
  rows: 3
  dimensions: 2
  id_field: _id
  vector_field: emb
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    from ldbbench.config import load_scenario
    from ldbbench.datasets.prepare import prepare_dataset

    prepare_dataset(
        scenario=load_scenario(scenario_path),
        output_dir=dataset_dir,
        limit=2,
        query_count=1,
        source_rows=[
            {"_id": "q", "emb": [1.0, 0.0]},
            {"_id": "a", "emb": [1.0, 0.0]},
            {"_id": "b", "emb": [0.0, 1.0]},
        ],
    )

    exit_code = main(
        [
            "dataset",
            "ground-truth",
            "--dataset-dir",
            str(dataset_dir),
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "progress: ground_truth: starting backend=exact" in captured.out
    assert "status: prepared" in captured.out
    assert (dataset_dir / "ground_truth.jsonl").exists()


def test_dataset_ground_truth_command_with_filter(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    dataset_dir = tmp_path / "dataset"
    scenario_path.write_text(
        """
name: smoke
dataset:
  provider: huggingface
  source: demo/source
  rows: 3
  dimensions: 2
  id_field: _id
  vector_field: emb
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    from ldbbench.config import load_scenario
    from ldbbench.datasets.prepare import prepare_dataset

    prepare_dataset(
        scenario=load_scenario(scenario_path),
        output_dir=dataset_dir,
        limit=2,
        query_count=1,
        source_rows=[
            {"_id": "q", "emb": [1.0, 0.0]},
            {"_id": "a", "emb": [1.0, 0.0]},
            {"_id": "b", "emb": [0.0, 1.0]},
        ],
    )

    exit_code = main(
        [
            "dataset",
            "ground-truth",
            "--dataset-dir",
            str(dataset_dir),
            "--top-k",
            "1",
            "--filter-name",
            "synthetic_bucket_50pct",
            "--filter-field",
            "filter_bucket_2",
            "--filter-value-source",
            "eligible-record-buckets",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "filter: synthetic_bucket_50pct" in captured.out
    assert (
        dataset_dir / "ground_truth.filtered.synthetic_bucket_50pct.jsonl"
    ).exists()


def test_dataset_optimize_command(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    dataset_dir = tmp_path / "dataset"
    scenario_path.write_text(
        """
name: smoke
dataset:
  provider: huggingface
  source: demo/source
  rows: 3
  dimensions: 2
  id_field: _id
  vector_field: emb
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    from ldbbench.config import load_scenario
    from ldbbench.datasets.prepare import prepare_dataset

    dataset = prepare_dataset(
        scenario=load_scenario(scenario_path),
        output_dir=dataset_dir,
        limit=2,
        query_count=1,
        source_rows=[
            {"_id": "q", "emb": [1.0, 0.0]},
            {"_id": "a", "emb": [1.0, 0.0]},
            {"_id": "b", "emb": [0.0, 1.0]},
        ],
    )
    dataset.records_msgpack_path.unlink()
    dataset.queries_msgpack_path.unlink()

    exit_code = main(
        ["dataset", "optimize", "--dataset-dir", str(dataset_dir), "--shards", "2"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: optimized" in captured.out
    assert "wrote_record_shards: 2" in captured.out
    assert (dataset_dir / "records.msgpack").exists()
    assert (dataset_dir / "queries.msgpack").exists()
    assert (dataset_dir / "records-00000.msgpack").exists()


def test_manifest_init_command(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    output_dir = tmp_path / "result"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: lambdadb
name: lambdadb-ci
endpoint: https://api.example.test
prepare:
  mode: existing
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "manifest",
            "init",
            "--scenario",
            str(scenario_path),
            "--target",
            str(target_path),
            "--out",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_manifest.json" in captured.out
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "target.redacted.yaml").exists()


def test_report_command(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "load": {
                    "batches": 1,
                    "duration_seconds": 1.0,
                    "errors": 0,
                    "records": 10,
                    "records_per_second": 10.0,
                    "status": "completed",
                },
                "query": {
                    "errors": 0,
                    "mode": "staged",
                    "queries": 20,
                    "queries_per_second": 20.0,
                    "recall_at_k": 1.0,
                    "recall_samples": 20,
                    "stages": [
                        {
                            "concurrency": 1,
                            "duration_seconds": 1.0,
                            "errors": 0,
                            "queries": 20,
                            "queries_per_second": 20.0,
                            "stage_index": 1,
                        }
                    ],
                },
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
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
                    "report_label": "qdrant-ci",
                    "vendor": "qdrant",
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "smoke.md"

    exit_code = main(["report", str(result_dir), "--out", str(report_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "runs: 1" in captured.out
    assert report_path.exists()
    assert (report_path.parent / "smoke-load.csv").exists()
    assert (report_path.parent / "smoke-query-stages.csv").exists()


def test_target_check_command(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-ci
endpoint: https://api.example.test
collection_name: smoke
prepare:
  mode: existing
""",
        encoding="utf-8",
    )

    exit_code = main(["target", "check", "--target", str(target_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: ok" in captured.out
    assert '"supported_query_consistency": [' in captured.out


def test_run_dry_run_writes_plan(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    output_dir = tmp_path / "result"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-ci
endpoint: https://api.example.test
prepare:
  mode: existing
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--dry-run",
            "--scenario",
            str(scenario_path),
            "--target",
            str(target_path),
            "--out",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dry_run: supported" in captured.out
    assert (output_dir / "run_manifest.json").exists()


def test_run_without_dry_run_fails(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-ci
endpoint: https://api.example.test
prepare:
  mode: existing
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--scenario",
                str(scenario_path),
                "--target",
                str(target_path),
                "--out",
                str(tmp_path / "result"),
            ]
        )

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "requires --dataset-dir" in captured.err
