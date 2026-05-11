from __future__ import annotations

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
    assert "status: planned" in captured.out
    assert (output_dir / "dataset_manifest.json").exists()


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


def test_target_check_command(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    target_path = tmp_path / "target.yaml"
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
    assert "only --dry-run is supported" in captured.err
