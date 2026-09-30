from __future__ import annotations

import json
import subprocess
from importlib import metadata

from ldbbench.config import load_scenario, load_target
from ldbbench.manifest import initialize_run_artifacts, tool_git_state


def test_initialize_run_artifacts_writes_manifest_and_redacted_target(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAMBDADB_ENDPOINT", "https://api.lambdadb.example")
    scenario_path = tmp_path / "scenario.yaml"
    target_path = tmp_path / "target.yaml"
    output_dir = tmp_path / "result"
    scenario_path.write_text(
        """
name: smoke
dataset:
  source: demo
  subset: en
  rows: 1000000
  dimensions: 1024
  seed: 123
load:
  write_mode: upsert
query:
  top_k: 10
  consistency: strong
""",
        encoding="utf-8",
    )
    target_path.write_text(
        """
vendor: lambdadb
name: lambdadb-test
endpoint: ${LAMBDADB_ENDPOINT}
api_key_env: LAMBDADB_API_KEY
region: us-east-1
prepare:
  mode: existing
metadata:
  deployment_mode: serverless
  report_label: lambdadb-ci
""",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)
    target = load_target(target_path)
    sdk_direct_url = {
        "url": "https://github.com/lambdadb/lambdadb.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "fedcba98",
            "requested_revision": "v0.9.0",
        },
    }
    monkeypatch.setattr(
        "ldbbench.manifest._sdk_direct_url",
        lambda _package: sdk_direct_url,
    )
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=output_dir,
        sdk_package="lambdadb",
        adapter_capabilities={
            "supported_query_consistency": ["eventual", "strong"],
        },
        dry_run_plan={"status": "supported"},
    )

    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    redacted_target = paths.target_redacted.read_text(encoding="utf-8")

    assert paths.scenario_resolved.exists()
    assert manifest["tool"]["sdk_package"] == "lambdadb"
    assert manifest["tool"]["sdk_version"] == metadata.version("lambdadb")
    assert manifest["tool"]["sdk_direct_url"] == sdk_direct_url
    assert {"git_commit", "git_dirty"} <= set(manifest["tool"])
    assert manifest["scenario"]["name"] == "smoke"
    assert manifest["scenario"]["query"]["consistency"] == "strong"
    assert manifest["target"]["report_label"] == "lambdadb-ci"
    assert manifest["target"]["endpoint"] == "https://<redacted-host>"
    assert manifest["target"]["adapter_capabilities"][
        "supported_query_consistency"
    ] == ["eventual", "strong"]
    assert manifest["dry_run_plan"]["status"] == "supported"
    assert "api.lambdadb.example" not in redacted_target


def _git(repo, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=bench", "-c", "user.email=bench@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_tool_git_state_uses_commit_recorded_by_git_install() -> None:
    direct_url = {
        "url": "https://github.com/lambdadb/lambdadb-bench",
        "vcs_info": {"vcs": "git", "commit_id": "0123abcd"},
    }

    assert tool_git_state(direct_url) == ("0123abcd", False)


def test_tool_git_state_reads_editable_checkout(tmp_path) -> None:
    checkout = tmp_path / "lambdadb-bench"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    (checkout / "README.md").write_text("bench\n", encoding="utf-8")
    _git(checkout, "add", "README.md")
    _git(checkout, "commit", "-q", "-m", "init")
    head = _git(checkout, "rev-parse", "HEAD")
    direct_url = {"url": checkout.as_uri(), "dir_info": {"editable": True}}

    (checkout / "results.json").write_text("{}\n", encoding="utf-8")
    clean = tool_git_state(direct_url)
    (checkout / "README.md").write_text("changed\n", encoding="utf-8")
    dirty = tool_git_state(direct_url)

    assert clean == (head, False)
    assert dirty == (head, True)


def test_tool_git_state_is_unknown_without_checkout_or_vcs_record(tmp_path) -> None:
    plain_dir = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    built_copy = {"url": tmp_path.as_uri(), "dir_info": {}}

    assert tool_git_state(None) == (None, None)
    assert tool_git_state(plain_dir) == (None, None)
    assert tool_git_state(built_copy) == (None, None)
