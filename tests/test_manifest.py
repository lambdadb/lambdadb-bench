from __future__ import annotations

import json

from ldbbench.config import load_scenario, load_target
from ldbbench.manifest import initialize_run_artifacts


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
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=scenario_path,
        target_path=target_path,
        output_dir=output_dir,
    )

    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    redacted_target = paths.target_redacted.read_text(encoding="utf-8")

    assert paths.scenario_resolved.exists()
    assert manifest["scenario"]["name"] == "smoke"
    assert manifest["scenario"]["query"]["consistency"] == "strong"
    assert manifest["target"]["report_label"] == "lambdadb-ci"
    assert manifest["target"]["endpoint"] == "https://<redacted-host>"
    assert "api.lambdadb.example" not in redacted_target

