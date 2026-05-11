from __future__ import annotations

import pytest

from ldbbench.config import (
    ConfigError,
    load_scenario,
    load_target,
    redact_target_config,
)


def test_load_scenario_validates_common_shape(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1000000
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
  stages:
    - concurrency: 8
      duration: 5m
""",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "smoke"
    assert scenario.load["write_mode"] == "upsert"


def test_load_target_expands_env_and_redacts_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-cloud
endpoint: ${QDRANT_URL}
api_key: inline-secret
api_key_env: QDRANT_API_KEY
prepare:
  mode: create
metadata:
  report_label: qdrant-test
""",
        encoding="utf-8",
    )

    target = load_target(target_path)
    redacted = redact_target_config(target)

    assert target.endpoint == "https://example.qdrant.io"
    assert target.prepare_mode == "create"
    assert redacted["endpoint"] == "https://<redacted-host>"
    assert redacted["api_key"] == "<redacted>"
    assert redacted["api_key_env"] == "QDRANT_API_KEY"


def test_load_lambdadb_target_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAMBDADB_ENDPOINT", "https://api.lambdadb.ai")
    monkeypatch.setenv("LAMBDADB_PROJECT_NAME", "demo")
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: lambdadb
name: lambdadb-ci
endpoint: ${LAMBDADB_ENDPOINT}
project_name: ${LAMBDADB_PROJECT_NAME}
api_key_env: LAMBDADB_API_KEY
collection_name: smoke
vector_field: dense
index_configs:
  dense:
    type: vector
    dimensions: 3
    similarity: cosine
""",
        encoding="utf-8",
    )

    target = load_target(target_path)

    assert target.endpoint == "https://api.lambdadb.ai"
    assert target.project_name == "demo"
    assert target.api_key_env == "LAMBDADB_API_KEY"
    assert target.collection_name == "smoke"
    assert target.vector_field == "dense"
    assert target.index_configs["dense"]["dimensions"] == 3


def test_conflicting_collection_names_fail(tmp_path) -> None:
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: lambdadb
name: lambdadb-ci
collection: old
collection_name: new
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="collection"):
        load_target(target_path)


def test_missing_env_var_fails(tmp_path) -> None:
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-cloud
endpoint: ${MISSING_URL}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="MISSING_URL"):
        load_target(target_path)


def test_invalid_consistency_fails(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: linearizable
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="query.consistency"):
        load_scenario(scenario_path)
