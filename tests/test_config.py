from __future__ import annotations

from pathlib import Path

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
  concurrency: 64
  processes: 4
  sharded_records: true
  shard_count: 16
query:
  consistency: eventual
  processes: 2
  partition_filter:
    field: url
    metadata_field: url
  stages:
    - concurrency: 8
      duration: 5m
""",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.name == "smoke"
    assert scenario.load["write_mode"] == "upsert"
    assert scenario.load["processes"] == 4
    assert scenario.load["sharded_records"] is True
    assert scenario.load["shard_count"] == 16
    assert scenario.query["processes"] == 2
    assert scenario.query["partition_filter"] == {
        "field": "url",
        "metadata_field": "url",
    }


def test_load_scenario_accepts_search_under_ingest_workload(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: search-under-ingest
workload: search_under_ingest
dataset:
  rows: 1000000
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
  top_k: 10
search_under_ingest:
  pattern: upload_and_ask
  probe_source: queries
  document_group_field: url
  max_probe_documents: 100
  duration: 5m
  min_chunks_per_document: 1
  max_chunks_per_document: 20
  probe_queries_per_document: 1
  probe_concurrency: 1
  top_k: 10
  consistency: strong
  poll_until_visible: true
  visibility_timeout: 5s
  visibility_poll_interval: 25ms
""",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.workload == "search_under_ingest"
    assert scenario.search_under_ingest["document_group_field"] == "url"
    assert scenario.search_under_ingest["consistency"] == "strong"


def test_load_scenario_accepts_parallel_search_under_ingest_pattern(
    tmp_path,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: parallel-search-under-ingest
workload: search_under_ingest
dataset:
  rows: 1000000
  dimensions: 1024
load:
  write_mode: upsert
query:
  consistency: eventual
  top_k: 10
search_under_ingest:
  pattern: parallel_upsert_query
  duration: 5m
  ingest_concurrency: 8
  query_concurrency: 16
  top_k: 10
  consistency: eventual
""",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)

    assert scenario.workload == "search_under_ingest"
    assert scenario.search_under_ingest["pattern"] == "parallel_upsert_query"
    assert scenario.search_under_ingest["ingest_concurrency"] == 8
    assert scenario.search_under_ingest["query_concurrency"] == 16


def test_load_target_expands_env_and_redacts_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_ENDPOINT", "https://example.qdrant.io")
    target_path = tmp_path / "target.yaml"
    target_path.write_text(
        """
vendor: qdrant
name: qdrant-cloud
endpoint: ${QDRANT_ENDPOINT}
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
partition_config:
  field_name: url
  data_type: keyword
  num_partitions: 16
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
    assert target.partition_config == {
        "field_name": "url",
        "data_type": "keyword",
        "num_partitions": 16,
    }


def test_example_targets_load_collection_names_from_env(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LAMBDADB_ENDPOINT", "https://api.lambdadb.ai")
    monkeypatch.setenv("LAMBDADB_PROJECT_NAME", "demo")
    monkeypatch.setenv("LAMBDADB_COLLECTION_NAME", "lambda-demo")
    monkeypatch.setenv("QDRANT_ENDPOINT", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "qdrant-demo")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "pinecone-demo")

    lambdadb = load_target(repo_root / "configs/lambdadb.example.yaml")
    lambdadb_partitioned = load_target(
        repo_root / "configs/lambdadb-partitioned.example.yaml"
    )
    qdrant = load_target(repo_root / "configs/qdrant-cloud.example.yaml")
    pinecone = load_target(repo_root / "configs/pinecone-serverless.example.yaml")

    assert lambdadb.collection_name == "lambda-demo"
    assert lambdadb_partitioned.collection_name == "lambda-demo"
    assert qdrant.collection_name == "qdrant-demo"
    assert pinecone.collection_name == "pinecone-demo"


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


def test_invalid_partition_filter_fails(tmp_path) -> None:
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
  consistency: eventual
  partition_filter:
    field: url
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="partition_filter.metadata_field"):
        load_scenario(scenario_path)


def test_invalid_process_counts_fail(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
  processes: 0
query:
  consistency: eventual
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="processes"):
        load_scenario(scenario_path)


def test_invalid_sharded_records_flag_fails(tmp_path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        """
name: smoke
dataset:
  rows: 1
  dimensions: 1024
load:
  write_mode: upsert
  sharded_records: enabled
query:
  consistency: eventual
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="sharded_records"):
        load_scenario(scenario_path)
