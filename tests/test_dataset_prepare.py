from __future__ import annotations

import json
from pathlib import Path

import msgpack
import pytest

from ldbbench.config import ConfigError, ScenarioConfig
from ldbbench.datasets import (
    default_dataset_output_dir,
    optimize_dataset,
    prepare_dataset,
)
from ldbbench.manifest import sha256_file


def make_scenario() -> ScenarioConfig:
    return ScenarioConfig.from_mapping(
        {
            "name": "cohere-smoke",
            "dataset": {
                "provider": "huggingface",
                "source": "demo/source",
                "subset": "en",
                "rows": 10,
                "dimensions": 1024,
                "id_field": "_id",
                "vector_field": "emb",
                "text_field": "text",
                "seed": 123,
            },
            "load": {"write_mode": "upsert"},
            "query": {"consistency": "eventual"},
        }
    )


def test_default_dataset_output_dir() -> None:
    assert default_dataset_output_dir(make_scenario()).as_posix() == (
        "data/datasets/cohere-smoke"
    )


def test_prepare_dataset_dry_run_writes_manifest_only(tmp_path) -> None:
    result = prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=3,
        dry_run=True,
    )

    assert result.manifest_path.exists()
    assert not result.raw_records_path.exists()
    assert result.manifest["status"] == "planned"
    assert result.manifest["dataset"]["requested_rows"] == 3
    assert result.manifest["dataset"]["requested_query_rows"] == 1000
    assert result.manifest["dataset"]["requested_source_rows"] == 1003
    assert result.manifest["dataset"]["written_rows"] == 0


def test_prepare_dataset_writes_normalized_records_and_queries(tmp_path) -> None:
    rows = [
        {"_id": "q", "emb": [9.0, 9.0], "text": "query", "url": "q-url"},
        {"_id": "a", "emb": [1.0, 0.0], "text": "alpha", "url": "a-url"},
        {"_id": "b", "emb": [0.0, 1.0], "text": "beta", "url": "b-url"},
        {"_id": "c", "emb": [1.0, 1.0], "text": "gamma", "url": "c-url"},
    ]

    result = prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=2,
        query_count=1,
        source_rows=rows,
    )

    raw = [
        json.loads(line)
        for line in result.raw_records_path.read_text(encoding="utf-8").splitlines()
    ]
    records = [
        json.loads(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    queries = [
        json.loads(line)
        for line in result.queries_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.manifest["status"] == "prepared"
    assert result.manifest["dataset"]["written_source_rows"] == 3
    assert result.manifest["dataset"]["written_rows"] == 2
    assert result.manifest["dataset"]["written_query_rows"] == 1
    assert result.manifest["artifacts"]["raw_records_sha256"]
    assert result.manifest["artifacts"]["records_sha256"]
    assert result.manifest["artifacts"]["queries_sha256"]
    assert result.manifest["artifacts"]["records_msgpack_sha256"]
    assert result.manifest["artifacts"]["queries_msgpack_sha256"]
    assert result.manifest["artifacts"]["raw_records_sha256"] == sha256_file(
        result.raw_records_path
    )
    assert result.manifest["artifacts"]["records_sha256"] == sha256_file(
        result.records_path
    )
    assert result.manifest["artifacts"]["queries_sha256"] == sha256_file(
        result.queries_path
    )
    assert result.manifest["artifacts"]["records_msgpack_sha256"] == sha256_file(
        result.records_msgpack_path
    )
    assert result.manifest["artifacts"]["queries_msgpack_sha256"] == sha256_file(
        result.queries_msgpack_path
    )
    assert raw == rows[:3]
    assert queries[0]["id"] == "q"
    assert records == [
        {
            "id": "a",
            "metadata": {"text": "alpha", "url": "a-url"},
            "vector": [1.0, 0.0],
        },
        {
            "id": "b",
            "metadata": {"text": "beta", "url": "b-url"},
            "vector": [0.0, 1.0],
        },
    ]
    with result.records_msgpack_path.open("rb") as file:
        msgpack_records = list(msgpack.Unpacker(file, raw=False))
    assert msgpack_records[0]["id"] == "a"
    assert msgpack_records[0]["vector"] == [1.0, 0.0]
    assert msgpack_records[0]["estimated_size_bytes"] > 0


def test_optimize_dataset_writes_msgpack_for_existing_jsonl(tmp_path) -> None:
    result = prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=2,
        query_count=1,
        source_rows=[
            {"_id": "q", "emb": [9.0, 9.0], "text": "query"},
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
        ],
    )
    result.records_msgpack_path.unlink()
    result.queries_msgpack_path.unlink()

    optimized = optimize_dataset(dataset_dir=result.output_dir)

    assert optimized.records_msgpack_path.exists()
    assert optimized.queries_msgpack_path.exists()
    assert optimized.manifest["artifacts"]["records_msgpack_sha256"] == sha256_file(
        optimized.records_msgpack_path
    )


def test_optimize_dataset_writes_record_shards(tmp_path) -> None:
    result = prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=3,
        query_count=1,
        source_rows=[
            {"_id": "q", "emb": [9.0, 9.0], "text": "query"},
            {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
            {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
            {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
        ],
    )

    optimized = optimize_dataset(dataset_dir=result.output_dir, shards=2)

    shards = optimized.manifest["artifacts"]["records_shards"]
    assert len(shards) == 2
    assert [item["records"] for item in shards] == [2, 1]
    assert [item["first_record_index"] for item in shards] == [1, 3]
    assert [item["last_record_index"] for item in shards] == [2, 3]
    for item in shards:
        assert Path(item["path"]).exists()
        assert item["sha256"] == sha256_file(item["path"])


def test_prepare_dataset_falls_back_to_row_id(tmp_path) -> None:
    scenario = make_scenario()
    rows = [
        {"emb": [1.0, 0.0], "text": "missing id"},
    ]

    result = prepare_dataset(
        scenario=scenario,
        output_dir=tmp_path,
        limit=1,
        query_count=0,
        source_rows=rows,
    )

    records = [
        json.loads(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["id"] == "row-0"


def test_prepare_dataset_rejects_negative_query_count(tmp_path) -> None:
    with pytest.raises(ConfigError, match="query count"):
        prepare_dataset(
            scenario=make_scenario(),
            output_dir=tmp_path,
            limit=1,
            query_count=-1,
            source_rows=[],
        )
