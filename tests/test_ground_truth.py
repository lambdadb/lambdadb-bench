from __future__ import annotations

import json

import pytest

from ldbbench.config import ConfigError, ScenarioConfig
from ldbbench.datasets.ground_truth import (
    VectorItem,
    exact_top_k,
    prepare_ground_truth,
    score_vectors,
)
from ldbbench.datasets.prepare import prepare_dataset


def make_scenario() -> ScenarioConfig:
    return ScenarioConfig.from_mapping(
        {
            "name": "ground-truth-smoke",
            "dataset": {
                "provider": "huggingface",
                "source": "demo/source",
                "subset": "en",
                "rows": 10,
                "dimensions": 2,
                "id_field": "_id",
                "vector_field": "emb",
                "text_field": "text",
                "metric": "cosine",
                "seed": 123,
            },
            "load": {"write_mode": "upsert"},
            "query": {"consistency": "eventual", "query_count": 1},
        }
    )


def prepare_fixture_dataset(tmp_path):
    rows = [
        {"_id": "query", "emb": [1.0, 0.0], "text": "query"},
        {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
        {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
        {"_id": "c", "emb": [0.8, 0.2], "text": "gamma"},
    ]
    return prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=3,
        query_count=1,
        source_rows=rows,
    )


def test_prepare_ground_truth_writes_exact_matches(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(dataset_dir=tmp_path, top_k=2)

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.manifest["status"] == "prepared"
    assert result.manifest["dataset"]["records"] == 3
    assert result.manifest["dataset"]["queries"] == 1
    assert result.manifest["artifacts"]["ground_truth_sha256"]
    assert truth["query_id"] == "query"
    assert [match["id"] for match in truth["matches"]] == ["a", "c"]
    assert truth["matches"][0]["rank"] == 1
    assert truth["matches"][0]["score"] == pytest.approx(1.0)


def test_prepare_ground_truth_dry_run_writes_manifest_only(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(dataset_dir=tmp_path, top_k=2, dry_run=True)

    assert result.manifest["status"] == "planned"
    assert not result.ground_truth_path.exists()
    assert result.manifest["artifacts"]["ground_truth_sha256"] is None


def test_prepare_ground_truth_limit_queries(tmp_path) -> None:
    rows = [
        {"_id": "q1", "emb": [1.0, 0.0], "text": "query"},
        {"_id": "q2", "emb": [0.0, 1.0], "text": "query"},
        {"_id": "a", "emb": [1.0, 0.0], "text": "alpha"},
        {"_id": "b", "emb": [0.0, 1.0], "text": "beta"},
    ]
    prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=2,
        query_count=2,
        source_rows=rows,
    )

    result = prepare_ground_truth(dataset_dir=tmp_path, top_k=1, limit_queries=1)

    assert result.manifest["dataset"]["queries"] == 1
    assert len(result.ground_truth_path.read_text(encoding="utf-8").splitlines()) == 1


def test_prepare_ground_truth_rejects_invalid_top_k(tmp_path) -> None:
    with pytest.raises(ConfigError, match="top_k"):
        prepare_ground_truth(dataset_dir=tmp_path, top_k=0)


def test_score_vectors_rejects_dimension_mismatch(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)
    result = prepare_ground_truth(dataset_dir=tmp_path, top_k=2)
    assert result.manifest["ground_truth"]["metric"] == "cosine"

    query = VectorItem(id="q", vector=[1.0], metadata={}, norm=1.0)
    record = VectorItem(id="r", vector=[1.0, 2.0], metadata={}, norm=1.0)
    with pytest.raises(ConfigError, match="dimension mismatch"):
        score_vectors(query=query, record=record, metric="cosine")


def test_exact_top_k_excludes_same_id() -> None:
    query = VectorItem(id="same", vector=[1.0, 0.0], metadata={}, norm=1.0)
    records = [
        VectorItem(id="same", vector=[1.0, 0.0], metadata={}, norm=1.0),
        VectorItem(id="other", vector=[0.5, 0.0], metadata={}, norm=0.5),
    ]

    matches = exact_top_k(query=query, records=records, top_k=2, metric="cosine")

    assert [match["id"] for match in matches] == ["other"]
