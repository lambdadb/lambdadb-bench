from __future__ import annotations

import json

from ldbbench.config import ScenarioConfig
from ldbbench.datasets import default_dataset_output_dir, prepare_dataset


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
                "vector_field": "emb",
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
    assert result.manifest["dataset"]["written_rows"] == 0


def test_prepare_dataset_writes_limited_raw_rows(tmp_path) -> None:
    rows = [
        {"id": "a", "emb": [1.0, 0.0], "text": "alpha"},
        {"id": "b", "emb": [0.0, 1.0], "text": "beta"},
        {"id": "c", "emb": [1.0, 1.0], "text": "gamma"},
    ]

    result = prepare_dataset(
        scenario=make_scenario(),
        output_dir=tmp_path,
        limit=2,
        source_rows=rows,
    )

    written = [
        json.loads(line)
        for line in result.raw_records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.manifest["status"] == "prepared"
    assert result.manifest["dataset"]["written_rows"] == 2
    assert result.manifest["artifacts"]["raw_records_sha256"]
    assert written == rows[:2]

