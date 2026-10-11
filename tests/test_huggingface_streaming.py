from __future__ import annotations

import json

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ldbbench.config import ScenarioConfig
from ldbbench.datasets.prepare import load_huggingface_rows, prepare_dataset


@pytest.fixture
def parquet_source(tmp_path, monkeypatch):
    def make_source(count=24, buffer_size=4, seed=42):
        paths = []
        start = 0
        for shard in range(4):
            length = count // 4 + (shard < count % 4)
            rows = [
                {"id": str(i), "emb": [float(i), 1.0]}
                for i in range(start, start + length)
            ]
            path = tmp_path / f"shard-{shard}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            paths.append(str(path))
            start += length
        source = datasets.load_dataset(
            "parquet", data_files=paths, split="train", streaming=True
        )

        def load_dataset(name, config, *, split, streaming):
            assert (name, config, split, streaming) == (
                "local/streaming-test",
                "en",
                "train",
                True,
            )
            return source

        monkeypatch.setattr(datasets, "load_dataset", load_dataset)
        return ScenarioConfig.from_mapping(
            {
                "name": "streaming-test",
                "dataset": {
                    "provider": "huggingface",
                    "source": "local/streaming-test",
                    "subset": "en",
                    "rows": count,
                    "dimensions": 2,
                    "seed": seed,
                    "shuffle_buffer_size": buffer_size,
                },
                "load": {"write_mode": "upsert"},
                "query": {"consistency": "eventual"},
            }
        )

    return make_source


@pytest.mark.parametrize(
    ("buffer_size", "expected"),
    [
        (
            4,
            [
                18,
                21,
                20,
                19,
                13,
                23,
                22,
                12,
                16,
                6,
                17,
                15,
                8,
                9,
                10,
                11,
                0,
                7,
                1,
                14,
                5,
                3,
                4,
                2,
            ],
        ),
        (
            100,
            [
                10,
                9,
                0,
                1,
                15,
                13,
                11,
                5,
                12,
                16,
                21,
                18,
                2,
                6,
                23,
                17,
                8,
                4,
                20,
                22,
                3,
                19,
                7,
                14,
            ],
        ),
    ],
)
def test_streaming_preserves_selection_and_final_buffer(
    parquet_source, buffer_size, expected
):
    scenario = parquet_source(buffer_size=buffer_size)
    # Fixed datasets 4.x selections cover shard order and both buffer drain paths.
    for _ in range(2):
        rows = list(load_huggingface_rows(scenario=scenario, limit=24))
        assert [int(row["id"]) for row in rows] == expected


def test_streaming_preserves_random_index_batch_boundary(parquet_source):
    scenario = parquet_source(count=1010)
    rows = list(load_huggingface_rows(scenario=scenario, limit=1010))
    ids = [int(row["id"]) for row in rows]
    assert ids[996:1008] == [242, 240, 236, 243, 245, 246, 248, 247, 239, 250, 244, 252]
    assert sorted(ids) == list(range(1010))


def test_streaming_prepare_preserves_query_record_split(parquet_source, tmp_path):
    scenario = parquet_source()
    result = prepare_dataset(
        scenario=scenario, output_dir=tmp_path / "prepared", limit=4, query_count=2
    )
    queries = [
        json.loads(line) for line in result.queries_path.read_text().splitlines()
    ]
    records = [
        json.loads(line) for line in result.records_path.read_text().splitlines()
    ]
    raw = [
        json.loads(line) for line in result.raw_records_path.read_text().splitlines()
    ]
    assert [row["id"] for row in queries] == ["18", "21"]
    assert [row["id"] for row in records] == ["20", "19", "13", "23"]
    assert [row["id"] for row in raw] == ["18", "21", "20", "19", "13", "23"]
    assert result.manifest["dataset"]["written_source_rows"] == 6
