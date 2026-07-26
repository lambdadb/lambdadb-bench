from __future__ import annotations

import json
import sys
import types

import pytest

from ldbbench.config import ConfigError, ScenarioConfig
from ldbbench.datasets.ground_truth import (
    FilteredFaissBucket,
    FilteredFaissQuery,
    FilterSpec,
    VectorItem,
    _top_k_result,
    exact_top_k,
    ground_truth_manifest_path,
    prepare_ground_truth,
    score_vectors,
    search_filtered_faiss_bucket,
)
from ldbbench.datasets.prepare import prepare_dataset


def make_scenario(metric: str = "cosine") -> ScenarioConfig:
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
                "metric": metric,
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


def prepare_boundary_tie_dataset(tmp_path):
    rows = [
        {"_id": "query", "emb": [0.0, 0.0], "text": "query"},
        {"_id": "query", "emb": [0.0, 0.0], "text": "self"},
        {"_id": "strict", "emb": [0.5, 0.0], "text": "strict"},
        {"_id": "boundary-a", "emb": [1.0, 0.0], "text": "boundary"},
        {"_id": "boundary-b", "emb": [1.0, 0.0], "text": "boundary"},
        {"_id": "boundary-c", "emb": [1.0, 0.0], "text": "boundary"},
        {"_id": "outside", "emb": [2.0, 0.0], "text": "outside"},
    ]
    return prepare_dataset(
        scenario=make_scenario(metric="euclidean"),
        output_dir=tmp_path,
        limit=6,
        query_count=1,
        source_rows=rows,
    )


def fake_l2_index(np, search_sizes: list[int] | None = None):
    class FakeIndexFlatL2:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.vectors = None

        def add(self, vectors) -> None:
            assert vectors.shape[1] == self.dimensions
            self.vectors = vectors.copy()

        def search(self, queries, top_k: int):
            if search_sizes is not None:
                search_sizes.append(top_k)
            diff = queries[:, None, :] - self.vectors[None, :, :]
            scores = np.sum(diff * diff, axis=2)
            order = np.argsort(scores, axis=1)[:, :top_k]
            sorted_scores = np.take_along_axis(scores, order, axis=1)
            return sorted_scores, order

    return FakeIndexFlatL2


def test_prepare_ground_truth_writes_exact_matches(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(dataset_dir=tmp_path, top_k=2)

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.ground_truth_path.name == "ground_truth.cosine.jsonl"
    assert result.manifest_path.name == "ground_truth.cosine.manifest.json"
    assert result.manifest["status"] == "prepared"
    assert result.manifest["dataset"]["records"] == 3
    assert result.manifest["dataset"]["queries"] == 1
    assert result.manifest["artifacts"]["ground_truth_sha256"]
    assert truth["query_id"] == "query"
    assert [match["id"] for match in truth["matches"]] == ["a", "c"]
    assert "recall_groups" not in truth
    assert truth["matches"][0]["rank"] == 1
    assert truth["matches"][0]["score"] == pytest.approx(1.0)


def test_prepare_ground_truth_writes_exact_euclidean_matches(
    tmp_path,
) -> None:
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        metric="euclidean",
    )

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.manifest["ground_truth"]["metric"] == "euclidean"
    assert [match["id"] for match in truth["matches"]] == ["a", "c"]
    assert truth["matches"][0]["score"] == pytest.approx(0.0)
    assert truth["matches"][1]["score"] == pytest.approx(0.08)


def test_exact_ground_truth_preserves_k_boundary_candidates(tmp_path) -> None:
    prepare_boundary_tie_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        metric="euclidean",
    )

    truth = json.loads(result.ground_truth_path.read_text(encoding="utf-8"))
    assert truth["recall_groups"] == {
        "k": 2,
        "strict_ids": ["strict"],
        "boundary_ids": ["boundary-a", "boundary-b", "boundary-c"],
    }


def test_prepare_ground_truth_rejects_l2_metric(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    with pytest.raises(ConfigError, match="metric 'l2' is not supported"):
        prepare_ground_truth(dataset_dir=tmp_path, top_k=2, metric="l2")


def test_prepare_ground_truth_keeps_metric_artifacts_separate(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    cosine = prepare_ground_truth(dataset_dir=tmp_path, top_k=2, metric="cosine")
    euclidean = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        metric="euclidean",
    )

    assert cosine.ground_truth_path.name == "ground_truth.cosine.jsonl"
    assert euclidean.ground_truth_path.name == "ground_truth.euclidean.jsonl"
    assert cosine.ground_truth_path.exists()
    assert euclidean.ground_truth_path.exists()


def test_ground_truth_manifest_path_supports_current_and_legacy_names(tmp_path) -> None:
    assert ground_truth_manifest_path(
        tmp_path / "ground_truth.euclidean.jsonl"
    ).name == "ground_truth.euclidean.manifest.json"
    assert ground_truth_manifest_path(
        tmp_path / "ground_truth.jsonl"
    ).name == "ground_truth_manifest.json"


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


def test_prepare_ground_truth_writes_filtered_exact_matches(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=1,
        filter_name="synthetic_bucket_50pct",
        filter_field="filter_bucket_2",
        filter_value_source="eligible-record-buckets",
    )

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.ground_truth_path.name == (
        "ground_truth.cosine.filtered.synthetic_bucket_50pct.jsonl"
    )
    assert result.manifest_path.name == (
        "ground_truth.cosine.filtered.synthetic_bucket_50pct.manifest.json"
    )
    assert result.manifest["ground_truth"]["filter"]["field"] == "filter_bucket_2"
    assert result.manifest["ground_truth"]["candidate_count"]["eligible_values"] >= 1
    assert truth["filter"] == {
        "field": "filter_bucket_2",
        "operator": "eq",
        "value": truth["filter"]["value"],
    }
    assert truth["candidate_count"] >= 1
    assert truth["expected_count"] == 1
    assert [match["id"] for match in truth["matches"]]


def test_filtered_ground_truth_keeps_metric_artifacts_separate(tmp_path) -> None:
    prepare_fixture_dataset(tmp_path)
    options = {
        "dataset_dir": tmp_path,
        "top_k": 1,
        "filter_name": "synthetic_bucket_50pct",
        "filter_field": "filter_bucket_2",
        "filter_value_source": "eligible-record-buckets",
    }

    cosine = prepare_ground_truth(metric="cosine", **options)
    euclidean = prepare_ground_truth(metric="euclidean", **options)

    assert cosine.ground_truth_path.name == (
        "ground_truth.cosine.filtered.synthetic_bucket_50pct.jsonl"
    )
    assert euclidean.ground_truth_path.name == (
        "ground_truth.euclidean.filtered.synthetic_bucket_50pct.jsonl"
    )
    assert cosine.ground_truth_path.exists()
    assert euclidean.ground_truth_path.exists()


def test_prepare_ground_truth_backfills_missing_filter_buckets(tmp_path) -> None:
    dataset = prepare_fixture_dataset(tmp_path)
    stripped = []
    for line in dataset.records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["metadata"] = {
            key: value
            for key, value in record["metadata"].items()
            if not key.startswith("filter_bucket_")
        }
        stripped.append(json.dumps(record, sort_keys=True))
    dataset.records_path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=1,
        filter_name="synthetic_bucket_5pct",
        filter_field="filter_bucket_20",
        filter_value_source="eligible-record-buckets",
    )

    truth = json.loads(
        result.ground_truth_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert truth["candidate_count"] >= 1
    assert truth["filter"]["field"] == "filter_bucket_20"


def test_prepare_ground_truth_writes_faiss_matches(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    fake_faiss = types.ModuleType("faiss")

    class FakeIndexFlatIP:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.vectors = None

        def add(self, vectors) -> None:
            assert vectors.shape[1] == self.dimensions
            self.vectors = vectors.copy()

        def search(self, queries, top_k: int):
            scores = queries @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :top_k]
            sorted_scores = np.take_along_axis(scores, order, axis=1)
            return sorted_scores, order

    def normalize_l2(vectors) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

    fake_faiss.IndexFlatIP = FakeIndexFlatIP
    fake_faiss.normalize_L2 = normalize_l2
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        backend="faiss",
        batch_size=2,
    )

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.manifest["status"] == "prepared"
    assert result.manifest["ground_truth"]["backend"] == "faiss"
    assert result.manifest["ground_truth"]["index_type"] == "IndexFlatIP"
    assert result.manifest["ground_truth"]["batch_size"] == 2
    assert result.manifest["ground_truth"]["normalize_vectors"] is True
    assert [match["id"] for match in truth["matches"]] == ["a", "c"]


def test_prepare_ground_truth_writes_faiss_euclidean_matches(
    tmp_path,
    monkeypatch,
) -> None:
    np = pytest.importorskip("numpy")
    fake_faiss = types.ModuleType("faiss")

    class FakeIndexFlatL2:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.vectors = None

        def add(self, vectors) -> None:
            assert vectors.shape[1] == self.dimensions
            self.vectors = vectors.copy()

        def search(self, queries, top_k: int):
            diff = queries[:, None, :] - self.vectors[None, :, :]
            scores = np.sum(diff * diff, axis=2)
            order = np.argsort(scores, axis=1)[:, :top_k]
            sorted_scores = np.take_along_axis(scores, order, axis=1)
            return sorted_scores, order

    fake_faiss.IndexFlatL2 = FakeIndexFlatL2
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    prepare_fixture_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        metric="euclidean",
        backend="faiss",
        batch_size=2,
    )

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.manifest["ground_truth"]["backend"] == "faiss"
    assert result.manifest["ground_truth"]["metric"] == "euclidean"
    assert result.manifest["ground_truth"]["index_type"] == "IndexFlatL2"
    assert result.manifest["ground_truth"]["normalize_vectors"] is False
    assert [match["id"] for match in truth["matches"]] == ["a", "c"]
    assert truth["matches"][0]["score"] == pytest.approx(0.0)


def test_faiss_ground_truth_preserves_complete_k_boundary_after_self_match(
    tmp_path,
    monkeypatch,
) -> None:
    np = pytest.importorskip("numpy")
    search_sizes: list[int] = []
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.IndexFlatL2 = fake_l2_index(np, search_sizes)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    prepare_boundary_tie_dataset(tmp_path)

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=2,
        metric="euclidean",
        backend="faiss",
    )

    truth = json.loads(result.ground_truth_path.read_text(encoding="utf-8"))
    assert [match["id"] for match in truth["matches"]] == [
        "strict",
        "boundary-a",
    ]
    assert truth["recall_groups"] == {
        "k": 2,
        "strict_ids": ["strict"],
        "boundary_ids": ["boundary-a", "boundary-b", "boundary-c"],
    }
    assert search_sizes == [4, 6]
    assert result.manifest["ground_truth"]["recall_semantics"] == (
        "k-boundary-tie-aware-v1"
    )
    assert result.manifest["ground_truth"]["tie_score_policy"] == "exact"


def test_filtered_faiss_uses_complete_k_boundary_logic() -> None:
    np = pytest.importorskip("numpy")
    fake_faiss = types.SimpleNamespace(IndexFlatL2=fake_l2_index(np))
    query = VectorItem(id="query", vector=[0.0, 0.0], metadata={}, norm=0.0)
    bucket = FilteredFaissBucket(
        filter_value="value",
        record_ids=[
            "query",
            "strict",
            "boundary-a",
            "boundary-b",
            "boundary-c",
            "outside",
        ],
        vectors=np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    rows: list[str | None] = [None]

    search_filtered_faiss_bucket(
        faiss=fake_faiss,
        np=np,
        bucket=bucket,
        queries=[FilteredFaissQuery(ordinal=0, filter_value="value", query=query)],
        result_rows=rows,
        top_k=2,
        metric="euclidean",
        normalize=False,
        batch_size=1,
        filter_spec=FilterSpec(
            name="filter",
            field="bucket",
            operator="eq",
            value_source="eligible-record-buckets",
            seed=0,
            min_candidates=2,
        ),
    )

    assert rows[0] is not None
    truth = json.loads(rows[0])
    assert truth["expected_count"] == 2
    assert truth["recall_groups"]["strict_ids"] == ["strict"]
    assert truth["recall_groups"]["boundary_ids"] == [
        "boundary-a",
        "boundary-b",
        "boundary-c",
    ]


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_k_boundary_groups_support_higher_is_better_metrics(metric: str) -> None:
    result = _top_k_result(
        [
            (3.0, "strict"),
            (2.0, "boundary-a"),
            (2.0, "boundary-b"),
            (1.0, "outside"),
        ],
        top_k=2,
        metric=metric,
    )

    assert [match["id"] for match in result.matches] == ["strict", "boundary-a"]
    assert result.recall_groups == {
        "k": 2,
        "strict_ids": ["strict"],
        "boundary_ids": ["boundary-a", "boundary-b"],
    }


def test_prepare_ground_truth_writes_filtered_faiss_matches(
    tmp_path,
    monkeypatch,
) -> None:
    np = pytest.importorskip("numpy")
    fake_faiss = types.ModuleType("faiss")

    class FakeIndexFlatIP:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions
            self.vectors = None

        def add(self, vectors) -> None:
            assert vectors.shape[1] == self.dimensions
            self.vectors = vectors.copy()

        def search(self, queries, top_k: int):
            scores = queries @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :top_k]
            sorted_scores = np.take_along_axis(scores, order, axis=1)
            return sorted_scores, order

    def normalize_l2(vectors) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

    fake_faiss.IndexFlatIP = FakeIndexFlatIP
    fake_faiss.normalize_L2 = normalize_l2
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    dataset = prepare_fixture_dataset(tmp_path)
    stripped = []
    for line in dataset.records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["metadata"].pop("filter_bucket_20")
        stripped.append(json.dumps(record, sort_keys=True))
    dataset.records_path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    result = prepare_ground_truth(
        dataset_dir=tmp_path,
        top_k=1,
        backend="faiss",
        batch_size=2,
        filter_name="synthetic_bucket_5pct",
        filter_field="filter_bucket_20",
        filter_value_source="eligible-record-buckets",
    )

    lines = result.ground_truth_path.read_text(encoding="utf-8").splitlines()
    truth = json.loads(lines[0])
    assert result.manifest["status"] == "prepared"
    assert result.manifest["ground_truth"]["backend"] == "faiss"
    assert result.manifest["ground_truth"]["filtered_index_values"] >= 1
    assert result.manifest["ground_truth"]["candidate_count"]["eligible_values"] >= 1
    assert truth["filter"]["field"] == "filter_bucket_20"
    assert truth["candidate_count"] >= 1
    assert [match["id"] for match in truth["matches"]]


def test_prepare_ground_truth_rejects_invalid_top_k(tmp_path) -> None:
    with pytest.raises(ConfigError, match="top_k"):
        prepare_ground_truth(dataset_dir=tmp_path, top_k=0)


def test_prepare_ground_truth_rejects_invalid_batch_size(tmp_path) -> None:
    with pytest.raises(ConfigError, match="batch size"):
        prepare_ground_truth(dataset_dir=tmp_path, top_k=1, batch_size=0)


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
