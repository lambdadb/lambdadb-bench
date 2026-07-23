from __future__ import annotations

import json

import pytest

from ldbbench.config import ConfigError, ScenarioConfig, TargetConfig
from ldbbench.datasets import deletion as deletion_module
from ldbbench.datasets.deletion import (
    dataset_records_sha256,
    deletion_checkpoint_count,
    load_deletion_plan,
    prepare_deletion_plan,
)
from ldbbench.datasets.ground_truth import prepare_ground_truth, read_faiss_records
from ldbbench.datasets.prepare import prepare_dataset
from ldbbench.runner.deletion import (
    load_deletion_state,
    validate_deletion_state_scenario,
    validate_ground_truth_deletion_state,
    write_deletion_state,
)


def _scenario() -> ScenarioConfig:
    return ScenarioConfig.from_mapping(
        {
            "name": "delete-smoke",
            "dataset": {
                "provider": "huggingface",
                "source": "demo/source",
                "rows": 3,
                "dimensions": 2,
                "id_field": "_id",
                "vector_field": "emb",
                "metric": "cosine",
            },
            "load": {"write_mode": "upsert"},
            "query": {"top_k": 2, "consistency": "eventual"},
            "delete": {
                "order": "sequential",
                "seed": 0,
                "checkpoints_pct": [50, 75],
                "batch_size": 1,
            },
        }
    )


def _dataset(tmp_path):
    return prepare_dataset(
        scenario=_scenario(),
        output_dir=tmp_path / "dataset",
        limit=3,
        query_count=1,
        source_rows=[
            {"_id": "query", "emb": [1.0, 0.0]},
            {"_id": "a", "emb": [1.0, 0.0]},
            {"_id": "b", "emb": [0.0, 1.0]},
            {"_id": "c", "emb": [0.8, 0.2]},
        ],
    )


def test_prepare_sequential_deletion_plan_uses_record_order(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = _dataset(tmp_path)
    original_sha256_file = deletion_module.sha256_file
    hashed_paths = []

    def tracked_sha256_file(path):
        hashed_paths.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(deletion_module, "sha256_file", tracked_sha256_file)

    result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    loaded = load_deletion_plan(
        result.plan_path,
        records_path=dataset.records_path,
        expected_records_sha256=dataset_records_sha256(dataset.manifest),
    )

    assert loaded.ids == ["a", "b", "c"]
    assert loaded.order == "sequential"
    assert loaded.total_records == 3
    assert result.manifest["artifacts"]["records_sha256"]
    assert result.manifest["artifacts"]["deletion_plan_sha256"]
    assert dataset.records_path not in hashed_paths


def test_load_deletion_plan_rejects_dataset_manifest_checksum_mismatch(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path)
    result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )

    with pytest.raises(ConfigError, match="does not match dataset manifest"):
        load_deletion_plan(
            result.plan_path,
            records_path=dataset.records_path,
            expected_records_sha256="stale-records-checksum",
        )


def test_random_deletion_plan_is_seeded_and_reproducible(tmp_path) -> None:
    dataset = _dataset(tmp_path)

    first = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="random",
        seed=7,
    )
    first_content = first.plan_path.read_text(encoding="utf-8")
    second = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="random",
        seed=7,
    )

    assert second.plan_path.read_text(encoding="utf-8") == first_content
    assert second.plan_path.name == "deletion_plan.random.seed-7.jsonl"


def test_survivor_ground_truth_replaces_deleted_neighbors(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    plan = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )

    result = prepare_ground_truth(
        dataset_dir=dataset.output_dir,
        top_k=2,
        deletion_plan_path=plan.plan_path,
        delete_checkpoint_pct=50,
    )

    row = json.loads(result.ground_truth_path.read_text(encoding="utf-8"))
    assert [match["id"] for match in row["matches"]] == ["c", "b"]
    assert result.ground_truth_path.name == (
        "ground_truth.cosine.deleted.sequential.pct-050.jsonl"
    )
    assert result.manifest["dataset"]["records"] == 2
    assert result.manifest["ground_truth"]["deletion"] == {
        "order": "sequential",
        "seed": 0,
        "checkpoint_pct": 50,
        "total_records": 3,
        "deleted_count": 1,
        "remaining_count": 2,
    }


def test_deletion_checkpoint_rejects_full_deletion() -> None:
    with pytest.raises(ConfigError, match="between 0 and 99"):
        deletion_checkpoint_count(10, 100)


def test_faiss_record_reader_excludes_deleted_ids(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    dataset = _dataset(tmp_path)

    ids, vectors = read_faiss_records(
        records_path=dataset.records_path,
        dimensions=2,
        expected_records=3,
        excluded_ids={"a"},
        np=np,
    )

    assert ids == ["b", "c"]
    np.testing.assert_allclose(vectors, [[0.0, 1.0], [0.8, 0.2]])


def test_deletion_state_rejects_count_inconsistent_with_checkpoint(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    plan_result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    plan = load_deletion_plan(
        plan_result.plan_path,
        records_path=dataset.records_path,
    )
    target = TargetConfig.from_mapping(
        {
            "vendor": "dryrun",
            "name": "delete-target",
            "collection_name": "documents",
            "prepare": {"mode": "existing"},
        }
    )
    state_path = tmp_path / "deletion_state.json"
    state = write_deletion_state(
        output_path=state_path,
        target=target,
        plan=plan,
        checkpoint_pct=50,
        previous_deleted_count=0,
        delete_summary={
            "status": "completed",
            "documents": 1,
            "requested_documents": 1,
            "visibility": {"status": "not_visible"},
        },
    )
    state["deletion"]["deleted_count"] = 2
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ConfigError, match="deleted_count does not match"):
        load_deletion_state(
            state_path,
            target=target,
            records_path=dataset.records_path,
            plan=plan,
        )


def test_deletion_state_rejects_different_target_endpoint(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    plan_result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    plan = load_deletion_plan(
        plan_result.plan_path,
        records_path=dataset.records_path,
    )
    original_target = TargetConfig.from_mapping(
        {
            "vendor": "qdrant",
            "name": "same-label",
            "endpoint": "https://a.example",
            "collection_name": "documents",
            "prepare": {"mode": "existing"},
        }
    )
    other_target = TargetConfig.from_mapping(
        {
            "vendor": "qdrant",
            "name": "same-label",
            "endpoint": "https://b.example",
            "collection_name": "documents",
            "prepare": {"mode": "existing"},
        }
    )
    state_path = tmp_path / "deletion_state.json"
    write_deletion_state(
        output_path=state_path,
        target=original_target,
        plan=plan,
        checkpoint_pct=50,
        previous_deleted_count=0,
        delete_summary={
            "status": "completed",
            "documents": 1,
            "requested_documents": 1,
            "visibility": {"status": "not_visible"},
        },
    )

    with pytest.raises(ConfigError, match="does not match selected target"):
        load_deletion_state(
            state_path,
            target=other_target,
            records_path=dataset.records_path,
            plan=plan,
        )


def test_deletion_state_rejects_scenario_order_mismatch(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    plan_result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    plan = load_deletion_plan(
        plan_result.plan_path,
        records_path=dataset.records_path,
    )
    target = TargetConfig.from_mapping(
        {
            "vendor": "dryrun",
            "name": "delete-target",
            "collection_name": "documents",
            "prepare": {"mode": "existing"},
        }
    )
    state = write_deletion_state(
        output_path=tmp_path / "deletion_state.json",
        target=target,
        plan=plan,
        checkpoint_pct=50,
        previous_deleted_count=0,
        delete_summary={
            "status": "completed",
            "documents": 1,
            "requested_documents": 1,
            "visibility": {"status": "not_visible"},
        },
    )
    mapping = dict(_scenario().raw)
    mapping["delete"] = {
        "order": "random",
        "seed": 7,
        "checkpoints_pct": [50],
    }

    with pytest.raises(ConfigError, match="order/seed"):
        validate_deletion_state_scenario(
            ScenarioConfig.from_mapping(mapping),
            state,
        )


def test_survivor_ground_truth_rejects_records_checksum_mismatch(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    plan_result = prepare_deletion_plan(
        dataset_dir=dataset.output_dir,
        order="sequential",
    )
    plan = load_deletion_plan(
        plan_result.plan_path,
        records_path=dataset.records_path,
    )
    ground_truth = prepare_ground_truth(
        dataset_dir=dataset.output_dir,
        top_k=1,
        deletion_plan_path=plan.path,
        delete_checkpoint_pct=50,
    )
    target = TargetConfig.from_mapping(
        {
            "vendor": "dryrun",
            "name": "delete-target",
            "collection_name": "documents",
            "prepare": {"mode": "existing"},
        }
    )
    state = write_deletion_state(
        output_path=tmp_path / "deletion_state.json",
        target=target,
        plan=plan,
        checkpoint_pct=50,
        previous_deleted_count=0,
        delete_summary={
            "status": "completed",
            "documents": 1,
            "requested_documents": 1,
            "visibility": {"status": "not_visible"},
        },
    )
    manifest = json.loads(ground_truth.manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["records_sha256"] = "stale-records-checksum"
    ground_truth.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConfigError, match="records checksum"):
        validate_ground_truth_deletion_state(
            ground_truth.ground_truth_path,
            state,
            queries_path=dataset.queries_path,
        )

    manifest["artifacts"]["records_sha256"] = state["dataset"]["records_sha256"]
    ground_truth.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with dataset.queries_path.open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ConfigError, match="queries checksum"):
        validate_ground_truth_deletion_state(
            ground_truth.ground_truth_path,
            state,
            queries_path=dataset.queries_path,
        )
