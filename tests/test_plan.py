from __future__ import annotations

from ldbbench.adapters.dryrun import LAMBDADB_DRYRUN, QDRANT_DRYRUN
from ldbbench.config import ScenarioConfig, TargetConfig
from ldbbench.runner import build_run_plan


def make_scenario(
    *,
    write_mode: str = "upsert",
    consistency: str = "eventual",
    partition_filter: bool = False,
    workload: str = "standard",
) -> ScenarioConfig:
    query = {"consistency": consistency}
    if partition_filter:
        query["partition_filter"] = {
            "field": "url",
            "metadata_field": "url",
        }
    mapping = {
        "name": "smoke",
        "workload": workload,
        "dataset": {"rows": 1, "dimensions": 1024},
        "load": {"write_mode": write_mode},
        "query": query,
    }
    if workload == "search_under_ingest":
        mapping["search_under_ingest"] = {
            "document_group_field": "url",
            "consistency": consistency,
        }
    return ScenarioConfig.from_mapping(mapping)


def make_target(
    *,
    vendor: str = "qdrant",
    prepare_mode: str = "existing",
) -> TargetConfig:
    return TargetConfig.from_mapping(
        {
            "vendor": vendor,
            "name": f"{vendor}-target",
            "endpoint": "https://example.test",
            "prepare": {"mode": prepare_mode},
        }
    )


def test_qdrant_strong_consistency_is_partial_na() -> None:
    plan = build_run_plan(
        scenario=make_scenario(consistency="strong"),
        target=make_target(vendor="qdrant"),
        capabilities=QDRANT_DRYRUN.capabilities,
    )

    assert plan.status == "partial"
    assert plan.can_run
    assert plan.not_applicable


def test_lambdadb_strong_consistency_is_supported() -> None:
    plan = build_run_plan(
        scenario=make_scenario(consistency="strong"),
        target=make_target(vendor="lambdadb"),
        capabilities=LAMBDADB_DRYRUN.capabilities,
    )

    assert plan.status == "supported"
    assert not plan.not_applicable


def test_search_under_ingest_strong_consistency_uses_workload_config() -> None:
    plan = build_run_plan(
        scenario=make_scenario(consistency="strong", workload="search_under_ingest"),
        target=make_target(vendor="qdrant"),
        capabilities=QDRANT_DRYRUN.capabilities,
    )

    assert plan.status == "partial"
    assert plan.query_consistency == "strong"
    assert "strong" in plan.not_applicable[0]


def test_qdrant_partition_filter_is_partial_na() -> None:
    plan = build_run_plan(
        scenario=make_scenario(partition_filter=True),
        target=make_target(vendor="qdrant"),
        capabilities=QDRANT_DRYRUN.capabilities,
    )

    assert plan.status == "partial"
    assert plan.can_run
    assert "partition_filter" in plan.not_applicable[0]


def test_lambdadb_partition_filter_is_supported() -> None:
    plan = build_run_plan(
        scenario=make_scenario(partition_filter=True),
        target=make_target(vendor="lambdadb"),
        capabilities=LAMBDADB_DRYRUN.capabilities,
    )

    assert plan.status == "supported"
    assert not plan.not_applicable


def test_recreate_requires_destructive_flag() -> None:
    plan = build_run_plan(
        scenario=make_scenario(),
        target=make_target(prepare_mode="recreate"),
        capabilities=QDRANT_DRYRUN.capabilities,
    )

    assert plan.status == "unsupported"
    assert "requires --allow-destructive" in plan.unsupported[0]


def test_recreate_allowed_with_destructive_flag() -> None:
    plan = build_run_plan(
        scenario=make_scenario(),
        target=make_target(prepare_mode="recreate"),
        capabilities=QDRANT_DRYRUN.capabilities,
        allow_destructive=True,
    )

    assert plan.status == "supported"


def test_unsupported_write_mode_blocks_run() -> None:
    plan = build_run_plan(
        scenario=make_scenario(write_mode="bulk_upsert"),
        target=make_target(vendor="qdrant"),
        capabilities=QDRANT_DRYRUN.capabilities,
    )

    assert plan.status == "unsupported"
    assert "bulk_upsert" in plan.unsupported[0]
