"""Command line interface for lambdadb-bench."""

from __future__ import annotations

import argparse
import json

from ldbbench.__about__ import __version__
from ldbbench.adapters import get_adapter
from ldbbench.config import ConfigError, load_scenario, load_target
from ldbbench.datasets import (
    default_dataset_output_dir,
    prepare_dataset,
    prepare_ground_truth,
)
from ldbbench.manifest import initialize_run_artifacts
from ldbbench.runner import build_run_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ldbbench",
        description=(
            "Reproducible benchmark harness for LambdaDB and comparable managed "
            "vector databases."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subcommands = parser.add_subparsers(dest="command")

    doctor = subcommands.add_parser(
        "doctor",
        help="Check the local CLI installation.",
    )
    doctor.set_defaults(func=run_doctor)

    config = subcommands.add_parser(
        "config",
        help="Validate benchmark configuration files.",
    )
    config_subcommands = config.add_subparsers(dest="config_command")
    validate = config_subcommands.add_parser(
        "validate",
        help="Validate a scenario and target config.",
    )
    validate.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    validate.add_argument("--target", required=True, help="Path to target YAML.")
    validate.set_defaults(func=run_config_validate)

    manifest = subcommands.add_parser(
        "manifest",
        help="Create run manifest artifacts.",
    )
    manifest_subcommands = manifest.add_subparsers(dest="manifest_command")
    init = manifest_subcommands.add_parser(
        "init",
        help="Initialize a result directory with reproducibility artifacts.",
    )
    init.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    init.add_argument("--target", required=True, help="Path to target YAML.")
    init.add_argument("--out", required=True, help="Output result directory.")
    init.set_defaults(func=run_manifest_init)

    dataset = subcommands.add_parser(
        "dataset",
        help="Prepare benchmark datasets.",
    )
    dataset_subcommands = dataset.add_subparsers(dest="dataset_command")
    prepare = dataset_subcommands.add_parser(
        "prepare",
        help="Prepare local dataset cache artifacts.",
    )
    prepare.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    prepare.add_argument(
        "--out",
        help="Output dataset cache directory. Defaults to data/datasets/<scenario>.",
    )
    prepare.add_argument(
        "--limit",
        type=int,
        help="Limit rows to prepare. Useful for smoke tests.",
    )
    prepare.add_argument(
        "--query-count",
        type=int,
        help="Number of held-out query rows to write.",
    )
    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the dataset manifest without downloading rows.",
    )
    prepare.set_defaults(func=run_dataset_prepare)
    ground_truth = dataset_subcommands.add_parser(
        "ground-truth",
        help="Compute exact ground truth for prepared dataset artifacts.",
    )
    ground_truth.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared dataset cache directory.",
    )
    ground_truth.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to store per query.",
    )
    ground_truth.add_argument(
        "--metric",
        choices=["cosine", "dot"],
        help="Distance/similarity metric. Defaults to dataset manifest metric.",
    )
    ground_truth.add_argument(
        "--backend",
        default="exact",
        choices=["exact"],
        help="Ground truth backend.",
    )
    ground_truth.add_argument(
        "--limit-queries",
        type=int,
        help="Limit the number of query rows to process.",
    )
    ground_truth.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the ground truth manifest without computing neighbors.",
    )
    ground_truth.set_defaults(func=run_dataset_ground_truth)

    target = subcommands.add_parser(
        "target",
        help="Check target metadata and adapter capabilities.",
    )
    target_subcommands = target.add_subparsers(dest="target_command")
    check = target_subcommands.add_parser(
        "check",
        help="Check a target config without running a benchmark.",
    )
    check.add_argument("--target", required=True, help="Path to target YAML.")
    check.set_defaults(func=run_target_check)

    run = subcommands.add_parser(
        "run",
        help="Run or dry-run a benchmark scenario.",
    )
    run.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    run.add_argument("--target", required=True, help="Path to target YAML.")
    run.add_argument("--out", required=True, help="Output result directory.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the run plan and write artifacts without contacting a database.",
    )
    run.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Allow destructive preparation modes such as recreate.",
    )
    run.set_defaults(func=run_benchmark)

    return parser


def run_doctor(_args: argparse.Namespace) -> int:
    print(f"ldbbench {__version__}")
    print("status: ok")
    return 0


def run_config_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    adapter = get_adapter(target.vendor)
    plan = build_run_plan(
        scenario=scenario,
        target=target,
        capabilities=adapter.capabilities,
    )
    print(f"scenario: {scenario.name}")
    print(f"target: {target.name} ({target.vendor})")
    print(f"plan: {plan.status}")
    for item in plan.unsupported:
        print(f"unsupported: {item}")
    for item in plan.not_applicable:
        print(f"n/a: {item}")
    for item in plan.warnings:
        print(f"warning: {item}")
    print("status: ok")
    return 0


def run_manifest_init(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    adapter = get_adapter(target.vendor)
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=args.scenario,
        target_path=args.target,
        output_dir=args.out,
        adapter_capabilities=adapter.capabilities.as_dict(),
    )
    print(f"wrote {paths.run_manifest}")
    print(f"wrote {paths.scenario_resolved}")
    print(f"wrote {paths.target_redacted}")
    return 0


def run_dataset_prepare(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    output_dir = args.out or default_dataset_output_dir(scenario)
    result = prepare_dataset(
        scenario=scenario,
        output_dir=output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
        query_count=args.query_count,
    )
    print(f"dataset: {scenario.name}")
    print(f"status: {result.manifest['status']}")
    print(
        "requested_source_rows: "
        f"{result.manifest['dataset']['requested_source_rows']}"
    )
    print(f"requested_rows: {result.manifest['dataset']['requested_rows']}")
    print(f"requested_query_rows: {result.manifest['dataset']['requested_query_rows']}")
    print(f"written_rows: {result.manifest['dataset']['written_rows']}")
    print(f"written_query_rows: {result.manifest['dataset']['written_query_rows']}")
    print(f"wrote {result.manifest_path}")
    if not args.dry_run:
        print(f"wrote {result.raw_records_path}")
    return 0


def run_dataset_ground_truth(args: argparse.Namespace) -> int:
    result = prepare_ground_truth(
        dataset_dir=args.dataset_dir,
        top_k=args.top_k,
        metric=args.metric,
        backend=args.backend,
        limit_queries=args.limit_queries,
        dry_run=args.dry_run,
    )
    print(f"status: {result.manifest['status']}")
    print(f"backend: {result.manifest['ground_truth']['backend']}")
    print(f"metric: {result.manifest['ground_truth']['metric']}")
    print(f"top_k: {result.manifest['ground_truth']['top_k']}")
    print(f"records: {result.manifest['dataset']['records']}")
    print(f"queries: {result.manifest['dataset']['queries']}")
    print(f"wrote {result.manifest_path}")
    if not args.dry_run:
        print(f"wrote {result.ground_truth_path}")
    return 0


def run_target_check(args: argparse.Namespace) -> int:
    target = load_target(args.target)
    adapter = get_adapter(target.vendor)
    result = adapter.check(target)
    print(f"target: {target.name} ({target.vendor})")
    print(f"status: {'ok' if result.ok else 'error'}")
    print(f"message: {result.message}")
    print("capabilities:")
    print(json.dumps(adapter.capabilities.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def run_benchmark(args: argparse.Namespace) -> int:
    if not args.dry_run:
        raise ConfigError(
            "only --dry-run is supported until real adapters are implemented"
        )

    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    adapter = get_adapter(target.vendor)
    plan = build_run_plan(
        scenario=scenario,
        target=target,
        capabilities=adapter.capabilities,
        allow_destructive=args.allow_destructive,
    )
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=args.scenario,
        target_path=args.target,
        output_dir=args.out,
        adapter_capabilities=adapter.capabilities.as_dict(),
        dry_run_plan=plan.as_dict(),
    )

    print(f"dry_run: {plan.status}")
    for item in plan.unsupported:
        print(f"unsupported: {item}")
    for item in plan.not_applicable:
        print(f"n/a: {item}")
    for item in plan.warnings:
        print(f"warning: {item}")
    print(f"wrote {paths.run_manifest}")
    return 0 if plan.can_run else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except ConfigError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
