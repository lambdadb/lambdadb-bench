"""Command line interface for lambdadb-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys

from ldbbench.__about__ import __version__
from ldbbench.adapters import get_adapter
from ldbbench.config import ConfigError, load_scenario, load_target
from ldbbench.datasets import (
    default_dataset_output_dir,
    optimize_dataset,
    prepare_dataset,
    prepare_ground_truth,
)
from ldbbench.manifest import initialize_run_artifacts
from ldbbench.report import generate_report
from ldbbench.runner import build_run_plan, execute_benchmark

RESOURCE_TRACKER_WARNING_FILTER = (
    "ignore:resource_tracker:UserWarning:multiprocessing.resource_tracker"
)


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
    optimize = dataset_subcommands.add_parser(
        "optimize",
        help="Build fast binary caches for an existing prepared dataset.",
    )
    optimize.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared dataset cache directory.",
    )
    optimize.set_defaults(func=run_dataset_optimize)
    ground_truth = dataset_subcommands.add_parser(
        "ground-truth",
        help="Compute ground truth for prepared dataset artifacts.",
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
        choices=["exact", "faiss"],
        help="Ground truth backend.",
    )
    ground_truth.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Query batch size for FAISS ground truth search.",
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
        "--dataset-dir",
        help="Prepared dataset directory containing records.jsonl and queries.jsonl.",
    )
    run.add_argument(
        "--ground-truth",
        help=(
            "Optional ground_truth.jsonl path. Defaults to "
            "<dataset-dir>/ground_truth.jsonl."
        ),
    )
    run.add_argument(
        "--max-records",
        type=int,
        help="Limit records loaded in this run. Useful for smoke tests.",
    )
    run.add_argument(
        "--max-queries",
        type=int,
        help="Limit queries executed in this run. Useful for smoke tests.",
    )
    run.add_argument(
        "--load-only",
        action="store_true",
        help="Load records and skip query execution.",
    )
    run.add_argument(
        "--query-only",
        action="store_true",
        help="Skip loading and run queries against an existing prepared target.",
    )
    run.add_argument(
        "--resume-load",
        action="store_true",
        help=(
            "Resume loading from an existing load_checkpoint.json in --out. "
            "Requires target prepare.mode: existing."
        ),
    )
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
    run.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Allow 1M+ real runs. Intended for explicit cost/time opt-in.",
    )
    run.set_defaults(func=run_benchmark)

    report = subcommands.add_parser(
        "report",
        help="Combine benchmark result directories into Markdown and CSV reports.",
    )
    report.add_argument(
        "result_dirs",
        nargs="+",
        help="Result directories containing summary.json and run_manifest.json.",
    )
    report.add_argument("--out", required=True, help="Output Markdown report path.")
    report.set_defaults(func=run_report)

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
    uses_huggingface = _uses_huggingface_provider(scenario.dataset)
    if not args.dry_run and uses_huggingface:
        _suppress_resource_tracker_warning()
    print(f"dataset: {scenario.name}", flush=True)
    if not args.dry_run:
        print("status: preparing", flush=True)
    result = prepare_dataset(
        scenario=scenario,
        output_dir=output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
        query_count=args.query_count,
        progress=print_progress,
    )
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
        print(f"wrote {result.records_path}")
        print(f"wrote {result.queries_path}")
        print(f"wrote {result.records_msgpack_path}")
        print(f"wrote {result.queries_msgpack_path}")
        args._force_exit_after_return = uses_huggingface
    return 0


def run_dataset_optimize(args: argparse.Namespace) -> int:
    print(f"dataset_dir: {args.dataset_dir}", flush=True)
    print("status: optimizing", flush=True)
    result = optimize_dataset(
        dataset_dir=args.dataset_dir,
        progress=print_progress,
    )
    print("status: optimized")
    print(f"wrote {result.records_msgpack_path}")
    print(f"wrote {result.queries_msgpack_path}")
    print(f"updated {result.manifest_path}")
    return 0


def run_dataset_ground_truth(args: argparse.Namespace) -> int:
    result = prepare_ground_truth(
        dataset_dir=args.dataset_dir,
        top_k=args.top_k,
        metric=args.metric,
        backend=args.backend,
        limit_queries=args.limit_queries,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        progress=print_progress,
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
    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    adapter = get_adapter(target.vendor, dry_run=args.dry_run)
    plan = build_run_plan(
        scenario=scenario,
        target=target,
        capabilities=adapter.capabilities,
        allow_destructive=args.allow_destructive,
    )
    if not args.dry_run:
        if not args.dataset_dir:
            raise ConfigError("run requires --dataset-dir unless --dry-run is set")
        result = execute_benchmark(
            scenario=scenario,
            target=target,
            adapter=adapter,
            scenario_path=args.scenario,
            target_path=args.target,
            output_dir=args.out,
            dataset_dir=args.dataset_dir,
            ground_truth_path=args.ground_truth,
            max_records=args.max_records,
            max_queries=args.max_queries,
            load_only=args.load_only,
            query_only=args.query_only,
            resume_load=args.resume_load,
            allow_destructive=args.allow_destructive,
            allow_large_run=args.allow_large_run,
            progress=print_progress,
        )
        print(f"run: {result.summary['status']}")
        print(f"records: {result.summary['load']['records']}")
        if result.summary["load"]["status"] == "skipped":
            print(f"load: skipped ({result.summary['load']['skip_reason']})")
        else:
            print(f"load_concurrency: {result.summary['load']['concurrency']}")
            print(f"load_processes: {result.summary['load']['processes']}")
            print(
                "load_records_per_second: "
                f"{result.summary['load']['records_per_second']}"
            )
            print(
                "load_batching_seconds: "
                f"{result.summary['load']['batching_duration_seconds']}"
            )
            print(
                "load_upsert_attempt_seconds: "
                f"{result.summary['load']['upsert_attempt_duration_seconds']}"
            )
        if result.summary["load"]["errors"]:
            print(f"load_errors: {result.summary['load']['errors']}")
            print(f"load_error_rate: {result.summary['load']['error_rate']}")
        if "visibility" in result.summary["load"]:
            print(f"visibility: {result.summary['load']['visibility']['status']}")
        print(f"queries: {result.summary['query']['queries']}")
        print(f"query_processes: {result.summary['query']['processes']}")
        if result.summary["query"]["errors"]:
            print(f"errors: {result.summary['query']['errors']}")
            print(f"error_rate: {result.summary['query']['error_rate']}")
        if result.summary["query"]["recall_at_k"] is not None:
            print(f"recall_at_k: {result.summary['query']['recall_at_k']}")
        print(f"wrote {result.ingest_events_path}")
        print(f"wrote {result.query_events_path}")
        if result.load_checkpoint_path.exists():
            print(f"wrote {result.load_checkpoint_path}")
        print(f"wrote {result.summary_path}")
        return 0

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


def run_report(args: argparse.Namespace) -> int:
    result = generate_report(args.result_dirs, output_path=args.out)
    print(f"runs: {result.run_count}")
    print(f"wrote {result.markdown_path}")
    print(f"wrote {result.load_csv_path}")
    print(f"wrote {result.query_csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    cli_invocation = argv is None
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        exit_code = args.func(args)
    except ConfigError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    if cli_invocation and getattr(args, "_force_exit_after_return", False):
        # Hugging Face streaming can leave PyArrow worker threads waiting during
        # interpreter shutdown on macOS after all dataset artifacts are written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


def _uses_huggingface_provider(dataset: dict[str, object]) -> bool:
    return str(dataset.get("provider", "huggingface")) == "huggingface"


def _suppress_resource_tracker_warning() -> None:
    filters = [
        item.strip()
        for item in os.environ.get("PYTHONWARNINGS", "").split(",")
        if item.strip()
    ]
    if RESOURCE_TRACKER_WARNING_FILTER not in filters:
        filters.append(RESOURCE_TRACKER_WARNING_FILTER)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def print_progress(message: str) -> None:
    print(f"progress: {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
