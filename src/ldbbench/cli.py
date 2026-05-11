"""Command line interface for lambdadb-bench."""

from __future__ import annotations

import argparse
import json

from ldbbench.__about__ import __version__
from ldbbench.adapters import get_adapter
from ldbbench.config import ConfigError, load_scenario, load_target
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
