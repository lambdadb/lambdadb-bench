"""Command line interface for lambdadb-bench."""

from __future__ import annotations

import argparse

from ldbbench.__about__ import __version__


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

    return parser


def run_doctor(_args: argparse.Namespace) -> int:
    print(f"ldbbench {__version__}")
    print("status: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

