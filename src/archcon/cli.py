"""Command-line entry point for archcon."""

from __future__ import annotations

import argparse

from .app import start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archcon",
        description="Start the local ArchCon browser interface.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind to. Keep 127.0.0.1 for local-only access.",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        help="TCP port for the local server (default: 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
