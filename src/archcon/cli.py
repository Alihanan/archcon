"""Command-line entry point for ArchCon."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archcon",
        description="Start the ArchCon browser interface or execute one exported training config.",
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
    parser.add_argument(
        "--run-config",
        metavar="JSON",
        help="Run one exported ArchCon training JSON headlessly instead of starting Gradio.",
    )
    parser.add_argument(
        "--data-dir",
        help="Override the ArchCon data directory for a headless --run-config job.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory for headless model/result folders (default: ./models).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_config:
        # Lazy import keeps Gradio/FastAPI out of normal PBS batch jobs.
        from .batch import run_training_request

        run_training_request(
            args.run_config,
            data_dir=args.data_dir,
            output_root=args.output_root,
        )
        return

    if args.data_dir or args.output_root:
        raise SystemExit("--data-dir/--output-root are only valid together with --run-config.")

    # Gradio is imported only for the interactive browser path.
    from .app import start

    start(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
