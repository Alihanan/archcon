#!/usr/bin/env python3
"""Generate a donor-safe ArchCon paper sweep from installed data stores."""

from __future__ import annotations

import argparse
from pathlib import Path

from archcon.batch import (
    build_run_request,
    generate_sweep_bundle,
    recommended_comparison_grid_json,
)
from archcon.data.defaults import project_data_layout
from archcon.data.geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from archcon.data.training import TrainingConfig
from archcon.data.training_sources import (
    METHOD_PER_DATASET_STANDARDIZED,
    TRAINING_PREPROCESSING_OPTIONS,
    create_shared_preprocessing_split,
)

ALIASES = {
    "standardized": METHOD_PER_DATASET_STANDARDIZED,
    "per-gse": METHOD_PER_GSE_RMA,
    "per-study": METHOD_PER_GSE_RMA,
    "global": METHOD_GLOBAL_RMA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fresh donor-safe sweep. Use --methods per-gse for an "
            "early 480-run per-study-only sweep, or --methods all after the "
            "combined Global-RMA rebuild has completed."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", *ALIASES],
    )
    parser.add_argument("--ncpus", type=int, default=1)
    parser.add_argument("--memory", default="10gb")
    parser.add_argument("--scratch", default="4gb")
    parser.add_argument("--walltime", default="24:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    data = (args.data_dir or project / "data").expanduser().resolve()
    if "all" in args.methods:
        if len(args.methods) != 1:
            raise SystemExit("--methods all cannot be combined with another method.")
        methods = tuple(TRAINING_PREPROCESSING_OPTIONS)
    else:
        methods = tuple(ALIASES[value] for value in args.methods)

    layout = project_data_layout(data)
    split = create_shared_preprocessing_split(
        layout,
        seed=42,
        train_fraction=0.90,
        validation_fraction=0.05,
        methods=methods,
    )
    base = build_run_request(
        method=methods[0],
        split_seed=42,
        train_fraction=0.90,
        validation_fraction=0.05,
        training=TrainingConfig(),
    )
    result = generate_sweep_bundle(
        base_request=base,
        grid_text=recommended_comparison_grid_json(methods),
        destination_root=project / "sweeps",
        sweep_name=args.sweep_name,
        project_dir=str(project),
        data_dir=str(data),
        python_executable=str(project / ".venv/bin/python"),
        ncpus=args.ncpus,
        memory=args.memory,
        scratch=args.scratch,
        walltime=args.walltime,
        ngpus=0,
        split_frame=split,
        data_layout=layout,
    )
    print(result)


if __name__ == "__main__":
    main()
