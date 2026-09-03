"""Command-line entry point for frozen-z and clinical eGFR evaluation."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import as_file, files
from pathlib import Path

from .data.defaults import project_data_layout
from .data.downstream import (
    CLINICAL_COLUMNS,
    aligned_ikem_matrix,
    extract_checkpoint_embedding,
    finalize_nested_selection,
    load_egfr_wide,
    prepare_nested_mixed_model_design,
    run_lme4_benchmark,
    save_embeddings,
    save_molecular_selection,
    scan_validation_checkpoints,
    score_molecular_selection_records,
    select_molecular_group_winners,
    summarize_mixed_model_results,
)

LINE = "=" * 88


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install archcon[training].") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select one checkpoint per preprocessing×architecture group using pooled molecular "
            "validation+test MSE, then select and evaluate the encoder with nested donor-grouped "
            "eGFR CV."
        )
    )
    parser.add_argument("--sweep-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-incomplete-sweep",
        action="store_true",
        help="Compatibility flag; incomplete-sweep evaluation is now the default.",
    )
    parser.add_argument(
        "--require-complete-sweep",
        action="store_true",
        help="Refuse evaluation unless every generated sweep run has completed.",
    )
    parser.add_argument(
        "--include-running-checkpoints",
        action="store_true",
        help=(
            "Include readable best.pt checkpoints from jobs without run_summary.json. "
            "These are best-so-far snapshots and may change while evaluation runs."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help="Donor-grouped inner folds used to select among molecular group winners.",
    )
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument(
        "--rscript",
        default="Rscript",
        help="Rscript executable name or absolute path.",
    )
    parser.add_argument(
        "--stratify-column",
        default="KDRI_8",
        help="Column used to stratify donor-grouped folds; use 'none' to disable.",
    )
    parser.add_argument(
        "--preprocessing-winners",
        action="store_true",
        help=(
            "Compatibility flag. The final workflow always selects one winner from every "
            "preprocessing×architecture group."
        ),
    )
    parser.add_argument(
        "--no-non-stadniuk",
        action="store_true",
        help="Do not add the best validation-selected non-Stadniuk encoder.",
    )
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument(
        "--no-clinical",
        action="store_true",
        help=(
            "Disable clinical baselines and clinical-augmented molecular models. "
            "By default KDRI, donor age, and cold-ischemia hours are included."
        ),
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Extract and save z without invoking R/lme4.",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root.expanduser().resolve()
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else sweep_root / "results"
    )
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else sweep_root.parent.parent / "data"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else sweep_root / "downstream" / "molecular_egfr"
    )
    prepared_root = sweep_root / "prepared"
    if not prepared_root.is_dir():
        raise SystemExit(f"Missing prepared directory: {prepared_root}")
    if not results_root.is_dir():
        raise SystemExit(f"Missing results directory: {results_root}")

    readable_records, warnings = scan_validation_checkpoints(results_root)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not readable_records:
        raise SystemExit("No readable run_*/best.pt checkpoints were found.")
    expected_runs = len(list((sweep_root / "configs").glob("run_*.json")))
    completed_names = {
        path.parent.name for path in results_root.glob("run_*/run_summary.json")
    }
    completed_records = [
        record for record in readable_records if record.run in completed_names
    ]
    records = readable_records if args.include_running_checkpoints else completed_records
    if not records:
        raise SystemExit(
            "No eligible readable best.pt checkpoint was found. Use "
            "--include-running-checkpoints to include best-so-far checkpoints."
        )
    complete = (
        expected_runs > 0
        and len(completed_records) == expected_runs
        and len(completed_names) == expected_runs
    )
    if args.require_complete_sweep and not args.allow_incomplete_sweep and not complete:
        raise SystemExit(
            "Sweep is incomplete: "
            f"{len(completed_records)}/{expected_runs} completed readable checkpoints and "
            f"{len(completed_names)}/{expected_runs} completed run summaries."
        )
    groups = {(record.method, record.architecture) for record in records}
    if complete and len(groups) != 6:
        readable = ", ".join(f"{method} × {architecture}" for method, architecture in sorted(groups))
        raise SystemExit(
            f"Expected six preprocessing×architecture groups, found {len(groups)}: {readable}."
        )
    print(LINE)
    print("MOLECULAR CONFIGURATION SELECTION" if complete else "PRELIMINARY MOLECULAR SELECTION")
    print(LINE)
    if not complete:
        running_count = len(records) - len(completed_records)
        print(
            f"PRELIMINARY: using {len(records)}/{expected_runs or '?'} readable checkpoints "
            f"({len(completed_records)} completed + {running_count} in-progress) from "
            f"{len(groups)}/6 currently represented preprocessing×architecture groups."
        )
        if args.include_running_checkpoints:
            print(
                "In-progress best.pt files are best-so-far snapshots and can be replaced by "
                "their training jobs during this evaluation."
            )
        print("Results will change as additional sweep runs complete.\n")
    print(
        f"Scoring {len(records)} validation-selected best.pt checkpoints on the pooled "
        "molecular validation+test rows."
    )
    print(
        "Scores are compared only within each preprocessing × architecture group; "
        "raw MSE is not compared across preprocessing arms."
    )

    layout = project_data_layout(data_dir)
    device = choose_device(args.device)

    def report_progress(index, total, record):
        if index == 1 or index == total or index % 25 == 0:
            print(f"  [{index:>4}/{total}] {record.run} · {record.method} · {record.architecture}")

    scored = score_molecular_selection_records(
        records,
        layout,
        prepared_root,
        device=device,
        batch_size=args.batch_size,
        progress=report_progress,
    )
    selected = select_molecular_group_winners(
        scored, expected_groups=6 if complete else None
    )
    score_path, group_path = save_molecular_selection(
        scored, selected, output_root / "molecular_selection"
    )

    print("\n" + LINE)
    print(
        "SIX MOLECULAR GROUP WINNERS"
        if complete
        else f"CURRENT GROUP WINNERS ({len(selected)}/6 GROUPS)"
    )
    print(LINE)
    for model_id, label, record in selected:
        print(
            f"{record.run:<10} pooled MSE={record.molecular_selection_mse:.10g} | "
            f"VAL={record.molecular_validation_mse:.10g} | TEST={record.molecular_test_mse:.10g} | "
            f"{record.method} | {record.architecture} | z={record.latent_dim}"
        )
    print(f"\nAll molecular scores: {score_path}")
    print(f"Group winners: {group_path}")
    print(
        "The former molecular test partition is now part of model selection and is not "
        "reported as an untouched test set."
    )
    print(
        "NOTE: Per-dataset RMA is study-isolated. Global RMA remains transductive because "
        "all GEO arrays contributed to its shared normalization."
    )

    embeddings = []
    for model_id, label, record in selected:
        print(f"Encoding all supervised samples with {model_id} on {device}...")
        embeddings.append(
            extract_checkpoint_embedding(
                record,
                model_id,
                label,
                layout,
                prepared_root,
                device=device,
                batch_size=args.batch_size,
            )
        )

    embedding_root, save_warnings = save_embeddings(embeddings, output_root / "embeddings")
    for warning in save_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Saved embeddings: {embedding_root}")
    if args.embeddings_only:
        return

    winner_input_dim = int(embeddings[0].record.checkpoint["input_dim"])
    expression, expression_samples = aligned_ikem_matrix(
        layout, prepared_root, winner_input_dim
    )
    egfr = load_egfr_wide(layout, expression_samples["sample_id"])
    stratify_column = None if args.stratify_column.lower() == "none" else args.stratify_column
    benchmark_root = output_root / "mixed_models"
    design_path, specs_path, specs = prepare_nested_mixed_model_design(
        egfr,
        embeddings,
        expression,
        expression_samples,
        benchmark_root,
        n_splits=args.folds,
        n_repeats=args.repeats,
        inner_splits=args.inner_folds,
        seed=args.cv_seed,
        stratify_column=stratify_column,
        include_pca=not args.no_pca,
        include_clinical=not args.no_clinical,
    )
    print(
        f"Prepared {len(specs)} nested model fits from {egfr['patient'].nunique()} patients "
        f"and {egfr['donor'].nunique()} donor groups. Inner folds select the encoder; "
        "outer folds evaluate that selection."
    )
    r_resource = files("archcon.assets").joinpath("molecular_mixed_models.R")
    with as_file(r_resource) as r_script:
        metrics_path, predictions_path = run_lme4_benchmark(
            design_path,
            specs_path,
            r_script,
            benchmark_root,
            rscript=args.rscript,
            output_prefix="nested_all_",
        )
    metrics_path, predictions_path, _selections, candidate_ranking = finalize_nested_selection(
        metrics_path, predictions_path, embeddings, benchmark_root
    )
    summary, pairwise, clinical_incremental = summarize_mixed_model_results(
        metrics_path, predictions_path, benchmark_root
    )

    print("\n" + LINE)
    print("eGFR ENCODER SELECTION")
    print(LINE)
    ranking_columns = [
        "model_label",
        "mean_cv_rmse",
        "nested_selection_count",
        "nested_selection_fraction",
        "run",
    ]
    print(
        candidate_ranking[ranking_columns].to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    deployment = candidate_ranking.iloc[0]
    print(
        "\nFull-data CV deployment winner: "
        f"{deployment['run']} · {deployment['preprocessing']} · "
        f"{deployment['architecture']} · z={int(deployment['latent_dim'])}"
    )
    print(
        "Its ordinary CV score is used to lock a future deployment model; the unbiased "
        "performance estimate below comes from nested outer folds."
    )

    print("\n" + LINE)
    print("NESTED-CV eGFR MIXED-MODEL SUMMARY")
    print(LINE)
    if not args.no_clinical:
        print("Clinical variables: " + ", ".join(CLINICAL_COLUMNS))
        print("Clinical imputation and scaling are fitted within each training fold.\n")
    columns = [
        "model_label",
        "mean_rmse",
        "pooled_rmse",
        "mean_delta_vs_time",
        "lcb_delta_vs_time",
        "positive_folds_vs_time",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(
        "\nNested-selected z against baselines "
        "(positive gain means selected z has lower RMSE):"
    )
    pair_columns = [
        "model_label",
        "mean_winner_gain",
        "lcb_winner_gain",
        "winner_better_fraction",
    ]
    print(pairwise[pair_columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if not clinical_incremental.empty:
        print(
            "\nIncremental value beyond clinical baselines "
            "(positive gain means the augmented model has lower RMSE):"
        )
        clinical_columns = [
            "comparison_label",
            "mean_gain",
            "lcb_gain",
            "augmented_better_fraction",
        ]
        print(
            clinical_incremental[clinical_columns].to_string(
                index=False, float_format=lambda value: f"{value:.6g}"
            )
        )
    print(f"\nComplete results: {benchmark_root}")
