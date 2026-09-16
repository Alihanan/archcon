"""Command-line entry point for frozen-z and clinical eGFR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd

from .data.defaults import project_data_layout
from .data.downstream import (
    CLINICAL_COLUMNS,
    aligned_ikem_matrix,
    evaluate_frozen_molecular_test_records,
    extract_checkpoint_embedding,
    load_egfr_wide,
    prepare_mixed_model_design,
    run_lme4_benchmark,
    save_embeddings,
    save_molecular_selection,
    scan_validation_checkpoints,
    score_molecular_selection_records,
    select_molecular_group_winners,
    summarize_fixed_encoder_results,
)
from .data.ikem_preprocessing import prepare_ikem_evaluation_sources
from .data.probe_lasso import (
    ProbeExpressionArm,
    ProbeLassoConfig,
    evaluate_probe_lasso_mixed_models,
    summarize_probe_lasso_against_time,
)

LINE = "=" * 88


def expected_sweep_groups(config_directory: Path) -> set[tuple[str, str]]:
    """Read the preprocessing×architecture groups actually generated in a sweep."""

    groups: set[tuple[str, str]] = set()
    for path in sorted(config_directory.glob("run_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        training = payload.get("training", {})
        method = str(payload.get("method", "")).strip()
        architecture = str(training.get("architecture_family", "")).strip()
        if not method or not architecture:
            raise ValueError(f"Generated run config lacks method/architecture: {path}")
        groups.add((method, architecture))
    return groups


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
            "Select one checkpoint per preprocessing×architecture group using the predeclared "
            "GEO/IKEM molecular-validation score, evaluate GEO test only after freezing those "
            "choices, and evaluate every frozen encoder with donor-grouped eGFR CV."
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
        help=(
            "Write validation-only diagnostics for the currently completed runs, then stop "
            "before GEO test and eGFR evaluation."
        ),
    )
    parser.add_argument(
        "--require-complete-sweep",
        action="store_true",
        help="Compatibility flag; complete-sweep evaluation is already the default.",
    )
    parser.add_argument(
        "--include-running-checkpoints",
        action="store_true",
        help=(
            "Include readable best.pt checkpoints from jobs without run_summary.json. "
            "These are best-so-far snapshots and may change while evaluation runs."
        ),
    )
    parser.add_argument(
        "--evaluate-all-readable-checkpoints",
        action="store_true",
        help=(
            "Deprecated exploratory option. The final-paper workflow refuses eGFR-based "
            "inspection of non-frozen hyperparameter candidates."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
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
        help=(
            "Deprecated compatibility flag; the paper workflow retains every "
            "preprocessing×architecture group winner declared by this sweep."
        ),
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
    parser.add_argument(
        "--rebuild-ikem-preprocessing",
        action="store_true",
        help=(
            "Rebuild cached method-matched IKEM matrices even when their recorded source "
            "and frozen-pretraining provenance still match."
        ),
    )
    parser.add_argument(
        "--reuse-baseline-root",
        type=Path,
        default=None,
        help=(
            "Completed output of archcon-evaluate-egfr-baselines. Matching time, "
            "clinical, PCA, and probe-LASSO folds are reused instead of recomputed."
        ),
    )
    parser.add_argument(
        "--no-probe-lasso",
        action="store_true",
        help="Skip the all-probe LASSO-AIC and LASSO-BIC mixed-model baselines.",
    )
    parser.add_argument(
        "--lasso-alpha-fractions",
        default="1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
        help="Descending fractions of each fold's lambda_max used for LASSO tuning.",
    )
    parser.add_argument("--lasso-max-iter", type=int, default=5_000)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.evaluate_all_readable_checkpoints:
        raise SystemExit(
            "--evaluate-all-readable-checkpoints is incompatible with the final-paper "
            "contract: eGFR outcomes must not inspect hyperparameter candidates."
        )

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

    print(
        "Scanning best.pt files in metadata-only mode; model and optimizer tensors are not "
        "retained in RAM...",
        flush=True,
    )

    def report_scan(index, total, readable):
        if index == 1 or index == total or index % 50 == 0:
            print(
                f"  scanned {index:>4}/{total} result folders; "
                f"{readable} readable checkpoints",
                flush=True,
            )

    readable_records, warnings = scan_validation_checkpoints(
        results_root, progress=report_scan
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not readable_records:
        raise SystemExit("No readable run_*/best.pt checkpoints were found.")
    config_directory = sweep_root / "configs"
    expected_names = {path.stem for path in config_directory.glob("run_*.json")}
    expected_runs = len(expected_names)
    if not expected_names:
        raise SystemExit(f"No generated run configs were found under {sweep_root / 'configs'}.")
    all_completed_names = {
        path.parent.name for path in results_root.glob("run_*/run_summary.json")
    }
    unexpected_names = sorted(all_completed_names.difference(expected_names))
    if unexpected_names:
        print(
            "WARNING: ignoring result folders with no matching generated config: "
            + ", ".join(unexpected_names[:10]),
            file=sys.stderr,
        )
    completed_names = all_completed_names.intersection(expected_names)
    readable_records = [record for record in readable_records if record.run in expected_names]
    completed_records = [
        record for record in readable_records if record.run in completed_names
    ]
    records = readable_records if args.include_running_checkpoints else completed_records
    if not records:
        raise SystemExit(
            "No eligible readable best.pt checkpoint was found. Use "
            "--include-running-checkpoints to include best-so-far checkpoints."
        )
    completed_readable_names = {record.run for record in completed_records}
    complete = completed_names == expected_names == completed_readable_names
    if args.require_complete_sweep and args.allow_incomplete_sweep:
        raise SystemExit(
            "Choose either --require-complete-sweep or --allow-incomplete-sweep, not both."
        )
    if not complete and not args.allow_incomplete_sweep:
        raise SystemExit(
            "Sweep is incomplete: "
            f"{len(completed_records)}/{expected_runs} completed readable checkpoints and "
            f"{len(completed_names)}/{expected_runs} completed run summaries. Use "
            "--allow-incomplete-sweep for validation diagnostics only; that mode never "
            "touches GEO test or eGFR outcomes."
        )
    groups = {(record.method, record.architecture) for record in records}
    expected_groups = expected_sweep_groups(config_directory)
    if complete and groups != expected_groups:
        readable = ", ".join(
            f"{method} × {architecture}" for method, architecture in sorted(groups)
        )
        raise SystemExit(
            "Completed checkpoints do not cover the preprocessing×architecture groups "
            f"declared by the sweep configs. Observed {len(groups)}/{len(expected_groups)}: "
            f"{readable}."
        )
    print(LINE)
    print("MOLECULAR CONFIGURATION SELECTION" if complete else "PRELIMINARY MOLECULAR SELECTION")
    print(LINE)
    if not complete:
        running_count = len(records) - len(completed_records)
        print(
            f"PRELIMINARY: using {len(records)}/{expected_runs or '?'} readable checkpoints "
            f"({len(completed_records)} completed + {running_count} in-progress) from "
            f"{len(groups)}/{len(expected_groups)} currently represented "
            "preprocessing×architecture groups."
        )
        if args.include_running_checkpoints:
            print(
                "In-progress best.pt files are best-so-far snapshots and can be replaced by "
                "their training jobs during this evaluation."
            )
        print("Results will change as additional sweep runs complete.\n")
    print(
        f"Checking {len(records)} best.pt files against the frozen molecular-validation "
        "identities and checkpoint scores."
    )
    print(
        "Selection score = 0.50 × GEO clean-validation MSE + 0.50 × donor-balanced "
        "IKEM clean-validation MSE. Scores are compared only within each preprocessing "
        "× architecture group."
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
    group_winners = select_molecular_group_winners(
        scored, expected_groups=len(expected_groups) if complete else None
    )

    if not complete:
        score_path, group_path = save_molecular_selection(
            scored, group_winners, output_root / "molecular_selection"
        )
        print("\n" + LINE)
        print(
            "PRELIMINARY VALIDATION-ONLY WINNERS "
            f"({len(group_winners)}/{len(expected_groups)} GROUPS)"
        )
        print(LINE)
        for _, _, record in group_winners:
            print(
                f"{record.run:<10} score={record.molecular_selection_mse:.10g} | "
                f"GEO={record.geo_validation_mse:.10g} | "
                f"IKEM donor-balanced={record.ikem_validation_mse:.10g} "
                f"(donor SD={record.ikem_validation_mse_donor_sd:.10g}) | "
                f"{record.method} | {record.architecture} | z={record.latent_dim}"
            )
        print(f"\nValidation scores: {score_path}")
        print(f"Current group winners: {group_path}")
        print(
            "Stopped before GEO test and eGFR evaluation because the sweep is incomplete."
        )
        return

    print(
        "Evaluating the 584-row GEO test partition for the frozen group winners only..."
    )
    group_winners = evaluate_frozen_molecular_test_records(
        group_winners,
        layout,
        prepared_root,
        device=device,
        batch_size=args.batch_size,
        progress=report_progress,
    )
    score_path, group_path = save_molecular_selection(
        scored, group_winners, output_root / "molecular_selection"
    )

    print("\n" + LINE)
    print(
        f"{len(group_winners)} FROZEN MOLECULAR GROUP WINNERS"
    )
    print(LINE)
    for _, _, record in group_winners:
        print(
            f"{record.run:<10} score={record.molecular_selection_mse:.10g} | "
            f"GEO val={record.geo_validation_mse:.10g} | "
            f"IKEM val donor-balanced={record.ikem_validation_mse:.10g} "
            f"(donor SD={record.ikem_validation_mse_donor_sd:.10g}) | "
            f"GEO test={record.molecular_test_mse:.10g} | "
            f"{record.method} | {record.architecture} | z={record.latent_dim}"
        )
    print(f"\nAll molecular scores: {score_path}")
    print(f"Group winners: {group_path}")
    print(
        "GEO test was evaluated only after validation froze all group winners; it did "
        "not participate in checkpoint or hyperparameter selection."
    )
    print(
        "NOTE: Per-dataset standardization uses one source dataset at a time; each GEO "
        "dataset stays in one molecular split, while IKEM uses frozen outcome-blind "
        "pretraining-train parameters. IKEM per-dataset RMA likewise uses only frozen "
        "outcome-free pretraining-train parameters and transforms every evaluation CEL "
        "independently. Global RMA must carry matching GEO-train-reference provenance."
    )

    selected = group_winners

    print("\nPreparing method-matched IKEM inputs without using eGFR outcomes or CV folds...")
    required_ikem_methods = tuple(
        dict.fromkeys(record.method for _, _, record in selected)
    )
    ikem_sources = prepare_ikem_evaluation_sources(
        layout,
        prepared_root,
        output_root / "ikem_preprocessing",
        methods=required_ikem_methods,
        force=args.rebuild_ikem_preprocessing,
        batch_size=args.batch_size,
    )
    for method, prepared_ikem in ikem_sources.items():
        provenance = prepared_ikem.provenance
        qualification = (
            "ERROR: transductive preprocessing"
            if provenance["transductive_across_egfr_folds"]
            else "frozen pretraining reference; inductive per sample"
        )
        if provenance["transductive_across_egfr_folds"]:
            raise RuntimeError(
                f"Refusing eGFR evaluation with transductive preprocessing: {method}."
            )
        print(f"  {method}: {qualification}")

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
                source=ikem_sources[record.method].source,
            )
        )

    embedding_root, save_warnings = save_embeddings(embeddings, output_root / "embeddings")
    for warning in save_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Saved embeddings: {embedding_root}")
    if args.embeddings_only:
        return

    input_dimensions = {int(result.record.input_dim) for result in embeddings}
    if len(input_dimensions) != 1:
        raise RuntimeError(
            f"Frozen encoders disagree on input dimension: {sorted(input_dimensions)}"
        )
    winner_input_dim = input_dimensions.pop()
    expressions: dict[str, object] = {}
    expression_samples: dict[str, object] = {}
    for method in required_ikem_methods:
        matrix, samples = aligned_ikem_matrix(
            layout,
            prepared_root,
            winner_input_dim,
            source=ikem_sources[method].source,
        )
        expressions[method] = matrix
        expression_samples[method] = samples
    reference_samples = expression_samples[required_ikem_methods[0]]
    reference_ids = reference_samples["sample_id"].astype(str).tolist()
    for method, samples in expression_samples.items():
        if samples["sample_id"].astype(str).tolist() != reference_ids:
            raise RuntimeError(f"IKEM sample order differs after {method} preprocessing.")
    egfr = load_egfr_wide(layout, reference_ids)
    stratify_column = None if args.stratify_column.lower() == "none" else args.stratify_column
    benchmark_root = output_root / "mixed_models"

    reused_baseline_metrics = reused_baseline_predictions = None
    reused_lasso_metrics = None
    if args.reuse_baseline_root is not None:
        baseline_root = args.reuse_baseline_root.expanduser().resolve()
        contract_path = baseline_root / "baseline_contract.json"
        if not contract_path.is_file():
            raise SystemExit(f"Missing reusable baseline contract: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        patient_hash = hashlib.sha256(
            "\n".join(egfr["patient"].astype(str)).encode("utf-8")
        ).hexdigest()
        expected_contract = {
            "format": 2,
            "complete": True,
            "checkpoint_free": True,
            "input_dim": winner_input_dim,
            "patient_ids_sha256": patient_hash,
            "folds": args.folds,
            "repeats": args.repeats,
            "cv_seed": args.cv_seed,
            "stratify_column": stratify_column,
            "include_clinical": not args.no_clinical,
            "include_pca": not args.no_pca,
        }
        if not args.no_probe_lasso:
            expected_contract.update(
                {
                    "include_lasso": True,
                    "lasso_models": ["LASSO-AIC", "LASSO-BIC"],
                }
            )
        mismatches = {
            key: (contract.get(key), expected)
            for key, expected in expected_contract.items()
            if contract.get(key) != expected
        }
        baseline_methods = set(str(value) for value in contract.get("methods", []))
        missing_methods = sorted(set(required_ikem_methods) - baseline_methods)
        if missing_methods:
            mismatches["methods"] = (
                sorted(baseline_methods),
                f"must include {missing_methods}",
            )
        if mismatches:
            raise SystemExit(
                "Reusable baseline contract does not match this evaluation: "
                + "; ".join(
                    f"{key}={observed!r}, expected {expected!r}"
                    for key, (observed, expected) in mismatches.items()
                )
            )
        reused_baseline_metrics = baseline_root / str(contract["mixed_model_metrics"])
        reused_baseline_predictions = baseline_root / str(
            contract["mixed_model_predictions"]
        )
        if not reused_baseline_metrics.is_file() or not reused_baseline_predictions.is_file():
            raise SystemExit("Reusable baseline contract points to missing result files.")
        candidate_lasso = baseline_root / "probe_lasso" / "fold_metrics.csv"
        if (
            not args.no_probe_lasso
            and contract.get("include_lasso") is True
            and candidate_lasso.is_file()
        ):
            reused_lasso_metrics = pd.read_csv(candidate_lasso)
        print(f"Reusing checkpoint-free baseline results from: {baseline_root}")

    lasso_fold_metrics = None
    if reused_lasso_metrics is not None:
        lasso_fold_metrics = reused_lasso_metrics
    elif not args.no_probe_lasso and args.reuse_baseline_root is None:
        try:
            alpha_fractions = tuple(
                float(value.strip())
                for value in args.lasso_alpha_fractions.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise SystemExit("--lasso-alpha-fractions must be comma-separated numbers.") from exc
        lasso_arms = {
            method: ProbeExpressionArm(
                method=method,
                matrix=expressions[method],
                samples=expression_samples[method],
                probe_ids=ikem_sources[method].source.probe_ids,
            )
            for method in required_ikem_methods
        }
        print(
            "Running all-probe LASSO-AIC and LASSO-BIC mixed models from one "
            "shared training-fold path per preprocessing strategy..."
        )
        lasso_fold_metrics, lasso_summary = evaluate_probe_lasso_mixed_models(
            egfr,
            lasso_arms,
            output_root / "probe_lasso",
            n_splits=args.folds,
            n_repeats=args.repeats,
            seed=args.cv_seed,
            stratify_column=stratify_column,
            config=ProbeLassoConfig(
                alpha_fractions=alpha_fractions,
                max_iter=args.lasso_max_iter,
                tolerance=args.lasso_tolerance,
            ),
        )
        print("\nALL-PROBE LASSO MIXED-MODEL BASELINES")
        print(lasso_summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    design_path, specs_path, specs = prepare_mixed_model_design(
        egfr,
        embeddings,
        expressions,
        expression_samples,
        benchmark_root,
        n_splits=args.folds,
        n_repeats=args.repeats,
        seed=args.cv_seed,
        stratify_column=stratify_column,
        include_pca=not args.no_pca and args.reuse_baseline_root is None,
        include_clinical=not args.no_clinical,
        include_standalone_baselines=args.reuse_baseline_root is None,
    )
    print(
        f"Prepared {len(specs)} fixed-model fits from {egfr['patient'].nunique()} patients "
        f"and {egfr['donor'].nunique()} donor groups. Every encoder was frozen by molecular "
        "validation before outcomes were loaded."
    )
    r_resource = files("archcon.assets").joinpath("molecular_mixed_models.R")
    with as_file(r_resource) as r_script:
        metrics_path, predictions_path = run_lme4_benchmark(
            design_path,
            specs_path,
            r_script,
            benchmark_root,
            rscript=args.rscript,
            output_prefix="fixed_",
        )
    if reused_baseline_metrics is not None and reused_baseline_predictions is not None:
        encoder_metrics = pd.read_csv(metrics_path)
        baseline_metrics = pd.read_csv(reused_baseline_metrics)
        duplicate_keys = ["model_id", "repeat", "fold"]
        if set(encoder_metrics["model_id"]).intersection(baseline_metrics["model_id"]):
            raise RuntimeError("Reusable baseline and encoder model IDs overlap.")
        combined_metrics = pd.concat(
            [baseline_metrics, encoder_metrics], ignore_index=True, sort=False
        )
        if combined_metrics.duplicated(duplicate_keys).any():
            raise RuntimeError("Combined reusable baseline metrics contain duplicate folds.")
        combined_predictions = pd.concat(
            [pd.read_csv(reused_baseline_predictions), pd.read_csv(predictions_path)],
            ignore_index=True,
            sort=False,
        )
        metrics_path = benchmark_root / "combined_fixed_fold_metrics.csv"
        predictions_path = benchmark_root / "combined_fixed_oof_predictions.csv"
        combined_metrics.to_csv(metrics_path, index=False)
        combined_predictions.to_csv(predictions_path, index=False)
    summary, fixed_encoder_summary, comparisons = summarize_fixed_encoder_results(
        metrics_path, predictions_path, embeddings, benchmark_root
    )
    lasso_comparison = None
    if lasso_fold_metrics is not None:
        lasso_comparison = summarize_probe_lasso_against_time(
            lasso_fold_metrics,
            pd.read_csv(metrics_path),
            output_root / "probe_lasso",
        )

    print("\n" + LINE)
    print("FIXED-ENCODER eGFR EVALUATION")
    print(LINE)
    fixed_columns = [
        "model_label",
        "mean_rmse",
        "mean_rmse_ci95_low",
        "mean_rmse_ci95_high",
        "pooled_rmse",
        "mean_delta_vs_time",
        "positive_folds_vs_time",
        "run",
    ]
    print(
        fixed_encoder_summary[fixed_columns].to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    print(
        "\nNo encoder ranking or deployment winner is derived from these outcomes. "
        "Each row is a separately reported, molecular-validation-frozen analysis."
    )
    if not args.no_clinical:
        print("Clinical variables: " + ", ".join(CLINICAL_COLUMNS))
        print("Clinical imputation and scaling are fitted within each training fold.")
    if not comparisons.empty:
        print(
            "\nMatched-fold comparisons (positive gain favors the frozen encoder model):"
        )
    if lasso_comparison is not None:
        print("\nAll-probe LASSO compared with the same-fold lme4 time-only baseline:")
        print(
            lasso_comparison[
                [
                    "model_label",
                    "mean_rmse",
                    "mean_rmse_ci95_low",
                    "mean_rmse_ci95_high",
                    "mean_delta_vs_time",
                    "positive_folds_vs_time",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
        )
        comparison_columns = [
            "candidate_id",
            "comparison",
            "mean_gain",
            "gain_ci95_low",
            "gain_ci95_high",
            "candidate_better_fraction",
        ]
        print(
            comparisons[comparison_columns].to_string(
                index=False, float_format=lambda value: f"{value:.6g}"
            )
        )
    print(f"\nAll-model summary rows: {len(summary)}")
    print(f"\nComplete results: {benchmark_root}")
