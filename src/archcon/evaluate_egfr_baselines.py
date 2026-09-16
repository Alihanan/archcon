"""Checkpoint-free longitudinal eGFR baseline evaluation."""

from __future__ import annotations

import argparse
import hashlib
from importlib.resources import as_file, files
import json
from pathlib import Path

import pandas as pd

from .data.defaults import project_data_layout
from .data.downstream import (
    aligned_ikem_matrix,
    load_egfr_wide,
    prepare_mixed_model_design,
    run_lme4_benchmark,
    summarize_baseline_mixed_model_results,
)
from .data.geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from .data.ikem_preprocessing import prepare_ikem_evaluation_sources
from .data.probe_lasso import (
    ProbeExpressionArm,
    ProbeLassoConfig,
    evaluate_probe_lasso_mixed_models,
    summarize_probe_lasso_against_time,
)
from .data.training_sources import METHOD_PER_DATASET_STANDARDIZED

LINE = "=" * 88
METHOD_ALIASES = {
    "standardized": METHOD_PER_DATASET_STANDARDIZED,
    "standardization": METHOD_PER_DATASET_STANDARDIZED,
    "per-gse": METHOD_PER_GSE_RMA,
    "per-study": METHOD_PER_GSE_RMA,
    "per-study-rma": METHOD_PER_GSE_RMA,
    "global": METHOD_GLOBAL_RMA,
    "global-rma": METHOD_GLOBAL_RMA,
}


def parse_methods(value: str) -> tuple[str, ...]:
    """Resolve concise CLI aliases while accepting canonical method names."""

    canonical = {
        METHOD_PER_DATASET_STANDARDIZED.lower(): METHOD_PER_DATASET_STANDARDIZED,
        METHOD_PER_GSE_RMA.lower(): METHOD_PER_GSE_RMA,
        METHOD_GLOBAL_RMA.lower(): METHOD_GLOBAL_RMA,
    }
    result: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        method = METHOD_ALIASES.get(token.lower(), canonical.get(token.lower()))
        if method is None:
            choices = ", ".join(sorted(METHOD_ALIASES))
            raise ValueError(f"Unknown preprocessing method {token!r}; aliases: {choices}")
        if method not in result:
            result.append(method)
    if not result:
        raise ValueError("At least one preprocessing method is required.")
    return tuple(result)


def parse_positive_ints(value: str, option: str) -> tuple[int, ...]:
    try:
        result = tuple(
            dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip())
        )
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated integers.") from exc
    if not result or any(item < 1 for item in result):
        raise ValueError(f"{option} must contain positive integers.")
    return result


def parse_alpha_fractions(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--lasso-alpha-fractions must contain numbers.") from exc
    return result


def write_contract(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all-probe LASSO-AIC/LASSO-BIC, fold-fitted input PCA, time-only, "
            "and clinical longitudinal eGFR baselines without neural checkpoints."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help=(
            "Frozen sweep prepared/ directory containing sample_index.csv, "
            "probe_index.csv, and preprocessing parameters. No checkpoints are read."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--methods",
        default="standardized,per-gse",
        help=(
            "Comma-separated preprocessing aliases. Supported: standardized, per-gse, "
            "global. Global may be added after its CEL rebuild completes."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--stratify-column", default="KDRI_8")
    parser.add_argument("--pca-dimensions", default="3,8,16")
    parser.add_argument(
        "--lasso-alpha-fractions",
        default="1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    parser.add_argument("--lasso-max-iter", type=int, default=5_000)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-4)
    parser.add_argument("--no-lasso", action="store_true")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--no-clinical", action="store_true")
    parser.add_argument(
        "--mixed-models-design-only",
        action="store_true",
        help="Save the lme4 design/specification files but do not invoke Rscript.",
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild-ikem-preprocessing", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else project_root / "data-per-gse-ready"
    )
    prepared_root = args.prepared_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "evaluations" / f"egfr-baselines-seed-{args.cv_seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        methods = parse_methods(args.methods)
        pca_dimensions = parse_positive_ints(args.pca_dimensions, "--pca-dimensions")
        alpha_fractions = parse_alpha_fractions(args.lasso_alpha_fractions)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.folds < 2 or args.repeats < 1:
        raise SystemExit("folds must be >=2; repeats must be >=1.")

    probe_index_path = prepared_root / "probe_index.csv"
    if not probe_index_path.is_file():
        raise SystemExit(f"Missing frozen probe index: {probe_index_path}")
    probe_index = pd.read_csv(probe_index_path)
    if "probe_id" not in probe_index or probe_index["probe_id"].duplicated().any():
        raise SystemExit("Frozen probe_index.csv lacks unique probe_id values.")
    input_dim = len(probe_index)

    layout = project_data_layout(data_dir)
    print(LINE)
    print("CHECKPOINT-FREE eGFR BASELINES")
    print(LINE)
    print(f"Data directory: {data_dir}")
    print(f"Prepared split/probes: {prepared_root}")
    print(f"Output directory: {output_root}")
    print(f"Preprocessing strategies: {', '.join(methods)}")
    print(f"Canonical probes: {input_dim:,}")
    print("Neural checkpoints: not read")

    prepared_sources = prepare_ikem_evaluation_sources(
        layout,
        prepared_root,
        output_root / "ikem_preprocessing",
        methods=methods,
        force=args.rebuild_ikem_preprocessing,
        batch_size=args.batch_size,
    )
    expressions: dict[str, object] = {}
    samples_by_method: dict[str, pd.DataFrame] = {}
    reference_ids: list[str] | None = None
    arms: dict[str, ProbeExpressionArm] = {}
    for method in methods:
        prepared = prepared_sources[method]
        if prepared.provenance.get("transductive_across_egfr_folds") is not False:
            raise RuntimeError(f"Refusing transductive IKEM preprocessing: {method}")
        matrix, samples = aligned_ikem_matrix(
            layout,
            prepared_root,
            input_dim,
            source=prepared.source,
        )
        ids = samples["sample_id"].astype(str).tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"IKEM sample order differs for {method}.")
        expressions[method] = matrix
        samples_by_method[method] = samples
        arms[method] = ProbeExpressionArm(
            method=method,
            matrix=matrix,
            samples=samples,
            probe_ids=prepared.source.probe_ids,
        )
        print(f"  {method}: {matrix.shape[0]} samples × {matrix.shape[1]:,} probes")
    assert reference_ids is not None

    egfr = load_egfr_wide(layout, reference_ids)
    stratify_column = None if args.stratify_column.lower() == "none" else args.stratify_column
    print(
        f"Measured-eGFR cohort: {len(egfr)} biopsies from "
        f"{egfr['donor'].nunique()} donor groups"
    )
    contract_path = output_root / "baseline_contract.json"
    patient_ids = egfr["patient"].astype(str).tolist()
    contract: dict[str, object] = {
        "format": 2,
        "complete": False,
        "checkpoint_free": True,
        "methods": sorted(methods),
        "input_dim": input_dim,
        "patient_ids_sha256": hashlib.sha256(
            "\n".join(patient_ids).encode("utf-8")
        ).hexdigest(),
        "n_patients": len(patient_ids),
        "n_donors": int(egfr["donor"].nunique()),
        "folds": args.folds,
        "repeats": args.repeats,
        "cv_seed": args.cv_seed,
        "stratify_column": stratify_column,
        "pca_dimensions": list(pca_dimensions) if not args.no_pca else [],
        "include_clinical": not args.no_clinical,
        "include_pca": not args.no_pca,
        "include_lasso": not args.no_lasso,
        "lasso_models": ["LASSO-AIC", "LASSO-BIC"] if not args.no_lasso else [],
        "lasso_selection": (
            "minimum training-fold marginal AIC/BIC on one shared path"
            if not args.no_lasso
            else None
        ),
    }
    write_contract(contract_path, contract)

    lasso_metrics = None
    if not args.no_lasso:
        print(
            "\nRunning resumable all-probe LASSO-AIC and LASSO-BIC models "
            "from shared training-fold paths..."
        )
        lasso_metrics, lasso_summary = evaluate_probe_lasso_mixed_models(
            egfr,
            arms,
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
        print(lasso_summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    mixed_root = output_root / "mixed_models"
    design_path, specs_path, specs = prepare_mixed_model_design(
        egfr,
        [],
        expressions,
        samples_by_method,
        mixed_root,
        n_splits=args.folds,
        n_repeats=args.repeats,
        seed=args.cv_seed,
        stratify_column=stratify_column,
        include_pca=not args.no_pca,
        include_clinical=not args.no_clinical,
        pca_dimensions=pca_dimensions if not args.no_pca else (),
    )
    print(f"\nSaved {len(specs)} lme4 baseline fit specifications: {specs_path}")
    if args.mixed_models_design_only:
        print("Stopped after design generation as requested.")
        return

    r_resource = files("archcon.assets").joinpath("molecular_mixed_models.R")
    with as_file(r_resource) as r_script:
        metrics_path, predictions_path = run_lme4_benchmark(
            design_path,
            specs_path,
            r_script,
            mixed_root,
            rscript=args.rscript,
            output_prefix="baseline_",
        )
    mixed_summary, _ = summarize_baseline_mixed_model_results(
        metrics_path, predictions_path, mixed_root
    )
    print("\nTIME / CLINICAL / INPUT-PCA MIXED-MODEL BASELINES")
    print(mixed_summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    combined_parts = [mixed_summary.assign(baseline_family="lme4")]
    if lasso_metrics is not None:
        lasso_comparison = summarize_probe_lasso_against_time(
            lasso_metrics,
            pd.read_csv(metrics_path),
            output_root / "probe_lasso",
        )
        combined_parts.append(lasso_comparison.assign(baseline_family="probe_lasso"))
    combined = pd.concat(combined_parts, ignore_index=True, sort=False)
    combined.to_csv(output_root / "combined_baseline_summary.csv", index=False)
    contract["complete"] = True
    contract["mixed_model_metrics"] = str(metrics_path.relative_to(output_root))
    contract["mixed_model_predictions"] = str(predictions_path.relative_to(output_root))
    contract["combined_summary"] = "combined_baseline_summary.csv"
    write_contract(contract_path, contract)
    print(f"\nComplete reusable baseline results: {output_root}")


if __name__ == "__main__":
    main()
