"""Nested donor-CV eGFR evaluation with reconstruction/soft-InfoNCE fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.resources import as_file, files
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import pandas as pd

from .data.contrastive_egfr import (
    STADNIUK_CONTRASTIVE_COLUMNS,
    FineTuneControls,
    contrastive_candidate_grid,
)
from .data.defaults import ProjectDataLayout, project_data_layout
from .data.downstream import (
    ValidationCheckpoint,
    aligned_ikem_matrix,
    evaluate_frozen_molecular_test_records,
    load_egfr_wide,
    save_molecular_selection,
    scan_validation_checkpoints,
    score_molecular_selection_records,
    select_molecular_group_winners,
)
from .data.ikem_preprocessing import PreparedIKEMSource, prepare_ikem_evaluation_sources
from .data.nested_cv import (
    NestedSplitConfig,
    create_or_load_nested_split_manifest,
    manifest_sha256,
)
from .data.nested_egfr import EncoderArm, MethodExpression, evaluate_nested_egfr
from .data.probe_lasso import (
    ProbeExpressionArm,
    ProbeLassoConfig,
    evaluate_probe_lasso_mixed_models,
    summarize_probe_lasso_against_time,
)


LINE = "=" * 88


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return result or "item"


def _parse_floats(value: str, option: str) -> tuple[float, ...]:
    try:
        parsed = tuple(
            dict.fromkeys(float(part.strip()) for part in value.split(",") if part.strip())
        )
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated numbers.") from exc
    if not parsed or any(item <= 0.0 for item in parsed):
        raise ValueError(f"{option} must contain positive numbers.")
    return parsed


def _parse_positive_ints(value: str, option: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip())
        )
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated integers.") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError(f"{option} must contain positive integers.")
    return parsed


def _parse_columns(value: str, option: str) -> tuple[str, ...]:
    parsed = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not parsed:
        raise ValueError(f"{option} must contain at least one column name.")
    return parsed


def _expected_sweep_groups(config_directory: Path) -> set[tuple[str, str]]:
    groups: set[tuple[str, str]] = set()
    for path in sorted(config_directory.glob("run_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        training = payload.get("training", {})
        method = str(payload.get("method", "")).strip()
        architecture = str(training.get("architecture_family", "")).strip()
        if not method or not architecture:
            raise ValueError(f"Run config lacks method/architecture: {path}")
        groups.add((method, architecture))
    if not groups:
        raise ValueError(f"No generated run configs were found under {config_directory}.")
    return groups


def _choose_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required; install archcon[training].") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _progress_scan(index: int, total: int, readable: int) -> None:
    if index == 1 or index == total or index % 50 == 0:
        print(
            f"  scanned {index:>4}/{total}; {readable} readable best.pt checkpoints",
            flush=True,
        )


def _progress_record(index: int, total: int, record: ValidationCheckpoint) -> None:
    if index == 1 or index == total or index % 50 == 0:
        print(
            f"  [{index:>4}/{total}] {record.run} · {record.method} · "
            f"{record.architecture}",
            flush=True,
        )


def _freeze_sweep_winners(
    sweep_root: Path,
    layout: ProjectDataLayout,
    output_root: Path,
    *,
    device: str,
    batch_size: int,
    audit_geo_test: bool,
) -> tuple[Path, list[tuple[str, str, ValidationCheckpoint]]]:
    """Require a complete sweep and freeze its validation-selected group winners."""

    prepared_root = sweep_root / "prepared"
    results_root = sweep_root / "results"
    config_root = sweep_root / "configs"
    for required in (prepared_root, results_root, config_root):
        if not required.is_dir():
            raise FileNotFoundError(f"Incomplete sweep directory: {required}")

    config_names = {path.stem for path in config_root.glob("run_*.json")}
    expected_groups = _expected_sweep_groups(config_root)
    completed_names = {
        path.parent.name for path in results_root.glob("run_*/run_summary.json")
    }.intersection(config_names)
    records, warnings = scan_validation_checkpoints(results_root, progress=_progress_scan)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    records = [record for record in records if record.run in completed_names]
    readable_names = {record.run for record in records}
    if completed_names != config_names or readable_names != config_names:
        raise RuntimeError(
            f"Sweep {sweep_root.name!r} is not complete and readable: "
            f"configs={len(config_names)}, summaries={len(completed_names)}, "
            f"eligible best.pt={len(readable_names)}. Nested outcome evaluation refuses "
            "best-so-far checkpoints."
        )
    observed_groups = {(record.method, record.architecture) for record in records}
    if observed_groups != expected_groups:
        raise RuntimeError(
            f"Sweep {sweep_root.name!r} checkpoint groups differ from its configs."
        )

    scored = score_molecular_selection_records(
        records,
        layout,
        prepared_root,
        device=device,
        batch_size=batch_size,
        progress=_progress_record,
    )
    winners = select_molecular_group_winners(
        scored, expected_groups=len(expected_groups)
    )
    if audit_geo_test:
        winners = evaluate_frozen_molecular_test_records(
            winners,
            layout,
            prepared_root,
            device=device,
            batch_size=batch_size,
            progress=_progress_record,
        )
    selection_root = output_root / _slug(sweep_root.name)
    save_molecular_selection(scored, winners, selection_root)
    return prepared_root, winners


def _expected_outer_assignments(manifest: pd.DataFrame) -> pd.DataFrame:
    expected = manifest.loc[
        :, ["outer_repeat", "outer_fold", "outer_role", "patient", "donor"]
    ].rename(
        columns={
            "outer_repeat": "repeat",
            "outer_fold": "fold",
            "outer_role": "partition",
        }
    )
    columns = ["repeat", "fold", "partition", "patient", "donor"]
    expected["repeat"] = expected["repeat"].astype(int)
    expected["fold"] = expected["fold"].astype(int)
    for column in ("partition", "patient", "donor"):
        expected[column] = expected[column].astype(str)
    return expected.loc[:, columns].sort_values(columns).reset_index(drop=True)


def _load_reusable_lasso(
    requested_root: Path,
    manifest: pd.DataFrame,
    methods: Iterable[str],
) -> tuple[Path, pd.DataFrame]:
    """Load prior LASSO fits only after proving their outer roles are identical."""

    root = requested_root.expanduser().resolve()
    if (root / "probe_lasso").is_dir():
        root = root / "probe_lasso"
    assignments_path = root / "outer_fold_assignments.csv"
    metrics_path = root / "fold_metrics.csv"
    if not assignments_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Reusable LASSO root lacks assignments or fold metrics: {root}"
        )
    observed = pd.read_csv(assignments_path)
    expected = _expected_outer_assignments(manifest)
    columns = list(expected.columns)
    missing = set(columns).difference(observed.columns)
    if missing:
        raise ValueError(f"Reusable LASSO assignments lack columns: {sorted(missing)}")
    observed = observed.loc[:, columns].copy()
    observed["repeat"] = observed["repeat"].astype(int)
    observed["fold"] = observed["fold"].astype(int)
    for column in ("partition", "patient", "donor"):
        observed[column] = observed[column].astype(str)
    observed = observed.sort_values(columns).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(
            "Reusable LASSO folds differ from nested_split_manifest.csv; refusing reuse."
        )
    metrics = pd.read_csv(metrics_path)
    required_methods = set(map(str, methods))
    observed_methods = set(metrics["preprocessing"].astype(str))
    if not required_methods.issubset(observed_methods):
        raise ValueError(
            f"Reusable LASSO output lacks methods: {sorted(required_methods-observed_methods)}"
        )
    return root, metrics.loc[
        metrics["preprocessing"].astype(str).isin(required_methods)
    ].copy()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fully nested donor-grouped eGFR evaluation. Molecular preprocessing "
            "and checkpoint selection remain frozen; inner validation selects none versus "
            "soft-InfoNCE and PCA dimension, then each choice is refitted on full outer "
            "training donors before one outer-test evaluation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        action="append",
        required=True,
        help="Completed sweep root; repeat once for standardized, per-GSE, and global RMA.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--stratify-column", default="KDRI_8")
    parser.add_argument("--contrastive-weights", default="0.01,0.05,0.1")
    parser.add_argument("--temperatures", default="0.05,0.1,0.2")
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.0)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--pca-dimensions", default="3,8,16")
    parser.add_argument(
        "--contrastive-clinical-columns",
        default=",".join(STADNIUK_CONTRASTIVE_COLUMNS),
        help=(
            "Comma-separated columns from Klasifikator_20_3_24_v2.xlsx used only to "
            "define soft-InfoNCE similarities. Defaults to Stadniuk's prespecified "
            "eight-feature pool; eGFR values are never features."
        ),
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--expected-encoder-arms", type=int, default=6)
    parser.add_argument("--expected-egfr-biopsies", type=int, default=254)
    parser.add_argument("--expected-egfr-donors", type=int, default=163)
    parser.add_argument("--rebuild-ikem-preprocessing", action="store_true")
    parser.add_argument(
        "--skip-geo-test-audit",
        action="store_true",
        help="Skip the post-freeze GEO test audit; it never affects eGFR selection.",
    )
    lasso = parser.add_mutually_exclusive_group()
    lasso.add_argument(
        "--run-lasso",
        action="store_true",
        help="Also fit LASSO-AIC/BIC on the exact persisted outer folds.",
    )
    lasso.add_argument(
        "--reuse-lasso-root",
        type=Path,
        default=None,
        help="Reuse prior probe_lasso output only if its donor assignments match exactly.",
    )
    parser.add_argument(
        "--lasso-alpha-fractions",
        default="1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    parser.add_argument("--lasso-max-iter", type=int, default=5_000)
    parser.add_argument("--lasso-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        weights = _parse_floats(args.contrastive_weights, "--contrastive-weights")
        temperatures = _parse_floats(args.temperatures, "--temperatures")
        pca_dimensions = _parse_positive_ints(args.pca_dimensions, "--pca-dimensions")
        contrastive_columns = _parse_columns(
            args.contrastive_clinical_columns,
            "--contrastive-clinical-columns",
        )
        lasso_fractions = _parse_floats(
            args.lasso_alpha_fractions, "--lasso-alpha-fractions"
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.folds < 2 or args.repeats < 1 or args.inner_folds < 2:
        raise SystemExit("folds and inner-folds must be >=2; repeats must be >=1.")

    project_root = args.project_root.expanduser().resolve()
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else project_root / "data"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "evaluations" / f"egfr-nested-infonce-seed-{args.cv_seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_roots = tuple(path.expanduser().resolve() for path in args.sweep_root)
    if len(set(sweep_roots)) != len(sweep_roots):
        raise SystemExit("Duplicate --sweep-root values are not allowed.")
    device = _choose_device(args.device)
    layout = project_data_layout(data_dir)
    stratify_column = None if args.stratify_column.lower() == "none" else args.stratify_column

    print(LINE)
    print("NESTED eGFR · RECONSTRUCTION CONTROL VS SOFT-INFONCE")
    print(LINE)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_root}")
    print(f"Runtime device: {device}")
    print("Frozen RMA/standardization references will not be refitted.")

    selected: list[tuple[str, str, ValidationCheckpoint]] = []
    prepared_by_method: dict[str, Path] = {}
    selection_root = output_root / "molecular_selection"
    for sweep in sweep_roots:
        print(f"\nFreezing molecular-validation winners from {sweep.name}...")
        prepared, winners = _freeze_sweep_winners(
            sweep,
            layout,
            selection_root,
            device=device,
            batch_size=args.batch_size,
            audit_geo_test=not args.skip_geo_test_audit,
        )
        for _, _, record in winners:
            previous = prepared_by_method.setdefault(record.method, prepared)
            if previous != prepared:
                raise RuntimeError(
                    f"Preprocessing method {record.method!r} occurs in multiple sweeps."
                )
        selected.extend(winners)
    group_keys = [(record.method, record.architecture) for _, _, record in selected]
    if len(group_keys) != len(set(group_keys)):
        raise RuntimeError("Duplicate preprocessing×architecture winner across sweeps.")
    if args.expected_encoder_arms > 0 and len(selected) != args.expected_encoder_arms:
        raise RuntimeError(
            f"Expected {args.expected_encoder_arms} frozen encoder arms, found {len(selected)}."
        )

    dimensions = {int(record.input_dim) for _, _, record in selected}
    if len(dimensions) != 1:
        raise RuntimeError(f"Frozen encoders disagree on probe count: {sorted(dimensions)}")
    input_dim = dimensions.pop()
    methods: dict[str, MethodExpression] = {}
    prepared_sources: dict[str, PreparedIKEMSource] = {}
    reference_ids: list[str] | None = None
    for method, prepared_root in prepared_by_method.items():
        prepared = prepare_ikem_evaluation_sources(
            layout,
            prepared_root,
            output_root / "ikem_preprocessing" / _slug(method),
            methods=(method,),
            force=args.rebuild_ikem_preprocessing,
            batch_size=args.batch_size,
        )[method]
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
        methods[method] = MethodExpression(method=method, matrix=matrix, samples=samples)
        prepared_sources[method] = prepared
        print(f"  {method}: {matrix.shape[0]} samples × {matrix.shape[1]:,} frozen probes")
    assert reference_ids is not None

    patients = load_egfr_wide(layout, reference_ids)
    if args.expected_egfr_biopsies > 0 and len(patients) != args.expected_egfr_biopsies:
        raise RuntimeError(
            f"Expected {args.expected_egfr_biopsies} measured-eGFR biopsies, "
            f"found {len(patients)}."
        )
    donor_count = int(patients["donor"].nunique())
    if args.expected_egfr_donors > 0 and donor_count != args.expected_egfr_donors:
        raise RuntimeError(
            f"Expected {args.expected_egfr_donors} measured-eGFR donors, "
            f"found {donor_count}."
        )
    print(f"Measured-eGFR cohort: {len(patients)} biopsies / {donor_count} donors")

    split_config = NestedSplitConfig(
        outer_splits=args.folds,
        outer_repeats=args.repeats,
        inner_splits=args.inner_folds,
        seed=args.cv_seed,
        stratify_column=stratify_column,
    )
    manifest_path = output_root / "nested_split_manifest.csv"
    manifest = create_or_load_nested_split_manifest(patients, manifest_path, split_config)
    print(f"Shared nested donor manifest: {manifest_path}")

    candidates = contrastive_candidate_grid(weights, temperatures)
    controls = FineTuneControls(
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        learning_rate=args.finetune_learning_rate,
        weight_decay=args.finetune_weight_decay,
        projection_dim=args.projection_dim,
        gradient_clip=args.gradient_clip,
        deterministic=True,
    )
    encoder_arms = tuple(
        EncoderArm(model_id=model_id, label=label, record=record)
        for model_id, label, record in selected
    )
    print(
        f"Inner grid per encoder: {len(candidates)} candidates "
        f"(1 reconstruction control + {len(candidates)-1} soft-InfoNCE)."
    )
    print(f"PCA dimensions selected on the same inner donors: {pca_dimensions}")

    r_resource = files("archcon.assets").joinpath("molecular_mixed_models.R")
    with as_file(r_resource) as r_script:
        nested = evaluate_nested_egfr(
            layout,
            patients,
            manifest,
            encoder_arms,
            methods,
            candidates,
            controls,
            pca_dimensions,
            output_root / "nested_models",
            r_script,
            rscript=args.rscript,
            device=device,
            seed=args.cv_seed,
            contrastive_columns=contrastive_columns,
        )

    nested_summary = pd.read_csv(nested.summary_path)
    combined_parts = [nested_summary]
    lasso_source: str | None = None
    lasso_metrics: pd.DataFrame | None = None
    lasso_root = output_root / "probe_lasso"
    if args.reuse_lasso_root is not None:
        reused_root, lasso_metrics = _load_reusable_lasso(
            args.reuse_lasso_root, manifest, methods
        )
        lasso_source = str(reused_root)
        print(f"Reusing donor-identical LASSO output: {reused_root}")
    elif args.run_lasso:
        probe_arms = {
            method: ProbeExpressionArm(
                method=method,
                matrix=expression.matrix,
                samples=expression.samples,
                probe_ids=prepared_sources[method].source.probe_ids,
            )
            for method, expression in methods.items()
        }
        lasso_metrics, _ = evaluate_probe_lasso_mixed_models(
            patients,
            probe_arms,
            lasso_root,
            n_splits=args.folds,
            n_repeats=args.repeats,
            seed=args.cv_seed,
            stratify_column=stratify_column,
            config=ProbeLassoConfig(
                alpha_fractions=lasso_fractions,
                max_iter=args.lasso_max_iter,
                tolerance=args.lasso_tolerance,
            ),
            split_manifest=manifest,
        )
        lasso_source = str(lasso_root)
    if lasso_metrics is not None:
        lasso_summary = summarize_probe_lasso_against_time(
            lasso_metrics,
            pd.read_csv(nested.metrics_path),
            lasso_root,
        )
        lasso_summary["model_family"] = "probe_lasso"
        combined_parts.append(lasso_summary)

    combined = pd.concat(combined_parts, ignore_index=True, sort=False)
    combined_path = output_root / "nested_complete_model_summary.csv"
    combined.to_csv(combined_path, index=False)
    _write_json(
        output_root / "nested_command_summary.json",
        {
            "format": 1,
            "complete": True,
            "data_dir": str(data_dir),
            "sweep_roots": list(map(str, sweep_roots)),
            "device": device,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": manifest_sha256(manifest),
            "split_config": asdict(split_config),
            "encoder_arms": len(encoder_arms),
            "contrastive_candidates": len(candidates),
            "fine_tune_controls": asdict(controls),
            "contrastive_clinical_columns": list(contrastive_columns),
            "pca_dimensions": list(pca_dimensions),
            "lasso_source": lasso_source,
            "combined_summary": str(combined_path),
        },
    )
    print("\n" + LINE)
    print("NESTED eGFR EVALUATION COMPLETE")
    print(LINE)
    print(combined.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nCombined comparison: {combined_path}")
    print(f"Encoder selections by outer fold: {nested.encoder_selections_path}")
    print(f"PCA selections by outer fold: {nested.pca_selections_path}")


if __name__ == "__main__":
    main()
