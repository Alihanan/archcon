"""Combine completed nested eGFR results with earlier frozen-encoder results.

This reads existing CSVs only: no CEL processing, model fitting, or neural inference.
Earlier fixed-encoder runs are marked preliminary because their checkpoints may
have been chosen before the molecular sweeps finished.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OLDER_EVALUATIONS = (
    "exploratory-egfr-standardized-20260917T120536Z",
    "exploratory-egfr-per-gse-20260917T103139Z",
)
VARIANTS = (
    ("", "z only"),
    ("_kdri", "z + KDRI"),
    ("_clinical", "z + full clinical"),
)
OUTPUT_COLUMNS = (
    "comparison_id",
    "evaluation_stage",
    "model_id",
    "model_label",
    "model_family",
    "preprocessing",
    "architecture",
    "variant",
    "encoder_state",
    "source_run",
    "checkpoint_matches_final",
    "mean_rmse",
    "sd_rmse",
    "pooled_rmse",
    "mean_mae",
    "mean_rmse_gain_vs_full_clinical",
    "fraction_folds_better_than_full_clinical",
    "mean_delta_vs_time",
    "positive_folds_vs_time",
    "n_folds",
    "source_evaluation",
)


def required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required existing evaluation file is missing: {path}")
    return pd.read_csv(path)


def as_single_index(frame: pd.DataFrame, key: str, path: Path) -> pd.DataFrame:
    if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{path} needs one nonempty row per {key}")
    return frame.set_index(key, verify_integrity=True)


def old_gains_against_clinical(metrics: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    keys = ["repeat", "fold"]
    required = {"model_id", *keys, "rmse"}
    if not required.issubset(metrics):
        raise ValueError(f"Earlier fold metrics lack: {sorted(required - set(metrics))}")
    if metrics.duplicated(["model_id", *keys]).any():
        raise ValueError("Earlier evaluation has duplicate model/fold metrics")
    reference = metrics.loc[
        metrics.model_id.eq("clinical_full"), [*keys, "rmse"]
    ].rename(columns={"rmse": "clinical_rmse"})
    if reference.empty or reference.duplicated(keys).any():
        raise ValueError("Earlier evaluation needs one full-clinical baseline per fold")
    selected = metrics.loc[metrics.model_id.isin(names), ["model_id", *keys, "rmse"]]
    paired = selected.merge(reference, on=keys, how="left", validate="many_to_one")
    if paired.clinical_rmse.isna().any() or set(paired.model_id) != set(names):
        raise ValueError("An earlier encoder is missing a matched clinical fold")
    counts = paired.groupby("model_id").size()
    if not counts.eq(len(reference)).all():
        raise ValueError("Some earlier encoders have incomplete outer folds")
    paired["gain"] = paired.clinical_rmse - paired.rmse
    return paired.groupby("model_id").gain.agg(
        mean_gain="mean",
        better_fraction=lambda values: float((values > 0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    final_root = project / "evaluations" / "egfr-nested-infonce-seed-0"
    final_path = final_root / "final_egfr_comparison.csv"
    output_path = (args.output or final_root / "combined_existing_egfr_results.csv")
    output_path = output_path.expanduser().resolve()

    final = required_csv(final_path).copy()
    catalog_path = final_root / "nested_models" / "nested_model_catalog.csv"
    catalog = as_single_index(required_csv(catalog_path), "model_id", catalog_path)
    if final.model_id.duplicated().any() or len(final) != 36:
        raise ValueError("Expected 36 distinct completed nested-evaluation models")
    final["source_run"] = final.model_id.map(catalog.source_run)
    final["evaluation_stage"] = "completed_nested"
    final["comparison_id"] = "nested__" + final.model_id
    final["encoder_state"] = np.where(
        final.model_family.eq("nested_finetuned_encoder"),
        "50-epoch fine-tuning; reconstruction-only or soft-InfoNCE selected per fold",
        "not applicable",
    )
    final["checkpoint_matches_final"] = "reference"
    final["source_evaluation"] = str(final_path)
    combined: list[pd.DataFrame] = [final]

    for evaluation_name in OLDER_EVALUATIONS:
        source_root = project / "evaluations" / evaluation_name
        mixed = source_root / "mixed_models"
        fixed_path = mixed / "fixed_encoder_cv_summary.csv"
        all_path = mixed / "all_model_cv_summary.csv"
        metrics_path = mixed / "fold_metrics.csv"
        fixed = as_single_index(required_csv(fixed_path), "model_id", fixed_path)
        all_models = as_single_index(required_csv(all_path), "model_id", all_path)
        fold_metrics = required_csv(metrics_path)
        if len(fixed) != 2:
            raise ValueError(f"Expected two preliminary encoders in {fixed_path}")

        rows = []
        for base_id, checkpoint in fixed.iterrows():
            for suffix, variant in VARIANTS:
                model_id = f"{base_id}{suffix}"
                if model_id not in all_models.index:
                    raise ValueError(f"Missing earlier mixed model {model_id}: {all_path}")
                metric = all_models.loc[model_id]
                row = metric.to_dict()
                row.update({
                    "model_id": model_id,
                    "model_family": "preliminary_frozen_encoder",
                    "preprocessing": checkpoint.preprocessing,
                    "architecture": checkpoint.architecture,
                    "variant": variant,
                    "source_run": checkpoint.run,
                    "evaluation_stage": "preliminary_frozen_checkpoint",
                    "comparison_id": f"frozen__{model_id}",
                    "encoder_state": "original molecular checkpoint; no IKEM eGFR fine-tuning",
                    "source_evaluation": str(source_root),
                    "checkpoint_matches_final": (
                        "yes" if model_id in catalog.index
                        and str(catalog.at[model_id, "source_run"]) == str(checkpoint.run)
                        else "no or unknown"
                    ),
                })
                rows.append(row)

        older = pd.DataFrame(rows)
        gains = old_gains_against_clinical(fold_metrics, older.model_id.tolist())
        older["mean_rmse_gain_vs_full_clinical"] = older.model_id.map(gains.mean_gain)
        older["fraction_folds_better_than_full_clinical"] = older.model_id.map(
            gains.better_fraction
        )
        combined.append(older)

    table = pd.concat(combined, ignore_index=True, sort=False)
    if len(table) != 48 or table.comparison_id.duplicated().any():
        raise ValueError("Expected 36 final plus 12 distinct preliminary encoder rows")
    if table.n_folds.isna().any() or not table.n_folds.eq(25).all():
        raise ValueError("Some source results do not report all 25 outer folds")
    table["_order"] = table.model_id.map(
        {"time_only": 0, "clinical_kdri": 1, "clinical_full": 2}
    ).fillna(3)
    table = table.sort_values(
        ["_order", "preprocessing", "architecture", "model_id", "evaluation_stage"],
        kind="stable",
        na_position="first",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table[list(OUTPUT_COLUMNS)].to_csv(output_path, index=False)
    print(f"Saved {len(table)} existing models: {output_path}")
    print("36 completed nested results + 12 earlier, preliminary frozen encoders")
    print("Earlier checkpoint matches:",
          table.loc[table.evaluation_stage.eq("preliminary_frozen_checkpoint"),
                    "checkpoint_matches_final"].value_counts().to_dict())
    print("Earlier outer-test donor membership has not been checked against the nested run.")


if __name__ == "__main__":
    main()
