"""Combine completed nested eGFR models with freshly evaluated frozen encoders.

Run the frozen evaluator once per completed molecular sweep first. This script
only reads saved CSVs and checkpoint metadata; it never fits a model.

It refuses to combine results when outer test observations, checkpoint identity,
selected runs, or clinical reference fits disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLD = ["repeat", "fold"]
TEST = [*FOLD, "patient", "donor", "time"]
ARM_SUFFIXES = ("", "_kdri", "_clinical")


def csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def json_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def model_folds(frame: pd.DataFrame, model_ids: set[str], source: Path) -> pd.DataFrame:
    required = {"model_id", *FOLD, "rmse", "mae"}
    if not required.issubset(frame):
        raise ValueError(f"{source}: missing metric columns {sorted(required - set(frame))}")
    chosen = frame.loc[frame.model_id.isin(model_ids)].copy()
    if set(chosen.model_id) != model_ids:
        raise ValueError(f"{source}: missing model IDs {sorted(model_ids-set(chosen.model_id))}")
    if chosen.duplicated(["model_id", *FOLD]).any():
        raise ValueError(f"{source}: repeated model/repeat/fold metrics")
    expected = {(repeat, fold) for repeat in range(5) for fold in range(5)}
    for model_id, block in chosen.groupby("model_id"):
        observed = set(block[FOLD].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"{source}: {model_id} is missing outer folds")
    return chosen


def check_predictions(
    predictions: pd.DataFrame,
    nested_clinical: pd.DataFrame,
    manifest_test: pd.DataFrame,
    model_ids: set[str],
    source: Path,
) -> None:
    """Check every model's held-out biopsy, time point, and observed outcome."""

    required = {"model_id", *TEST, "egfr"}
    if not required.issubset(predictions):
        raise ValueError(f"{source}: missing prediction columns {sorted(required-set(predictions))}")
    expected_biopsies = set(manifest_test.itertuples(index=False, name=None))
    reference = nested_clinical[[*TEST, "egfr"]]
    if reference.duplicated(TEST).any():
        raise ValueError("Nested clinical predictions have duplicate test observations")
    for model_id in sorted(model_ids | {"clinical_full"}):
        observed = predictions.loc[predictions.model_id.eq(model_id), [*TEST, "egfr"]]
        if observed.empty or observed.duplicated(TEST).any():
            raise ValueError(f"{source}: {model_id} lacks unique test observations")
        biopsies = set(observed[TEST[:-1]].drop_duplicates().itertuples(index=False, name=None))
        if biopsies != expected_biopsies:
            raise ValueError(f"{source}: {model_id} has different outer test biopsies")
        paired = observed.merge(
            reference, on=TEST, how="outer", validate="one_to_one", indicator=True,
            suffixes=("_old", "_nested"),
        )
        if not paired._merge.eq("both").all() or not np.allclose(
            paired.egfr_old, paired.egfr_nested, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"{source}: {model_id} has different eGFR test observations")


def check_checkpoint(model: dict, nested_identity: dict, source: Path) -> None:
    """Verify the fresh frozen run used the checkpoint recorded by nested CV."""

    checkpoint = Path(model["checkpoint"]).expanduser().resolve()
    nested_path = Path(nested_identity["path"]).expanduser().resolve()
    if checkpoint != nested_path:
        raise ValueError(f"{source}: checkpoint path changed: {checkpoint} != {nested_path}")
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        int(nested_identity["size"]), int(nested_identity["mtime_ns"])
    ):
        raise ValueError(f"{source}: checkpoint changed since the nested evaluation: {checkpoint}")
    for field in ("run", "method", "architecture", "latent_dim"):
        if str(model[field]) != str(nested_identity[field]):
            raise ValueError(f"{source}: checkpoint {field} differs for {checkpoint}")
    if not np.isclose(
        float(model["validation_selection_score"]),
        float(nested_identity["molecular_selection_mse"]),
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError(f"{source}: checkpoint molecular-validation score changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--nested-root", type=Path, default=None)
    parser.add_argument(
        "--frozen-root", action="append", type=Path, required=True,
        help="Completed frozen-encoder evaluation; provide three, one per preprocessing.",
    )
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    nested_root = (args.nested_root or project / "evaluations/egfr-nested-infonce-seed-0").resolve()
    if len(args.frozen_root) != 3:
        parser.error("Provide exactly three --frozen-root values")

    complete = json_file(nested_root / "nested_command_summary.json")
    if complete.get("complete") is not True:
        raise ValueError("Nested evaluation has not completed")
    comparison = csv(nested_root / "final_egfr_comparison.csv")
    if len(comparison) != 36 or comparison.model_id.duplicated().any():
        raise ValueError("Expected 36 distinct completed nested model summaries")
    if not comparison.n_folds.eq(25).all():
        raise ValueError("Nested summaries do not all contain 25 outer folds")
    catalog = csv(nested_root / "nested_models/nested_model_catalog.csv")
    catalog = catalog.set_index("model_id", verify_integrity=True)
    contract = json_file(nested_root / "nested_models/nested_evaluation_contract.json")
    checkpoint_by_arm = {
        item["model_id"]: item["checkpoint"] for item in contract["encoders"]
    }
    if len(checkpoint_by_arm) != 6:
        raise ValueError("Expected six distinct nested source checkpoints")
    manifest = csv(nested_root / "nested_split_manifest.csv")
    manifest_test = manifest.loc[manifest.outer_role.eq("test")].rename(
        columns={"outer_repeat": "repeat", "outer_fold": "fold"}
    )[[*FOLD, "patient", "donor"]].drop_duplicates()
    nested_metrics_path = nested_root / "nested_models/nested_outer_fold_metrics.csv"
    nested_metrics = csv(nested_metrics_path)
    nested_metrics = model_folds(
        nested_metrics,
        {"clinical_full", *(arm + suffix for arm in checkpoint_by_arm for suffix in ARM_SUFFIXES)},
        nested_metrics_path,
    )
    nested_clinical = csv(
        nested_root / "nested_models/nested_outer_oof_predictions.csv",
        columns=["model_id", *TEST, "egfr"],
    ).query('model_id == "clinical_full"')
    if nested_clinical.empty:
        raise ValueError("Nested clinical reference has no predictions")

    frozen_rows = []
    paired_rows = []
    seen_arms: set[str] = set()
    for requested in args.frozen_root:
        root = requested.expanduser().resolve()
        mixed = root / "mixed_models"
        arm_summary = csv(mixed / "fixed_encoder_cv_summary.csv")
        if len(arm_summary) != 2 or arm_summary.model_id.duplicated().any():
            raise ValueError(f"{root}: expected two frozen architecture winners")
        arm_ids = set(arm_summary.model_id)
        if seen_arms & arm_ids:
            raise ValueError(f"{root}: duplicate architecture/preprocessing arm")
        seen_arms.update(arm_ids)
        expected_ids = {arm + suffix for arm in arm_ids for suffix in ARM_SUFFIXES}
        embedding_models = json_file(root / "embeddings/metadata.json")["models"]
        embeddings = {item["model_id"]: item for item in embedding_models}
        if len(embeddings) != 2 or set(embeddings) != arm_ids:
            raise ValueError(f"{root}: frozen embeddings and selected arms differ")
        for arm_id in arm_ids:
            if arm_id not in checkpoint_by_arm:
                raise ValueError(f"{root}: arm absent from nested evaluation: {arm_id}")
            check_checkpoint(embeddings[arm_id], checkpoint_by_arm[arm_id], root)

        metric_path = mixed / "fold_metrics_with_deltas.csv"
        metrics = csv(metric_path)
        frozen = model_folds(metrics, expected_ids | {"clinical_full"}, metric_path)
        comparison_path = mixed / "all_model_cv_summary.csv"
        summaries = csv(comparison_path).set_index("model_id", verify_integrity=True)
        if not expected_ids.issubset(summaries.index):
            raise ValueError(f"{root}: missing frozen mixed-model summaries")
        prediction_candidates = (
            mixed / "combined_fixed_oof_predictions.csv",
            mixed / "fixed_oof_predictions.csv",
        )
        prediction_path = next((p for p in prediction_candidates if p.is_file()), None)
        if prediction_path is None:
            raise FileNotFoundError(f"{root}: missing frozen out-of-fold predictions")
        predictions = csv(prediction_path, columns=["model_id", *TEST, "egfr"])
        check_predictions(predictions, nested_clinical, manifest_test, expected_ids, prediction_path)

        clinical_old = frozen.loc[frozen.model_id.eq("clinical_full"), [*FOLD, "rmse"]]
        clinical_new = nested_metrics.loc[
            nested_metrics.model_id.eq("clinical_full"), [*FOLD, "rmse"]
        ]
        clinical_pair = clinical_old.merge(
            clinical_new, on=FOLD, suffixes=("_frozen", "_nested"), validate="one_to_one"
        )
        if len(clinical_pair) != 25 or not np.allclose(
            clinical_pair.rmse_frozen, clinical_pair.rmse_nested, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"{root}: matched full-clinical baseline RMSE changed")
        clinical = clinical_new.rename(columns={"rmse": "clinical_rmse"})

        for model_id in sorted(expected_ids):
            metrics_old = frozen.loc[frozen.model_id.eq(model_id), [*FOLD, "rmse"]]
            metrics_new = nested_metrics.loc[
                nested_metrics.model_id.eq(model_id), [*FOLD, "rmse"]
            ]
            pair = metrics_old.merge(
                metrics_new, on=FOLD, suffixes=("_frozen", "_finetuned"),
                validate="one_to_one",
            )
            if len(pair) != 25:
                raise ValueError(f"{root}: missing paired folds for {model_id}")
            gain = pair.rmse_frozen - pair.rmse_finetuned
            base_id = next(arm for arm in arm_ids if model_id in {arm+s for s in ARM_SUFFIXES})
            identity = embeddings[base_id]
            summary = summaries.loc[model_id].to_dict()
            if not np.isclose(float(summary["mean_rmse"]), float(metrics_old.rmse.mean())):
                raise ValueError(f"{root}: frozen RMSE summary and folds disagree for {model_id}")
            improvement_over_clinical = metrics_old.merge(
                clinical, on=FOLD, validate="one_to_one"
            )
            summary.update({
                "model_id": model_id,
                "comparison_id": "frozen__" + model_id,
                "evaluation_stage": "completed_frozen_checkpoint",
                "model_family": "frozen_pretrained_encoder",
                "encoder_state": "pretrained checkpoint; no eGFR fine-tuning",
                "preprocessing": catalog.at[model_id, "preprocessing"],
                "architecture": catalog.at[model_id, "architecture"],
                "variant": catalog.at[model_id, "variant"],
                "source_run": identity["run"],
                "mean_rmse_gain_vs_full_clinical": float(
                    (improvement_over_clinical.clinical_rmse - improvement_over_clinical.rmse).mean()
                ),
                "fraction_folds_better_than_full_clinical": float(
                    (improvement_over_clinical.clinical_rmse > improvement_over_clinical.rmse).mean()
                ),
                "source_evaluation": str(root),
                "matched_nested_folds": True,
                "matched_nested_checkpoint": True,
            })
            frozen_rows.append(summary)
            paired_rows.append({
                "model_id": model_id,
                "preprocessing": summary["preprocessing"],
                "architecture": summary["architecture"],
                "variant": summary["variant"],
                "source_run": identity["run"],
                "mean_rmse_frozen": float(pair.rmse_frozen.mean()),
                "mean_rmse_finetuned": float(pair.rmse_finetuned.mean()),
                "mean_rmse_gain_of_finetuning": float(gain.mean()),
                "fraction_folds_finetuned_better": float((gain > 0).mean()),
                "n_matched_folds": len(pair),
            })

    if seen_arms != set(checkpoint_by_arm) or len(frozen_rows) != 18:
        raise ValueError("Expected all six checkpoint-matched encoder arms and 18 frozen models")
    comparison = comparison.copy()
    comparison["comparison_id"] = "nested__" + comparison.model_id.astype(str)
    comparison["evaluation_stage"] = "completed_nested"
    comparison["encoder_state"] = np.where(
        comparison.model_family.eq("nested_finetuned_encoder"),
        "50-epoch IKEM fine-tuning; reconstruction or soft-InfoNCE selected inside fold",
        "not applicable",
    )
    comparison["matched_nested_folds"] = True
    comparison["matched_nested_checkpoint"] = pd.NA
    comparison["source_evaluation"] = str(nested_root)
    table = pd.concat([comparison, pd.DataFrame(frozen_rows)], ignore_index=True, sort=False)
    if len(table) != 54 or table.comparison_id.duplicated().any():
        raise ValueError("Expected 36 nested plus 18 frozen distinct comparison rows")
    out = nested_root / "all_completed_egfr_models.csv"
    paired_out = nested_root / "paired_frozen_vs_finetuned.csv"
    table.to_csv(out, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_out, index=False)
    print(f"Combined 36 nested and 18 checkpoint-matched frozen results: {out}")
    print(f"Paired 25-fold frozen-versus-fine-tuned differences: {paired_out}")
    print("Verified outer biopsy/time/eGFR identities and matched clinical baselines.")


if __name__ == "__main__":
    main()
