"""Leakage-safe probe-level LASSO linear mixed-model baseline.

The model is a Gaussian random-intercept linear mixed model

    eGFR = categorical time + probe expression + patient intercept + error

with an L1 penalty on probe coefficients only.  The random-intercept variance
ratio is estimated from training data, its marginal covariance is whitened,
and the unpenalized time effects are projected out before a coordinate-descent
LASSO path is fitted.  Penalty fractions are selected by inner donor-grouped
cross-validation inside every outer donor-grouped evaluation fold.

This module deliberately contains no neural-network code.  It is a direct
all-probe baseline, fitted independently for each frozen preprocessing arm.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from .downstream import TIME_LEVELS, egfr_long, repeated_donor_folds


@dataclass(frozen=True)
class ProbeLassoConfig:
    """Numerical and nested-CV controls for the all-probe baseline."""

    alpha_fractions: tuple[float, ...] = (
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
    )
    inner_splits: int = 5
    max_iter: int = 5_000
    tolerance: float = 1e-4

    def validate(self) -> None:
        if self.inner_splits < 2:
            raise ValueError("Probe-LASSO inner_splits must be at least two.")
        fractions = np.asarray(self.alpha_fractions, dtype=np.float64)
        if (
            fractions.ndim != 1
            or len(fractions) < 2
            or not np.isfinite(fractions).all()
            or np.any(fractions <= 0.0)
            or np.any(fractions > 1.0)
        ):
            raise ValueError("alpha_fractions must be finite values in (0, 1].")
        if np.any(np.diff(fractions) >= 0.0):
            raise ValueError("alpha_fractions must be strictly decreasing.")
        if self.max_iter < 1 or self.tolerance <= 0.0:
            raise ValueError("Invalid coordinate-descent controls.")


@dataclass(frozen=True)
class ProbeExpressionArm:
    """One method-specific IKEM matrix in canonical checkpoint probe order."""

    method: str
    matrix: np.ndarray
    samples: pd.DataFrame
    probe_ids: tuple[str, ...] | None = None


def _slug(value: str) -> str:
    result = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in result.split("_") if part)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _time_design(long_frame: pd.DataFrame) -> np.ndarray:
    """Return intercept plus three treatment-coded time indicators."""

    time = pd.Categorical(long_frame["time"], categories=TIME_LEVELS, ordered=True)
    codes = np.asarray(time.codes, dtype=np.int64)
    if np.any(codes < 0):
        raise ValueError("Longitudinal eGFR contains an unknown time level.")
    design = np.ones((len(long_frame), len(TIME_LEVELS)), dtype=np.float64)
    for column, level in enumerate(range(1, len(TIME_LEVELS)), start=1):
        design[:, column] = codes == level
    return design


def _expand_expression(
    patient_expression: np.ndarray,
    patients: pd.DataFrame,
    long_frame: pd.DataFrame,
) -> np.ndarray:
    lookup = {patient: index for index, patient in enumerate(patients["patient"].astype(str))}
    indices = np.asarray([lookup[value] for value in long_frame["patient"].astype(str)], dtype=np.int64)
    return np.asarray(patient_expression[indices], dtype=np.float64)


def _standardize_from_train(
    matrix: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(matrix[train_indices], dtype=np.float64)
    center = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = ((np.asarray(matrix, dtype=np.float64) - center) / scale).astype(
        np.float32
    )
    if not np.isfinite(standardized).all():
        raise ValueError("Probe expression contains non-finite values after fold-local scaling.")
    return standardized, center.astype(np.float32), scale.astype(np.float32)


def _whiten_random_intercept(
    values: np.ndarray,
    groups: Sequence[str],
    variance_ratio: float,
) -> np.ndarray:
    """Apply V^(-1/2) for V = I + variance_ratio * Z Z'."""

    result = np.asarray(values, dtype=np.float64).copy()
    group_values = np.asarray(groups, dtype=str)
    for group in np.unique(group_values):
        rows = np.flatnonzero(group_values == group)
        coefficient = (1.0 / math.sqrt(1.0 + variance_ratio * len(rows)) - 1.0) / len(rows)
        block = result[rows]
        result[rows] = block + coefficient * block.sum(axis=0, keepdims=True)
    return result


def _profile_variance_objective(
    log_ratio: float,
    y: np.ndarray,
    time_design: np.ndarray,
    groups: np.ndarray,
) -> float:
    ratio = math.exp(float(log_ratio))
    yw = _whiten_random_intercept(y[:, None], groups, ratio)[:, 0]
    tw = _whiten_random_intercept(time_design, groups, ratio)
    coefficients = np.linalg.lstsq(tw, yw, rcond=None)[0]
    residual = yw - tw @ coefficients
    degrees = max(len(yw) - np.linalg.matrix_rank(tw), 1)
    rss = max(float(residual @ residual), np.finfo(float).tiny)
    log_determinant = sum(
        math.log1p(ratio * int(count))
        for count in pd.Series(groups).value_counts().to_numpy()
    )
    information = tw.T @ tw
    sign, logdet_information = np.linalg.slogdet(information)
    if sign <= 0:
        return float("inf")
    return degrees * math.log(rss / degrees) + log_determinant + logdet_information


def _estimate_variance_ratio(
    y: np.ndarray,
    time_design: np.ndarray,
    groups: np.ndarray,
) -> float:
    """One-dimensional REML profile search, using training rows only."""

    lower, upper = -9.0, 9.0
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_value = _profile_variance_objective(left, y, time_design, groups)
    right_value = _profile_variance_objective(right, y, time_design, groups)
    for _ in range(64):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - golden * (upper - lower)
            left_value = _profile_variance_objective(left, y, time_design, groups)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + golden * (upper - lower)
            right_value = _profile_variance_objective(right, y, time_design, groups)
    return math.exp((lower + upper) / 2.0)


def _residualize_unpenalized(
    y: np.ndarray,
    expression: np.ndarray,
    time_design: np.ndarray,
    groups: np.ndarray,
    variance_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yw = _whiten_random_intercept(y[:, None], groups, variance_ratio)[:, 0]
    xw = _whiten_random_intercept(expression, groups, variance_ratio)
    tw = _whiten_random_intercept(time_design, groups, variance_ratio)
    inverse = np.linalg.pinv(tw.T @ tw)
    projection_y = inverse @ (tw.T @ yw)
    projection_x = inverse @ (tw.T @ xw)
    return yw - tw @ projection_y, xw - tw @ projection_x, tw, yw


def _fit_path(
    y: np.ndarray,
    expression: np.ndarray,
    time_design: np.ndarray,
    groups: np.ndarray,
    config: ProbeLassoConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    ratio = _estimate_variance_ratio(y, time_design, groups)
    residual_y, residual_x, _, _ = _residualize_unpenalized(
        y, expression, time_design, groups, ratio
    )
    alpha_max = float(np.max(np.abs(residual_x.T @ residual_y)) / max(len(y), 1))
    alpha_max = max(alpha_max, np.finfo(float).eps)
    requested = alpha_max * np.asarray(config.alpha_fractions, dtype=np.float64)
    returned, coefficients, _ = lasso_path(
        residual_x,
        residual_y,
        alphas=requested,
        max_iter=config.max_iter,
        tol=config.tolerance,
    )
    if not np.allclose(returned, requested, rtol=1e-10, atol=0.0):
        raise RuntimeError("scikit-learn returned an unexpected LASSO penalty grid.")
    return coefficients, requested, ratio


def _time_coefficients(
    y: np.ndarray,
    expression: np.ndarray,
    time_design: np.ndarray,
    groups: np.ndarray,
    variance_ratio: float,
    probe_coefficients: np.ndarray,
) -> np.ndarray:
    yw = _whiten_random_intercept(y[:, None], groups, variance_ratio)[:, 0]
    xw = _whiten_random_intercept(expression, groups, variance_ratio)
    tw = _whiten_random_intercept(time_design, groups, variance_ratio)
    return np.linalg.lstsq(tw, yw - xw @ probe_coefficients, rcond=None)[0]


def _donor_balanced_rmse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = frame.loc[:, ["donor", "egfr"]].copy()
    scored["squared_error"] = (scored["egfr"].to_numpy(dtype=float) - prediction) ** 2
    return float(np.sqrt(scored.groupby("donor")["squared_error"].mean().mean()))


def _aligned_patient_expression(
    arm: ProbeExpressionArm,
    patients: pd.DataFrame,
) -> np.ndarray:
    ids = arm.samples["sample_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{arm.method} contains duplicate normalized IKEM sample IDs.")
    lookup = {sample_id: row for row, sample_id in enumerate(ids)}
    missing = [patient for patient in patients["patient"].astype(str) if patient not in lookup]
    if missing:
        raise ValueError(f"{arm.method} is missing eGFR molecular rows: {missing[:5]}")
    rows = np.asarray([lookup[value] for value in patients["patient"].astype(str)], dtype=np.int64)
    matrix = np.asarray(arm.matrix[rows], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError(f"{arm.method} has an invalid all-probe expression matrix.")
    if arm.probe_ids is not None and len(arm.probe_ids) != matrix.shape[1]:
        raise ValueError(f"{arm.method} probe identifiers do not match its matrix width.")
    return matrix


def evaluate_probe_lasso_mixed_models(
    egfr_wide: pd.DataFrame,
    arms: Mapping[str, ProbeExpressionArm],
    output_root: str | Path,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    stratify_column: str | None = "KDRI_8",
    config: ProbeLassoConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested donor-CV once per preprocessing strategy and save results."""

    if not arms:
        raise ValueError("At least one preprocessing arm is required for probe LASSO.")
    config = ProbeLassoConfig() if config is None else config
    config.validate()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    patients = egfr_wide.reset_index(drop=True).copy()
    if patients["patient"].duplicated().any():
        raise ValueError("Probe-LASSO input must contain one row per biopsy.")
    long_all = egfr_long(patients)
    outer_folds = repeated_donor_folds(
        patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        stratify_column=stratify_column,
    )
    matrices = {method: _aligned_patient_expression(arm, patients) for method, arm in arms.items()}
    resume_contract = {
        "patients": patients["patient"].astype(str).tolist(),
        "donors": patients["donor"].astype(str).tolist(),
        "arms": {method: list(matrix.shape) for method, matrix in matrices.items()},
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "stratify_column": stratify_column,
        "config": {
            "alpha_fractions": list(config.alpha_fractions),
            "inner_splits": config.inner_splits,
            "max_iter": config.max_iter,
            "tolerance": config.tolerance,
        },
    }
    contract_hash = hashlib.sha256(
        json.dumps(resume_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    parts_root = root / "parts" / contract_hash
    parts_root.mkdir(parents=True, exist_ok=True)

    fold_assignments: list[dict[str, object]] = []
    for repeat, fold, train_indices, test_indices in outer_folds:
        for partition, indices in (("train", train_indices), ("test", test_indices)):
            for index in indices:
                fold_assignments.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "partition": partition,
                        "patient": str(patients.iloc[index]["patient"]),
                        "donor": str(patients.iloc[index]["donor"]),
                    }
                )

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for repeat, fold, outer_train, outer_test in outer_folds:
        outer_train_patients = patients.iloc[outer_train].reset_index(drop=True)
        inner_count = min(config.inner_splits, outer_train_patients["donor"].nunique())
        if inner_count < 2:
            raise ValueError("An outer fold has fewer than two donor groups for tuning.")
        inner_folds = repeated_donor_folds(
            outer_train_patients,
            n_splits=inner_count,
            n_repeats=1,
            seed=seed + 10_000 * repeat + 101 * fold + 1,
            stratify_column=stratify_column,
        )

        for method, arm in arms.items():
            part_name = f"r{repeat:03d}_f{fold:03d}_{_slug(method)}"
            metric_part = parts_root / f"{part_name}.metric.json"
            prediction_part = parts_root / f"{part_name}.predictions.csv"
            coefficient_part = parts_root / f"{part_name}.coefficients.csv"
            tuning_part = parts_root / f"{part_name}.tuning.csv"
            if all(
                path.is_file()
                for path in (
                    metric_part,
                    prediction_part,
                    coefficient_part,
                    tuning_part,
                )
            ):
                fold_rows.append(json.loads(metric_part.read_text(encoding="utf-8")))
                prediction_rows.append(pd.read_csv(prediction_part))
                saved_coefficients = pd.read_csv(coefficient_part)
                coefficient_rows.extend(saved_coefficients.to_dict("records"))
                tuning_rows.extend(pd.read_csv(tuning_part).to_dict("records"))
                continue

            tuning_start = len(tuning_rows)
            matrix = matrices[method]
            inner_scores = np.zeros((len(inner_folds), len(config.alpha_fractions)))
            for inner_number, (_, _, inner_train_local, inner_test_local) in enumerate(inner_folds):
                inner_train = outer_train[inner_train_local]
                inner_test = outer_train[inner_test_local]
                standardized, _, _ = _standardize_from_train(matrix, inner_train)
                inner_train_ids = set(patients.iloc[inner_train]["patient"].astype(str))
                inner_test_ids = set(patients.iloc[inner_test]["patient"].astype(str))
                train_long = long_all.loc[long_all["patient"].isin(inner_train_ids)].reset_index(drop=True)
                test_long = long_all.loc[long_all["patient"].isin(inner_test_ids)].reset_index(drop=True)
                train_expression = _expand_expression(standardized, patients, train_long)
                test_expression = _expand_expression(standardized, patients, test_long)
                train_time = _time_design(train_long)
                test_time = _time_design(test_long)
                train_y = train_long["egfr"].to_numpy(dtype=np.float64)
                train_groups = train_long["patient"].astype(str).to_numpy()
                path, alphas, ratio = _fit_path(
                    train_y, train_expression, train_time, train_groups, config
                )
                for alpha_index in range(path.shape[1]):
                    beta = path[:, alpha_index]
                    gamma = _time_coefficients(
                        train_y,
                        train_expression,
                        train_time,
                        train_groups,
                        ratio,
                        beta,
                    )
                    prediction = test_time @ gamma + test_expression @ beta
                    score = _donor_balanced_rmse(test_long, prediction)
                    inner_scores[inner_number, alpha_index] = score
                    tuning_rows.append(
                        {
                            "preprocessing": method,
                            "repeat": repeat,
                            "fold": fold,
                            "inner_fold": inner_number,
                            "alpha_fraction": config.alpha_fractions[alpha_index],
                            "alpha": alphas[alpha_index],
                            "donor_balanced_rmse": score,
                            "n_nonzero": int(np.count_nonzero(beta)),
                        }
                    )

            mean_scores = inner_scores.mean(axis=0)
            selected_index = int(np.argmin(mean_scores))
            selected_fraction = config.alpha_fractions[selected_index]

            standardized, _, _ = _standardize_from_train(matrix, outer_train)
            train_ids = set(patients.iloc[outer_train]["patient"].astype(str))
            test_ids = set(patients.iloc[outer_test]["patient"].astype(str))
            train_long = long_all.loc[long_all["patient"].isin(train_ids)].reset_index(drop=True)
            test_long = long_all.loc[long_all["patient"].isin(test_ids)].reset_index(drop=True)
            train_expression = _expand_expression(standardized, patients, train_long)
            test_expression = _expand_expression(standardized, patients, test_long)
            train_time = _time_design(train_long)
            test_time = _time_design(test_long)
            train_y = train_long["egfr"].to_numpy(dtype=np.float64)
            train_groups = train_long["patient"].astype(str).to_numpy()
            path, alphas, ratio = _fit_path(
                train_y, train_expression, train_time, train_groups, config
            )
            beta = path[:, selected_index]
            gamma = _time_coefficients(
                train_y, train_expression, train_time, train_groups, ratio, beta
            )
            prediction = test_time @ gamma + test_expression @ beta
            errors = test_long["egfr"].to_numpy(dtype=float) - prediction
            nonzero = np.flatnonzero(beta)

            train_prediction = train_time @ gamma + train_expression @ beta
            train_error = train_y - train_prediction
            marginal_scale = max(float(np.mean(train_error**2)), np.finfo(float).tiny)
            degrees = len(gamma) + len(nonzero) + 2
            bic = len(train_y) * math.log(marginal_scale) + degrees * math.log(len(train_y))
            model_id = f"probe_lasso_{_slug(method)}"
            fold_rows.append(
                {
                    "model_id": model_id,
                    "model_label": f"All-probe LASSO mixed model - {method}",
                    "preprocessing": method,
                    "repeat": repeat,
                    "fold": fold,
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "donor_balanced_rmse": _donor_balanced_rmse(test_long, prediction),
                    "mae": float(np.mean(np.abs(errors))),
                    "selected_alpha_fraction": selected_fraction,
                    "selected_alpha": float(alphas[selected_index]),
                    "inner_cv_rmse": float(mean_scores[selected_index]),
                    "n_nonzero": len(nonzero),
                    "random_intercept_variance_ratio": ratio,
                    "training_bic_diagnostic": bic,
                    "n_train_patients": len(outer_train),
                    "n_test_patients": len(outer_test),
                    "n_train_donors": patients.iloc[outer_train]["donor"].nunique(),
                    "n_test_donors": patients.iloc[outer_test]["donor"].nunique(),
                }
            )
            predictions = test_long.loc[:, ["patient", "donor", "time", "egfr"]].copy()
            predictions.insert(0, "fold", fold)
            predictions.insert(0, "repeat", repeat)
            predictions.insert(0, "preprocessing", method)
            predictions.insert(0, "model_id", model_id)
            predictions["prediction"] = prediction
            prediction_rows.append(predictions)

            probe_ids = arm.probe_ids
            for probe_index in nonzero:
                coefficient_rows.append(
                    {
                        "preprocessing": method,
                        "repeat": repeat,
                        "fold": fold,
                        "probe_index_python": int(probe_index),
                        "probe_id": (
                            probe_ids[probe_index]
                            if probe_ids is not None
                            else str(probe_index)
                        ),
                        "coefficient": float(beta[probe_index]),
                    }
                )

            coefficient_columns = (
                "preprocessing",
                "repeat",
                "fold",
                "probe_index_python",
                "probe_id",
                "coefficient",
            )
            current_coefficients = pd.DataFrame(
                [
                    row
                    for row in coefficient_rows
                    if row["preprocessing"] == method
                    and row["repeat"] == repeat
                    and row["fold"] == fold
                ],
                columns=coefficient_columns,
            )
            _atomic_csv(predictions, prediction_part)
            _atomic_csv(current_coefficients, coefficient_part)
            _atomic_csv(pd.DataFrame(tuning_rows[tuning_start:]), tuning_part)
            _atomic_json(fold_rows[-1], metric_part)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    coefficients = pd.DataFrame(
        coefficient_rows,
        columns=(
            "preprocessing",
            "repeat",
            "fold",
            "probe_index_python",
            "probe_id",
            "coefficient",
        ),
    )
    tuning = pd.DataFrame(tuning_rows)
    summary = (
        fold_metrics.groupby(["model_id", "model_label", "preprocessing"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_donor_balanced_rmse=("donor_balanced_rmse", "mean"),
            mean_mae=("mae", "mean"),
            median_nonzero=("n_nonzero", "median"),
            n_folds=("rmse", "count"),
        )
    )
    summary["se_rmse"] = summary["sd_rmse"] / np.sqrt(summary["n_folds"])
    summary["mean_rmse_ci95_low"] = summary["mean_rmse"] - 1.96 * summary["se_rmse"]
    summary["mean_rmse_ci95_high"] = summary["mean_rmse"] + 1.96 * summary["se_rmse"]

    _atomic_csv(fold_metrics, root / "fold_metrics.csv")
    _atomic_csv(predictions, root / "oof_predictions.csv")
    _atomic_csv(coefficients, root / "selected_probe_coefficients.csv")
    _atomic_csv(tuning, root / "inner_cv_tuning.csv")
    _atomic_csv(summary, root / "summary.csv")
    _atomic_csv(pd.DataFrame(fold_assignments), root / "outer_fold_assignments.csv")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": 1,
                "model": "Gaussian L1-penalized marginal linear mixed model",
                "random_intercept": "patient/biopsy",
                "unpenalized_fixed_effects": "categorical time",
                "penalized_features": "all aligned microarray probes",
                "outer_split": "repeated donor-grouped CV shared across preprocessing arms",
                "tuning": "inner donor-grouped CV within each outer training fold",
                "selection_metric": "donor-balanced RMSE",
                "bic_policy": "training-fold diagnostic only; never used on outer test data",
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "seed": seed,
                "stratify_column": stratify_column,
                "config": {
                    "alpha_fractions": list(config.alpha_fractions),
                    "inner_splits": config.inner_splits,
                    "max_iter": config.max_iter,
                    "tolerance": config.tolerance,
                },
                "preprocessing_arms": list(arms),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return fold_metrics, summary


def summarize_probe_lasso_against_time(
    lasso_metrics: pd.DataFrame,
    mixed_model_metrics: pd.DataFrame,
    output_root: str | Path,
) -> pd.DataFrame:
    """Create matched-fold comparisons against the common lme4 time-only model."""

    required_lasso = {
        "model_id",
        "model_label",
        "preprocessing",
        "repeat",
        "fold",
        "rmse",
        "mae",
    }
    missing = required_lasso.difference(lasso_metrics.columns)
    if missing:
        raise ValueError(f"Probe-LASSO metrics are missing columns: {sorted(missing)}")
    required_mixed = {"model_id", "repeat", "fold", "rmse"}
    missing = required_mixed.difference(mixed_model_metrics.columns)
    if missing:
        raise ValueError(f"Mixed-model metrics are missing columns: {sorted(missing)}")
    time = mixed_model_metrics.loc[
        mixed_model_metrics["model_id"].eq("time_only"),
        ["repeat", "fold", "rmse"],
    ].rename(columns={"rmse": "time_only_rmse"})
    if time.duplicated(["repeat", "fold"]).any() or time.empty:
        raise ValueError("The mixed-model output has invalid time-only fold rows.")
    matched = lasso_metrics.merge(
        time, on=["repeat", "fold"], how="left", validate="many_to_one"
    )
    if matched["time_only_rmse"].isna().any():
        raise ValueError("Probe-LASSO folds do not match the lme4 time-only folds.")
    matched["delta_vs_time_only"] = matched["time_only_rmse"] - matched["rmse"]
    summary = (
        matched.groupby(["model_id", "model_label", "preprocessing"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_delta_vs_time=("delta_vs_time_only", "mean"),
            sd_delta_vs_time=("delta_vs_time_only", "std"),
            positive_folds_vs_time=(
                "delta_vs_time_only", lambda values: float((values > 0).mean())
            ),
            n_folds=("rmse", "count"),
        )
    )
    summary["se_rmse"] = summary["sd_rmse"] / np.sqrt(summary["n_folds"])
    summary["mean_rmse_ci95_low"] = summary["mean_rmse"] - 1.96 * summary["se_rmse"]
    summary["mean_rmse_ci95_high"] = summary["mean_rmse"] + 1.96 * summary["se_rmse"]
    summary["se_delta_vs_time"] = summary["sd_delta_vs_time"] / np.sqrt(
        summary["n_folds"]
    )
    summary["delta_vs_time_ci95_low"] = (
        summary["mean_delta_vs_time"] - 1.96 * summary["se_delta_vs_time"]
    )
    summary["delta_vs_time_ci95_high"] = (
        summary["mean_delta_vs_time"] + 1.96 * summary["se_delta_vs_time"]
    )
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(matched, root / "fold_metrics_with_time_comparison.csv")
    _atomic_csv(summary, root / "summary_with_time_comparison.csv")
    return summary
