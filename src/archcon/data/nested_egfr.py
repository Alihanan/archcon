"""Resumable nested eGFR evaluation for fine-tuned encoders and input PCA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .contrastive_egfr import (
    STADNIUK_CONTRASTIVE_COLUMNS,
    ContrastiveCandidate,
    FineTuneControls,
    checkpoint_identity,
    fine_tune_checkpoint,
)
from .defaults import ProjectDataLayout
from .downstream import (
    CLINICAL_COLUMNS,
    ValidationCheckpoint,
    egfr_long,
    run_lme4_benchmark,
)
from .loading import normalize_sample_id
from .nested_cv import manifest_sha256, nested_folds_from_manifest
from .supervised import EGFR_COLUMNS, IKEM_ROLE_TRAIN


NESTED_EVALUATION_FORMAT = 1


@dataclass(frozen=True)
class EncoderArm:
    """One molecular-validation-frozen encoder entering eGFR fine-tuning."""

    model_id: str
    label: str
    record: ValidationCheckpoint


@dataclass(frozen=True)
class MethodExpression:
    """One frozen preprocessing arm containing all aligned IKEM biopsies."""

    method: str
    matrix: np.ndarray
    samples: pd.DataFrame


@dataclass(frozen=True)
class NestedEvaluationResult:
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    encoder_selections_path: Path
    pca_selections_path: Path
    inner_scores_path: Path
    model_catalog_path: Path


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return result or "model"


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray, *, rows_per_block: int = 16) -> str:
    """Hash a potentially memory-mapped matrix without making a second full copy."""

    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    for start in range(0, len(array), int(rows_per_block)):
        block = np.ascontiguousarray(array[start : start + int(rows_per_block)])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    encoded = selected.to_csv(index=False, na_rep="NA", float_format="%.17g")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _parse_feature_table(
    path: Path,
    sample_ids: Iterable[str],
    columns: Iterable[str],
) -> pd.DataFrame:
    """Load ordered numeric features for measured and no-eGFR IKEM biopsies."""

    feature_columns = tuple(dict.fromkeys(map(str, columns)))
    if not feature_columns:
        raise ValueError("At least one contrastive clinical feature is required.")
    frame = pd.read_excel(path).copy()
    id_candidates = ("Sample_ID", "sample_id", "patient", "Patient", "id", "ID")
    id_column = next((column for column in id_candidates if column in frame.columns), None)
    if id_column is None:
        raise ValueError(f"Could not identify sample IDs in clinical workbook: {path}")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Clinical workbook lacks contrastive columns: {missing}")
    frame["sample_id"] = frame[id_column].map(normalize_sample_id).str.upper()
    if frame["sample_id"].duplicated().any():
        duplicate = frame.loc[frame["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"IKEM clinical workbook has duplicate sample {duplicate}.")
    wanted = [normalize_sample_id(value).upper() for value in sample_ids]
    indexed = frame.set_index("sample_id")
    absent = [value for value in wanted if value not in indexed.index]
    if absent:
        raise ValueError(f"Clinical workbook lacks molecular samples: {absent[:5]}")
    result = indexed.loc[wanted, list(feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    entirely_missing = result.columns[result.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(
            f"Contrastive clinical features contain no numeric values: {entirely_missing}"
        )
    result.insert(0, "sample_id", wanted)
    return result.reset_index(drop=True)


def _aligned_expression(
    method: MethodExpression,
    sample_ids: Iterable[str],
) -> np.ndarray:
    ids = method.samples["sample_id"].astype(str).str.upper()
    if ids.duplicated().any():
        raise ValueError(f"{method.method} has duplicate IKEM sample IDs.")
    lookup = {value: index for index, value in enumerate(ids)}
    wanted = [normalize_sample_id(value).upper() for value in sample_ids]
    absent = [value for value in wanted if value not in lookup]
    if absent:
        raise ValueError(f"{method.method} lacks eGFR samples: {absent[:5]}")
    rows = np.asarray([lookup[value] for value in wanted], dtype=np.int64)
    return np.ascontiguousarray(method.matrix[rows], dtype=np.float32)


def _training_role_ids(method: MethodExpression, role: str) -> list[str]:
    if "training_role" not in method.samples.columns:
        raise ValueError(
            f"{method.method} sample index lacks audited training_role metadata."
        )
    mask = method.samples["training_role"].astype(str).eq(role)
    ids = method.samples.loc[mask, "sample_id"].astype(str).str.upper().tolist()
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"{method.method} has invalid {role!r} identities.")
    return ids


def _standardize_from_train(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    train64 = np.asarray(train, dtype=np.float64)
    mean = train64.mean(axis=0)
    scale = train64.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    result = [((train64 - mean) / scale).astype(np.float32)]
    result.extend(
        ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)
        for values in others
    )
    return tuple(result)


def _clinical_from_train(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    train64 = np.asarray(train, dtype=np.float64)
    medians = np.nanmedian(train64, axis=0)
    if np.isnan(medians).any():
        bad = np.flatnonzero(np.isnan(medians)).tolist()
        raise ValueError(f"Clinical training fold has entirely missing columns: {bad}")
    imputed_train = np.where(np.isnan(train64), medians, train64)
    mean = imputed_train.mean(axis=0)
    scale = imputed_train.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    def transform(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        imputed = np.where(np.isnan(array), medians, array)
        return ((imputed - mean) / scale).astype(np.float32)

    return (transform(train64), *(transform(values) for values in others))


def _pca_candidates(
    expression: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    dimensions: tuple[int, ...],
    *,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    scaled_train, scaled_validation = _standardize_from_train(
        expression[train_indices], expression[validation_indices]
    )
    maximum = min(max(dimensions), len(train_indices) - 1, expression.shape[1])
    if maximum < 1:
        raise ValueError("PCA training partition is too small.")
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=int(seed))
    full_train = pca.fit_transform(scaled_train)
    full_validation = pca.transform(scaled_validation)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dimension in dimensions:
        used = min(int(dimension), maximum)
        train, validation = _standardize_from_train(
            full_train[:, :used], full_validation[:, :used]
        )
        result[int(dimension)] = (train, validation)
    return result


class MixedModelDesign:
    """Small builder for the CSV contract consumed by molecular_mixed_models.R."""

    def __init__(self, patients: pd.DataFrame) -> None:
        self.patients = patients.reset_index(drop=True).copy()
        self.long = egfr_long(self.patients)
        self.rows: list[pd.DataFrame] = []
        self.specifications: list[dict[str, object]] = []

    def add_fit(
        self,
        *,
        fit_id: str,
        stage: str,
        model_id: str,
        model_label: str,
        candidate_id: str,
        repeat: int,
        fold: int,
        inner_fold: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        train_main: np.ndarray,
        test_main: np.ndarray,
        train_interactions: np.ndarray | None = None,
        test_interactions: np.ndarray | None = None,
        feature_description: str = "",
    ) -> None:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        train_main = np.asarray(train_main, dtype=np.float32)
        test_main = np.asarray(test_main, dtype=np.float32)
        if train_main.ndim != 2 or test_main.ndim != 2:
            raise ValueError("Mixed-model main features must be matrices.")
        if train_main.shape[0] != len(train_indices) or test_main.shape[0] != len(test_indices):
            raise ValueError("Mixed-model main feature rows do not match identities.")
        if train_interactions is None:
            train_interactions = np.empty((len(train_indices), 0), dtype=np.float32)
        if test_interactions is None:
            test_interactions = np.empty((len(test_indices), 0), dtype=np.float32)
        train_interactions = np.asarray(train_interactions, dtype=np.float32)
        test_interactions = np.asarray(test_interactions, dtype=np.float32)
        if (
            train_interactions.shape[0] != len(train_indices)
            or test_interactions.shape[0] != len(test_indices)
        ):
            raise ValueError("Mixed-model interaction feature rows do not match identities.")
        if train_main.shape[1] != test_main.shape[1] or (
            train_interactions.shape[1] != test_interactions.shape[1]
        ):
            raise ValueError("Train/test mixed-model feature dimensions differ.")
        train_features = np.column_stack((train_main, train_interactions))
        test_features = np.column_stack((test_main, test_interactions))
        n_main = int(train_main.shape[1])
        n_interactions = int(train_interactions.shape[1])
        self.specifications.append(
            {
                "fit_id": fit_id,
                "stage": stage,
                "model_id": model_id,
                "model_label": model_label,
                "candidate_id": candidate_id,
                "repeat": int(repeat),
                "fold": int(fold),
                "inner_fold": int(inner_fold),
                "n_features": n_main + n_interactions,
                "n_main_features": n_main,
                "n_time_interaction_features": n_interactions,
                "feature_description": feature_description,
            }
        )
        for partition, indices, features in (
            ("train", train_indices, train_features),
            ("test", test_indices, test_features),
        ):
            patient_ids = self.patients.iloc[indices]["patient"].astype(str).tolist()
            block = self.long.loc[
                self.long["patient"].astype(str).isin(patient_ids),
                ["patient", "donor", "time", "egfr"],
            ].copy()
            feature_lookup = {
                str(self.patients.iloc[index]["patient"]): features[position]
                for position, index in enumerate(indices)
            }
            block.insert(0, "partition", partition)
            block.insert(0, "fit_id", fit_id)
            block.insert(0, "fold", int(fold))
            block.insert(0, "repeat", int(repeat))
            block.insert(0, "model_id", model_id)
            for feature_index in range(features.shape[1]):
                block[f"x{feature_index + 1}"] = [
                    float(feature_lookup[str(patient)][feature_index])
                    for patient in block["patient"]
                ]
            self.rows.append(block)

    def write(self, root: Path, prefix: str) -> tuple[Path, Path]:
        if not self.rows or not self.specifications:
            raise ValueError("Cannot write an empty mixed-model design.")
        design = pd.concat(self.rows, ignore_index=True, sort=False)
        specs = pd.DataFrame(self.specifications)
        if specs["fit_id"].duplicated().any():
            raise RuntimeError("Mixed-model fit IDs are not unique.")
        design_path = root / f"{prefix}_design.csv"
        specs_path = root / f"{prefix}_specs.csv"
        _atomic_csv(design, design_path)
        _atomic_csv(specs, specs_path)
        return design_path, specs_path


def _cache_fine_tune(
    cache_root: Path,
    *,
    record: ValidationCheckpoint,
    train_expression: np.ndarray,
    train_clinical: np.ndarray,
    clinical_reference: np.ndarray,
    train_ids: list[str],
    encode_expression: np.ndarray,
    encode_ids: list[str],
    external_ids: list[str],
    candidate: ContrastiveCandidate,
    controls: FineTuneControls,
    device: str,
    seed: int,
) -> np.ndarray:
    cache_root.mkdir(parents=True, exist_ok=True)
    array_path = cache_root / "embedding.npz"
    metadata_path = cache_root / "metadata.json"
    contract = {
        "format": NESTED_EVALUATION_FORMAT,
        "checkpoint": checkpoint_identity(record),
        "candidate": asdict(candidate),
        "controls": asdict(controls),
        "seed": int(seed),
        "train_ids": list(train_ids),
        "encode_ids": list(encode_ids),
        "external_no_egfr_ids": list(external_ids),
        "clinical_reference_ids": list(train_ids[: len(clinical_reference)]),
    }
    if array_path.is_file() or metadata_path.is_file():
        if not array_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Incomplete fine-tuning cache: {cache_root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract") != contract:
            raise RuntimeError(
                f"Fine-tuning cache contract changed: {cache_root}. Use a new output root."
            )
        with np.load(array_path, allow_pickle=False) as saved:
            saved_ids = saved["sample_id"].astype(str).tolist()
            if saved_ids != encode_ids:
                raise RuntimeError(f"Fine-tuning cache identities changed: {cache_root}")
            print(f"    cache: {cache_root.name}", flush=True)
            return np.asarray(saved["z"], dtype=np.float32)
    print(
        f"    fine-tune: {candidate.candidate_id} · {len(train_expression)} samples · "
        f"{controls.epochs} epochs",
        flush=True,
    )
    result = fine_tune_checkpoint(
        record,
        train_expression,
        train_clinical,
        clinical_reference,
        encode_expression,
        candidate,
        controls,
        device=device,
        seed=seed,
    )
    _atomic_npz(
        array_path,
        z=result.z,
        sample_id=np.asarray(encode_ids, dtype=str),
    )
    _atomic_json({"contract": contract, "result": result.metadata()}, metadata_path)
    print(
        f"      final reconstruction={result.reconstruction_history[-1]:.6g}; "
        f"contrastive={result.contrastive_history[-1]:.6g}",
        flush=True,
    )
    return result.z


def _empty(n_rows: int) -> np.ndarray:
    return np.empty((int(n_rows), 0), dtype=np.float32)


def _selection_row(
    metrics: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    group_column: str,
    group_value: str,
) -> pd.Series:
    candidates = catalog.loc[catalog[group_column].astype(str).eq(str(group_value))].copy()
    scored = candidates.merge(
        metrics.loc[:, ["model_id", "rmse"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    if scored["rmse"].isna().any():
        missing = scored.loc[scored["rmse"].isna(), "model_id"].tolist()
        raise RuntimeError(f"Inner mixed models omitted candidates: {missing}")
    return scored.sort_values(["rmse", "candidate_order", "model_id"]).iloc[0]


def _summarize_outer(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    key = ["repeat", "fold"]
    if metrics.duplicated(["model_id", *key]).any():
        raise ValueError("Outer mixed-model metrics contain duplicate model/fold rows.")
    time = metrics.loc[metrics["model_id"].eq("time_only"), key + ["rmse"]].rename(
        columns={"rmse": "time_rmse"}
    )
    if time.duplicated(key).any() or len(time) != metrics[key].drop_duplicates().shape[0]:
        raise ValueError("Every outer fold must contain exactly one time-only fit.")
    merged = metrics.merge(time, on=key, validate="many_to_one")
    merged["delta_vs_time"] = merged["time_rmse"] - merged["rmse"]
    pooled = (
        predictions.assign(
            squared_error=(predictions["egfr"] - predictions["prediction"]) ** 2
        )
        .groupby("model_id", as_index=False)["squared_error"]
        .mean()
    )
    pooled["pooled_rmse"] = np.sqrt(pooled["squared_error"])
    summary = (
        merged.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_delta_vs_time=("delta_vs_time", "mean"),
            sd_delta_vs_time=("delta_vs_time", "std"),
            positive_folds_vs_time=(
                "delta_vs_time", lambda values: float((values > 0).mean())
            ),
            n_folds=("rmse", "count"),
        )
        .merge(pooled[["model_id", "pooled_rmse"]], on="model_id", how="left")
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
    metadata = catalog.drop_duplicates("model_id")
    summary = summary.merge(metadata, on="model_id", how="left", validate="one_to_one")
    return summary.sort_values(["mean_rmse", "model_id"]).reset_index(drop=True)


def evaluate_nested_egfr(
    layout: ProjectDataLayout,
    patients: pd.DataFrame,
    split_manifest: pd.DataFrame,
    encoder_arms: Iterable[EncoderArm],
    methods: Mapping[str, MethodExpression],
    candidates: tuple[ContrastiveCandidate, ...],
    controls: FineTuneControls,
    pca_dimensions: tuple[int, ...],
    output_root: str | Path,
    r_script: str | Path,
    *,
    rscript: str = "Rscript",
    device: str = "cpu",
    seed: int = 0,
    expected_external_no_egfr: int | None = 24,
    contrastive_columns: tuple[str, ...] = STADNIUK_CONTRASTIVE_COLUMNS,
) -> NestedEvaluationResult:
    """Run inner selection and outer refit/evaluation for every frozen encoder."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    arms = tuple(encoder_arms)
    if not arms:
        raise ValueError("At least one frozen encoder arm is required.")
    if len({arm.model_id for arm in arms}) != len(arms):
        raise ValueError("Encoder model IDs must be unique.")
    if any(arm.record.method not in methods for arm in arms):
        raise ValueError("Every encoder needs a method-matched IKEM matrix.")
    if not pca_dimensions or any(int(value) < 1 for value in pca_dimensions):
        raise ValueError("PCA dimensions must be positive and non-empty.")
    for candidate in candidates:
        candidate.validate()
    controls.validate()

    patients = patients.reset_index(drop=True).copy()
    patient_ids = patients["patient"].astype(str).str.upper().tolist()
    patient_clinical = patients.loc[:, list(CLINICAL_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    folds = nested_folds_from_manifest(patients, split_manifest)
    method_patient: dict[str, np.ndarray] = {}
    method_external: dict[str, np.ndarray] = {}
    external_ids: list[str] | None = None
    for method_name, method in methods.items():
        current_external_ids = _training_role_ids(method, IKEM_ROLE_TRAIN)
        if external_ids is None:
            external_ids = current_external_ids
        elif current_external_ids != external_ids:
            raise RuntimeError("No-eGFR reconstruction identities differ by preprocessing.")
        method_patient[method_name] = _aligned_expression(method, patient_ids)
        method_external[method_name] = _aligned_expression(method, current_external_ids)
    assert external_ids is not None
    if expected_external_no_egfr is not None and len(external_ids) != int(
        expected_external_no_egfr
    ):
        raise RuntimeError(
            f"Expected {expected_external_no_egfr} no-eGFR training biopsies, "
            f"found {len(external_ids)}."
        )
    contrastive_columns = tuple(dict.fromkeys(map(str, contrastive_columns)))
    contrastive_table = _parse_feature_table(
        layout.clinical_table,
        [*patient_ids, *external_ids],
        contrastive_columns,
    )
    contrastive_values = contrastive_table.loc[
        :, list(contrastive_columns)
    ].to_numpy(dtype=np.float64)
    patient_contrastive = contrastive_values[: len(patient_ids)]
    external_contrastive = contrastive_values[len(patient_ids) :]

    outcome_columns = [column for column in EGFR_COLUMNS if column in patients.columns]
    patient_contract_columns = [
        "patient",
        "donor",
        *outcome_columns,
        *CLINICAL_COLUMNS,
    ]
    method_contracts = {}
    for name, value in methods.items():
        sample_columns = [
            column
            for column in (
                "sample_id",
                "donor_id",
                "training_role",
                "pretraining_split",
                "split_unit",
            )
            if column in value.samples.columns
        ]
        method_contracts[name] = {
            "shape": list(value.matrix.shape),
            "dtype": str(value.matrix.dtype),
            "matrix_sha256": _array_sha256(value.matrix),
            "sample_index_sha256": _frame_sha256(value.samples, sample_columns),
        }
    resolved_r_script = Path(r_script).expanduser().resolve()
    if not resolved_r_script.is_file():
        raise FileNotFoundError(f"Mixed-model R helper is missing: {resolved_r_script}")
    contract = {
        "format": NESTED_EVALUATION_FORMAT,
        "patient_ids": patient_ids,
        "manifest_sha256": manifest_sha256(split_manifest),
        "encoders": [
            {"model_id": arm.model_id, "checkpoint": checkpoint_identity(arm.record)}
            for arm in arms
        ],
        "methods": method_contracts,
        "external_no_egfr_ids": external_ids,
        "contrastive_clinical_source": str(layout.clinical_table),
        "contrastive_clinical_columns": list(contrastive_columns),
        "measured_contrastive_clinical_sha256": _array_sha256(patient_contrastive),
        "external_contrastive_clinical_sha256": _array_sha256(external_contrastive),
        "patient_values_sha256": _frame_sha256(patients, patient_contract_columns),
        "candidates": [asdict(candidate) for candidate in candidates],
        "fine_tune_controls": asdict(controls),
        "pca_dimensions": list(map(int, pca_dimensions)),
        "seed": int(seed),
        "selection_model": "full clinical + molecular representation",
        "outer_predictions": "population-level random effects excluded",
        "frozen_molecular_preprocessing": True,
        "r_script": str(resolved_r_script),
        "r_script_sha256": _file_sha256(resolved_r_script),
        "rscript_executable": str(rscript),
    }
    contract["contract_sha256"] = _json_hash(contract)
    contract_path = root / "nested_evaluation_contract.json"
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError(
                "Nested eGFR output root belongs to a different contract. "
                "Use a new output directory."
            )
    else:
        _atomic_json(contract, contract_path)

    all_outer_metrics: list[pd.DataFrame] = []
    all_outer_predictions: list[pd.DataFrame] = []
    all_encoder_selections: list[pd.DataFrame] = []
    all_pca_selections: list[pd.DataFrame] = []
    all_inner_scores: list[pd.DataFrame] = []
    catalog_rows: list[dict[str, object]] = [
        {
            "model_id": "time_only",
            "model_family": "clinical_baseline",
            "preprocessing": pd.NA,
            "architecture": pd.NA,
            "variant": "time only",
        },
        {
            "model_id": "clinical_kdri",
            "model_family": "clinical_baseline",
            "preprocessing": pd.NA,
            "architecture": pd.NA,
            "variant": "KDRI",
        },
        {
            "model_id": "clinical_full",
            "model_family": "clinical_baseline",
            "preprocessing": pd.NA,
            "architecture": pd.NA,
            "variant": "full clinical",
        },
    ]
    # Build the complete catalog before inspecting fold checkpoints.  A fully
    # resumed evaluation must produce the same metadata as a fresh run even
    # when every fold takes the early ``continue`` path below.
    for arm in arms:
        encoder_metadata = {
            "model_family": "nested_finetuned_encoder",
            "preprocessing": arm.record.method,
            "architecture": arm.record.architecture,
            "source_run": arm.record.run,
            "latent_dim": int(arm.record.latent_dim),
            "molecular_validation_selection_score": (
                arm.record.molecular_selection_mse
            ),
            "geo_test_mse_post_freeze": arm.record.molecular_test_mse,
        }
        catalog_rows.extend(
            [
                {"model_id": arm.model_id, "variant": "z only", **encoder_metadata},
                {
                    "model_id": f"{arm.model_id}_kdri",
                    "variant": "z + KDRI",
                    **encoder_metadata,
                },
                {
                    "model_id": f"{arm.model_id}_clinical",
                    "variant": "z + full clinical",
                    **encoder_metadata,
                },
            ]
        )
    for method_name in method_patient:
        pca_metadata = {
            "model_family": "nested_selected_pca",
            "preprocessing": method_name,
            "architecture": pd.NA,
            "source_run": pd.NA,
            "latent_dim": pd.NA,
            "molecular_validation_selection_score": pd.NA,
            "geo_test_mse_post_freeze": pd.NA,
        }
        base_id = f"pca_{_slug(method_name)}"
        catalog_rows.extend(
            [
                {"model_id": base_id, "variant": "PCA only", **pca_metadata},
                {
                    "model_id": f"{base_id}_kdri",
                    "variant": "PCA + KDRI",
                    **pca_metadata,
                },
                {
                    "model_id": f"{base_id}_clinical",
                    "variant": "PCA + full clinical",
                    **pca_metadata,
                },
            ]
        )

    for fold_position, fold in enumerate(folds):
        fold_name = f"r{fold.outer_repeat:03d}_f{fold.outer_fold:03d}"
        fold_root = root / "parts" / fold_name
        fold_root.mkdir(parents=True, exist_ok=True)
        completion_path = fold_root / "complete.json"
        required_outputs = {
            "outer_metrics": fold_root / "outer_fold_metrics.csv",
            "outer_predictions": fold_root / "outer_oof_predictions.csv",
            "encoder_selections": fold_root / "encoder_selections.csv",
            "pca_selections": fold_root / "pca_selections.csv",
            "inner_scores": fold_root / "inner_candidate_scores.csv",
        }
        if completion_path.is_file() and all(path.is_file() for path in required_outputs.values()):
            completed = json.loads(completion_path.read_text(encoding="utf-8"))
            if completed.get("contract_sha256") != contract["contract_sha256"]:
                raise RuntimeError(f"Completed fold has a stale contract: {fold_root}")
            all_outer_metrics.append(pd.read_csv(required_outputs["outer_metrics"]))
            all_outer_predictions.append(pd.read_csv(required_outputs["outer_predictions"]))
            all_encoder_selections.append(pd.read_csv(required_outputs["encoder_selections"]))
            all_pca_selections.append(pd.read_csv(required_outputs["pca_selections"]))
            all_inner_scores.append(pd.read_csv(required_outputs["inner_scores"]))
            continue

        print(
            f"[{fold_position + 1}/{len(folds)}] outer repeat {fold.outer_repeat}, "
            f"fold {fold.outer_fold}: inner candidate selection",
            flush=True,
        )
        inner_design = MixedModelDesign(patients)
        inner_catalog: list[dict[str, object]] = []
        inner_train_clinical, inner_validation_clinical = _clinical_from_train(
            patient_clinical[fold.inner_train], patient_clinical[fold.inner_validation]
        )
        inner_encode_indices = np.concatenate((fold.inner_train, fold.inner_validation))
        inner_encode_ids = [patient_ids[index] for index in inner_encode_indices]
        for arm_index, arm in enumerate(arms):
            measured = method_patient[arm.record.method]
            training_expression = np.concatenate(
                (measured[fold.inner_train], method_external[arm.record.method]), axis=0
            )
            training_clinical = np.concatenate(
                (patient_contrastive[fold.inner_train], external_contrastive), axis=0
            )
            training_ids = [patient_ids[index] for index in fold.inner_train] + external_ids
            for candidate_order, candidate in enumerate(candidates):
                training_seed = (
                    int(seed)
                    + fold.outer_repeat * 100_003
                    + fold.outer_fold * 2_003
                    + arm_index * 101
                )
                cache = (
                    fold_root
                    / "inner_embeddings"
                    / arm.model_id
                    / candidate.candidate_id
                )
                z = _cache_fine_tune(
                    cache,
                    record=arm.record,
                    train_expression=training_expression,
                    train_clinical=training_clinical,
                    clinical_reference=patient_contrastive[fold.inner_train],
                    train_ids=training_ids,
                    encode_expression=measured[inner_encode_indices],
                    encode_ids=inner_encode_ids,
                    external_ids=external_ids,
                    candidate=candidate,
                    controls=controls,
                    device=device,
                    seed=training_seed,
                )
                n_train = len(fold.inner_train)
                train_z, validation_z = _standardize_from_train(
                    z[:n_train], z[n_train:]
                )
                model_id = f"{arm.model_id}__{candidate.candidate_id}"
                inner_design.add_fit(
                    fit_id=f"inner_{fold_name}_{_slug(model_id)}",
                    stage="inner_selection",
                    model_id=model_id,
                    model_label=f"{arm.label} · {candidate.candidate_id}",
                    candidate_id=arm.model_id,
                    repeat=fold.outer_repeat,
                    fold=fold.outer_fold,
                    inner_fold=0,
                    train_indices=fold.inner_train,
                    test_indices=fold.inner_validation,
                    train_main=train_z,
                    test_main=validation_z,
                    train_interactions=inner_train_clinical,
                    test_interactions=inner_validation_clinical,
                    feature_description="fine-tuned z + full clinical × time",
                )
                inner_catalog.append(
                    {
                        "model_id": model_id,
                        "candidate_type": "encoder",
                        "encoder_id": arm.model_id,
                        "preprocessing": arm.record.method,
                        "architecture": arm.record.architecture,
                        "contrastive_mode": candidate.mode,
                        "contrastive_weight": candidate.weight,
                        "temperature": candidate.temperature,
                        "pca_dimension": pd.NA,
                        "candidate_order": candidate_order,
                    }
                )

        for method_name, expression in method_patient.items():
            pca = _pca_candidates(
                expression,
                fold.inner_train,
                fold.inner_validation,
                pca_dimensions,
                seed=int(seed) + fold.outer_repeat * 101 + fold.outer_fold,
            )
            for candidate_order, dimension in enumerate(pca_dimensions):
                train_pca, validation_pca = pca[int(dimension)]
                model_id = f"pca_{_slug(method_name)}__d{int(dimension)}"
                inner_design.add_fit(
                    fit_id=f"inner_{fold_name}_{model_id}",
                    stage="inner_selection",
                    model_id=model_id,
                    model_label=f"{method_name} PCA({dimension}) + full clinical",
                    candidate_id=f"pca_{_slug(method_name)}",
                    repeat=fold.outer_repeat,
                    fold=fold.outer_fold,
                    inner_fold=0,
                    train_indices=fold.inner_train,
                    test_indices=fold.inner_validation,
                    train_main=train_pca,
                    test_main=validation_pca,
                    train_interactions=inner_train_clinical,
                    test_interactions=inner_validation_clinical,
                    feature_description="fold-fitted PCA + full clinical × time",
                )
                inner_catalog.append(
                    {
                        "model_id": model_id,
                        "candidate_type": "pca",
                        "encoder_id": f"pca_{_slug(method_name)}",
                        "preprocessing": method_name,
                        "architecture": pd.NA,
                        "contrastive_mode": pd.NA,
                        "contrastive_weight": pd.NA,
                        "temperature": pd.NA,
                        "pca_dimension": int(dimension),
                        "candidate_order": candidate_order,
                    }
                )

        inner_design_path, inner_specs_path = inner_design.write(fold_root, "inner")
        inner_metrics_path, _ = run_lme4_benchmark(
            inner_design_path,
            inner_specs_path,
            r_script,
            fold_root,
            rscript=rscript,
            output_prefix="inner_",
        )
        inner_metrics = pd.read_csv(inner_metrics_path)
        inner_catalog_frame = pd.DataFrame(inner_catalog)
        inner_scores = inner_catalog_frame.merge(
            inner_metrics.loc[:, ["model_id", "rmse", "mae", "singular"]],
            on="model_id",
            validate="one_to_one",
        )
        inner_scores.insert(0, "outer_fold", fold.outer_fold)
        inner_scores.insert(0, "outer_repeat", fold.outer_repeat)
        _atomic_csv(inner_scores, required_outputs["inner_scores"])

        encoder_selection_rows: list[dict[str, object]] = []
        selected_encoder_candidates: dict[str, ContrastiveCandidate] = {}
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
        for arm in arms:
            selected = _selection_row(
                inner_metrics,
                inner_catalog_frame,
                group_column="encoder_id",
                group_value=arm.model_id,
            )
            candidate_name = str(selected["model_id"]).split("__", 1)[1]
            chosen = candidate_lookup[candidate_name]
            selected_encoder_candidates[arm.model_id] = chosen
            encoder_selection_rows.append(
                {
                    "outer_repeat": fold.outer_repeat,
                    "outer_fold": fold.outer_fold,
                    "encoder_id": arm.model_id,
                    "preprocessing": arm.record.method,
                    "architecture": arm.record.architecture,
                    "selected_mode": chosen.mode,
                    "selected_contrastive_weight": chosen.weight,
                    "selected_temperature": chosen.temperature,
                    "inner_validation_rmse": float(selected["rmse"]),
                    "selection_model": "full clinical + z",
                }
            )
        encoder_selections = pd.DataFrame(encoder_selection_rows)
        _atomic_csv(encoder_selections, required_outputs["encoder_selections"])
        for row in encoder_selection_rows:
            print(
                f"    selected {row['encoder_id']}: {row['selected_mode']} "
                f"(weight={row['selected_contrastive_weight']:.6g}, "
                f"temperature={row['selected_temperature']:.6g}, "
                f"inner RMSE={row['inner_validation_rmse']:.6g})",
                flush=True,
            )

        pca_selection_rows: list[dict[str, object]] = []
        selected_pca_dimensions: dict[str, int] = {}
        for method_name in method_patient:
            group = f"pca_{_slug(method_name)}"
            selected = _selection_row(
                inner_metrics,
                inner_catalog_frame,
                group_column="encoder_id",
                group_value=group,
            )
            dimension = int(selected["pca_dimension"])
            selected_pca_dimensions[method_name] = dimension
            pca_selection_rows.append(
                {
                    "outer_repeat": fold.outer_repeat,
                    "outer_fold": fold.outer_fold,
                    "preprocessing": method_name,
                    "selected_pca_dimension": dimension,
                    "inner_validation_rmse": float(selected["rmse"]),
                    "selection_model": "full clinical + PCA",
                }
            )
        pca_selections = pd.DataFrame(pca_selection_rows)
        _atomic_csv(pca_selections, required_outputs["pca_selections"])
        for row in pca_selection_rows:
            print(
                f"    selected {row['preprocessing']} PCA dimension "
                f"{row['selected_pca_dimension']} "
                f"(inner RMSE={row['inner_validation_rmse']:.6g})",
                flush=True,
            )

        print(
            f"[{fold_position + 1}/{len(folds)}] outer repeat {fold.outer_repeat}, "
            f"fold {fold.outer_fold}: refit selected models and evaluate outer test",
            flush=True,
        )
        outer_design = MixedModelDesign(patients)
        outer_train_clinical, outer_test_clinical = _clinical_from_train(
            patient_clinical[fold.outer_train], patient_clinical[fold.outer_test]
        )
        kdri_index = list(CLINICAL_COLUMNS).index("KDRI_8")
        outer_design.add_fit(
            fit_id=f"outer_{fold_name}_time_only",
            stage="outer_evaluation",
            model_id="time_only",
            model_label="Time only",
            candidate_id="",
            repeat=fold.outer_repeat,
            fold=fold.outer_fold,
            inner_fold=-1,
            train_indices=fold.outer_train,
            test_indices=fold.outer_test,
            train_main=_empty(len(fold.outer_train)),
            test_main=_empty(len(fold.outer_test)),
        )
        outer_design.add_fit(
            fit_id=f"outer_{fold_name}_clinical_kdri",
            stage="outer_evaluation",
            model_id="clinical_kdri",
            model_label="KDRI × time",
            candidate_id="",
            repeat=fold.outer_repeat,
            fold=fold.outer_fold,
            inner_fold=-1,
            train_indices=fold.outer_train,
            test_indices=fold.outer_test,
            train_main=_empty(len(fold.outer_train)),
            test_main=_empty(len(fold.outer_test)),
            train_interactions=outer_train_clinical[:, [kdri_index]],
            test_interactions=outer_test_clinical[:, [kdri_index]],
            feature_description="KDRI_8",
        )
        outer_design.add_fit(
            fit_id=f"outer_{fold_name}_clinical_full",
            stage="outer_evaluation",
            model_id="clinical_full",
            model_label="Clinical (KDRI + donor age + cold ischemia) × time",
            candidate_id="",
            repeat=fold.outer_repeat,
            fold=fold.outer_fold,
            inner_fold=-1,
            train_indices=fold.outer_train,
            test_indices=fold.outer_test,
            train_main=_empty(len(fold.outer_train)),
            test_main=_empty(len(fold.outer_test)),
            train_interactions=outer_train_clinical,
            test_interactions=outer_test_clinical,
            feature_description=" + ".join(CLINICAL_COLUMNS),
        )

        outer_encode_indices = np.concatenate((fold.outer_train, fold.outer_test))
        outer_encode_ids = [patient_ids[index] for index in outer_encode_indices]
        for arm_index, arm in enumerate(arms):
            measured = method_patient[arm.record.method]
            chosen = selected_encoder_candidates[arm.model_id]
            training_expression = np.concatenate(
                (measured[fold.outer_train], method_external[arm.record.method]), axis=0
            )
            training_clinical = np.concatenate(
                (patient_contrastive[fold.outer_train], external_contrastive), axis=0
            )
            training_ids = [patient_ids[index] for index in fold.outer_train] + external_ids
            refit_seed = (
                int(seed)
                + 500_009
                + fold.outer_repeat * 100_003
                + fold.outer_fold * 2_003
                + arm_index * 101
            )
            cache = fold_root / "outer_embeddings" / arm.model_id / chosen.candidate_id
            z = _cache_fine_tune(
                cache,
                record=arm.record,
                train_expression=training_expression,
                train_clinical=training_clinical,
                clinical_reference=patient_contrastive[fold.outer_train],
                train_ids=training_ids,
                encode_expression=measured[outer_encode_indices],
                encode_ids=outer_encode_ids,
                external_ids=external_ids,
                candidate=chosen,
                controls=controls,
                device=device,
                seed=refit_seed,
            )
            n_train = len(fold.outer_train)
            train_z, test_z = _standardize_from_train(z[:n_train], z[n_train:])
            variants = (
                (arm.model_id, arm.label + " · nested fine-tuned z", None, None, "z only"),
                (
                    f"{arm.model_id}_kdri",
                    arm.label + " · nested fine-tuned z + KDRI",
                    outer_train_clinical[:, [kdri_index]],
                    outer_test_clinical[:, [kdri_index]],
                    "z + KDRI",
                ),
                (
                    f"{arm.model_id}_clinical",
                    arm.label + " · nested fine-tuned z + full clinical",
                    outer_train_clinical,
                    outer_test_clinical,
                    "z + full clinical",
                ),
            )
            for model_id, label, interaction_train, interaction_test, variant in variants:
                outer_design.add_fit(
                    fit_id=f"outer_{fold_name}_{_slug(model_id)}",
                    stage="outer_evaluation",
                    model_id=model_id,
                    model_label=label,
                    candidate_id=arm.model_id,
                    repeat=fold.outer_repeat,
                    fold=fold.outer_fold,
                    inner_fold=-1,
                    train_indices=fold.outer_train,
                    test_indices=fold.outer_test,
                    train_main=train_z,
                    test_main=test_z,
                    train_interactions=interaction_train,
                    test_interactions=interaction_test,
                    feature_description=variant,
                )

        for method_name, expression in method_patient.items():
            dimension = selected_pca_dimensions[method_name]
            outer_pca = _pca_candidates(
                expression,
                fold.outer_train,
                fold.outer_test,
                (dimension,),
                seed=int(seed) + 700_001 + fold.outer_repeat * 101 + fold.outer_fold,
            )[dimension]
            train_pca, test_pca = outer_pca
            base_id = f"pca_{_slug(method_name)}"
            variants = (
                (base_id, f"{method_name} · nested-selected PCA", None, None, "PCA only"),
                (
                    f"{base_id}_kdri",
                    f"{method_name} · nested-selected PCA + KDRI",
                    outer_train_clinical[:, [kdri_index]],
                    outer_test_clinical[:, [kdri_index]],
                    "PCA + KDRI",
                ),
                (
                    f"{base_id}_clinical",
                    f"{method_name} · nested-selected PCA + full clinical",
                    outer_train_clinical,
                    outer_test_clinical,
                    "PCA + full clinical",
                ),
            )
            for model_id, label, interaction_train, interaction_test, variant in variants:
                outer_design.add_fit(
                    fit_id=f"outer_{fold_name}_{model_id}",
                    stage="outer_evaluation",
                    model_id=model_id,
                    model_label=label,
                    candidate_id=base_id,
                    repeat=fold.outer_repeat,
                    fold=fold.outer_fold,
                    inner_fold=-1,
                    train_indices=fold.outer_train,
                    test_indices=fold.outer_test,
                    train_main=train_pca,
                    test_main=test_pca,
                    train_interactions=interaction_train,
                    test_interactions=interaction_test,
                    feature_description=f"selected PCA({dimension}); {variant}",
                )

        outer_design_path, outer_specs_path = outer_design.write(fold_root, "outer")
        outer_metrics_path, outer_predictions_path = run_lme4_benchmark(
            outer_design_path,
            outer_specs_path,
            r_script,
            fold_root,
            rscript=rscript,
            output_prefix="outer_",
        )
        # Normalize to stable filenames independent of run_lme4_benchmark naming.
        outer_metrics = pd.read_csv(outer_metrics_path)
        outer_predictions = pd.read_csv(outer_predictions_path)
        _atomic_csv(outer_metrics, required_outputs["outer_metrics"])
        _atomic_csv(outer_predictions, required_outputs["outer_predictions"])
        _atomic_json(
            {
                "contract_sha256": contract["contract_sha256"],
                "outer_repeat": fold.outer_repeat,
                "outer_fold": fold.outer_fold,
                "complete": True,
            },
            completion_path,
        )
        all_outer_metrics.append(outer_metrics)
        all_outer_predictions.append(outer_predictions)
        all_encoder_selections.append(encoder_selections)
        all_pca_selections.append(pca_selections)
        all_inner_scores.append(inner_scores)

    outer_metrics = pd.concat(all_outer_metrics, ignore_index=True, sort=False)
    outer_predictions = pd.concat(all_outer_predictions, ignore_index=True, sort=False)
    encoder_selections = pd.concat(all_encoder_selections, ignore_index=True, sort=False)
    pca_selections = pd.concat(all_pca_selections, ignore_index=True, sort=False)
    inner_scores = pd.concat(all_inner_scores, ignore_index=True, sort=False)
    catalog = pd.DataFrame(catalog_rows).drop_duplicates("model_id").sort_values("model_id")
    expected_fold_count = len(folds)
    observed_counts = outer_metrics.groupby("model_id").size()
    incomplete = observed_counts.loc[observed_counts.ne(expected_fold_count)]
    if not incomplete.empty:
        raise RuntimeError(f"Outer model fold counts are incomplete: {incomplete.to_dict()}")

    metrics_path = root / "nested_outer_fold_metrics.csv"
    predictions_path = root / "nested_outer_oof_predictions.csv"
    encoder_selections_path = root / "nested_encoder_selections.csv"
    pca_selections_path = root / "nested_pca_selections.csv"
    inner_scores_path = root / "nested_inner_candidate_scores.csv"
    model_catalog_path = root / "nested_model_catalog.csv"
    _atomic_csv(outer_metrics, metrics_path)
    _atomic_csv(outer_predictions, predictions_path)
    _atomic_csv(encoder_selections, encoder_selections_path)
    _atomic_csv(pca_selections, pca_selections_path)
    _atomic_csv(inner_scores, inner_scores_path)
    _atomic_csv(catalog, model_catalog_path)
    summary = _summarize_outer(outer_metrics, outer_predictions, catalog)
    summary_path = root / "nested_model_summary.csv"
    _atomic_csv(summary, summary_path)
    return NestedEvaluationResult(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        encoder_selections_path=encoder_selections_path,
        pca_selections_path=pca_selections_path,
        inner_scores_path=inner_scores_path,
        model_catalog_path=model_catalog_path,
    )
