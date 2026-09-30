"""Persisted donor-grouped train/validation/test partitions for eGFR models.

The molecular preprocessing reference is deliberately absent from this module:
RMA and the outcome-blind IKEM standardization are frozen before these roles are
created.  This file only assigns measured-eGFR donors to downstream modelling
roles and provides one auditable manifest shared by neural fine-tuning, PCA,
probe LASSO, and longitudinal mixed models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .downstream import repeated_donor_folds


MANIFEST_FORMAT = 1
OUTER_TRAIN = "train"
OUTER_TEST = "test"
INNER_TRAIN = "train"
INNER_VALIDATION = "validation"
INNER_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class NestedSplitConfig:
    """Controls for one reusable nested donor-CV manifest."""

    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 5
    seed: int = 0
    stratify_column: str | None = "KDRI_8"

    def validate(self) -> None:
        if int(self.outer_splits) < 2:
            raise ValueError("outer_splits must be at least two.")
        if int(self.outer_repeats) < 1:
            raise ValueError("outer_repeats must be positive.")
        if int(self.inner_splits) < 2:
            raise ValueError("inner_splits must be at least two.")


@dataclass(frozen=True)
class NestedFold:
    """Patient-row indices for one outer fold and its single inner holdout."""

    outer_repeat: int
    outer_fold: int
    inner_train: np.ndarray
    inner_validation: np.ndarray
    outer_train: np.ndarray
    outer_test: np.ndarray


def _patient_contract(patients: pd.DataFrame) -> list[dict[str, str]]:
    required = {"patient", "donor"}
    missing = required.difference(patients.columns)
    if missing:
        raise ValueError(f"eGFR cohort lacks split columns: {sorted(missing)}")
    if patients["patient"].astype(str).duplicated().any():
        raise ValueError("Nested split input must contain one row per biopsy/patient.")
    if patients[["patient", "donor"]].isna().any().any():
        raise ValueError("Nested split identities cannot be missing.")
    return (
        patients.loc[:, ["patient", "donor"]]
        .astype(str)
        .sort_values(["donor", "patient"])
        .to_dict("records")
    )


def _contract_payload(
    patients: pd.DataFrame,
    config: NestedSplitConfig,
) -> dict[str, object]:
    config.validate()
    return {
        "format": MANIFEST_FORMAT,
        "config": asdict(config),
        "cohort": _patient_contract(patients),
    }


def _contract_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_nested_split_manifest(
    patients: pd.DataFrame,
    config: NestedSplitConfig,
) -> pd.DataFrame:
    """Create one inner holdout inside every repeated outer donor fold.

    ``inner_splits`` controls the validation fraction (1 / inner_splits).  A
    fold-specific deterministic seed is used and the first resulting fold is
    retained as the single inner validation set.  The manifest contains one row
    per biopsy for each outer fold; donor consistency is checked separately.
    """

    config.validate()
    patients = patients.reset_index(drop=True).copy()
    _patient_contract(patients)
    outer_folds = repeated_donor_folds(
        patients,
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        seed=config.seed,
        stratify_column=config.stratify_column,
    )
    rows: list[dict[str, object]] = []
    for outer_repeat, outer_fold, outer_train, outer_test in outer_folds:
        outer_subset = patients.iloc[outer_train].reset_index(drop=True)
        inner_seed = (
            int(config.seed)
            + 100_003
            + int(outer_repeat) * 1_009
            + int(outer_fold) * 37
        )
        inner_candidates = repeated_donor_folds(
            outer_subset,
            n_splits=config.inner_splits,
            n_repeats=1,
            seed=inner_seed,
            stratify_column=config.stratify_column,
        )
        _, _, relative_inner_train, relative_inner_validation = inner_candidates[0]
        inner_train = outer_train[relative_inner_train]
        inner_validation = outer_train[relative_inner_validation]
        roles: dict[int, tuple[str, str]] = {}
        roles.update(
            (int(index), (OUTER_TRAIN, INNER_TRAIN)) for index in inner_train
        )
        roles.update(
            (int(index), (OUTER_TRAIN, INNER_VALIDATION))
            for index in inner_validation
        )
        roles.update(
            (int(index), (OUTER_TEST, INNER_NOT_APPLICABLE)) for index in outer_test
        )
        if set(roles) != set(range(len(patients))):
            raise RuntimeError("Nested split did not assign every biopsy exactly once.")
        for index in range(len(patients)):
            outer_role, inner_role = roles[index]
            rows.append(
                {
                    "outer_repeat": int(outer_repeat),
                    "outer_fold": int(outer_fold),
                    "patient": str(patients.iloc[index]["patient"]),
                    "donor": str(patients.iloc[index]["donor"]),
                    "outer_role": outer_role,
                    "inner_role": inner_role,
                }
            )
    manifest = pd.DataFrame(rows)
    validate_nested_split_manifest(manifest, patients, config)
    return manifest


def validate_nested_split_manifest(
    manifest: pd.DataFrame,
    patients: pd.DataFrame,
    config: NestedSplitConfig,
) -> None:
    """Reject identity drift, missing roles, and all donor leakage."""

    config.validate()
    required = {
        "outer_repeat",
        "outer_fold",
        "patient",
        "donor",
        "outer_role",
        "inner_role",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Nested split manifest lacks columns: {sorted(missing)}")
    expected_pairs = set(
        map(tuple, patients.loc[:, ["patient", "donor"]].astype(str).to_numpy())
    )
    expected_folds = {
        (repeat, fold)
        for repeat in range(int(config.outer_repeats))
        for fold in range(int(config.outer_splits))
    }
    observed_folds = set(
        map(
            tuple,
            manifest.loc[:, ["outer_repeat", "outer_fold"]]
            .drop_duplicates()
            .astype(int)
            .to_numpy(),
        )
    )
    if observed_folds != expected_folds:
        raise ValueError(
            "Nested split fold identities differ from the requested contract: "
            f"observed={sorted(observed_folds)}, expected={sorted(expected_folds)}"
        )
    for (repeat, fold), block in manifest.groupby(
        ["outer_repeat", "outer_fold"], sort=True
    ):
        pairs = set(map(tuple, block.loc[:, ["patient", "donor"]].astype(str).to_numpy()))
        if pairs != expected_pairs or block["patient"].astype(str).duplicated().any():
            raise ValueError(
                f"Nested split cohort changed in outer repeat {repeat}, fold {fold}."
            )
        allowed_outer = {OUTER_TRAIN, OUTER_TEST}
        if set(block["outer_role"].astype(str)) != allowed_outer:
            raise ValueError(f"Outer fold {repeat}/{fold} lacks train or test rows.")
        allowed_inner = {INNER_TRAIN, INNER_VALIDATION, INNER_NOT_APPLICABLE}
        if not set(block["inner_role"].astype(str)).issubset(allowed_inner):
            raise ValueError(f"Outer fold {repeat}/{fold} contains unknown inner roles.")
        outer_test = block["outer_role"].eq(OUTER_TEST)
        if not block.loc[outer_test, "inner_role"].eq(INNER_NOT_APPLICABLE).all():
            raise ValueError("Outer-test biopsies must never receive an inner role.")
        outer_train_roles = set(block.loc[~outer_test, "inner_role"].astype(str))
        if outer_train_roles != {INNER_TRAIN, INNER_VALIDATION}:
            raise ValueError("Every outer train fold needs inner train and validation rows.")
        donor_roles = (
            block.groupby("donor", sort=False)
            .agg(outer_roles=("outer_role", "nunique"), inner_roles=("inner_role", "nunique"))
        )
        if (donor_roles["outer_roles"] != 1).any() or (donor_roles["inner_roles"] != 1).any():
            raise ValueError(
                f"Donor leakage detected in outer repeat {repeat}, fold {fold}."
            )


def create_or_load_nested_split_manifest(
    patients: pd.DataFrame,
    path: str | Path,
    config: NestedSplitConfig,
) -> pd.DataFrame:
    """Reuse an exact manifest or create it atomically on first invocation."""

    target = Path(path).expanduser().resolve()
    metadata_path = target.with_suffix(target.suffix + ".json")
    payload = _contract_payload(patients, config)
    payload["contract_sha256"] = _contract_hash(payload)
    if target.is_file() or metadata_path.is_file():
        if not target.is_file() or not metadata_path.is_file():
            raise RuntimeError(
                "Nested split manifest is only partially present; use a new output path."
            )
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed_metadata != payload:
            raise RuntimeError(
                "Existing nested split manifest has a different cohort or CV contract. "
                "Use a new output directory instead of silently regenerating folds."
            )
        manifest = pd.read_csv(target)
        validate_nested_split_manifest(manifest, patients, config)
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_nested_split_manifest(patients, config)
    _atomic_csv(manifest, target)
    _atomic_json(payload, metadata_path)
    return manifest


def nested_folds_from_manifest(
    patients: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[NestedFold]:
    """Translate the persisted identities back to patient-row indices."""

    patients = patients.reset_index(drop=True)
    required = {
        "outer_repeat",
        "outer_fold",
        "patient",
        "donor",
        "outer_role",
        "inner_role",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Nested split manifest lacks columns: {sorted(missing)}")
    lookup = {str(value): index for index, value in enumerate(patients["patient"])}
    if len(lookup) != len(patients):
        raise ValueError("Patient identities must be unique before loading folds.")
    expected_identity = {
        str(row.patient): str(row.donor)
        for row in patients.loc[:, ["patient", "donor"]].itertuples(index=False)
    }
    folds: list[NestedFold] = []
    for (repeat, fold), block in manifest.groupby(
        ["outer_repeat", "outer_fold"], sort=True
    ):
        if block["patient"].astype(str).duplicated().any():
            raise ValueError(
                f"Nested manifest repeats a biopsy in outer repeat {repeat}, fold {fold}."
            )
        unknown = sorted(set(block["patient"].astype(str)) - set(lookup))
        if unknown:
            raise ValueError(f"Nested manifest contains unknown patients: {unknown[:5]}")
        missing_patients = sorted(set(lookup) - set(block["patient"].astype(str)))
        if missing_patients:
            raise ValueError(
                f"Nested manifest omits patients in outer repeat {repeat}, fold {fold}: "
                f"{missing_patients[:5]}"
            )
        observed_identity = dict(
            zip(block["patient"].astype(str), block["donor"].astype(str), strict=True)
        )
        if observed_identity != expected_identity:
            raise ValueError(
                f"Nested manifest donor identities changed in repeat {repeat}, fold {fold}."
            )
        outer_roles = set(block["outer_role"].astype(str))
        if outer_roles != {OUTER_TRAIN, OUTER_TEST}:
            raise ValueError(
                f"Nested manifest outer roles are incomplete in repeat {repeat}, fold {fold}."
            )
        test_mask = block["outer_role"].astype(str).eq(OUTER_TEST)
        if not block.loc[test_mask, "inner_role"].astype(str).eq(
            INNER_NOT_APPLICABLE
        ).all():
            raise ValueError("Outer-test biopsies must have inner_role=not_applicable.")
        train_inner_roles = set(block.loc[~test_mask, "inner_role"].astype(str))
        if train_inner_roles != {INNER_TRAIN, INNER_VALIDATION}:
            raise ValueError("Outer training must contain inner train and validation rows.")
        donor_roles = block.groupby("donor", sort=False).agg(
            outer_roles=("outer_role", "nunique"),
            inner_roles=("inner_role", "nunique"),
        )
        if (donor_roles != 1).any().any():
            raise ValueError(
                f"Donor leakage detected in outer repeat {repeat}, fold {fold}."
            )

        def indices(outer_role: str, inner_role: str | None = None) -> np.ndarray:
            mask = block["outer_role"].astype(str).eq(outer_role)
            if inner_role is not None:
                mask &= block["inner_role"].astype(str).eq(inner_role)
            return np.asarray(
                [lookup[value] for value in block.loc[mask, "patient"].astype(str)],
                dtype=np.int64,
            )

        outer_train = indices(OUTER_TRAIN)
        outer_test = indices(OUTER_TEST)
        inner_train = indices(OUTER_TRAIN, INNER_TRAIN)
        inner_validation = indices(OUTER_TRAIN, INNER_VALIDATION)
        if set(inner_train).intersection(inner_validation):
            raise RuntimeError("Inner train and validation indices overlap.")
        if set(outer_train) != set(inner_train).union(inner_validation):
            raise RuntimeError("Inner roles do not exactly partition outer training rows.")
        if not all(
            len(indices) > 0
            for indices in (inner_train, inner_validation, outer_train, outer_test)
        ):
            raise ValueError("Nested manifest contains an empty modelling partition.")
        folds.append(
            NestedFold(
                outer_repeat=int(repeat),
                outer_fold=int(fold),
                inner_train=inner_train,
                inner_validation=inner_validation,
                outer_train=outer_train,
                outer_test=outer_test,
            )
        )
    return folds


def outer_folds_from_manifest(
    patients: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Return the outer folds in the legacy tuple form used by probe LASSO."""

    return [
        (fold.outer_repeat, fold.outer_fold, fold.outer_train, fold.outer_test)
        for fold in nested_folds_from_manifest(patients, manifest)
    ]


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """Stable digest used by resumable downstream result contracts."""

    columns = [
        "outer_repeat",
        "outer_fold",
        "patient",
        "donor",
        "outer_role",
        "inner_role",
    ]
    canonical = manifest.loc[:, columns].sort_values(columns).to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
