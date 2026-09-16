"""Helpers for the small kidney-donor dataset used after public-expression pretraining.

The web UI deliberately calls this the *supervised dataset*: unlike GEO, it has
clinical/outcome tables.  A subset of molecular samples has no eGFR follow-up;
those rows may be used for outcome-blind representation learning.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .defaults import ProjectDataLayout
from .loading import load_table, normalize_sample_id

EGFR_COLUMNS = ("egfr_7d", "egfr_3m", "egfr_6m", "egfr_12m")
_ID_CANDIDATES = ("patient", "Patient", "Sample_ID", "sample_id", "id", "ID")

IKEM_PRETRAINING_SPLIT_SEED = 20260915
IKEM_VALIDATION_FRACTION = 0.20
IKEM_ROLE_TRAIN = "pretrain_train_no_egfr"
IKEM_ROLE_VALIDATION = "pretrain_validation_no_egfr"
IKEM_ROLE_RELATED_HELD_OUT = "held_out_related_to_measured_egfr"
IKEM_ROLE_MEASURED_HELD_OUT = "held_out_measured_egfr"
IKEM_PAPER_VALIDATION_DONORS = ("D076", "D097", "D184", "D204")
IKEM_PAPER_VALIDATION_SAMPLES = (
    "D076_L",
    "D076_P",
    "D097_L",
    "D097_P",
    "D184_P",
    "D204_L",
)
IKEM_PAPER_RELATED_HELD_OUT_SAMPLES = ("D006_L", "D037_P", "D106_P", "D118_L")


@dataclass(frozen=True)
class SupervisedDatasetStatus:
    """Outcome-availability classification for molecular sample identities."""

    table: pd.DataFrame
    egfr_columns: tuple[str, ...]
    egfr_id_column: str | None

    @property
    def n_samples(self) -> int:
        return int(len(self.table))

    @property
    def n_with_egfr(self) -> int:
        return int(self.table["has_egfr"].sum())

    @property
    def n_without_egfr(self) -> int:
        return int((~self.table["has_egfr"]).sum())

    @property
    def n_pretraining_eligible(self) -> int:
        return int(self.table["use_for_molecular_pretraining"].sum())

    @property
    def n_related_without_egfr(self) -> int:
        return int(
            self.table["training_role"].eq(IKEM_ROLE_RELATED_HELD_OUT).sum()
        )


def _find_id_column(frame: pd.DataFrame) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in _ID_CANDIDATES:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(
        "Could not identify the eGFR sample-ID column. Expected one of: "
        + ", ".join(_ID_CANDIDATES)
    )


def donor_id_from_sample_id(value: object) -> str:
    """Return the canonical donor identity for an IKEM biopsy ID.

    Canonical rows end in ``_L`` or ``_P``. A non-canonical ID is treated as a
    one-biopsy donor so small external/test datasets still remain donor-safe.
    """

    sample_id = normalize_sample_id(value).upper()
    match = re.fullmatch(r"(.+)_([LP])", sample_id)
    return match.group(1) if match else sample_id


def _tissue_from_sample_id(value: object) -> str:
    sample_id = normalize_sample_id(value).upper()
    match = re.fullmatch(r".+_([LP])", sample_id)
    return match.group(1) if match else "U"


def _stable_donor_score(donor_id: str, seed: int) -> int:
    """Mirror the row-order-independent donor ranking used by the R rebuild."""

    match = re.fullmatch(r"D0*([0-9]+)", str(donor_id).upper())
    if match:
        donor_number = int(match.group(1))
        return (
            donor_number * 2_654_435_761
            + (int(seed) % 100_000) * 1_013_904_223
        ) % 2_147_483_647
    digest = hashlib.sha256(
        f"archcon-ikem-split-v1|{int(seed)}|{str(donor_id).upper()}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def classify_supervised_samples(
    layout: ProjectDataLayout,
    sample_ids: Iterable[object],
) -> SupervisedDatasetStatus:
    """Classify molecular samples by whether any post-transplant eGFR is available.

    A molecular sample with no matching outcome row, or a matching row whose known
    eGFR measurements are all missing, is classified as ``without eGFR``.
    """

    # Biopsy suffixes in the source workbooks are not consistently cased
    # (notably D205_p), whereas the CEL manifest uses canonical uppercase IDs.
    # Identity matching must therefore be case-insensitive and all downstream
    # sample keys use the uppercase canonical representation.
    ids = [normalize_sample_id(value).upper() for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Supervised expression data contain duplicate sample IDs.")
    result = pd.DataFrame(
        {
            "sample_id": ids,
            "sample_key": [f"SUPERVISED:{value}" for value in ids],
            "donor_id": [donor_id_from_sample_id(value) for value in ids],
            "tissue": [_tissue_from_sample_id(value) for value in ids],
            "has_egfr": False,
        }
    )
    if not layout.egfr_table.is_file():
        raise FileNotFoundError(
            f"Supervised-dataset separation requires the eGFR table: {layout.egfr_table}"
        )

    egfr = load_table(layout.egfr_table)
    id_column = _find_id_column(egfr)
    egfr = egfr.copy()
    egfr[id_column] = egfr[id_column].map(normalize_sample_id).str.upper()
    known_columns = tuple(column for column in EGFR_COLUMNS if column in egfr.columns)
    if not known_columns:
        raise ValueError(
            "The eGFR table contains none of the canonical longitudinal fields: "
            + ", ".join(EGFR_COLUMNS)
        )

    numeric = egfr.loc[:, known_columns].apply(pd.to_numeric, errors="coerce")
    row_has_finite_egfr = np.isfinite(numeric.to_numpy(dtype=float)).any(axis=1)
    valid_ids = set(egfr.loc[row_has_finite_egfr, id_column].dropna().astype(str))
    result["has_egfr"] = result["sample_id"].isin(valid_ids)
    donor_has_egfr = result.groupby("donor_id", sort=False)["has_egfr"].transform("any")
    result["donor_has_egfr"] = donor_has_egfr.astype(bool)
    result["training_role"] = np.select(
        [
            result["has_egfr"],
            ~result["has_egfr"] & result["donor_has_egfr"],
        ],
        [IKEM_ROLE_MEASURED_HELD_OUT, IKEM_ROLE_RELATED_HELD_OUT],
        default="pretrain_eligible_no_egfr",
    )
    result["outcome_group"] = np.select(
        [
            result["has_egfr"],
            result["training_role"].eq(IKEM_ROLE_RELATED_HELD_OUT),
        ],
        ["eGFR available", "no eGFR · related donor"],
        default="no eGFR · donor-clean",
    )
    result["use_for_molecular_pretraining"] = ~result["donor_has_egfr"]
    return SupervisedDatasetStatus(
        table=result,
        egfr_columns=known_columns,
        egfr_id_column=id_column,
    )


def split_outcome_blind_samples(
    status: SupervisedDatasetStatus,
    *,
    seed: int,
    train_fraction: float | None = None,
    validation_fraction: float = IKEM_VALIDATION_FRACTION,
    expected_validation_donors: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Split donor-clean, no-eGFR biopsies into donor-level train/validation.

    Donors are stratified by their available ``L``/``P`` biopsy pattern. There
    is deliberately no IKEM molecular test partition: GEO retains the blinded
    test set, while all eligible IKEM rows contribute to pretraining or its
    target-domain validation criterion.
    """

    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("IKEM validation_fraction must be in (0, 1).")
    implied_train_fraction = 1.0 - validation_fraction
    if train_fraction is not None and not np.isclose(
        float(train_fraction), implied_train_fraction
    ):
        raise ValueError(
            "IKEM has no molecular test partition; train_fraction must equal "
            "1 - validation_fraction."
        )

    eligible = status.table.loc[
        status.table["use_for_molecular_pretraining"]
    ].copy().reset_index(drop=True)
    n = len(eligible)
    if n == 0:
        return eligible.assign(split=pd.Series(dtype=str))
    donors = (
        eligible.groupby("donor_id", sort=True)["tissue"]
        .agg(lambda values: "".join(sorted(set(map(str, values)))))
        .rename("tissue_pattern")
        .reset_index()
    )
    if len(donors) < 2:
        raise ValueError("At least two donor-clean IKEM donors are required.")
    donors["score"] = [
        _stable_donor_score(donor, int(seed)) for donor in donors["donor_id"]
    ]

    validation_donors: set[str] = set()
    for _, stratum in donors.groupby("tissue_pattern", sort=True):
        if len(stratum) < 2:
            continue
        n_validation = max(1, int(round(len(stratum) * validation_fraction)))
        n_validation = min(n_validation, len(stratum) - 1)
        ranked = stratum.sort_values(["score", "donor_id"], kind="stable")
        validation_donors.update(ranked.head(n_validation)["donor_id"].astype(str))
    if not validation_donors:
        ranked = donors.sort_values(["score", "donor_id"], kind="stable")
        validation_donors.add(str(ranked.iloc[0]["donor_id"]))
    if len(validation_donors) == len(donors):
        raise ValueError("IKEM donor split left no training donor.")

    if expected_validation_donors is not None:
        expected = {str(value).upper() for value in expected_validation_donors}
        observed = {value.upper() for value in validation_donors}
        if observed != expected:
            raise ValueError(
                "IKEM validation donors differ from the frozen paper contract: "
                f"observed={sorted(observed)}, expected={sorted(expected)}."
            )

    eligible["split"] = np.where(
        eligible["donor_id"].isin(validation_donors), "validation", "train"
    )
    eligible["training_role"] = np.where(
        eligible["split"].eq("validation"), IKEM_ROLE_VALIDATION, IKEM_ROLE_TRAIN
    )
    eligible["pretraining_split"] = eligible["split"]
    eligible["dataset_role"] = "IKEM · donor-clean no eGFR"
    eligible["split_unit"] = "donor"
    eligible["seed"] = int(seed)
    eligible["train_fraction"] = implied_train_fraction
    eligible["validation_fraction"] = validation_fraction
    eligible["test_fraction"] = 0.0

    train_donors = set(eligible.loc[eligible["split"].eq("train"), "donor_id"])
    validation_donors_observed = set(
        eligible.loc[eligible["split"].eq("validation"), "donor_id"]
    )
    held_out_donors = set(
        status.table.loc[status.table["donor_has_egfr"], "donor_id"]
    )
    if (
        train_donors & validation_donors_observed
        or train_donors & held_out_donors
        or validation_donors_observed & held_out_donors
    ):
        raise RuntimeError("IKEM donors overlap across molecular/outcome roles.")
    return eligible


def supervised_overview_markdown(status: SupervisedDatasetStatus, source_label: str) -> str:
    columns = ", ".join(status.egfr_columns) if status.egfr_columns else "ID match only"
    return f"""
### Supervised dataset

This kidney-donor cohort contains molecular expression profiles and, for many samples,
post-transplant outcomes.  ArchCon keeps the two roles separate:

- **{status.n_pretraining_eligible:,} donor-clean samples without eGFR** can join
  molecular-only pretraining;
- **{status.n_related_without_egfr:,} no-eGFR samples** are excluded because another
  biopsy from the same donor has measured eGFR;
- **{status.n_with_egfr:,} samples with eGFR** remain completely outside pretraining and are
  reserved for downstream prediction/evaluation.

Expression source: `{source_label}`  
Outcome fields detected: `{columns}`
"""


def plot_outcome_group_counts(status: SupervisedDatasetStatus):
    counts = status.table["outcome_group"].value_counts().reindex(
        ["no eGFR · donor-clean", "no eGFR · related donor", "eGFR available"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Samples")
    ax.set_title("Which molecular samples have an outcome?")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_supervised_expression_pca(matrix: np.ndarray, status: SupervisedDatasetStatus):
    """Small diagnostic PCA colored only by outcome availability."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        return None
    coordinates = PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(values)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = status.table["outcome_group"].astype(str).to_numpy()
    for group in (
        "no eGFR · donor-clean",
        "no eGFR · related donor",
        "eGFR available",
    ):
        mask = groups == group
        if np.any(mask):
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=24, alpha=0.75, label=group)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Supervised-dataset expression PCA")
    ax.legend()
    fig.tight_layout()
    return fig
