"""Helpers for the small kidney-donor dataset used after public-expression pretraining.

The web UI deliberately calls this the *supervised dataset*: unlike GEO, it has
clinical/outcome tables.  A subset of molecular samples has no eGFR follow-up;
those rows may be used for outcome-blind representation learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .defaults import ProjectDataLayout
from .loading import load_table, normalize_sample_id

EGFR_COLUMNS = ("egfr_7d", "egfr_3m", "egfr_6m", "egfr_12m")
_ID_CANDIDATES = ("patient", "Patient", "Sample_ID", "sample_id", "id", "ID")


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


def classify_supervised_samples(
    layout: ProjectDataLayout,
    sample_ids: Iterable[object],
) -> SupervisedDatasetStatus:
    """Classify molecular samples by whether any post-transplant eGFR is available.

    A molecular sample with no matching outcome row, or a matching row whose known
    eGFR measurements are all missing, is classified as ``without eGFR``.
    """

    ids = [normalize_sample_id(value) for value in sample_ids]
    result = pd.DataFrame(
        {
            "sample_id": ids,
            "sample_key": [f"SUPERVISED:{value}" for value in ids],
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
    egfr[id_column] = egfr[id_column].map(normalize_sample_id)
    known_columns = tuple(column for column in EGFR_COLUMNS if column in egfr.columns)

    if known_columns:
        valid = egfr.dropna(subset=list(known_columns), how="all")
    else:
        # If the source uses a different outcome naming scheme, matching an ID is
        # the most conservative available definition.  The UI reports that no
        # canonical eGFR columns were detected.
        valid = egfr
    valid_ids = set(valid[id_column].dropna().astype(str))
    result["has_egfr"] = result["sample_id"].isin(valid_ids)
    result["outcome_group"] = np.where(
        result["has_egfr"], "eGFR available", "no eGFR"
    )
    result["use_for_molecular_pretraining"] = ~result["has_egfr"]
    return SupervisedDatasetStatus(
        table=result,
        egfr_columns=known_columns,
        egfr_id_column=id_column,
    )


def split_outcome_blind_samples(
    status: SupervisedDatasetStatus,
    *,
    seed: int,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
) -> pd.DataFrame:
    """Create deterministic 90/5/5 partitions for supervised samples lacking eGFR."""

    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("train, validation, and test fractions must all be positive.")

    eligible = status.table.loc[~status.table["has_egfr"]].copy().reset_index(drop=True)
    n = len(eligible)
    if n == 0:
        return eligible.assign(split=pd.Series(dtype=str))
    if n < 3:
        raise ValueError(
            "At least three supervised samples without eGFR are required for a 90/5/5 split."
        )

    # Round the two small partitions first and leave the remainder to training.
    n_validation = max(1, int(round(n * float(validation_fraction))))
    n_test = max(1, int(round(n * float(test_fraction))))
    if n_validation + n_test >= n:
        n_validation = 1
        n_test = 1
    n_train = n - n_validation - n_test

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_validation]] = "validation"
    labels[order[n_train + n_validation :]] = "test"
    eligible["split"] = labels.astype(str)
    eligible["dataset_role"] = "supervised dataset · no eGFR"
    eligible["split_unit"] = "sample"
    eligible["seed"] = int(seed)
    eligible["train_fraction"] = float(train_fraction)
    eligible["validation_fraction"] = float(validation_fraction)
    eligible["test_fraction"] = float(test_fraction)
    return eligible


def supervised_overview_markdown(status: SupervisedDatasetStatus, source_label: str) -> str:
    columns = ", ".join(status.egfr_columns) if status.egfr_columns else "ID match only"
    return f"""
### Supervised dataset

This kidney-donor cohort contains molecular expression profiles and, for many samples,
post-transplant outcomes.  ArchCon keeps the two roles separate:

- **{status.n_without_egfr:,} samples without eGFR** can join molecular-only pretraining;
- **{status.n_with_egfr:,} samples with eGFR** remain completely outside pretraining and are
  reserved for downstream prediction/evaluation.

Expression source: `{source_label}`  
Outcome fields detected: `{columns}`
"""


def plot_outcome_group_counts(status: SupervisedDatasetStatus):
    counts = status.table["outcome_group"].value_counts().reindex(
        ["no eGFR", "eGFR available"], fill_value=0
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
    for group in ("no eGFR", "eGFR available"):
        mask = groups == group
        if np.any(mask):
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=24, alpha=0.75, label=group)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Supervised-dataset expression PCA")
    ax.legend()
    fig.tight_layout()
    return fig
