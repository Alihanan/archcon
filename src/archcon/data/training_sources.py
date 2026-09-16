"""Frozen, donor-safe molecular pretraining sources for web and batch runs.

GEO keeps its connected-study 90/5/5 partition. IKEM is gated by donor: only
biopsies from donors with no finite longitudinal eGFR enter molecular
pretraining, and those donors are split 80/20 into train/validation with no
IKEM test partition. The final jobs consume only frozen prepared artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from .._compat import strict_zip
from .defaults import ProjectDataLayout
from .geo_rma import (
    METHOD_GLOBAL_RMA,
    METHOD_PER_GSE_RMA,
    METHOD_RAW,
    SCOPE_AGGREGATE,
    load_geo_expression_store,
)
from .pretraining import create_train_validation_split
from .loading import normalize_sample_id
from .supervised import (
    IKEM_PAPER_RELATED_HELD_OUT_SAMPLES,
    IKEM_PAPER_VALIDATION_DONORS,
    IKEM_PAPER_VALIDATION_SAMPLES,
    IKEM_PRETRAINING_SPLIT_SEED,
    IKEM_ROLE_MEASURED_HELD_OUT,
    IKEM_ROLE_RELATED_HELD_OUT,
    IKEM_ROLE_TRAIN,
    IKEM_ROLE_VALIDATION,
    IKEM_VALIDATION_FRACTION,
    donor_id_from_sample_id,
)


METHOD_PER_DATASET_STANDARDIZED = "Per-dataset standardization"
# Source-compatibility alias for code importing the former constant.  Its value
# deliberately resolves to the replacement method; old checkpoint strings are
# not silently accepted as the new preprocessing.
METHOD_STADNIUK_RESCALED = METHOD_PER_DATASET_STANDARDIZED
# Comparison arms used by Stage 05 and the exported sweep.  Keep the distinction
# explicit: the legacy Global RMA label is retained for existing sweep configs,
# while prepared jobs require leakage-safe train-reference provenance.
TRAINING_PREPROCESSING_OPTIONS = [
    METHOD_PER_DATASET_STANDARDIZED,
    METHOD_PER_GSE_RMA,
    METHOD_GLOBAL_RMA,
]

_SAMPLE_ID_CANDIDATES = ("sample_id", "Sample_ID", "GSM", "sample", "id")
_PROBE_ID_CANDIDATES = ("probe_id", "probe", "probeset_id", "ID", "id")
_ROW_CANDIDATES = ("row_index_python", "global_row_python", "sample_row_python")
PREPARED_FORMAT = 5
VALIDATION_GEO_WEIGHT = 0.50
GEO_PAPER_SPLIT_COUNTS = {"train": 10_522, "validation": 585, "test": 584}


class StackedExpressionMatrix:
    """Virtual row stack over a primary matrix and selected supplemental rows.

    ``secondary_columns`` maps the primary feature order onto the supplemental
    matrix.  This keeps interactive/local use safe when the supervised store
    contains a few extra probes, without materializing the full matrix.  Batch
    sweeps use the separately prepared on-disk matrix instead.
    """

    def __init__(
        self,
        primary: np.ndarray,
        secondary: np.ndarray,
        secondary_rows: np.ndarray,
        secondary_columns: np.ndarray | None = None,
    ):
        if primary.ndim != 2 or secondary.ndim != 2:
            raise ValueError("Both expression matrices must be two-dimensional.")
        self.primary = primary
        self.secondary = secondary
        self.secondary_rows = np.asarray(secondary_rows, dtype=np.int64)
        self.secondary_columns = (
            np.arange(int(secondary.shape[1]), dtype=np.int64)
            if secondary_columns is None
            else np.asarray(secondary_columns, dtype=np.int64)
        )
        if len(self.secondary_columns) != int(primary.shape[1]):
            raise ValueError("Supplemental probe mapping must match the primary probe count.")
        if np.any(self.secondary_columns < 0) or np.any(
            self.secondary_columns >= int(secondary.shape[1])
        ):
            raise ValueError("Supplemental probe mapping contains out-of-range columns.")
        self.shape = (int(primary.shape[0]) + len(self.secondary_rows), int(primary.shape[1]))
        self.ndim = 2
        self.size = int(self.shape[0] * self.shape[1])
        self.dtype = np.dtype("float32")
        self.flags = SimpleNamespace(c_contiguous=True)

    def __getitem__(self, key):
        rows, columns = key
        logical = np.arange(self.shape[0], dtype=np.int64)[rows]
        logical = np.atleast_1d(logical).astype(np.int64, copy=False)
        target_columns = np.arange(self.shape[1], dtype=np.int64)[columns]
        target_columns = np.atleast_1d(target_columns).astype(np.int64, copy=False)
        result = np.empty((len(logical), len(target_columns)), dtype=np.float32)
        primary_mask = logical < int(self.primary.shape[0])
        if np.any(primary_mask):
            primary_rows = logical[primary_mask]
            result[primary_mask] = np.asarray(
                self.primary[np.ix_(primary_rows, target_columns)], dtype=np.float32
            )
        if np.any(~primary_mask):
            supplemental_positions = logical[~primary_mask] - int(self.primary.shape[0])
            source_rows = self.secondary_rows[supplemental_positions]
            source_columns = self.secondary_columns[target_columns]
            result[~primary_mask] = np.asarray(
                self.secondary[np.ix_(source_rows, source_columns)], dtype=np.float32
            )
        return result


class PreparedStackedExpressionMatrix:
    """Read-only virtual matrix driven only by frozen row/column mappings.

    ``primary_columns`` maps the canonical frozen feature order onto the native
    preprocessing matrix.  This is prepared once during sweep generation, so a
    generated training job never has to compare or reorder probe identifiers.
    """

    def __init__(
        self,
        primary: np.ndarray,
        primary_rows: np.ndarray,
        primary_columns: np.ndarray,
        supplemental: np.ndarray,
        *,
        primary_group_codes: np.ndarray | None = None,
        primary_centers: np.ndarray | None = None,
        primary_scales: np.ndarray | None = None,
        supplemental_center: np.ndarray | None = None,
        supplemental_scale: np.ndarray | None = None,
    ):
        if primary.ndim != 2 or supplemental.ndim != 2:
            raise ValueError("Prepared expression matrices must be two-dimensional.")
        self.primary = primary
        self.primary_rows = np.asarray(primary_rows, dtype=np.int64)
        self.primary_columns = np.asarray(primary_columns, dtype=np.int64)
        self.supplemental = supplemental
        self.primary_group_codes = primary_group_codes
        self.primary_centers = primary_centers
        self.primary_scales = primary_scales
        self.supplemental_center = supplemental_center
        self.supplemental_scale = supplemental_scale
        if np.any(self.primary_rows < 0) or np.any(self.primary_rows >= int(primary.shape[0])):
            raise ValueError("Prepared GEO row mapping contains out-of-range rows.")
        if np.any(self.primary_columns < 0) or np.any(
            self.primary_columns >= int(primary.shape[1])
        ):
            raise ValueError("Prepared GEO probe mapping contains out-of-range columns.")
        if len(np.unique(self.primary_columns)) != len(self.primary_columns):
            raise ValueError("Prepared GEO probe mapping contains duplicate native columns.")
        if int(supplemental.shape[1]) != len(self.primary_columns):
            raise ValueError(
                "Prepared supervised matrix must already use the canonical frozen probe count."
            )
        standardization_items = (
            primary_group_codes,
            primary_centers,
            primary_scales,
            supplemental_center,
            supplemental_scale,
        )
        if any(item is not None for item in standardization_items):
            if any(item is None for item in standardization_items):
                raise ValueError("Prepared standardization metadata is incomplete.")
            if len(primary_group_codes) != len(self.primary_rows):
                raise ValueError("Prepared standardization group map has the wrong row count.")
            if primary_centers.shape != primary_scales.shape:
                raise ValueError("Prepared standardization centers/scales differ in shape.")
            if int(primary_centers.shape[1]) != len(self.primary_columns):
                raise ValueError("Prepared standardization has the wrong probe count.")
            if np.any(primary_group_codes < 0) or np.any(
                primary_group_codes >= int(primary_centers.shape[0])
            ):
                raise ValueError("Prepared standardization contains an invalid group code.")
            if supplemental_center.shape != (len(self.primary_columns),) or \
               supplemental_scale.shape != (len(self.primary_columns),):
                raise ValueError("Prepared supervised standardization has the wrong probe count.")
        self.shape = (len(self.primary_rows) + int(supplemental.shape[0]), len(self.primary_columns))
        self.ndim = 2
        self.size = int(self.shape[0] * self.shape[1])
        self.dtype = np.dtype("float32")
        self.flags = SimpleNamespace(c_contiguous=True)

    def __getitem__(self, key):
        rows, columns = key
        logical = np.arange(self.shape[0], dtype=np.int64)[rows]
        logical = np.atleast_1d(logical).astype(np.int64, copy=False)
        target_columns = np.arange(self.shape[1], dtype=np.int64)[columns]
        target_columns = np.atleast_1d(target_columns).astype(np.int64, copy=False)
        result = np.empty((len(logical), len(target_columns)), dtype=np.float32)
        geo_mask = logical < len(self.primary_rows)
        if np.any(geo_mask):
            source_rows = self.primary_rows[logical[geo_mask]]
            source_columns = self.primary_columns[target_columns]
            values = np.asarray(
                self.primary[np.ix_(source_rows, source_columns)], dtype=np.float32
            )
            if self.primary_group_codes is not None:
                groups = np.asarray(
                    self.primary_group_codes[logical[geo_mask]], dtype=np.int64
                )
                centers = np.asarray(
                    self.primary_centers[np.ix_(groups, target_columns)], dtype=np.float32
                )
                scales = np.asarray(
                    self.primary_scales[np.ix_(groups, target_columns)], dtype=np.float32
                )
                values = (values - centers) / scales
            result[geo_mask] = values
        if np.any(~geo_mask):
            supplemental_rows = logical[~geo_mask] - len(self.primary_rows)
            values = np.asarray(
                self.supplemental[np.ix_(supplemental_rows, target_columns)], dtype=np.float32
            )
            if self.supplemental_center is not None:
                center = np.asarray(self.supplemental_center[target_columns], dtype=np.float32)
                scale = np.asarray(self.supplemental_scale[target_columns], dtype=np.float32)
                values = (values - center) / scale
            result[~geo_mask] = values
        return result


@dataclass(frozen=True)
class ExpressionMatrixSource:
    """One row-addressable expression matrix plus identity/probe metadata."""

    label: str
    matrix: object
    sample_index: pd.DataFrame
    probe_ids: tuple[str, ...] | None
    root: Path

    @property
    def n_samples(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_probes(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def sample_id_column(self) -> str:
        return _find_column(self.sample_index, _SAMPLE_ID_CANDIDATES, "sample identifier")


@dataclass(frozen=True)
class ValidationPartition:
    """Frozen validation-domain labels aligned exactly to ``validation_rows``."""

    domains: np.ndarray
    donor_ids: np.ndarray
    n_geo: int
    n_ikem: int
    n_ikem_donors: int
    geo_weight: float


def _membership_sha256(namespace: str, sample_ids: list[str] | tuple[str, ...]) -> str:
    values = sorted(
        f"{str(namespace).upper()}:{str(sample_id).strip().upper()}"
        for sample_id in sample_ids
    )
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validation_partition_from_split(split: pd.DataFrame) -> ValidationPartition:
    """Build aligned domain labels from an already frozen identity-level split."""

    validation = split.loc[split["split"].astype(str).str.lower().eq("validation")].copy()
    if "source_kind" in validation.columns:
        domains = validation["source_kind"].astype(str).str.lower().to_numpy(dtype=str)
    elif "sample_key" in validation.columns:
        domains = np.where(
            validation["sample_key"].astype(str).str.startswith("GEO:"), "geo", "ikem"
        )
    else:
        raise ValueError("Validation split lacks frozen source-domain identities.")
    donor_series = validation.get("donor_id", pd.Series([""] * len(validation)))
    donor_ids = donor_series.fillna("").astype(str).str.upper().to_numpy(dtype=str)
    ikem_mask = domains == "ikem"
    if set(domains) != {"geo", "ikem"} or np.any(
        np.char.str_len(donor_ids[ikem_mask]) == 0
    ):
        raise ValueError("Paper validation requires GEO rows and donor-labelled IKEM rows.")
    return ValidationPartition(
        domains=np.asarray(domains, dtype=str),
        donor_ids=donor_ids,
        n_geo=int((domains == "geo").sum()),
        n_ikem=int(ikem_mask.sum()),
        n_ikem_donors=int(len(set(donor_ids[ikem_mask]))),
        geo_weight=VALIDATION_GEO_WEIGHT,
    )


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], description: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not identify {description}; expected one of: {', '.join(candidates)}."
    )


def _load_probe_ids(path: Path, n_probes: int) -> tuple[str, ...] | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    for candidate in _PROBE_ID_CANDIDATES:
        if candidate in frame.columns:
            values = tuple(frame[candidate].astype(str))
            if len(values) != int(n_probes):
                raise ValueError(
                    f"{path.name} has {len(values):,} probe IDs but expression matrix has "
                    f"{int(n_probes):,} columns."
                )
            return values
    return None


def _normalize_sample_index(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    result = frame.copy()
    row_column = next((name for name in _ROW_CANDIDATES if name in result.columns), None)
    if row_column is None:
        result.insert(0, "row_index_python", np.arange(len(result), dtype=np.int64))
    else:
        result["row_index_python"] = pd.to_numeric(result[row_column], errors="raise").astype(
            np.int64
        )
    if len(result) != int(n_rows):
        raise ValueError(
            f"sample_index.csv has {len(result):,} rows but expression matrix has "
            f"{int(n_rows):,}."
        )
    if sorted(result["row_index_python"].tolist()) != list(range(int(n_rows))):
        raise ValueError("sample_index.csv row indices are not a complete 0-based matrix mapping.")
    return result.sort_values("row_index_python").reset_index(drop=True)


def _load_simple_numpy_store(
    root: Path,
    label: str,
    *,
    matrix_filename: str = "expression.npy",
) -> ExpressionMatrixSource:
    expression_path = root / matrix_filename
    sample_path = root / "sample_index.csv"
    if not expression_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError(
            f"{label} requires {expression_path} and {sample_path}."
        )
    matrix = np.load(expression_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError(f"{expression_path} must be a two-dimensional NumPy matrix.")
    sample_index = _normalize_sample_index(pd.read_csv(sample_path), matrix.shape[0])
    _find_column(sample_index, _SAMPLE_ID_CANDIDATES, "sample identifier")
    probe_ids = _load_probe_ids(root / "probe_index.csv", matrix.shape[1])
    return ExpressionMatrixSource(
        label=label,
        matrix=matrix,
        sample_index=sample_index,
        probe_ids=probe_ids,
        root=root,
    )


def load_training_source(layout: ProjectDataLayout, method: str) -> ExpressionMatrixSource:
    """Load one of the three GEO preprocessing matrices used by the comparison grid."""
    method = str(method)
    if method == METHOD_PER_DATASET_STANDARDIZED:
        # The replacement arm starts from the unmodified historical GEO matrix.
        # Dataset-specific standardization parameters are frozen later, after
        # the shared train/validation/test identities are known.
        store = load_geo_expression_store(layout.geo_rma_store)
        matrix, _, _ = store.matrix_rows(METHOD_RAW, SCOPE_AGGREGATE, None)
        sample_index = _normalize_sample_index(store.sample_index, matrix.shape[0])
        probe_ids = _load_probe_ids(store.root / "probe_index.csv", matrix.shape[1])
        return ExpressionMatrixSource(
            label=method,
            matrix=matrix,
            sample_index=sample_index,
            probe_ids=probe_ids,
            root=store.root,
        )
    if method not in {METHOD_RAW, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA}:
        raise ValueError(f"Unknown training preprocessing: {method}")

    store = load_geo_expression_store(layout.geo_rma_store)
    matrix, _, _ = store.matrix_rows(method, SCOPE_AGGREGATE, None)
    sample_index = _normalize_sample_index(store.sample_index, matrix.shape[0])
    probe_ids = _load_probe_ids(store.root / "probe_index.csv", matrix.shape[1])
    return ExpressionMatrixSource(
        label=method,
        matrix=matrix,
        sample_index=sample_index,
        probe_ids=probe_ids,
        root=store.root,
    )


def _source_with_sample_keys(source: ExpressionMatrixSource, prefix: str, role: str) -> pd.DataFrame:
    frame = source.sample_index.copy()
    ids = frame[source.sample_id_column].astype(str)
    frame["sample_id"] = ids
    frame["sample_key"] = [f"{prefix}:{value}" for value in ids]
    frame["dataset_role"] = role
    frame["source_kind"] = str(prefix).lower()
    return frame


def _assert_canonical_geo_split_counts(frame: pd.DataFrame) -> None:
    """Fail closed if the canonical 11,691-row GEO split drifts."""

    if len(frame) != sum(GEO_PAPER_SPLIT_COUNTS.values()):
        return
    observed = frame["split"].astype(str).str.lower().value_counts().to_dict()
    if observed != GEO_PAPER_SPLIT_COUNTS:
        raise RuntimeError(
            "Canonical GEO split counts differ from the frozen paper contract: "
            f"observed={observed}, expected={GEO_PAPER_SPLIT_COUNTS}."
        )


_IKEM_METHOD_MATRIX = {
    METHOD_PER_DATASET_STANDARDIZED: "raw_original.npy",
    METHOD_PER_GSE_RMA: "rma_per_gse.npy",
    METHOD_GLOBAL_RMA: "rma_global.npy",
}

_IKEM_METHOD_PROVENANCE = {
    METHOD_PER_DATASET_STANDARDIZED: "raw_original",
    METHOD_PER_GSE_RMA: "rma_per_gse",
    METHOD_GLOBAL_RMA: "rma_global",
}


def _validated_ikem_role_manifest(source: ExpressionMatrixSource) -> pd.DataFrame:
    """Validate the audited R-produced donor roles without recomputing them."""

    required = {"training_role", "pretraining_split", "donor_id"}
    missing = sorted(required.difference(source.sample_index.columns))
    if missing:
        raise RuntimeError(
            "The IKEM CEL store lacks donor-safe role columns: " + ", ".join(missing)
        )
    frame = source.sample_index.copy()
    frame["sample_id"] = frame[source.sample_id_column].map(normalize_sample_id)
    frame["donor_id"] = frame["donor_id"].astype(str).str.strip().str.upper()
    frame["training_role"] = frame["training_role"].astype(str).str.strip()
    if frame["sample_id"].duplicated().any() or (frame["donor_id"] == "").any():
        raise RuntimeError("The IKEM CEL role manifest has duplicate samples or blank donors.")
    parsed_donors = frame["sample_id"].map(donor_id_from_sample_id).str.upper()
    if not frame["donor_id"].equals(parsed_donors):
        raise RuntimeError("IKEM donor IDs disagree with canonical biopsy IDs.")
    allowed = {
        IKEM_ROLE_TRAIN,
        IKEM_ROLE_VALIDATION,
        IKEM_ROLE_RELATED_HELD_OUT,
        IKEM_ROLE_MEASURED_HELD_OUT,
    }
    unknown = sorted(set(frame["training_role"]) - allowed)
    if unknown:
        raise RuntimeError(f"The IKEM CEL store contains unknown training roles: {unknown}")
    split = frame["pretraining_split"].fillna("").astype(str).str.strip().str.lower()
    expected_split = np.select(
        [
            frame["training_role"].eq(IKEM_ROLE_TRAIN),
            frame["training_role"].eq(IKEM_ROLE_VALIDATION),
        ],
        ["train", "validation"],
        default="",
    )
    if not np.array_equal(split.to_numpy(dtype=str), expected_split.astype(str)):
        raise RuntimeError("IKEM training_role and pretraining_split columns disagree.")
    train_donors = set(frame.loc[frame["training_role"].eq(IKEM_ROLE_TRAIN), "donor_id"])
    validation_donors = set(
        frame.loc[frame["training_role"].eq(IKEM_ROLE_VALIDATION), "donor_id"]
    )
    held_out_donors = set(
        frame.loc[
            frame["training_role"].isin(
                [IKEM_ROLE_RELATED_HELD_OUT, IKEM_ROLE_MEASURED_HELD_OUT]
            ),
            "donor_id",
        ]
    )
    if (
        train_donors & validation_donors
        or train_donors & held_out_donors
        or validation_donors & held_out_donors
    ):
        raise RuntimeError("IKEM donors overlap across train, validation, and held-out roles.")

    # Fail closed for the canonical paper cohort while still permitting compact
    # synthetic/external stores in tests and exploratory use.
    if len(frame) == 288:
        counts = frame["training_role"].value_counts().to_dict()
        expected_counts = {
            IKEM_ROLE_TRAIN: 24,
            IKEM_ROLE_VALIDATION: 6,
            IKEM_ROLE_RELATED_HELD_OUT: 4,
            IKEM_ROLE_MEASURED_HELD_OUT: 254,
        }
        if counts != expected_counts:
            raise RuntimeError(
                f"Canonical IKEM role counts differ from the paper contract: {counts}."
            )
        if validation_donors != set(IKEM_PAPER_VALIDATION_DONORS):
            raise RuntimeError(
                "Canonical IKEM validation donors differ from the frozen paper contract."
            )
        validation_samples = set(
            frame.loc[frame["training_role"].eq(IKEM_ROLE_VALIDATION), "sample_id"]
        )
        related_samples = set(
            frame.loc[
                frame["training_role"].eq(IKEM_ROLE_RELATED_HELD_OUT), "sample_id"
            ]
        )
        if validation_samples != set(IKEM_PAPER_VALIDATION_SAMPLES):
            raise RuntimeError(
                "Canonical IKEM validation biopsies differ from the frozen paper contract."
            )
        if related_samples != set(IKEM_PAPER_RELATED_HELD_OUT_SAMPLES):
            raise RuntimeError(
                "Canonical IKEM related-donor exclusions differ from the paper contract."
            )
    return frame


def _assert_frozen_ikem_split_matches_manifest(
    split: pd.DataFrame,
    source: ExpressionMatrixSource,
) -> None:
    """Require prepared IKEM identities/roles to equal the audited CEL manifest."""

    manifest = _validated_ikem_role_manifest(source)
    eligible = manifest.loc[
        manifest["training_role"].isin([IKEM_ROLE_TRAIN, IKEM_ROLE_VALIDATION])
    ].copy()
    eligible["sample_key"] = eligible["sample_id"].map(
        lambda value: f"SUPERVISED:{normalize_sample_id(value).upper()}"
    )
    eligible["split"] = np.where(
        eligible["training_role"].eq(IKEM_ROLE_TRAIN), "train", "validation"
    )
    expected = eligible.set_index("sample_key")[["split", "donor_id", "training_role"]]

    supplied = split.copy()
    supplied["sample_key"] = supplied["sample_key"].map(
        lambda value: (
            "SUPERVISED:"
            + normalize_sample_id(str(value).split(":", 1)[-1]).upper()
        )
    )
    if supplied["sample_key"].duplicated().any():
        raise ValueError("Frozen IKEM split contains duplicate normalized sample identities.")
    supplied["split"] = supplied["split"].astype(str).str.lower()
    supplied["donor_id"] = supplied["donor_id"].astype(str).str.strip().str.upper()
    supplied["training_role"] = supplied["training_role"].astype(str).str.strip()
    observed = supplied.set_index("sample_key")[["split", "donor_id", "training_role"]]

    if set(observed.index) != set(expected.index):
        missing = sorted(set(expected.index) - set(observed.index))[:5]
        extra = sorted(set(observed.index) - set(expected.index))[:5]
        raise ValueError(
            "Frozen IKEM identities differ from the audited donor-clean CEL manifest: "
            f"missing={missing}, extra={extra}."
        )
    observed = observed.loc[expected.index]
    if not observed.equals(expected):
        mismatches = observed.ne(expected).any(axis=1)
        raise ValueError(
            "Frozen IKEM train/validation roles differ from the audited CEL manifest; "
            f"examples: {observed.index[mismatches].tolist()[:5]}."
        )


def _validate_ikem_role_count_provenance(
    manifest: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    """Cross-check every role count recorded by the installed V6 store."""

    counts = manifest["training_role"].value_counts().to_dict()
    expected = {
        "cohort_samples": len(manifest),
        "no_measured_egfr_biopsies": (
            counts.get(IKEM_ROLE_TRAIN, 0)
            + counts.get(IKEM_ROLE_VALIDATION, 0)
            + counts.get(IKEM_ROLE_RELATED_HELD_OUT, 0)
        ),
        "donor_clean_pretraining_eligible_samples": (
            counts.get(IKEM_ROLE_TRAIN, 0) + counts.get(IKEM_ROLE_VALIDATION, 0)
        ),
        "pretraining_train_samples": counts.get(IKEM_ROLE_TRAIN, 0),
        "pretraining_validation_samples": counts.get(IKEM_ROLE_VALIDATION, 0),
        "related_no_egfr_held_out_samples": counts.get(
            IKEM_ROLE_RELATED_HELD_OUT, 0
        ),
        "held_out_measured_egfr_samples": counts.get(IKEM_ROLE_MEASURED_HELD_OUT, 0),
    }
    missing = [name for name in expected if name not in provenance]
    if missing:
        raise RuntimeError(
            "The IKEM CEL store lacks complete V6 role-count provenance: "
            + ", ".join(missing)
        )
    observed = {name: int(provenance[name]) for name in expected}
    if observed != expected:
        raise RuntimeError(
            "IKEM provenance role counts disagree with sample_index.csv: "
            f"observed={observed}, expected={expected}."
        )


def _validate_ikem_provenance(
    source: ExpressionMatrixSource,
    provenance: dict[str, object],
    method: str,
) -> None:
    if int(provenance.get("format", -1)) != 4:
        raise RuntimeError(
            "The paper pipeline requires donor-safe IKEM CEL provenance format 4. "
            "Rerun the V6 CEL/RMA rebuild."
        )
    if str(provenance.get("outcome_gate_unit", "")).lower() != "donor" or str(
        provenance.get("split_unit", "")
    ).lower() != "donor":
        raise RuntimeError("IKEM preprocessing provenance is not donor-gated/split.")
    if int(provenance.get("split_seed", -1)) != IKEM_PRETRAINING_SPLIT_SEED or not np.isclose(
        float(provenance.get("validation_fraction", -1.0)), IKEM_VALIDATION_FRACTION
    ):
        raise RuntimeError("IKEM preprocessing provenance uses a different frozen split policy.")

    manifest = _validated_ikem_role_manifest(source)
    _validate_ikem_role_count_provenance(manifest, provenance)
    methods_info = provenance.get("methods", {})
    native_method = _IKEM_METHOD_PROVENANCE[method]
    method_info = methods_info.get(native_method, {}) if isinstance(methods_info, dict) else {}
    if (
        not isinstance(method_info, dict)
        or method_info.get("transductive_across_egfr_folds") is not False
        or method_info.get("uses_outcome_values_in_fit") is not False
        or method_info.get("uses_egfr_cv_fold") is not False
    ):
        raise RuntimeError(
            f"IKEM {method} is not certified as an outcome-value-free, "
            "non-transductive transform."
        )

    train = manifest.loc[manifest["training_role"].eq(IKEM_ROLE_TRAIN)]
    validation = manifest.loc[manifest["training_role"].eq(IKEM_ROLE_VALIDATION)]
    declared_train = int(provenance.get("pretraining_train_samples", -1))
    declared_validation = int(provenance.get("pretraining_validation_samples", -1))
    if declared_train != len(train) or declared_validation != len(validation):
        raise RuntimeError("IKEM provenance sample counts disagree with sample_index.csv.")
    declared_donors = {
        str(value).upper() for value in provenance.get("pretraining_validation_donors", [])
    }
    if declared_donors != set(validation["donor_id"]):
        raise RuntimeError("IKEM provenance validation donors disagree with sample_index.csv.")
    expected_train_hash = _membership_sha256(
        "SUPERVISED", train["sample_id"].astype(str).tolist()
    )
    expected_validation_hash = _membership_sha256(
        "SUPERVISED", validation["sample_id"].astype(str).tolist()
    )
    if (
        provenance.get("ikem_train_sample_ids_sha256") != expected_train_hash
        or provenance.get("ikem_validation_sample_ids_sha256")
        != expected_validation_hash
    ):
        raise RuntimeError(
            "IKEM provenance membership hashes disagree with sample_index.csv."
        )


def load_ikem_source(
    layout: ProjectDataLayout,
    method: str | None = None,
) -> ExpressionMatrixSource | None:
    """Load the preferred private-CEL IKEM store when available."""
    if layout.ikem_store.is_dir():
        if method is not None:
            method = str(method)
            try:
                filename = _IKEM_METHOD_MATRIX[method]
            except KeyError as exc:
                raise ValueError(f"Unknown IKEM preprocessing method: {method}") from exc
            provenance_path = layout.ikem_store / "preprocessing_provenance.json"
            if not provenance_path.is_file():
                raise RuntimeError(
                    "Method-specific IKEM CEL matrices have no provenance. Run the "
                    "GSE290167 CEL/RMA rebuild before generating or evaluating a sweep."
                )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            source = _load_simple_numpy_store(
                layout.ikem_store,
                f"IKEM private CEL · {method}",
                matrix_filename=filename,
            )
            _validate_ikem_provenance(source, provenance, method)
            return source
        default_filename = (
            "expression.npy"
            if (layout.ikem_store / "expression.npy").is_file()
            else "rma_per_gse.npy"
        )
        source = _load_simple_numpy_store(
            layout.ikem_store,
            "Supervised dataset",
            matrix_filename=default_filename,
        )
        if (layout.ikem_store / "preprocessing_provenance.json").is_file():
            provenance = json.loads(
                (layout.ikem_store / "preprocessing_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            if int(provenance.get("format", -1)) != 4:
                raise RuntimeError(
                    "The paper pipeline requires donor-safe IKEM CEL provenance format 4."
                )
            manifest = _validated_ikem_role_manifest(source)
            _validate_ikem_role_count_provenance(manifest, provenance)
        return source
    if not layout.expression_matrix.is_file():
        return None

    from .loading import load_expression_matrix

    expression = load_expression_matrix(layout.expression_matrix, orientation="auto")
    frame = expression.frame
    matrix = frame.to_numpy(dtype=np.float32, copy=False)
    sample_index = pd.DataFrame(
        {
            "GSM": frame.index.astype(str),
            "row_index_python": np.arange(len(frame), dtype=np.int64),
        }
    )
    return ExpressionMatrixSource(
        label="Supervised dataset",
        matrix=matrix,
        sample_index=sample_index,
        probe_ids=tuple(str(column) for column in frame.columns),
        root=layout.root,
    )


def _ikem_pretraining_split(
    layout: ProjectDataLayout,
    source: ExpressionMatrixSource,
) -> pd.DataFrame:
    """Load the outcome gate and donor split frozen by the CEL/RMA rebuild.

    The R rebuild is the single authority for outcome availability because it
    reads the workbook, verifies all canonical identities, freezes roles before
    fitting any preprocessing, and records membership hashes in provenance.
    Re-reading the Excel workbook with a second library here can disagree on
    cached/formula cells and would violate the freeze-once contract.
    """

    manifest = _validated_ikem_role_manifest(source)
    result = manifest.loc[
        manifest["training_role"].isin([IKEM_ROLE_TRAIN, IKEM_ROLE_VALIDATION])
    ].copy()
    result["sample_key"] = result["sample_id"].map(
        lambda value: f"SUPERVISED:{normalize_sample_id(value)}"
    )
    result["tissue"] = result["sample_id"].map(
        lambda value: str(value).rsplit("_", 1)[-1].upper()
    )
    result["has_egfr"] = False
    result["donor_has_egfr"] = False
    result["use_for_molecular_pretraining"] = True
    result["outcome_group"] = "no eGFR · donor-clean"
    result["split"] = result["pretraining_split"].astype(str).str.lower()
    result["pretraining_split"] = result["split"]
    result["dataset_role"] = "IKEM · donor-clean no eGFR"
    result["split_unit"] = "donor"
    result["seed"] = IKEM_PRETRAINING_SPLIT_SEED
    result["train_fraction"] = 1.0 - IKEM_VALIDATION_FRACTION
    result["validation_fraction"] = IKEM_VALIDATION_FRACTION
    result["test_fraction"] = 0.0
    return result.reset_index(drop=True)


def _probe_alignment_indices(
    target: ExpressionMatrixSource,
    supplemental: ExpressionMatrixSource,
) -> np.ndarray:
    """Map target probe order onto a supplemental expression matrix.

    The supervised kidney store may contain the original 42,921 post-QC probes
    while the GEO comparison uses the 42,917-probe shared space.  Alignment is
    therefore by probe identifier, never by silently truncating columns.
    """

    if target.probe_ids is None or supplemental.probe_ids is None:
        if target.n_probes == supplemental.n_probes:
            return np.arange(target.n_probes, dtype=np.int64)
        raise ValueError(
            f"Cannot align {supplemental.label} ({supplemental.n_probes:,} probes) to "
            f"{target.label} ({target.n_probes:,} probes) because probe IDs are missing."
        )
    if len(set(target.probe_ids)) != len(target.probe_ids):
        raise ValueError(f"{target.label} probe_index.csv contains duplicate probe IDs.")
    if len(set(supplemental.probe_ids)) != len(supplemental.probe_ids):
        raise ValueError(f"{supplemental.label} probe_index.csv contains duplicate probe IDs.")
    lookup = {probe: index for index, probe in enumerate(supplemental.probe_ids)}
    missing = [probe for probe in target.probe_ids if probe not in lookup]
    if missing:
        raise ValueError(
            f"{supplemental.label} is missing {len(missing):,} probes required by "
            f"{target.label}; examples: {missing[:5]}."
        )
    return np.asarray([lookup[probe] for probe in target.probe_ids], dtype=np.int64)


def _fit_feature_standardizer(
    matrix: object,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a finite population mean/SD without materializing the full matrix."""

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if len(rows) == 0:
        raise ValueError("Cannot fit a standardizer without training rows.")
    total = np.zeros(len(columns), dtype=np.float64)
    total2 = np.zeros(len(columns), dtype=np.float64)
    count = 0
    for start in range(0, len(rows), max(1, int(batch_size))):
        block_rows = rows[start : start + max(1, int(batch_size))]
        values = np.asarray(matrix[np.ix_(block_rows, columns)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Per-dataset standardization encountered non-finite values.")
        total += values.sum(axis=0)
        total2 += np.square(values).sum(axis=0)
        count += int(values.shape[0])
    center = total / count
    variance = np.maximum(total2 / count - np.square(center), 0.0)
    scale = np.sqrt(variance)
    # Constant and numerically near-constant probes remain finite and map to 0.
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _standardization_dataset_labels(geo_split: pd.DataFrame) -> list[str]:
    """Return one stable source-dataset label for every frozen GEO sample."""

    for column in ("GSE", "canonical_GSE", "source_GSE", "split_group"):
        if column in geo_split.columns and geo_split[column].notna().all():
            return geo_split[column].astype(str).tolist()
    raise ValueError(
        "Per-dataset standardization requires GSE/source-study metadata in the frozen split."
    )


def _method_file_stem(method: str) -> str:
    mapping = {
        METHOD_PER_DATASET_STANDARDIZED: "standardized",
        METHOD_PER_GSE_RMA: "per_gse_rma",
        METHOD_GLOBAL_RMA: "global_rma",
        METHOD_RAW: "raw",
    }
    try:
        return mapping[str(method)]
    except KeyError as exc:
        raise ValueError(f"Unknown preprocessing method for prepared assets: {method}") from exc


def _validate_global_reference_for_prepared_sweep(
    store_root: Path,
    prepared_root: Path,
) -> None:
    """Require the global matrix to be fitted from this sweep's training rows."""
    from archcon.rebuild_global_normalization import (
        METHOD_ID,
        PROVENANCE_FILENAME,
        PROVENANCE_FORMAT,
        _sha256,
    )

    provenance_path = store_root / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise RuntimeError(
            "Refusing the legacy Global RMA matrix because it has no train-reference "
            "provenance and may include held-out arrays. Run "
            "archcon-rebuild-global-normalization --data-dir ... --sweep-root ... --replace."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    accepted_methods = {METHOD_ID, "cel_level_train_reference_rma"}
    if (
        int(provenance.get("format", -1)) != PROVENANCE_FORMAT
        or provenance.get("method") not in accepted_methods
    ):
        raise RuntimeError(f"Unsupported global-normalization provenance: {provenance_path}")
    split_path = prepared_root / "sample_index.csv"
    prepared = pd.read_csv(split_path)
    required = {"sample_key", "sample_id", "source_kind", "split"}
    if not required.issubset(prepared.columns):
        raise RuntimeError("Prepared sample index lacks global-reference membership metadata.")
    train = prepared.loc[prepared["split"].astype(str).str.lower().eq("train")]
    geo_ids = train.loc[
        train["source_kind"].astype(str).str.lower().eq("geo"), "sample_id"
    ].astype(str).tolist()
    ikem_ids = train.loc[
        train["source_kind"].astype(str).str.lower().eq("ikem"), "sample_id"
    ].astype(str).tolist()
    expected_geo = str(provenance.get("geo_train_sample_ids_sha256", ""))
    expected_ikem = str(provenance.get("ikem_train_sample_ids_sha256", ""))
    if provenance.get("method") == "cel_level_train_reference_rma":
        if (
            not expected_geo
            or not expected_ikem
            or expected_geo != _membership_sha256("GEO", geo_ids)
            or expected_ikem != _membership_sha256("SUPERVISED", ikem_ids)
            or int(provenance.get("geo_train_samples", -1)) != len(geo_ids)
            or int(provenance.get("ikem_no_egfr_train_samples", -1)) != len(ikem_ids)
        ):
            raise RuntimeError(
                "The installed exact global-RMA reference was fitted for different GEO/IKEM "
                "training identities. Resume the V6 CEL/RMA rebuild before Global RMA jobs."
            )
        return

    # The summarized Python fallback has no IKEM contribution and therefore is
    # not a final-paper global reference when donor-clean IKEM rows are present.
    if ikem_ids:
        raise RuntimeError(
            "The Python probe-set fallback global normalizer excludes IKEM training arrays. "
            "The final paper sweep requires the exact V6 CEL-level combined reference."
        )
    expected = str(provenance.get("frozen_split_sha256", ""))
    if not expected or expected != _sha256(split_path):
        raise RuntimeError(
            "The installed fallback global-normalization matrix was fitted for a different "
            "frozen split."
        )


def load_pretraining_source(
    layout: ProjectDataLayout,
    method: str,
    *,
    include_supervised_without_egfr: bool = True,
) -> ExpressionMatrixSource:
    """Load GEO plus outcome-blind supervised samples for molecular pretraining.

    The supervised cohort uses the CEL-derived representation matching the GEO
    arm: raw PM for standardization, frozen IKEM-train-reference RMA for
    per-GSE RMA, and frozen combined GEO+IKEM-train-reference RMA for Global
    RMA. Every biopsy from a donor with any finite eGFR is excluded, and none
    contributes to a fitted preprocessing parameter.
    """

    geo = load_training_source(layout, method)
    geo_index = _source_with_sample_keys(geo, "GEO", "unsupervised data · GEO")
    geo_index["row_index_python"] = np.arange(len(geo_index), dtype=np.int64)
    if not include_supervised_without_egfr:
        return ExpressionMatrixSource(
            label=geo.label,
            matrix=geo.matrix,
            sample_index=geo_index,
            probe_ids=geo.probe_ids,
            root=geo.root,
        )

    supervised = load_ikem_source(layout, method=method)
    if supervised is None:
        return ExpressionMatrixSource(
            label=geo.label,
            matrix=geo.matrix,
            sample_index=geo_index,
            probe_ids=geo.probe_ids,
            root=geo.root,
        )
    supervised_columns = _probe_alignment_indices(geo, supervised)
    eligible = _ikem_pretraining_split(layout, supervised).reset_index(drop=True)
    if eligible.empty:
        return ExpressionMatrixSource(
            label=geo.label,
            matrix=geo.matrix,
            sample_index=geo_index,
            probe_ids=geo.probe_ids,
            root=geo.root,
        )

    supervised_ids = supervised.sample_index[supervised.sample_id_column].map(normalize_sample_id)
    lookup = dict(
        strict_zip(supervised_ids, supervised.sample_index["row_index_python"].astype(np.int64))
    )
    source_rows = np.asarray([lookup[value] for value in eligible["sample_id"].astype(str)], dtype=np.int64)
    supplemental_index = pd.DataFrame(
        {
            "sample_id": eligible["sample_id"].astype(str),
            "sample_key": eligible["sample_key"].astype(str),
            "dataset_role": eligible["dataset_role"].astype(str),
            "source_kind": "ikem",
            "donor_id": eligible["donor_id"].astype(str),
            "training_role": eligible["training_role"].astype(str),
            "pretraining_split": eligible["split"].astype(str),
            "source_row_index": source_rows,
            "row_index_python": np.arange(
                len(geo_index), len(geo_index) + len(eligible), dtype=np.int64
            ),
        }
    )
    combined_index = pd.concat([geo_index, supplemental_index], ignore_index=True, sort=False)
    matrix = StackedExpressionMatrix(
        geo.matrix, supervised.matrix, source_rows, secondary_columns=supervised_columns
    )
    return ExpressionMatrixSource(
        label=f"{method} + donor-clean IKEM no-eGFR",
        matrix=matrix,
        sample_index=combined_index,
        probe_ids=geo.probe_ids,
        root=geo.root,
    )



def prepare_pretraining_assets(
    layout: ProjectDataLayout,
    split: pd.DataFrame,
    destination: str | Path,
    *,
    methods: list[str] | tuple[str, ...] = tuple(TRAINING_PREPROCESSING_OPTIONS),
) -> Path:
    """Freeze all sample/probe mappings needed by generated batch jobs.

    This function is intentionally run once when the sweep is generated.  It
    aligns the supervised expression matrix to the GEO 42,917-probe space,
    materializes only donor-clean outcome-blind IKEM rows, freezes a canonical
    logical sample order, saves method-specific GEO row maps, and writes the
    final train/validation/test integer row arrays.  Generated jobs only mmap
    these artifacts; they do not classify outcomes, align probes, resplit data,
    or rebuild sample mappings.
    """

    methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not methods:
        raise ValueError("At least one preprocessing method is required.")
    required = {"sample_key", "split"}
    missing_columns = sorted(required.difference(split.columns))
    if missing_columns:
        raise ValueError(
            "Prepared pretraining requires a frozen split with columns: "
            + ", ".join(sorted(required))
        )

    frame = split.copy().reset_index(drop=True)
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["split"] = frame["split"].astype(str).str.lower()
    if frame["sample_key"].duplicated().any():
        raise ValueError("Frozen pretraining split contains duplicate sample keys.")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Frozen pretraining split must contain train, validation, and test rows.")

    geo_split = frame.loc[frame["sample_key"].str.startswith("GEO:")].copy()
    supervised_split = frame.loc[
        frame["sample_key"].str.startswith("SUPERVISED:")
    ].copy()
    known = len(geo_split) + len(supervised_split)
    if known != len(frame):
        unknown = frame.loc[
            ~frame.index.isin([*geo_split.index, *supervised_split.index]), "sample_key"
        ].head(5)
        raise ValueError(f"Unknown frozen sample-key namespace: {unknown.tolist()}")
    if geo_split.empty:
        raise ValueError("Frozen pretraining split contains no GEO samples.")
    if set(geo_split["split"]) != {"train", "validation", "test"}:
        raise ValueError("Frozen GEO rows must contain train, validation, and test.")
    _assert_canonical_geo_split_counts(geo_split)
    if not supervised_split.empty:
        if set(supervised_split["split"]) != {"train", "validation"}:
            raise ValueError("Frozen IKEM rows must contain train/validation and no test rows.")
        required_ikem = {"donor_id", "training_role", "split_unit"}
        missing_ikem = sorted(required_ikem.difference(supervised_split.columns))
        if missing_ikem:
            raise ValueError(
                "Frozen IKEM split lacks donor-role metadata: " + ", ".join(missing_ikem)
            )
        if not supervised_split["split_unit"].astype(str).str.lower().eq("donor").all():
            raise ValueError("Every frozen IKEM row must use donor as the split unit.")
        expected_roles = np.where(
            supervised_split["split"].eq("train"), IKEM_ROLE_TRAIN, IKEM_ROLE_VALIDATION
        )
        if not np.array_equal(
            supervised_split["training_role"].astype(str).to_numpy(), expected_roles
        ):
            raise ValueError("Frozen IKEM training roles disagree with split labels.")
        donor_partitions = supervised_split.groupby("donor_id")["split"].nunique()
        if (donor_partitions > 1).any():
            raise ValueError("A frozen IKEM donor crosses train and validation.")
        audited_ikem = load_ikem_source(layout, method=methods[0])
        if audited_ikem is None:
            raise FileNotFoundError(
                "Frozen split contains IKEM samples, but no audited IKEM CEL "
                "expression store exists."
            )
        _assert_frozen_ikem_split_matches_manifest(supervised_split, audited_ikem)

    sources = {method: load_training_source(layout, method) for method in methods}

    # Canonical feature space is chosen once. Prefer the reconstructed per-study
    # RMA store because its 42,917-probe index is the project's common GEO/
    # supervised feature definition. Other preprocessing matrices may store the
    # exact same probe IDs in a different native column order; that is not an
    # error. We freeze a method-specific native-column map below.
    reference_method = (
        METHOD_PER_GSE_RMA if METHOD_PER_GSE_RMA in sources else methods[0]
    )
    reference = sources[reference_method]
    method_columns_arrays: dict[str, np.ndarray] = {}
    for method, source in sources.items():
        try:
            columns = _probe_alignment_indices(reference, source)
        except ValueError as exc:
            raise ValueError(
                f"Cannot align {method} to canonical {reference.label} probe space: {exc}"
            ) from exc
        if len(columns) != reference.n_probes:
            raise ValueError(
                f"Frozen probe map for {method} has {len(columns):,} columns; "
                f"expected {reference.n_probes:,}."
            )
        method_columns_arrays[method] = columns

    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    geo_keys = geo_split["sample_key"].astype(str).tolist()
    supervised_keys = supervised_split["sample_key"].astype(str).tolist()
    logical_keys = [*geo_keys, *supervised_keys]
    split_lookup = dict(strict_zip(frame["sample_key"], frame["split"]))

    # Freeze method-specific native GEO row AND probe-column maps once.
    # Generated jobs never infer either mapping.
    method_rows: dict[str, str] = {}
    method_columns: dict[str, str] = {}
    method_rows_arrays: dict[str, np.ndarray] = {}
    for method, source in sources.items():
        keys = _source_with_sample_keys(source, "GEO", "unsupervised data · GEO")[
            "sample_key"
        ].astype(str)
        if keys.duplicated().any():
            raise ValueError(f"{method} contains duplicate GEO sample keys.")
        lookup = dict(
            strict_zip(keys, source.sample_index["row_index_python"].astype(np.int64))
        )
        missing = [key for key in geo_keys if key not in lookup]
        if missing:
            raise ValueError(
                f"Frozen split contains {len(missing):,} GEO samples absent from {method}; "
                f"examples: {missing[:5]}."
            )
        rows = np.asarray([lookup[key] for key in geo_keys], dtype=np.int64)
        stem = _method_file_stem(method)
        row_filename = f"geo_rows_{stem}.npy"
        column_filename = f"geo_columns_{stem}.npy"
        np.save(root / row_filename, rows, allow_pickle=False)
        np.save(root / column_filename, method_columns_arrays[method], allow_pickle=False)
        method_rows[method] = row_filename
        method_columns[method] = column_filename
        method_rows_arrays[method] = rows

    # Materialize one donor-clean IKEM matrix per preprocessing arm.
    # Sharing the historical IKEM cohort-RMA matrix across all three arms was
    # inconsistent: GEO raw-standardization and GEO train-reference RMA must be
    # paired with the corresponding CEL-derived IKEM representations.
    supplemental_files: dict[str, str] = {}
    supplemental_paths: dict[str, Path] = {}
    for method in methods:
        filename = f"ikem_donor_clean_no_egfr_{_method_file_stem(method)}.npy"
        path = root / filename
        supplemental_files[method] = filename
        supplemental_paths[method] = path
        if supervised_keys:
            supervised = load_ikem_source(layout, method=method)
            if supervised is None:
                raise FileNotFoundError(
                    "Frozen split contains IKEM samples, but no audited IKEM CEL "
                    "expression store exists."
                )
            probe_columns = _probe_alignment_indices(reference, supervised)
            supervised_ids = supervised.sample_index[
                supervised.sample_id_column
            ].map(normalize_sample_id)
            if supervised_ids.duplicated().any():
                raise ValueError(
                    f"IKEM {method} store contains duplicate sample identities."
                )
            sample_lookup = dict(
                strict_zip(
                    [f"SUPERVISED:{value}" for value in supervised_ids],
                    supervised.sample_index["row_index_python"].astype(np.int64),
                )
            )
            missing = [key for key in supervised_keys if key not in sample_lookup]
            if missing:
                raise ValueError(
                    f"Frozen split contains {len(missing):,} IKEM samples absent "
                    f"from the IKEM {method} store; examples: {missing[:5]}."
                )
            source_rows = np.asarray(
                [sample_lookup[key] for key in supervised_keys], dtype=np.int64
            )
            output = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(len(source_rows), reference.n_probes),
            )
            block = 16
            for start in range(0, len(source_rows), block):
                stop = min(start + block, len(source_rows))
                output[start:stop] = np.asarray(
                    supervised.matrix[np.ix_(source_rows[start:stop], probe_columns)],
                    dtype=np.float32,
                )
            output.flush()
            del output
        else:
            np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(0, reference.n_probes),
            ).flush()

    # Freeze the replacement for the former range-rescaling arm. Public GEO
    # samples are standardized inside their own source dataset. The connected-
    # GSE split guarantees that a dataset belongs to exactly one molecular
    # partition. For IKEM, parameters are fitted only on the outcome-blind
    # supervised rows assigned to the molecular-pretraining TRAIN partition;
    # the same frozen parameters are later applied to eGFR-bearing IKEM rows.
    method_extra_files: dict[str, dict[str, str]] = {}
    standardization_metadata: dict[str, object] | None = None
    if METHOD_PER_DATASET_STANDARDIZED in sources:
        dataset_labels = _standardization_dataset_labels(geo_split)
        geo_partitions = geo_split["split"].astype(str).str.lower().tolist()
        group_order = list(dict.fromkeys(dataset_labels))
        group_lookup = {label: index for index, label in enumerate(group_order)}
        group_codes = np.asarray(
            [group_lookup[label] for label in dataset_labels], dtype=np.int32
        )
        group_rows = method_rows_arrays[METHOD_PER_DATASET_STANDARDIZED]
        group_columns = method_columns_arrays[METHOD_PER_DATASET_STANDARDIZED]
        standardization_source = sources[METHOD_PER_DATASET_STANDARDIZED]
        centers = np.empty((len(group_order), reference.n_probes), dtype=np.float32)
        scales = np.empty_like(centers)
        group_records: list[dict[str, object]] = []
        for label, code in group_lookup.items():
            positions = np.flatnonzero(group_codes == code)
            partitions = sorted({geo_partitions[index] for index in positions})
            if len(partitions) != 1:
                raise ValueError(
                    f"Dataset {label!r} crosses frozen molecular partitions: {partitions}."
                )
            center, scale = _fit_feature_standardizer(
                standardization_source.matrix,
                group_rows[positions],
                group_columns,
            )
            centers[code] = center
            scales[code] = scale
            group_records.append(
                {
                    "group_code": int(code),
                    "dataset": label,
                    "split": partitions[0],
                    "n_samples": int(len(positions)),
                }
            )

        supplemental_matrix = np.load(
            supplemental_paths[METHOD_PER_DATASET_STANDARDIZED],
            mmap_mode="r",
            allow_pickle=False,
        )
        supervised_train_positions = np.asarray(
            [
                index
                for index, key in enumerate(supervised_keys)
                if split_lookup[key] == "train"
            ],
            dtype=np.int64,
        )
        if len(supervised_train_positions) == 0:
            raise ValueError(
                "Per-dataset standardization requires at least one outcome-blind IKEM "
                "sample in the frozen molecular-pretraining training partition."
            )
        supplemental_center, supplemental_scale = _fit_feature_standardizer(
            supplemental_matrix,
            supervised_train_positions,
            np.arange(reference.n_probes, dtype=np.int64),
        )

        extra_names = {
            "primary_group_codes": "standardization_geo_group_codes.npy",
            "primary_centers": "standardization_geo_centers.npy",
            "primary_scales": "standardization_geo_scales.npy",
            "supplemental_center": "standardization_ikem_train_center.npy",
            "supplemental_scale": "standardization_ikem_train_scale.npy",
            "groups": "standardization_geo_groups.csv",
        }
        np.save(root / extra_names["primary_group_codes"], group_codes, allow_pickle=False)
        np.save(root / extra_names["primary_centers"], centers, allow_pickle=False)
        np.save(root / extra_names["primary_scales"], scales, allow_pickle=False)
        np.save(
            root / extra_names["supplemental_center"], supplemental_center, allow_pickle=False
        )
        np.save(
            root / extra_names["supplemental_scale"], supplemental_scale, allow_pickle=False
        )
        pd.DataFrame(group_records).to_csv(root / extra_names["groups"], index=False)
        method_extra_files[METHOD_PER_DATASET_STANDARDIZED] = extra_names
        standardization_metadata = {
            "formula": "(x - per_probe_mean) / max(per_probe_population_sd, 1e-6)",
            "geo_fit_scope": (
                "each complete source dataset; each dataset occurs in one frozen split only"
            ),
            "ikem_fit_scope": "donor-clean no-eGFR IKEM pretraining-train rows only",
            "n_geo_datasets": int(len(group_order)),
            "n_ikem_reference_rows": int(len(supervised_train_positions)),
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
            "uses_egfr_cv_fold": False,
        }

    ordered = frame.set_index("sample_key", drop=False).loc[logical_keys].reset_index(drop=True)
    sample_index = pd.DataFrame(
        {
            "row_index_python": np.arange(len(logical_keys), dtype=np.int64),
            "sample_key": logical_keys,
            "sample_id": [key.split(":", 1)[1] for key in logical_keys],
            "source_kind": ["geo"] * len(geo_keys) + ["ikem"] * len(supervised_keys),
            "dataset_role": ordered["dataset_role"].astype(str).tolist(),
            "split": [split_lookup[key] for key in logical_keys],
            "donor_id": ordered.get("donor_id", pd.Series([pd.NA] * len(ordered))),
            "training_role": ordered.get(
                "training_role", pd.Series([pd.NA] * len(ordered))
            ),
            "split_unit": ordered.get("split_unit", pd.Series([pd.NA] * len(ordered))),
        }
    )
    sample_index.to_csv(root / "sample_index.csv", index=False)

    labels = sample_index["split"].astype(str)
    train_rows = sample_index.loc[labels.eq("train"), "row_index_python"].to_numpy(
        dtype=np.int64
    )
    validation_rows = sample_index.loc[
        labels.eq("validation"), "row_index_python"
    ].to_numpy(dtype=np.int64)
    test_rows = sample_index.loc[labels.eq("test"), "row_index_python"].to_numpy(
        dtype=np.int64
    )
    np.save(root / "train_rows.npy", train_rows, allow_pickle=False)
    np.save(root / "validation_rows.npy", validation_rows, allow_pickle=False)
    np.save(root / "test_rows.npy", test_rows, allow_pickle=False)

    if reference.probe_ids is not None:
        pd.DataFrame(
            {
                "probe_index_python": np.arange(reference.n_probes, dtype=np.int64),
                "probe_id": reference.probe_ids,
            }
        ).to_csv(root / "probe_index.csv", index=False)

    metadata = {
        "format": PREPARED_FORMAT,
        "n_samples": int(len(sample_index)),
        "n_geo": int(len(geo_keys)),
        "n_ikem_donor_clean_no_egfr": int(len(supervised_keys)),
        "n_geo_train": int((geo_split["split"] == "train").sum()),
        "n_geo_validation": int((geo_split["split"] == "validation").sum()),
        "n_geo_test": int((geo_split["split"] == "test").sum()),
        "n_ikem_train": int((supervised_split["split"] == "train").sum()),
        "n_ikem_validation": int((supervised_split["split"] == "validation").sum()),
        "n_probes": int(reference.n_probes),
        "train_rows": "train_rows.npy",
        "validation_rows": "validation_rows.npy",
        "test_rows": "test_rows.npy",
        "sample_index": "sample_index.csv",
        "supplemental_matrices": supplemental_files,
        "canonical_method": reference_method,
        "method_geo_rows": method_rows,
        "method_geo_columns": method_columns,
        "method_extra_files": method_extra_files,
        "standardization": standardization_metadata,
        "validation_selection": {
            "metric": "clean_reconstruction_mse",
            "geo_weight": VALIDATION_GEO_WEIGHT,
            "ikem_weight": 1.0 - VALIDATION_GEO_WEIGHT,
            "geo_aggregation": "elementwise mean across GEO validation samples",
            "ikem_aggregation": (
                "mean per biopsy across probes, mean biopsies within donor, then mean donors"
            ),
            "uses_geo_test": False,
            "uses_egfr_values": False,
        },
        "policy": (
            "GEO connected-study 90/5/5 identities and donor-clean IKEM 80/20 "
            "train/validation identities, row/probe mappings, and standardization parameters "
            "were frozen once during sweep generation. Generated jobs only read these files."
        ),
    }
    (root / "prepared.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return root


def _load_prepared_metadata(prepared_root: str | Path) -> tuple[Path, dict[str, object]]:
    root = Path(prepared_root).expanduser().resolve()
    path = root / "prepared.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared pretraining metadata not found: {path}. Regenerate the sweep before submitting."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata.get("format", -1)) != PREPARED_FORMAT:
        raise ValueError(
            "Unsupported prepared-pretraining format. Regenerate the donor-safe "
            "paper sweep after installing the V6 private-CEL IKEM matrices."
        )
    return root, metadata


def load_prepared_pretraining_source(
    layout: ProjectDataLayout,
    method: str,
    prepared_root: str | Path,
) -> ExpressionMatrixSource:
    """Load a pretraining source using only mappings frozen at sweep generation."""

    root, metadata = _load_prepared_metadata(prepared_root)
    method = str(method)
    method_rows = metadata.get("method_geo_rows", {})
    method_columns = metadata.get("method_geo_columns", {})
    if not isinstance(method_rows, dict) or method not in method_rows:
        raise ValueError(f"Prepared sweep has no frozen GEO row map for {method}.")
    if not isinstance(method_columns, dict) or method not in method_columns:
        raise ValueError(f"Prepared sweep has no frozen GEO probe map for {method}.")
    if method == METHOD_GLOBAL_RMA:
        _validate_global_reference_for_prepared_sweep(layout.geo_rma_store, root)
    geo = load_training_source(layout, method)
    primary_rows = np.load(root / str(method_rows[method]), mmap_mode="r", allow_pickle=False)
    primary_columns = np.load(
        root / str(method_columns[method]), mmap_mode="r", allow_pickle=False
    )
    supplemental_matrices = metadata.get("supplemental_matrices", {})
    if not isinstance(supplemental_matrices, dict) or method not in supplemental_matrices:
        raise ValueError(f"Prepared sweep has no method-matched IKEM matrix for {method}.")
    supplemental = np.load(
        root / str(supplemental_matrices[method]), mmap_mode="r", allow_pickle=False
    )
    sample_index = _normalize_sample_index(
        pd.read_csv(root / str(metadata["sample_index"])),
        len(primary_rows) + int(supplemental.shape[0]),
    )
    matrix_kwargs: dict[str, np.ndarray] = {}
    if method == METHOD_PER_DATASET_STANDARDIZED:
        all_extras = metadata.get("method_extra_files", {})
        extras = all_extras.get(method, {}) if isinstance(all_extras, dict) else {}
        required_extras = {
            "primary_group_codes",
            "primary_centers",
            "primary_scales",
            "supplemental_center",
            "supplemental_scale",
        }
        if not isinstance(extras, dict) or not required_extras.issubset(extras):
            raise ValueError(
                "Prepared sweep lacks frozen per-dataset standardization parameters."
            )
        matrix_kwargs = {
            name: np.load(root / str(extras[name]), mmap_mode="r", allow_pickle=False)
            for name in required_extras
        }
    matrix = PreparedStackedExpressionMatrix(
        geo.matrix,
        primary_rows,
        primary_columns,
        supplemental,
        **matrix_kwargs,
    )
    if int(matrix.shape[1]) != int(metadata["n_probes"]):
        raise ValueError("Prepared probe count no longer matches the selected GEO matrix.")
    return ExpressionMatrixSource(
        label=f"{method} + frozen supervised no-eGFR",
        matrix=matrix,
        sample_index=sample_index,
        probe_ids=_load_probe_ids(root / "probe_index.csv", matrix.shape[1]),
        root=root,
    )


def load_prepared_split_rows(
    prepared_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load final frozen train/validation/test logical row arrays verbatim."""

    root, metadata = _load_prepared_metadata(prepared_root)
    arrays = tuple(
        np.asarray(
            np.load(root / str(metadata[name]), mmap_mode="r", allow_pickle=False),
            dtype=np.int64,
        )
        for name in ("train_rows", "validation_rows", "test_rows")
    )
    train_rows, validation_rows, test_rows = arrays
    if min(len(train_rows), len(validation_rows), len(test_rows)) < 1:
        raise ValueError("Prepared split must contain non-empty train, validation, and test arrays.")
    all_rows = np.concatenate(arrays)
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError("Prepared train/validation/test arrays overlap.")
    if np.any(all_rows < 0) or np.any(all_rows >= int(metadata["n_samples"])):
        raise ValueError("Prepared split contains out-of-range logical row indices.")
    return train_rows, validation_rows, test_rows


def load_prepared_validation_partition(
    prepared_root: str | Path,
) -> ValidationPartition:
    """Load frozen validation domains/donors in validation-row order."""

    root, metadata = _load_prepared_metadata(prepared_root)
    _, validation_rows, _ = load_prepared_split_rows(root)
    sample_index = _normalize_sample_index(
        pd.read_csv(root / str(metadata["sample_index"])), int(metadata["n_samples"])
    )
    validation = sample_index.iloc[validation_rows].copy()
    domains = validation["source_kind"].astype(str).str.lower().to_numpy(dtype=str)
    if set(domains) != {"geo", "ikem"}:
        raise ValueError(
            "Paper validation must contain separate GEO and IKEM rows; "
            f"observed domains: {sorted(set(domains))}."
        )
    donor_ids = validation["donor_id"].fillna("").astype(str).str.upper().to_numpy(dtype=str)
    ikem_mask = domains == "ikem"
    if np.any(np.char.str_len(donor_ids[ikem_mask]) == 0):
        raise ValueError("A frozen IKEM validation row has no donor ID.")
    policy = metadata.get("validation_selection", {})
    if not isinstance(policy, dict) or policy.get("uses_geo_test") is not False:
        raise ValueError("Prepared validation-selection policy is missing or unsafe.")
    geo_weight = float(policy.get("geo_weight", float("nan")))
    if not np.isclose(geo_weight, VALIDATION_GEO_WEIGHT):
        raise ValueError("Prepared validation weighting differs from the paper contract.")
    return ValidationPartition(
        domains=domains,
        donor_ids=donor_ids,
        n_geo=int((domains == "geo").sum()),
        n_ikem=int(ikem_mask.sum()),
        n_ikem_donors=int(len(set(donor_ids[ikem_mask]))),
        geo_weight=geo_weight,
    )

def assert_probe_alignment(
    training_source: ExpressionMatrixSource,
    evaluation_source: ExpressionMatrixSource,
) -> None:
    """Require evaluation features to match training columns exactly."""
    if training_source.n_probes != evaluation_source.n_probes:
        raise ValueError(
            f"{evaluation_source.label} has {evaluation_source.n_probes:,} probes but "
            f"{training_source.label} has {training_source.n_probes:,}."
        )
    if training_source.probe_ids is not None and evaluation_source.probe_ids is not None:
        if training_source.probe_ids != evaluation_source.probe_ids:
            raise ValueError(
                f"Probe order differs between {training_source.label} and "
                f"{evaluation_source.label}; refusing to compute misleading supervised-dataset metrics."
            )


def create_shared_preprocessing_split(
    layout: ProjectDataLayout,
    seed: int,
    train_fraction: float = 0.90,
    methods: list[str] | tuple[str, ...] = tuple(TRAINING_PREPROCESSING_OPTIONS),
    validation_fraction: float = 0.05,
    ikem_seed: int = IKEM_PRETRAINING_SPLIT_SEED,
    ikem_validation_fraction: float = IKEM_VALIDATION_FRACTION,
) -> pd.DataFrame:
    """Create the GEO 90/5/5 plus donor-clean IKEM 80/20 split.

    GEO is assigned by connected source-GSE component. IKEM roles must match
    the audited CEL manifest and are assigned by donor, with no IKEM test rows.
    """
    methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not methods:
        raise ValueError("At least one preprocessing method is required.")

    sources = [load_training_source(layout, method) for method in methods]
    id_sets: list[set[str]] = []
    for source in sources:
        id_column = source.sample_id_column
        ids = source.sample_index[id_column].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"{source.label} sample_index contains duplicate sample IDs.")
        id_sets.append(set(ids))
    shared = set.intersection(*id_sets)
    if len(shared) < 3:
        raise ValueError("Requested preprocessing stores have fewer than three shared samples.")

    store = load_geo_expression_store(layout.geo_rma_store)
    split = create_train_validation_split(
        store,
        seed=int(seed),
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        eligible_gsms=shared,
    ).reset_index(drop=True)
    split["sample_id"] = split["GSM"].astype(str)
    split["sample_key"] = [f"GEO:{value}" for value in split["sample_id"]]
    split["dataset_role"] = "unsupervised data · GEO"
    split["source_kind"] = "geo"
    split["split_unit"] = "connected source-GSE component"
    _assert_canonical_geo_split_counts(split)

    supervised = load_ikem_source(layout)
    if supervised is None or not layout.egfr_table.is_file():
        return split
    if int(ikem_seed) != IKEM_PRETRAINING_SPLIT_SEED or not np.isclose(
        float(ikem_validation_fraction), IKEM_VALIDATION_FRACTION
    ):
        raise ValueError(
            "The final paper protocol freezes IKEM seed 20260915 and validation fraction 0.20."
        )
    supplemental = _ikem_pretraining_split(layout, supervised)
    if supplemental.empty:
        return split
    supplemental["GSM"] = pd.NA
    supplemental["GSE"] = pd.NA
    supplemental["source_GSE"] = pd.NA
    supplemental["row_index_python"] = pd.NA
    supplemental["source_kind"] = "ikem"
    columns = list(dict.fromkeys([*split.columns, *supplemental.columns]))
    # Cast both small metadata frames to object before concatenation. This keeps
    # intentionally missing GEO-only metadata on supplemental rows without
    # triggering pandas' deprecated all-NA dtype inference path. Numeric split
    # metadata is parsed explicitly by downstream validation where needed.
    return pd.concat(
        [
            split.reindex(columns=columns).astype(object),
            supplemental.reindex(columns=columns).astype(object),
        ],
        ignore_index=True,
    )



def validate_pretraining_split(
    layout: ProjectDataLayout,
    path: str | Path,
    *,
    methods: list[str] | tuple[str, ...] = tuple(TRAINING_PREPROCESSING_OPTIONS),
) -> pd.DataFrame:
    """Validate a saved combined molecular split against current data identities."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Split file not found: {source}")
    frame = pd.read_json(source) if source.suffix.lower() == ".json" else pd.read_csv(source)
    required = {"sample_key", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Split file is missing required columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["split"] = frame["split"].astype(str).str.lower()
    if frame["sample_key"].duplicated().any():
        raise ValueError("Split file contains duplicate molecular sample keys.")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("A strict molecular split must contain train, validation, and test rows.")

    seed = 42
    if "seed" in frame.columns and frame["seed"].notna().any():
        seed = int(float(frame["seed"].dropna().iloc[0]))
    train_fraction = 0.90
    validation_fraction = 0.05
    if "train_fraction" in frame.columns and frame["train_fraction"].notna().any():
        train_fraction = float(frame["train_fraction"].dropna().iloc[0])
    if "validation_fraction" in frame.columns and frame["validation_fraction"].notna().any():
        validation_fraction = float(frame["validation_fraction"].dropna().iloc[0])
    expected = create_shared_preprocessing_split(
        layout,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        methods=methods,
    )
    expected_keys = set(expected["sample_key"].astype(str))
    supplied_keys = set(frame["sample_key"].astype(str))
    if supplied_keys != expected_keys:
        missing_keys = sorted(expected_keys - supplied_keys)[:10]
        extra_keys = sorted(supplied_keys - expected_keys)[:10]
        raise ValueError(
            "Split sample universe does not match current molecular data. "
            f"Missing examples: {missing_keys}; extra examples: {extra_keys}."
        )

    expected_assignments = expected.set_index("sample_key")["split"].astype(str)
    supplied_assignments = frame.set_index("sample_key")["split"].astype(str)
    changed_assignments = [
        key
        for key in expected_assignments.index
        if supplied_assignments.get(key) != expected_assignments.get(key)
    ]
    if changed_assignments:
        raise ValueError(
            "Loaded split changes deterministic frozen GEO/IKEM assignments; examples: "
            f"{changed_assignments[:5]}."
        )

    # Reattach canonical metadata while preserving the user's assignments.
    assignments = frame[["sample_key", "split"]].copy()
    canonical = expected.drop(columns=["split"], errors="ignore")
    merged = canonical.merge(assignments, on="sample_key", how="left", validate="one_to_one")
    if "split_group" in merged.columns:
        geo = merged.loc[merged["dataset_role"].astype(str).str.contains("GEO", na=False)]
        leaking = geo.groupby("split_group")["split"].nunique()
        if (leaking > 1).any():
            raise ValueError("Loaded split leaks a connected GEO study component across partitions.")
    return merged.reset_index(drop=True)

def split_rows_for_source(
    split: pd.DataFrame,
    source: ExpressionMatrixSource,
    *,
    include_test: bool = False,
):
    """Map a fixed identity-level split onto one preprocessing matrix's rows.

    By default return the historical ``(train, validation)`` pair so interactive
    training code remains source-compatible.  ``include_test=True`` returns a
    third held-out array; callers must not feed those rows to training or model
    selection.
    """
    if "split" not in split.columns:
        raise ValueError("Split has no 'split' column.")

    if "sample_key" in split.columns and "sample_key" in source.sample_index.columns:
        keys = source.sample_index["sample_key"].astype(str)
        if keys.duplicated().any():
            raise ValueError(f"{source.label} has duplicate sample keys.")
        lookup = dict(
            strict_zip(keys, source.sample_index["row_index_python"].astype(np.int64))
        )
        requested = split["sample_key"].astype(str)
        missing = [value for value in requested if value not in lookup]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Fixed split contains {len(missing):,} samples missing from {source.label} "
                f"(for example: {preview})."
            )
        mapped = np.asarray([lookup[value] for value in requested], dtype=np.int64)
    elif "GSM" not in split.columns:
        if "row_index_python" not in split.columns:
            raise ValueError("Split needs sample_key, GSM, or row_index_python values.")
        rows = pd.to_numeric(split["row_index_python"], errors="raise").to_numpy(dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= source.n_samples):
            raise ValueError("Split row indices do not fit the selected expression matrix.")
        mapped = rows
    else:
        id_column = source.sample_id_column
        ids = source.sample_index[id_column].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"{source.label} has duplicate sample identities.")
        lookup = dict(
            strict_zip(ids, source.sample_index["row_index_python"].astype(np.int64))
        )
        requested = split["GSM"].astype(str)
        missing = [value for value in requested if value not in lookup]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Fixed split contains {len(missing):,} samples missing from {source.label} "
                f"(for example: {preview}). Build the split over the shared preprocessing "
                "intersection before launching the comparison sweep."
            )
        mapped = np.asarray([lookup[value] for value in requested], dtype=np.int64)

    labels = split["split"].astype(str).str.lower()
    train_rows = mapped[labels.eq("train").to_numpy()]
    validation_rows = mapped[labels.eq("validation").to_numpy()]
    test_rows = mapped[labels.eq("test").to_numpy()]
    if len(train_rows) < 1 or len(validation_rows) < 1:
        raise ValueError("Fixed split must contain at least one train and validation sample.")
    if include_test and len(test_rows) < 1:
        raise ValueError("Strict fixed split must contain at least one held-out GEO test sample.")

    used = np.concatenate([train_rows, validation_rows, test_rows])
    if len(np.unique(used)) != len(used):
        raise ValueError("Train/validation/test rows overlap or contain duplicates.")
    if include_test:
        return train_rows, validation_rows, test_rows
    return train_rows, validation_rows
