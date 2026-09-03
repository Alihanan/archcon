"""Molecular pretraining/evaluation sources shared by web and headless runs.

Version 0.5.8 keeps one frozen molecular 90/5/5 split across the comparison: public GEO
rows are assigned by connected source-GSE component, while supervised-dataset
rows with no eGFR are independently assigned at the same target fractions and
virtually appended.  Outcome-bearing supervised rows are never exposed to the
autoencoder.  GEO itself is compared in three representations: Stadniuk
rescaling, per-study RMA, and a legacy-named ``Global RMA`` arm rebuilt as
train-reference quantile normalization from the available probe-set PM medians.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .supervised import classify_supervised_samples, split_outcome_blind_samples


METHOD_STADNIUK_RESCALED = "Stadniuk rescaling"
# Comparison arms used by Stage 05 and the exported sweep.  Keep the distinction
# explicit: the legacy Global RMA label is retained for existing sweep configs,
# while prepared jobs require leakage-safe train-reference provenance.
TRAINING_PREPROCESSING_OPTIONS = [
    METHOD_STADNIUK_RESCALED,
    METHOD_PER_GSE_RMA,
    METHOD_GLOBAL_RMA,
]

_SAMPLE_ID_CANDIDATES = ("sample_id", "Sample_ID", "GSM", "sample", "id")
_PROBE_ID_CANDIDATES = ("probe_id", "probe", "probeset_id", "ID", "id")
_ROW_CANDIDATES = ("row_index_python", "global_row_python", "sample_row_python")


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
    ):
        if primary.ndim != 2 or supplemental.ndim != 2:
            raise ValueError("Prepared expression matrices must be two-dimensional.")
        self.primary = primary
        self.primary_rows = np.asarray(primary_rows, dtype=np.int64)
        self.primary_columns = np.asarray(primary_columns, dtype=np.int64)
        self.supplemental = supplemental
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
            result[geo_mask] = np.asarray(
                self.primary[np.ix_(source_rows, source_columns)], dtype=np.float32
            )
        if np.any(~geo_mask):
            supplemental_rows = logical[~geo_mask] - len(self.primary_rows)
            result[~geo_mask] = np.asarray(
                self.supplemental[np.ix_(supplemental_rows, target_columns)], dtype=np.float32
            )
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


def _load_simple_numpy_store(root: Path, label: str) -> ExpressionMatrixSource:
    expression_path = root / "expression.npy"
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
    if method == METHOD_STADNIUK_RESCALED:
        return _load_simple_numpy_store(layout.geo_stadniuk_store, method)
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
    return frame


def load_ikem_source(layout: ProjectDataLayout) -> ExpressionMatrixSource | None:
    """Load the supervised kidney-donor molecular store when available.

    Prefer the compact internal ``IKEM_NUMPY_STORE`` used for repeated cluster runs. For
    interactive/local compatibility, fall back to the historical
    ``expression_matrix.csv`` understood by ArchCon's existing data loader.
    """
    if layout.ikem_store.is_dir():
        return _load_simple_numpy_store(layout.ikem_store, "Supervised dataset")
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


def _method_file_stem(method: str) -> str:
    mapping = {
        METHOD_STADNIUK_RESCALED: "stadniuk",
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
    if (
        int(provenance.get("format", -1)) != PROVENANCE_FORMAT
        or provenance.get("method") != METHOD_ID
    ):
        raise RuntimeError(f"Unsupported global-normalization provenance: {provenance_path}")
    split_path = prepared_root / "sample_index.csv"
    expected = str(provenance.get("frozen_split_sha256", ""))
    observed = _sha256(split_path)
    if not expected or expected != observed:
        raise RuntimeError(
            "The installed global-normalization matrix was fitted for a different frozen "
            "split. Rebuild it with this sweep root before running Global RMA jobs."
        )


def load_pretraining_source(
    layout: ProjectDataLayout,
    method: str,
    *,
    include_supervised_without_egfr: bool = True,
) -> ExpressionMatrixSource:
    """Load GEO plus outcome-blind supervised samples for molecular pretraining.

    The supervised cohort keeps its existing RMA expression values. Samples that
    have any eGFR follow-up are never appended here.
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

    supervised = load_ikem_source(layout)
    if supervised is None:
        return ExpressionMatrixSource(
            label=geo.label,
            matrix=geo.matrix,
            sample_index=geo_index,
            probe_ids=geo.probe_ids,
            root=geo.root,
        )
    supervised_columns = _probe_alignment_indices(geo, supervised)
    status = classify_supervised_samples(
        layout, supervised.sample_index[supervised.sample_id_column].astype(str)
    )
    eligible = status.table.loc[~status.table["has_egfr"]].reset_index(drop=True)
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
            "dataset_role": "supervised dataset · no eGFR",
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
        label=f"{method} + supervised no-eGFR",
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
    materializes only the outcome-blind supervised rows, freezes a canonical
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

    # Freeze and materialize only the supervised rows that are already present
    # in the frozen split.  No outcome classification occurs in a generated job.
    supplemental_filename = "supervised_no_egfr_common.npy"
    supplemental_path = root / supplemental_filename
    if supervised_keys:
        supervised = load_ikem_source(layout)
        if supervised is None:
            raise FileNotFoundError(
                "Frozen split contains supervised samples, but no supervised expression store exists."
            )
        probe_columns = _probe_alignment_indices(reference, supervised)
        supervised_ids = supervised.sample_index[supervised.sample_id_column].map(
            normalize_sample_id
        )
        if supervised_ids.duplicated().any():
            raise ValueError("Supervised expression store contains duplicate sample identities.")
        sample_lookup = dict(
            strict_zip(
                [f"SUPERVISED:{value}" for value in supervised_ids],
                supervised.sample_index["row_index_python"].astype(np.int64),
            )
        )
        missing = [key for key in supervised_keys if key not in sample_lookup]
        if missing:
            raise ValueError(
                f"Frozen split contains {len(missing):,} supervised samples absent from the "
                f"expression store; examples: {missing[:5]}."
            )
        source_rows = np.asarray([sample_lookup[key] for key in supervised_keys], dtype=np.int64)
        output = np.lib.format.open_memmap(
            supplemental_path,
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
            supplemental_path,
            mode="w+",
            dtype=np.float32,
            shape=(0, reference.n_probes),
        ).flush()

    roles = ["unsupervised data · GEO"] * len(geo_keys) + [
        "supervised dataset · no eGFR"
    ] * len(supervised_keys)
    sample_ids = [key.split(":", 1)[1] for key in logical_keys]
    sample_index = pd.DataFrame(
        {
            "row_index_python": np.arange(len(logical_keys), dtype=np.int64),
            "sample_key": logical_keys,
            "sample_id": sample_ids,
            "dataset_role": roles,
            "split": [split_lookup[key] for key in logical_keys],
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
        "format": 2,
        "n_samples": int(len(sample_index)),
        "n_geo": int(len(geo_keys)),
        "n_supervised_no_egfr": int(len(supervised_keys)),
        "n_probes": int(reference.n_probes),
        "train_rows": "train_rows.npy",
        "validation_rows": "validation_rows.npy",
        "test_rows": "test_rows.npy",
        "sample_index": "sample_index.csv",
        "supplemental_matrix": supplemental_filename,
        "canonical_method": reference_method,
        "method_geo_rows": method_rows,
        "method_geo_columns": method_columns,
        "policy": (
            "All sample identities, GEO row mappings, GEO probe-column mappings, supervised "
            "eligibility/alignment, and 90/5/5 logical row indices were frozen once during "
            "sweep generation. Generated jobs only read these files."
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
    if int(metadata.get("format", -1)) != 2:
        raise ValueError(
            "Unsupported prepared-pretraining format. Regenerate the sweep with ArchCon 0.5.8."
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
    supplemental = np.load(
        root / str(metadata["supplemental_matrix"]), mmap_mode="r", allow_pickle=False
    )
    sample_index = _normalize_sample_index(
        pd.read_csv(root / str(metadata["sample_index"])),
        len(primary_rows) + int(supplemental.shape[0]),
    )
    matrix = PreparedStackedExpressionMatrix(
        geo.matrix, primary_rows, primary_columns, supplemental
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
) -> pd.DataFrame:
    """Create the combined molecular 90/5/5 split used for pretraining.

    GEO is assigned by connected source-GSE component. Supervised-dataset
    molecular samples without eGFR are assigned independently by sample using
    the same seed and target fractions. Samples with eGFR are absent entirely.
    The public function name is retained for API compatibility.
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
    split["split_unit"] = "connected source-GSE component"

    supervised = load_ikem_source(layout)
    if supervised is None or not layout.egfr_table.is_file():
        return split
    status = classify_supervised_samples(
        layout, supervised.sample_index[supervised.sample_id_column].astype(str)
    )
    supplemental = split_outcome_blind_samples(
        status,
        seed=int(seed),
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
    )
    if supplemental.empty:
        return split
    supplemental["GSM"] = pd.NA
    supplemental["GSE"] = pd.NA
    supplemental["source_GSE"] = pd.NA
    supplemental["row_index_python"] = pd.NA
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
