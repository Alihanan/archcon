"""Method-matched private-CEL IKEM inputs for downstream eGFR evaluation.

The private 288-CEL collection is canonical; GSE290167 is fallback-only. The
R rebuild creates three aligned matrices in ``IKEM_CEL_NUMPY_STORE``. This module
never estimates an expression-normalization parameter from eGFR-bearing rows:

* per-dataset standardization starts from raw PM probe-set medians and applies
  center/scale values frozen from outcome-blind IKEM pretraining-train rows;
* per-dataset RMA uses an IKEM-specific reference fitted only on outcome-free
  molecular-pretraining-train CELs and applies it independently to every other
  CEL;
* Global RMA uses CEL-level target/probe effects fitted on frozen GEO train plus
  donor-clean IKEM train arrays and applies them to every other CEL independently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .defaults import ProjectDataLayout
from .geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from .training_sources import (
    METHOD_PER_DATASET_STANDARDIZED,
    ExpressionMatrixSource,
    _membership_sha256,
    load_ikem_source,
)

FORMAT_VERSION = 4
PROVENANCE_FILENAME = "preprocessing_provenance.json"
MATRIX_FILES = {
    METHOD_PER_DATASET_STANDARDIZED: "per_dataset_standardization.npy",
    METHOD_PER_GSE_RMA: "per_dataset_rma.npy",
    METHOD_GLOBAL_RMA: "global_train_reference.npy",
}


@dataclass(frozen=True)
class PreparedIKEMSource:
    """One method-matched IKEM source and its saved provenance."""

    method: str
    source: ExpressionMatrixSource
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_probe_map(
    source: ExpressionMatrixSource,
    prepared_root: Path,
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    path = prepared_root / "probe_index.csv"
    if not path.is_file():
        if source.probe_ids is None:
            return np.arange(source.n_probes, dtype=np.int64), None
        return np.arange(source.n_probes, dtype=np.int64), source.probe_ids
    frame = pd.read_csv(path)
    if "probe_id" not in frame:
        raise ValueError(f"Prepared probe index lacks probe_id: {path}")
    target = tuple(frame["probe_id"].astype(str))
    if len(set(target)) != len(target):
        raise ValueError("Prepared probe index contains duplicate probe IDs.")
    if source.probe_ids is None:
        if source.n_probes != len(target):
            raise ValueError(
                "Cannot align IKEM expression without its probe_index.csv: "
                f"{source.n_probes:,} source probes vs {len(target):,} prepared probes."
            )
        return np.arange(len(target), dtype=np.int64), target
    if len(set(source.probe_ids)) != len(source.probe_ids):
        raise ValueError("IKEM probe index contains duplicate probe IDs.")
    lookup = {probe: index for index, probe in enumerate(source.probe_ids)}
    missing = [probe for probe in target if probe not in lookup]
    if missing:
        raise ValueError(
            f"IKEM CEL matrices are missing {len(missing):,} prepared probes; "
            f"examples: {missing[:5]}."
        )
    return np.asarray([lookup[probe] for probe in target], dtype=np.int64), target


def _write_matrix(
    path: Path,
    source: ExpressionMatrixSource,
    columns: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    batch_size: int,
) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(source.n_samples, len(columns)),
        fortran_order=False,
    )
    try:
        for start in range(0, source.n_samples, batch_size):
            stop = min(start + batch_size, source.n_samples)
            rows = np.arange(start, stop, dtype=np.int64)
            values = np.asarray(source.matrix[np.ix_(rows, columns)], dtype=np.float32)
            transformed = np.asarray(transform(values), dtype=np.float32)
            if transformed.shape != values.shape or not np.isfinite(transformed).all():
                raise ValueError(f"Invalid transformed IKEM block for {path.name}.")
            output[start:stop] = transformed
        output.flush()
    except Exception:
        output._mmap.close()
        partial.unlink(missing_ok=True)
        raise
    else:
        output._mmap.close()
        os.replace(partial, path)


def _load_saved_source(
    root: Path,
    matrix_path: Path,
    sample_index: pd.DataFrame,
    probe_ids: tuple[str, ...] | None,
    method: str,
) -> ExpressionMatrixSource:
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    return ExpressionMatrixSource(
        label=f"IKEM private CEL · {method}",
        matrix=matrix,
        sample_index=sample_index,
        probe_ids=probe_ids,
        root=root,
    )


def _load_standardization_parameters(
    prepared: Path,
    n_probes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    metadata = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    all_extras = metadata.get("method_extra_files", {})
    extras = (
        all_extras.get(METHOD_PER_DATASET_STANDARDIZED, {})
        if isinstance(all_extras, dict)
        else {}
    )
    center_name = extras.get("supplemental_center")
    scale_name = extras.get("supplemental_scale")
    if not center_name or not scale_name:
        raise ValueError(
            "Prepared sweep lacks frozen raw-IKEM standardization parameters. "
            "Regenerate it after installing the GSE290167 CEL matrices."
        )
    center_path = prepared / str(center_name)
    scale_path = prepared / str(scale_name)
    center = np.asarray(
        np.load(center_path, mmap_mode="r", allow_pickle=False), dtype=np.float32
    )
    scale = np.asarray(
        np.load(scale_path, mmap_mode="r", allow_pickle=False), dtype=np.float32
    )
    if center.shape != (n_probes,) or scale.shape != (n_probes,):
        raise ValueError("Frozen IKEM standardization parameters have the wrong shape.")
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("Frozen IKEM standardization parameters are invalid.")
    standardization = metadata.get("standardization", {})
    if (
        not isinstance(standardization, dict)
        or standardization.get("uses_outcome_values_in_fit") is not False
        or standardization.get("uses_egfr_cv_fold") is not False
    ):
        raise ValueError(
            "Prepared IKEM standardization is not certified as outcome-free. "
            "Regenerate the sweep."
        )
    n_train = (
        int(standardization.get("n_ikem_reference_rows", 0))
        if isinstance(standardization, dict)
        else 0
    )
    if n_train < 1:
        raise ValueError(
            "Prepared IKEM standardization has no frozen pretraining-train reference rows."
        )
    signature = {
        "center": str(center_path),
        "center_sha256": _sha256(center_path),
        "scale": str(scale_path),
        "scale_sha256": _sha256(scale_path),
        "n_ikem_pretraining_train": n_train,
    }
    return center, scale, signature


def prepare_ikem_evaluation_sources(
    layout: ProjectDataLayout,
    prepared_root: str | Path,
    destination: str | Path,
    *,
    methods: Iterable[str] | None = None,
    force: bool = False,
    batch_size: int = 16,
) -> dict[str, PreparedIKEMSource]:
    """Prepare and cache exact method-matched IKEM encoder inputs."""

    requested = (
        tuple(MATRIX_FILES)
        if methods is None
        else tuple(dict.fromkeys(str(method) for method in methods))
    )
    if not requested:
        raise ValueError("At least one IKEM preprocessing method must be requested.")
    unknown = sorted(set(requested) - set(MATRIX_FILES))
    if unknown:
        raise ValueError(f"Unknown IKEM preprocessing method(s): {unknown}")

    prepared = Path(prepared_root).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared_index = prepared / "sample_index.csv"
    cel_provenance_path = layout.ikem_store / "preprocessing_provenance.json"
    if not prepared_index.is_file() or not cel_provenance_path.is_file():
        raise FileNotFoundError(
            "The prepared split or private-CEL IKEM provenance is missing."
        )
    cel_provenance = json.loads(cel_provenance_path.read_text(encoding="utf-8"))
    if int(cel_provenance.get("format", -1)) != 4:
        raise RuntimeError(f"Unsupported IKEM CEL provenance: {cel_provenance_path}")
    cel_methods = cel_provenance.get("methods", {})
    local_rma_info = (
        cel_methods.get("rma_per_gse", {}) if isinstance(cel_methods, dict) else {}
    )
    if (
        not isinstance(local_rma_info, dict)
        or local_rma_info.get("transductive_across_egfr_folds") is not False
        or local_rma_info.get("uses_outcome_values_in_fit") is not False
        or local_rma_info.get("uses_egfr_cv_fold") is not False
    ):
        raise RuntimeError(
            "IKEM per-dataset RMA is not a certified frozen train-reference transform."
        )
    observed_split_hash = _sha256(prepared_index)
    prepared_samples = pd.read_csv(prepared_index)
    required_columns = {"sample_id", "source_kind", "split"}
    if not required_columns.issubset(prepared_samples.columns):
        raise RuntimeError("Prepared sample index lacks IKEM membership metadata.")
    ikem_rows = prepared_samples.loc[
        prepared_samples["source_kind"].astype(str).str.lower().eq("ikem")
    ]
    train_ids = ikem_rows.loc[
        ikem_rows["split"].astype(str).str.lower().eq("train"), "sample_id"
    ].astype(str).tolist()
    validation_ids = ikem_rows.loc[
        ikem_rows["split"].astype(str).str.lower().eq("validation"), "sample_id"
    ].astype(str).tolist()
    train_signature = _membership_sha256("SUPERVISED", train_ids)
    validation_signature = _membership_sha256("SUPERVISED", validation_ids)
    if (
        train_signature != str(cel_provenance.get("ikem_train_sample_ids_sha256", ""))
        or validation_signature
        != str(cel_provenance.get("ikem_validation_sample_ids_sha256", ""))
    ):
        raise RuntimeError(
            "The exact IKEM preprocessing matrices use different donor-safe train/validation "
            "identities. Resume the V6 CEL/RMA rebuild before downstream evaluation."
        )

    sources = {method: load_ikem_source(layout, method=method) for method in requested}
    if any(source is None for source in sources.values()):
        raise FileNotFoundError("A method-specific IKEM CEL matrix is missing.")
    reference = sources[requested[0]]
    assert reference is not None
    reference_ids = reference.sample_index[reference.sample_id_column].astype(str).tolist()
    method_columns: dict[str, np.ndarray] = {}
    probe_ids: tuple[str, ...] | None = None
    for method, source in sources.items():
        assert source is not None
        ids = source.sample_index[source.sample_id_column].astype(str).tolist()
        if ids != reference_ids:
            raise ValueError(f"IKEM sample order differs for {method}.")
        columns, current_probe_ids = _canonical_probe_map(source, prepared)
        method_columns[method] = columns
        if probe_ids is None:
            probe_ids = current_probe_ids
        elif current_probe_ids != probe_ids:
            raise ValueError(f"IKEM probe order differs for {method}.")

    center = scale = None
    standardization_signature: dict[str, object] = {}
    if METHOD_PER_DATASET_STANDARDIZED in requested:
        center, scale, standardization_signature = _load_standardization_parameters(
            prepared, len(method_columns[METHOD_PER_DATASET_STANDARDIZED])
        )

    source_signature: dict[str, dict[str, object]] = {}
    method_info = cel_provenance.get("methods", {})
    for method in requested:
        native = {
            METHOD_PER_DATASET_STANDARDIZED: "raw_original",
            METHOD_PER_GSE_RMA: "rma_per_gse",
            METHOD_GLOBAL_RMA: "rma_global",
        }[method]
        native_info = method_info.get(native, {}) if isinstance(method_info, dict) else {}
        native_path = layout.ikem_store / f"{native}.npy"
        source_signature[method] = {
            "matrix": str(native_path),
            "matrix_sha256": str(native_info.get("sha256", "")) or _sha256(native_path),
        }

    expected = {
        "format": FORMAT_VERSION,
        "source": "private IKEM CEL collection (GSE290167 fallback-only)",
        "prepared_sample_index_sha256": observed_split_hash,
        "ikem_train_sample_ids_sha256": train_signature,
        "ikem_validation_sample_ids_sha256": validation_signature,
        "cel_provenance_sha256": _sha256(cel_provenance_path),
        "shape": [reference.n_samples, len(method_columns[requested[0]])],
        "source_matrices": source_signature,
        "standardization": standardization_signature,
    }
    provenance_path = root / PROVENANCE_FILENAME
    previous = None
    if provenance_path.is_file() and not force:
        try:
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    cache_valid = previous is not None and all(
        previous.get(key) == value for key, value in expected.items()
    ) and all((root / MATRIX_FILES[method]).is_file() for method in requested)

    methods_metadata: dict[str, dict[str, object]] = {
        METHOD_PER_DATASET_STANDARDIZED: {
            "native_cel_matrix": "raw_original.npy",
            "algorithm": (
                "Per-probe standardization of CEL-derived raw PM probe-set medians "
                "using parameters fitted only on outcome-blind IKEM molecular-"
                "pretraining train rows."
            ),
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        METHOD_PER_GSE_RMA: {
            "native_cel_matrix": "rma_per_gse.npy",
            "algorithm": (
                "IKEM-specific RMA reference fitted only on frozen donor-clean no-eGFR "
                "pretraining-train CELs; validation and held-out CELs are transformed "
                "independently."
            ),
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        METHOD_GLOBAL_RMA: {
            "native_cel_matrix": "rma_global.npy",
            "algorithm": (
                "CEL-level RMA with quantile target and median-polish probe effects "
                "fitted only on frozen GEO train plus donor-clean IKEM train arrays; "
                "each remaining IKEM CEL is transformed independently."
            ),
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
    }

    if not cache_valid:
        for method in requested:
            source = sources[method]
            assert source is not None
            columns = method_columns[method]
            if method == METHOD_PER_DATASET_STANDARDIZED:
                assert center is not None and scale is not None
                transform = lambda values: (values - center) / scale
            else:
                transform = lambda values: values
            _write_matrix(
                root / MATRIX_FILES[method],
                source,
                columns,
                transform,
                batch_size=max(1, int(batch_size)),
            )
        reference.sample_index.to_csv(root / "sample_index.csv", index=False)
        if probe_ids is not None:
            pd.DataFrame(
                {
                    "probe_index_python": np.arange(len(probe_ids), dtype=np.int64),
                    "probe_id": probe_ids,
                }
            ).to_csv(root / "probe_index.csv", index=False)
        payload = {
            **expected,
            "methods": {method: methods_metadata[method] for method in requested},
        }
        temporary = provenance_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, provenance_path)

    sample_index = pd.read_csv(root / "sample_index.csv")
    result: dict[str, PreparedIKEMSource] = {}
    for method in requested:
        info = dict(methods_metadata[method])
        info["matrix"] = str(root / MATRIX_FILES[method])
        result[method] = PreparedIKEMSource(
            method=method,
            source=_load_saved_source(
                root,
                root / MATRIX_FILES[method],
                sample_index,
                probe_ids,
                method,
            ),
            provenance=info,
        )
    return result
