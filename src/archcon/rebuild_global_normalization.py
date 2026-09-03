"""Build leakage-safe train-reference normalization from the frozen sweep split.

The available ``raw_original.npy`` matrix contains one PM-median value per probe
set, not CEL-level probe intensities.  Consequently this command cannot recreate
exact RMA.  It performs the strongest valid correction available from that
matrix: fit one quantile reference on GEO training rows only, then apply the
frozen reference independently to every GEO row and log2-transform the result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .data.defaults import project_data_layout


PROVENANCE_FILENAME = "rma_global_provenance.json"
TARGET_FILENAME = "rma_global_train_reference_target.npy"
METHOD_ID = "train_reference_quantile_normalization_from_probe_set_pm_medians"
PROVENANCE_FORMAT = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sample_row_lookup(sample_index: pd.DataFrame, n_rows: int) -> dict[str, int]:
    id_column = next(
        (
            name
            for name in ("GSM", "sample_id", "Sample_ID", "sample", "id")
            if name in sample_index
        ),
        None,
    )
    if id_column is None:
        raise ValueError("GEO sample_index.csv has no recognizable sample-ID column.")
    row_column = next(
        (
            name
            for name in ("global_row_python", "row_index_python", "sample_row_python")
            if name in sample_index
        ),
        None,
    )
    rows = (
        np.arange(len(sample_index), dtype=np.int64)
        if row_column is None
        else pd.to_numeric(sample_index[row_column], errors="raise").to_numpy(dtype=np.int64)
    )
    if len(sample_index) != int(n_rows) or sorted(rows.tolist()) != list(range(int(n_rows))):
        raise ValueError("GEO sample_index.csv does not map every raw-matrix row exactly once.")
    ids = sample_index[id_column].astype(str).str.strip().str.upper()
    if ids.duplicated().any():
        raise ValueError("GEO sample_index.csv contains duplicate sample IDs.")
    return dict(zip(ids.tolist(), rows.tolist()))


def frozen_geo_partitions(
    sweep_root: str | Path,
    store_sample_index: pd.DataFrame,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Map the sweep's frozen GEO partitions to native ``raw_original.npy`` rows."""
    root = Path(sweep_root).expanduser().resolve()
    prepared_path = root / "prepared" / "sample_index.csv"
    if not prepared_path.is_file():
        raise FileNotFoundError(f"Frozen prepared sample index not found: {prepared_path}")
    prepared = pd.read_csv(prepared_path, low_memory=False)
    required = {"sample_key", "split"}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(f"{prepared_path} is missing columns: {', '.join(missing)}")

    geo = prepared.loc[prepared["sample_key"].astype(str).str.startswith("GEO:")].copy()
    if geo.empty:
        raise ValueError("The frozen sweep contains no GEO rows.")
    geo["sample_id"] = (
        geo["sample_key"].astype(str).str.split(":", n=1).str[1].str.strip().str.upper()
    )
    geo["split"] = geo["split"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(geo["split"]).difference({"train", "validation", "test"}))
    if unexpected:
        raise ValueError(f"Unexpected frozen split labels: {unexpected}")
    if geo["sample_id"].duplicated().any():
        raise ValueError("The frozen sweep contains duplicate GEO sample identities.")

    lookup = _sample_row_lookup(store_sample_index, n_rows)
    missing_ids = sorted(set(geo["sample_id"]).difference(lookup))
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids):,} frozen GEO samples are absent from GEO sample_index.csv; "
            f"examples: {missing_ids[:5]}"
        )
    partitions: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        ids = geo.loc[geo["split"].eq(name), "sample_id"].tolist()
        if not ids:
            raise ValueError(f"The frozen GEO {name} partition is empty.")
        partitions[name] = np.asarray([lookup[value] for value in ids], dtype=np.int64)
    return partitions


def fit_quantile_reference(
    raw: np.ndarray,
    train_rows: np.ndarray,
    *,
    chunk_size: int = 16,
    progress=None,
) -> np.ndarray:
    """Average sorted raw intensity vectors from training rows only."""
    if raw.ndim != 2:
        raise ValueError("raw_original.npy must be two-dimensional.")
    rows = np.asarray(train_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("At least one training row is required.")
    if np.any(rows < 0) or np.any(rows >= int(raw.shape[0])):
        raise ValueError("Training rows contain an out-of-range raw-matrix index.")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Training rows contain duplicates.")
    chunk_size = max(1, int(chunk_size))
    total = np.zeros(int(raw.shape[1]), dtype=np.float64)
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        values = np.asarray(raw[rows[start:stop]], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(
                "raw_original.npy contains non-finite or non-positive values in training rows."
            )
        total += np.sort(values, axis=1).sum(axis=0, dtype=np.float64)
        if progress is not None:
            progress(stop, len(rows), "fit")
    target = total / float(len(rows))
    if not np.isfinite(target).all() or np.any(target <= 0.0):
        raise ValueError("The fitted training reference contains invalid values.")
    return target


def normalize_one_row(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map one row to a frozen target, averaging target ranks for raw ties."""
    row = np.asarray(values, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if row.ndim != 1 or reference.ndim != 1 or len(row) != len(reference):
        raise ValueError("Row and target must be one-dimensional with equal length.")
    if not np.isfinite(row).all() or np.any(row <= 0.0):
        raise ValueError("Raw expression row contains non-finite or non-positive values.")
    order = np.argsort(row, kind="stable")
    sorted_values = row[order]
    mapped = np.log2(reference).astype(np.float32)
    boundaries = np.flatnonzero(np.diff(sorted_values) != 0.0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(row)]))
    tied = (stops - starts) > 1
    for start, stop in zip(starts[tied], stops[tied]):
        mapped[start:stop] = np.float32(np.log2(reference[start:stop].mean()))
    output = np.empty(len(row), dtype=np.float32)
    output[order] = mapped
    return output


def write_normalized_matrix(
    raw: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    *,
    progress=None,
) -> None:
    """Write a C-contiguous float32 train-reference normalized matrix."""
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=raw.shape,
        fortran_order=False,
    )
    try:
        for row_index in range(int(raw.shape[0])):
            output[row_index] = normalize_one_row(raw[row_index], target)
            if progress is not None:
                progress(row_index + 1, int(raw.shape[0]), "apply")
        output.flush()
    finally:
        del output


def _unique_backup(path: Path, timestamp: str) -> Path:
    candidate = path.with_name(f"{path.stem}.all_samples_backup_{timestamp}{path.suffix}")
    if candidate.exists():
        raise FileExistsError(f"Backup path already exists: {candidate}")
    return candidate


def _update_store_manifest(path: Path) -> None:
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    if "matrix_name" not in frame or "description" not in frame:
        return
    mask = frame["matrix_name"].astype(str).eq("rma_global")
    if not mask.any():
        return
    frame.loc[mask, "description"] = (
        "Train-reference quantile normalization of probe-set PM medians followed by log2; "
        "reference fitted only on frozen GEO training rows. Not exact CEL-level RMA."
    )
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def rebuild_global_normalization(
    *,
    data_dir: str | Path,
    sweep_root: str | Path,
    replace: bool,
    chunk_size: int = 16,
    progress=None,
) -> dict[str, object]:
    """Create and optionally install the train-reference matrix."""
    layout = project_data_layout(data_dir)
    store = layout.geo_rma_store
    raw_path = store / "raw_original.npy"
    sample_path = store / "sample_index.csv"
    destination = store / "rma_global.npy"
    provenance_path = store / PROVENANCE_FILENAME
    target_path = store / TARGET_FILENAME
    if not raw_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError(
            f"Expected {raw_path} and {sample_path}; the GEO NumPy store is incomplete."
        )

    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2:
        raise ValueError(f"{raw_path} must be two-dimensional.")
    sample_index = pd.read_csv(sample_path, low_memory=False)
    partitions = frozen_geo_partitions(sweep_root, sample_index, int(raw.shape[0]))
    target = fit_quantile_reference(
        raw,
        partitions["train"],
        chunk_size=chunk_size,
        progress=progress,
    )

    partial = store / "rma_global.train_reference.partial.npy"
    partial.unlink(missing_ok=True)
    try:
        write_normalized_matrix(raw, target, partial, progress=progress)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    check = np.load(partial, mmap_mode="r", allow_pickle=False)
    if check.shape != raw.shape or check.dtype != np.dtype("float32"):
        partial.unlink(missing_ok=True)
        raise RuntimeError("Generated matrix failed its shape/dtype validation.")
    if not np.isfinite(np.asarray(check[[0, len(check) // 2, len(check) - 1]])).all():
        partial.unlink(missing_ok=True)
        raise RuntimeError("Generated matrix contains non-finite values in validation rows.")
    del check

    prepared_path = Path(sweep_root).expanduser().resolve() / "prepared" / "sample_index.csv"
    created = datetime.now(timezone.utc)
    timestamp = created.strftime("%Y%m%dT%H%M%SZ")
    id_column = next(
        name
        for name in ("GSM", "sample_id", "Sample_ID", "sample", "id")
        if name in sample_index
    )
    row_column = next(
        (
            name
            for name in ("global_row_python", "row_index_python", "sample_row_python")
            if name in sample_index
        ),
        None,
    )
    ordered_index = sample_index.copy()
    ordered_index["_native_row"] = (
        np.arange(len(ordered_index), dtype=np.int64)
        if row_column is None
        else pd.to_numeric(ordered_index[row_column], errors="raise").to_numpy(dtype=np.int64)
    )
    ordered_index = ordered_index.sort_values("_native_row").reset_index(drop=True)
    train_ids = ordered_index.iloc[partitions["train"]]
    train_digest = hashlib.sha256(
        "\n".join(train_ids[id_column].astype(str).str.upper()).encode("utf-8")
    ).hexdigest()
    provenance: dict[str, object] = {
        "format": PROVENANCE_FORMAT,
        "method": METHOD_ID,
        "exact_cel_level_rma": False,
        "created_utc": created.isoformat(),
        "source_matrix": str(raw_path),
        "source_semantics": (
            "Probe-set PM medians; no CEL-level RMA background correction or probe-level "
            "median-polish effects are recoverable from this source."
        ),
        "frozen_split": str(prepared_path),
        "frozen_split_sha256": _sha256(prepared_path),
        "training_sample_ids_sha256": train_digest,
        "n_train_geo": int(len(partitions["train"])),
        "n_validation_geo": int(len(partitions["validation"])),
        "n_test_geo": int(len(partitions["test"])),
        "shape": [int(value) for value in raw.shape],
        "dtype": "float32",
        "algorithm": (
            "Average sorted raw probe-set vectors over frozen GEO training rows; map every "
            "row independently to that frozen target with tie averaging; log2 transform."
        ),
    }

    if not replace:
        candidate = store / "rma_global.train_reference.npy"
        os.replace(partial, candidate)
        np.save(store / "rma_global.train_reference_target.npy", target.astype(np.float32))
        (store / "rma_global.train_reference_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        provenance["output_matrix"] = str(candidate)
        return provenance

    backups: list[str] = []
    if destination.exists():
        backup = _unique_backup(destination, timestamp)
        os.replace(destination, backup)
        backups.append(str(backup))
    try:
        os.replace(partial, destination)
    except Exception:
        if backups and not destination.exists():
            os.replace(Path(backups[0]), destination)
        raise

    for stale_name in (
        "global_rma_streaming_validation.csv",
        "numpy_validation.json",
        PROVENANCE_FILENAME,
    ):
        stale = store / stale_name
        if stale.exists():
            backup = _unique_backup(stale, timestamp)
            os.replace(stale, backup)
            backups.append(str(backup))

    np.save(target_path, target.astype(np.float32))
    provenance["output_matrix"] = str(destination)
    provenance["backups"] = backups
    temporary_provenance = provenance_path.with_suffix(".json.part")
    temporary_provenance.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_provenance, provenance_path)
    _update_store_manifest(store / "store_manifest.csv")

    cache = layout.root / "training_cache" / "rma_global_row_major.npy"
    cache_complete = cache.with_suffix(cache.suffix + ".complete")
    cache.unlink(missing_ok=True)
    cache_complete.unlink(missing_ok=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace GEO_NUMPY_STORE/rma_global.npy and preserve timestamped backups.",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    def report(done: int, total: int, stage: str) -> None:
        interval = 100 if stage == "apply" else max(1, total // 20)
        if done == total or done == 1 or done % interval == 0:
            label = "Fitting training reference" if stage == "fit" else "Normalizing all rows"
            print(f"{label}: {done:,}/{total:,}", flush=True)

    try:
        result = rebuild_global_normalization(
            data_dir=args.data_dir,
            sweep_root=args.sweep_root,
            replace=args.replace,
            chunk_size=args.chunk_size,
            progress=report,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nTrain-reference global normalization completed.")
    print(f"Output: {result['output_matrix']}")
    print(
        "Training/validation/test GEO rows: "
        f"{result['n_train_geo']}/{result['n_validation_geo']}/{result['n_test_geo']}"
    )
    print("Exact CEL-level RMA: no (source is already probe-set summarized).")
    for backup in result.get("backups", []):
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
