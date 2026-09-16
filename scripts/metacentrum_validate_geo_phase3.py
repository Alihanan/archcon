#!/usr/bin/env python3
"""Memory-bounded integrity validation for ArchCon GEO RMA Phase 3 outputs.

The validator reads every stored value without loading a complete matrix into
RAM.  It checks the final Python-facing matrices, the large probe-level Phase 3
checkpoint, completion/progress markers, HDF5 metadata ordering, and exact GSM
membership relative to the frozen sweep split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


FINAL_DATASETS = (
    "expression/raw_original",
    "expression/rma_per_gse",
    "expression/rma_global",
)
ID_COLUMNS = ("GSM", "sample_id", "Sample_ID", "sample", "id")
PROBE_COLUMNS = ("probe_id", "probe", "probeset_id", "Probe.Set.ID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully scan and validate ArchCon Phase 3 GEO RMA outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("ARCHCON_PROJECT_ROOT", Path.cwd())),
    )
    parser.add_argument("--sweep-root", type=Path, default=None)
    parser.add_argument("--rebuild-root", type=Path, default=None)
    parser.add_argument(
        "--block-mib",
        type=float,
        default=48.0,
        help="Approximate raw matrix bytes read per block (default: 48 MiB).",
    )
    parser.add_argument(
        "--skip-probe-level",
        action="store_true",
        help="Skip the full scan of the 17+ GB normalized-PM checkpoint.",
    )
    parser.add_argument(
        "--pass3-block-size",
        type=int,
        default=128,
        help="Probe-set block size used by Phase 3 (default: 128).",
    )
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def normalize_ids(values: Any) -> list[str]:
    return [str(value).strip().upper() for value in values]


def choose_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No {label} column found. Available columns: {list(frame.columns)}")


def record_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)
    print(f"ERROR: {message}", flush=True)


def record_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)
    print(f"WARNING: {message}", flush=True)


def scan_dataset(
    dataset: h5py.Dataset,
    *,
    n_samples: int,
    n_features: int,
    block_mib: float,
    require_positive: bool,
    label: str,
) -> dict[str, Any]:
    expected_forward = (n_samples, n_features)
    expected_reverse = (n_features, n_samples)
    shape = tuple(int(x) for x in dataset.shape)

    if shape == expected_forward:
        sample_axis = 0
    elif shape == expected_reverse:
        sample_axis = 1
    else:
        raise ValueError(
            f"{label}: shape {shape} is neither {expected_forward} nor {expected_reverse}."
        )

    bytes_per_sample = max(1, n_features * dataset.dtype.itemsize)
    target_bytes = max(1, int(block_mib * 1024**2))
    batch_size = max(1, min(256, target_bytes // bytes_per_sample))

    # The working HDF5 was written in eight-array batches.  Aligning larger
    # reads to that boundary avoids unnecessary decompression work.
    if batch_size >= 8:
        batch_size = max(8, (batch_size // 8) * 8)

    feature_min = np.full(n_features, np.inf, dtype=np.float64)
    feature_max = np.full(n_features, -np.inf, dtype=np.float64)
    sample_means = np.full(n_samples, np.nan, dtype=np.float64)
    sample_stds = np.full(n_samples, np.nan, dtype=np.float64)

    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    zero_count = 0
    nonpositive_count = 0
    finite_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    global_min = math.inf
    global_max = -math.inf
    all_zero_samples = 0
    constant_samples = 0

    n_batches = math.ceil(n_samples / batch_size)
    progress_every = max(1, n_batches // 20)

    print(
        f"Scanning {label}: shape={shape}, dtype={dataset.dtype}, "
        f"sample_axis={sample_axis}, batch={batch_size}",
        flush=True,
    )

    for batch_number, start in enumerate(range(0, n_samples, batch_size), start=1):
        stop = min(start + batch_size, n_samples)
        if sample_axis == 0:
            block = np.asarray(dataset[start:stop, :])
        else:
            block = np.asarray(dataset[:, start:stop]).T
        block = np.ascontiguousarray(block)

        if block.shape != (stop - start, n_features):
            raise ValueError(
                f"{label}: read returned {block.shape}; expected "
                f"{(stop - start, n_features)}."
            )

        nan_count += int(np.count_nonzero(np.isnan(block)))
        posinf_count += int(np.count_nonzero(np.isposinf(block)))
        neginf_count += int(np.count_nonzero(np.isneginf(block)))
        zero_count += int(np.count_nonzero(block == 0))
        nonpositive_count += int(np.count_nonzero(block <= 0))

        finite_mask = np.isfinite(block)
        n_finite_block = int(np.count_nonzero(finite_mask))
        finite_count += n_finite_block

        if n_finite_block:
            finite_values = block if n_finite_block == block.size else block[finite_mask]
            block_min = float(np.min(finite_values))
            block_max = float(np.max(finite_values))
            global_min = min(global_min, block_min)
            global_max = max(global_max, block_max)
            total_sum += float(np.sum(finite_values, dtype=np.float64))
            total_sumsq += float(
                np.einsum(
                    "...,...->",
                    finite_values,
                    finite_values,
                    dtype=np.float64,
                    optimize=True,
                )
            )

        if n_finite_block != block.size:
            clean = np.where(finite_mask, block, np.nan)
            with np.errstate(all="ignore"):
                row_min = np.nanmin(clean, axis=1)
                row_max = np.nanmax(clean, axis=1)
                row_mean = np.nanmean(clean, axis=1, dtype=np.float64)
                row_std = np.nanstd(clean, axis=1, dtype=np.float64)
                block_feature_min = np.nanmin(clean, axis=0)
                block_feature_max = np.nanmax(clean, axis=0)
        else:
            row_min = np.min(block, axis=1)
            row_max = np.max(block, axis=1)
            row_mean = np.mean(block, axis=1, dtype=np.float64)
            row_std = np.std(block, axis=1, dtype=np.float64)
            block_feature_min = np.min(block, axis=0)
            block_feature_max = np.max(block, axis=0)

        sample_means[start:stop] = row_mean
        sample_stds[start:stop] = row_std
        all_zero_samples += int(np.count_nonzero((row_min == 0) & (row_max == 0)))
        constant_samples += int(np.count_nonzero(row_min == row_max))
        feature_min = np.fmin(feature_min, block_feature_min)
        feature_max = np.fmax(feature_max, block_feature_max)

        if (
            batch_number == 1
            or batch_number == n_batches
            or batch_number % progress_every == 0
        ):
            print(
                f"  {stop:>6}/{n_samples} samples "
                f"({100.0 * stop / n_samples:5.1f}%)",
                flush=True,
            )

        del block, finite_mask

    variance = (
        max(0.0, total_sumsq / finite_count - (total_sum / finite_count) ** 2)
        if finite_count
        else math.nan
    )
    all_zero_features = int(np.count_nonzero((feature_min == 0) & (feature_max == 0)))
    constant_features = int(np.count_nonzero(feature_min == feature_max))
    nonfinite_features = int(
        np.count_nonzero(~np.isfinite(feature_min) | ~np.isfinite(feature_max))
    )

    result = {
        "shape": list(shape),
        "dtype": str(dataset.dtype),
        "sample_axis": sample_axis,
        "values": int(np.prod(shape, dtype=np.int64)),
        "finite": finite_count,
        "nan": nan_count,
        "positive_inf": posinf_count,
        "negative_inf": neginf_count,
        "zero": zero_count,
        "nonpositive": nonpositive_count,
        "minimum": global_min if finite_count else None,
        "maximum": global_max if finite_count else None,
        "mean": total_sum / finite_count if finite_count else None,
        "standard_deviation": math.sqrt(variance) if finite_count else None,
        "all_zero_samples": all_zero_samples,
        "constant_samples": constant_samples,
        "all_zero_features": all_zero_features,
        "constant_features": constant_features,
        "features_without_finite_values": nonfinite_features,
        "require_positive": require_positive,
        "sample_means": sample_means,
        "sample_stds": sample_stds,
    }

    print(
        f"  min={result['minimum']!r}, max={result['maximum']!r}, "
        f"mean={result['mean']!r}, sd={result['standard_deviation']!r}",
        flush=True,
    )
    print(
        f"  NaN={nan_count}, +Inf={posinf_count}, -Inf={neginf_count}, "
        f"zero={zero_count}, nonpositive={nonpositive_count}",
        flush=True,
    )
    print(
        f"  all-zero samples/features={all_zero_samples}/{all_zero_features}; "
        f"constant samples/features={constant_samples}/{constant_features}",
        flush=True,
    )

    return result


def serializable_scan(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in scan.items()
        if key not in {"sample_means", "sample_stds"}
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if args.sweep_root is not None:
        sweep_root = args.sweep_root.expanduser().resolve()
    else:
        final_sweep = project_root / "sweeps" / "archcon-pretrain-0516"
        bootstrap_sweep = project_root / "sweeps" / "archcon-pretrain-0515"
        sweep_root = (
            final_sweep
            if (final_sweep / "prepared" / "sample_index.csv").is_file()
            else bootstrap_sweep
        )
    rebuild_root = (
        args.rebuild_root.expanduser().resolve()
        if args.rebuild_root is not None
        else project_root / "data" / "GEO_DWNLD_TRAIN_REFERENCE_REBUILD"
    )

    matrix_store = rebuild_root / "GEO_MATRIX_STORE"
    final_h5 = matrix_store / "geo_expression_store.h5"
    work_h5 = rebuild_root / ".GLOBAL_RMA_WORK" / "train_reference_rma_probe_level.h5"
    parameter_h5 = rebuild_root / ".GLOBAL_RMA_WORK" / "train_reference_rma_parameters.h5"
    completion_marker = matrix_store / "GLOBAL_RMA_COMPLETE.txt"
    pass3_done_file = (
        rebuild_root
        / ".GLOBAL_RMA_WORK"
        / "progress"
        / "train_reference_pass3_done_block.txt"
    )
    sample_csv = matrix_store / "sample_index.csv"
    gse_csv = matrix_store / "gse_index.csv"
    probe_csv = matrix_store / "probe_index.csv"
    frozen_candidates = (
        rebuild_root / "prepared_sample_index.csv",
        sweep_root / "prepared" / "sample_index.csv",
    )

    report: dict[str, Any] = {
        "project_root": str(project_root),
        "sweep_root": str(sweep_root),
        "rebuild_root": str(rebuild_root),
        "errors": [],
        "warnings": [],
        "datasets": {},
    }

    required = (
        final_h5,
        work_h5,
        completion_marker,
        pass3_done_file,
        sample_csv,
        gse_csv,
        probe_csv,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        for path in missing:
            record_error(report, f"Required Phase 3 artifact is missing: {path}")
        return 1

    sample_index = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    gse_index = pd.read_csv(gse_csv, dtype=str, keep_default_na=False)
    probe_index = pd.read_csv(probe_csv, dtype=str, keep_default_na=False)
    gsm_column = choose_column(sample_index, ID_COLUMNS, "sample identifier")
    probe_column = choose_column(probe_index, PROBE_COLUMNS, "probe identifier")
    current_gsm = normalize_ids(sample_index[gsm_column])
    current_probes = [str(value).strip() for value in probe_index[probe_column]]
    n_samples = len(current_gsm)
    n_probes = len(current_probes)

    print("=" * 80)
    print("PHASE 3 METADATA AND MEMBERSHIP")
    print("=" * 80)
    print(f"Samples: {n_samples}")
    print(f"Probe sets: {n_probes}")

    if len(set(current_gsm)) != n_samples:
        record_error(report, "GEO_MATRIX_STORE/sample_index.csv contains duplicate GSM IDs.")

    if "global_row_python" in sample_index.columns:
        sample_rows = pd.to_numeric(sample_index["global_row_python"], errors="coerce")
        if sample_rows.isna().any() or not np.array_equal(
            sample_rows.to_numpy(dtype=np.int64), np.arange(n_samples, dtype=np.int64)
        ):
            record_error(report, "sample_index.csv global_row_python is not exactly 0..N-1.")
    else:
        record_error(report, "sample_index.csv lacks global_row_python.")

    if "probe_index_python" in probe_index.columns:
        probe_rows = pd.to_numeric(probe_index["probe_index_python"], errors="coerce")
        if probe_rows.isna().any() or not np.array_equal(
            probe_rows.to_numpy(dtype=np.int64), np.arange(n_probes, dtype=np.int64)
        ):
            record_error(report, "probe_index.csv probe_index_python is not exactly 0..P-1.")
    else:
        record_error(report, "probe_index.csv lacks probe_index_python.")

    required_gse_columns = {
        "GSE",
        "n_samples",
        "start_row_python",
        "stop_row_python",
    }
    if not required_gse_columns.issubset(gse_index.columns):
        record_error(
            report,
            "gse_index.csv lacks columns: "
            + ", ".join(sorted(required_gse_columns - set(gse_index.columns))),
        )
    else:
        starts = pd.to_numeric(gse_index["start_row_python"], errors="coerce")
        stops = pd.to_numeric(gse_index["stop_row_python"], errors="coerce")
        counts = pd.to_numeric(gse_index["n_samples"], errors="coerce")
        if starts.isna().any() or stops.isna().any() or counts.isna().any():
            record_error(report, "gse_index.csv contains non-numeric row ranges/counts.")
        else:
            starts_np = starts.to_numpy(dtype=np.int64)
            stops_np = stops.to_numpy(dtype=np.int64)
            counts_np = counts.to_numpy(dtype=np.int64)
            contiguous = (
                len(starts_np) > 0
                and starts_np[0] == 0
                and stops_np[-1] == n_samples
                and np.array_equal(starts_np[1:], stops_np[:-1])
            )
            if not contiguous:
                record_error(report, "gse_index.csv ranges do not cover samples contiguously.")
            if not np.array_equal(stops_np - starts_np, counts_np):
                record_error(report, "gse_index.csv row ranges disagree with n_samples.")
            observed_gse_counts = sample_index.groupby("GSE", sort=False).size().to_dict()
            indexed_gse_counts = dict(
                zip(gse_index["GSE"].astype(str), counts_np.tolist(), strict=True)
            )
            if observed_gse_counts != indexed_gse_counts:
                record_error(report, "gse_index.csv counts/order disagree with sample_index.csv.")

    frozen_path = next((path for path in frozen_candidates if path.is_file()), None)
    if frozen_path is None:
        record_error(
            report,
            "No frozen sample index was found in rebuild root or sweep/prepared.",
        )
    else:
        frozen = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
        if "source_kind" in frozen.columns:
            frozen = frozen[
                frozen["source_kind"].astype(str).str.strip().str.lower() == "geo"
            ].copy()
        frozen_id_column = choose_column(frozen, ID_COLUMNS, "frozen sample identifier")
        frozen_gsm = normalize_ids(frozen[frozen_id_column])
        current_set = set(current_gsm)
        frozen_set = set(frozen_gsm)
        missing_from_store = sorted(frozen_set - current_set)
        unexpected_in_store = sorted(current_set - frozen_set)
        report["membership"] = {
            "frozen_file": str(frozen_path),
            "frozen_geo_samples": len(frozen_gsm),
            "matrix_store_samples": n_samples,
            "missing_from_store_count": len(missing_from_store),
            "unexpected_in_store_count": len(unexpected_in_store),
            "missing_from_store_first_20": missing_from_store[:20],
            "unexpected_in_store_first_20": unexpected_in_store[:20],
        }
        print(f"Frozen GEO samples: {len(frozen_gsm)} ({frozen_path})")
        if len(set(frozen_gsm)) != len(frozen_gsm):
            record_error(report, "Frozen GEO sample index contains duplicate IDs.")
        if missing_from_store:
            record_error(
                report,
                f"{len(missing_from_store)} frozen GEO GSM IDs are absent from the "
                f"matrix store. First IDs: {', '.join(missing_from_store[:10])}",
            )
        if unexpected_in_store:
            record_error(
                report,
                f"{len(unexpected_in_store)} matrix-store GSM IDs are absent from the "
                f"frozen split. First IDs: {', '.join(unexpected_in_store[:10])}",
            )

    expected_blocks = math.ceil(n_probes / args.pass3_block_size)
    done_blocks: list[int] = []
    for line in pass3_done_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done_blocks.append(int(line))
            except ValueError:
                record_error(report, f"Invalid PASS3 completion line: {line!r}")
    unique_done = sorted(set(done_blocks))
    expected_done = list(range(1, expected_blocks + 1))
    report["pass3_blocks"] = {
        "expected": expected_blocks,
        "lines": len(done_blocks),
        "unique": len(unique_done),
        "complete": unique_done == expected_done,
    }
    if unique_done != expected_done:
        missing_blocks = sorted(set(expected_done) - set(unique_done))
        extra_blocks = sorted(set(unique_done) - set(expected_done))
        record_error(
            report,
            f"PASS3 block checkpoint is incomplete/inconsistent: "
            f"missing={missing_blocks[:20]}, extra={extra_blocks[:20]}.",
        )
    else:
        print(f"PASS3 checkpoint: all {expected_blocks} blocks recorded.")

    marker_text = completion_marker.read_text(encoding="utf-8").strip()
    report["completion_marker"] = marker_text
    print(f"Completion marker: {marker_text}")
    if "TRAIN_REFERENCE_RMA_COMPLETE" not in marker_text:
        record_error(report, "GLOBAL_RMA_COMPLETE.txt has an unexpected status string.")
    marker_samples = re.search(r"samples=\s*(\d+)", marker_text)
    marker_probes = re.search(r"probes=\s*(\d+)", marker_text)
    if marker_samples is None or int(marker_samples.group(1)) != n_samples:
        record_error(report, "Completion-marker sample count disagrees with sample_index.csv.")
    if marker_probes is None or int(marker_probes.group(1)) != n_probes:
        record_error(report, "Completion-marker probe count disagrees with probe_index.csv.")

    print("=" * 80)
    print("FINAL HDF5 FULL SCAN")
    print("=" * 80)
    global_scan: dict[str, Any] | None = None
    with h5py.File(final_h5, "r") as handle:
        for name in FINAL_DATASETS:
            if name not in handle:
                record_error(report, f"Final HDF5 dataset is missing: /{name}")
                continue
            dataset = handle[name]
            try:
                scan = scan_dataset(
                    dataset,
                    n_samples=n_samples,
                    n_features=n_probes,
                    block_mib=args.block_mib,
                    require_positive=name == "expression/raw_original",
                    label=f"/{name}",
                )
            except Exception as exc:
                record_error(report, f"/{name} could not be scanned: {exc}")
                continue

            report["datasets"][name] = serializable_scan(scan)
            bad_nonfinite = scan["nan"] + scan["positive_inf"] + scan["negative_inf"]
            if bad_nonfinite:
                record_error(report, f"/{name} contains {bad_nonfinite} non-finite values.")
            if scan["all_zero_samples"] or scan["all_zero_features"]:
                record_error(
                    report,
                    f"/{name} contains all-zero samples/features: "
                    f"{scan['all_zero_samples']}/{scan['all_zero_features']}.",
                )
            if scan["constant_samples"]:
                record_error(
                    report,
                    f"/{name} contains {scan['constant_samples']} constant sample rows.",
                )
            if scan["constant_features"]:
                record_warning(
                    report,
                    f"/{name} contains {scan['constant_features']} constant probe columns; "
                    "review if this is unexpectedly large.",
                )
            if scan["require_positive"] and scan["nonpositive"]:
                record_error(
                    report,
                    f"/{name} should contain positive intensities but has "
                    f"{scan['nonpositive']} nonpositive values.",
                )
            if name in {"expression/rma_per_gse", "expression/rma_global"}:
                max_abs = max(abs(scan["minimum"] or 0), abs(scan["maximum"] or 0))
                if max_abs > 100:
                    record_warning(
                        report,
                        f"/{name} has |value| > 100 ({max_abs}); this is unusual for "
                        "log2 RMA expression.",
                    )
            if name == "expression/rma_global":
                global_scan = scan

        metadata_checks = (
            ("metadata/GSM", current_gsm, True),
            ("metadata/probe_id", current_probes, False),
        )
        for name, expected, uppercase in metadata_checks:
            if name not in handle:
                record_error(report, f"Final HDF5 metadata is missing: /{name}")
                continue
            observed = decode_strings(handle[name][...])
            if uppercase:
                observed = normalize_ids(observed)
            else:
                observed = [value.strip() for value in observed]
            if observed != expected:
                record_error(report, f"/{name} does not exactly match its CSV ordering.")

        if "metadata/global_row_python" in handle:
            rows = np.asarray(handle["metadata/global_row_python"][...]).reshape(-1)
            if not np.array_equal(rows.astype(np.int64), np.arange(n_samples, dtype=np.int64)):
                record_error(report, "/metadata/global_row_python is not exactly 0..N-1.")
        else:
            record_error(report, "Final HDF5 metadata is missing: /metadata/global_row_python")

        if "pretraining_split" in sample_index.columns:
            expected_splits = [
                str(value).strip().lower()
                for value in sample_index["pretraining_split"]
            ]
            if "metadata/pretraining_split" not in handle:
                record_error(
                    report,
                    "Final HDF5 metadata is missing: /metadata/pretraining_split",
                )
            else:
                observed_splits = [
                    value.strip().lower()
                    for value in decode_strings(handle["metadata/pretraining_split"][...])
                ]
                if observed_splits != expected_splits:
                    record_error(
                        report,
                        "/metadata/pretraining_split does not match sample_index.csv ordering.",
                    )

    if global_scan is not None and "pretraining_split" in sample_index.columns:
        print("=" * 80)
        print("GLOBAL-RMA SAMPLE DISTRIBUTIONS BY FROZEN SPLIT")
        print("=" * 80)
        labels = sample_index["pretraining_split"].astype(str).str.strip().str.lower().to_numpy()
        split_summary: dict[str, Any] = {}
        for label in sorted(set(labels)):
            selected = labels == label
            means = global_scan["sample_means"][selected]
            stds = global_scan["sample_stds"][selected]
            split_summary[label] = {
                "samples": int(np.count_nonzero(selected)),
                "mean_of_sample_means": float(np.mean(means)),
                "minimum_sample_mean": float(np.min(means)),
                "maximum_sample_mean": float(np.max(means)),
                "mean_within_sample_sd": float(np.mean(stds)),
            }
            print(
                f"{label:>10}: n={split_summary[label]['samples']}, "
                f"sample-mean range="
                f"[{split_summary[label]['minimum_sample_mean']:.6g}, "
                f"{split_summary[label]['maximum_sample_mean']:.6g}], "
                f"mean(within-sample SD)="
                f"{split_summary[label]['mean_within_sample_sd']:.6g}"
            )
        report["global_rma_split_summary"] = split_summary

    if not args.skip_probe_level:
        print("=" * 80)
        print("PROBE-LEVEL CHECKPOINT FULL SCAN")
        print("=" * 80)
        with h5py.File(work_h5, "r") as handle:
            name = "normalized_common_pm"
            if name not in handle:
                record_error(report, f"Probe-level HDF5 dataset is missing: /{name}")
            else:
                shape = tuple(int(value) for value in handle[name].shape)
                if n_samples not in shape:
                    record_error(
                        report,
                        f"Probe-level dataset shape {shape} has no sample axis of {n_samples}.",
                    )
                else:
                    n_common_pm = shape[1] if shape[0] == n_samples else shape[0]
                    try:
                        scan = scan_dataset(
                            handle[name],
                            n_samples=n_samples,
                            n_features=n_common_pm,
                            block_mib=args.block_mib,
                            require_positive=True,
                            label=f"/{name}",
                        )
                    except Exception as exc:
                        record_error(report, f"/{name} could not be scanned: {exc}")
                    else:
                        report["datasets"][f"work/{name}"] = serializable_scan(scan)
                        bad_nonfinite = (
                            scan["nan"] + scan["positive_inf"] + scan["negative_inf"]
                        )
                        if bad_nonfinite:
                            record_error(
                                report,
                                f"Probe-level checkpoint contains {bad_nonfinite} "
                                "non-finite values.",
                            )
                        if scan["nonpositive"]:
                            record_error(
                                report,
                                f"Probe-level checkpoint contains {scan['nonpositive']} "
                                "nonpositive values; PASS3 log2 would be invalid.",
                            )
                        if scan["all_zero_samples"] or scan["all_zero_features"]:
                            record_error(
                                report,
                                "Probe-level checkpoint contains all-zero sample/feature regions.",
                            )
                        if scan["constant_samples"]:
                            record_error(
                                report,
                                f"Probe-level checkpoint contains {scan['constant_samples']} "
                                "constant sample arrays.",
                            )

    report["stage4_parameter_file"] = {
        "path": str(parameter_h5),
        "exists": parameter_h5.is_file(),
    }
    if not parameter_h5.is_file():
        record_warning(
            report,
            "Phase 3 numerical outputs can still be valid, but Stage 4 remains blocked: "
            "train_reference_rma_parameters.h5 is absent.",
        )

    report_path = matrix_store / "phase3_full_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 80)
    if report["errors"]:
        print(f"PHASE 3 VALIDATION FAILED: {len(report['errors'])} hard error(s).")
        print("Do not build the parameter sidecar or run Stage 4 yet.")
        status = 1
    else:
        print("PHASE 3 NUMERICAL AND STRUCTURAL VALIDATION PASSED.")
        if parameter_h5.is_file():
            print("Stage 4 parameter sidecar is present.")
        else:
            print("Stage 4 is still blocked only by the missing parameter sidecar.")
        status = 0
    print(f"Report: {report_path}")
    print("=" * 80)
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nValidation interrupted; no source HDF5 file was modified.", file=sys.stderr)
        raise SystemExit(130)
