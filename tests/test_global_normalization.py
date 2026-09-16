from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archcon.data.defaults import project_data_layout
from archcon.data.geo_rma import METHOD_GLOBAL_RMA
from archcon.data.training_sources import load_prepared_pretraining_source
from archcon.rebuild_global_normalization import (
    PROVENANCE_FILENAME,
    fit_quantile_reference,
    frozen_geo_partitions,
    normalize_one_row,
    rebuild_global_normalization,
)


def test_reference_is_fitted_only_from_training_rows() -> None:
    raw = np.asarray(
        [
            [1.0, 4.0, 2.0],
            [2.0, 8.0, 4.0],
            [1000.0, 3000.0, 2000.0],
        ],
        dtype=np.float32,
    )
    target = fit_quantile_reference(raw, np.asarray([0, 1]))
    assert np.allclose(target, [1.5, 3.0, 6.0])
    assert not np.allclose(target, np.sort(raw, axis=1).mean(axis=0))


def test_apply_reference_handles_ties_without_using_other_rows() -> None:
    target = np.asarray([2.0, 4.0, 8.0])
    result = normalize_one_row(np.asarray([10.0, 10.0, 20.0]), target)
    assert np.allclose(result, [np.log2(3.0), np.log2(3.0), 3.0])


def test_frozen_partitions_map_gsm_to_native_rows(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    (sweep / "prepared").mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_key": ["GEO:GSM2", "GEO:GSM1", "GEO:GSM3", "SUPERVISED:X"],
            "split": ["train", "validation", "test", "train"],
        }
    ).to_csv(sweep / "prepared" / "sample_index.csv", index=False)
    store_index = pd.DataFrame(
        {"GSM": ["GSM1", "GSM2", "GSM3"], "global_row_python": [0, 1, 2]}
    )
    partitions = frozen_geo_partitions(sweep, store_index, 3)
    assert partitions["train"].tolist() == [1]
    assert partitions["validation"].tolist() == [0]
    assert partitions["test"].tolist() == [2]


def _write_minimal_store(root: Path) -> None:
    store = root / "GEO_NUMPY_STORE"
    store.mkdir(parents=True)
    raw = np.asarray(
        [[1.0, 4.0, 2.0], [2.0, 8.0, 4.0], [1000.0, 3000.0, 2000.0]],
        dtype=np.float32,
    )
    np.save(store / "raw_original.npy", raw)
    np.save(store / "rma_global.npy", raw + 99.0)
    np.save(store / "rma_per_gse.npy", raw + 10.0)
    np.save(store / "per_dataset_raw_original.npy", raw)
    np.save(store / "per_dataset_rma_per_gse.npy", raw + 10.0)
    pd.DataFrame(
        {"GSM": ["GSM1", "GSM2", "GSM3"], "global_row_python": [0, 1, 2]}
    ).to_csv(store / "sample_index.csv", index=False)
    pd.DataFrame({"GSM": ["GSM1", "GSM2", "GSM3"], "source_GSE": ["A", "B", "C"]}).to_csv(
        store / "source_sample_occurrences.csv", index=False
    )
    pd.DataFrame({"source_GSE": ["A", "B", "C"]}).to_csv(
        store / "source_gse_occurrence_index.csv", index=False
    )
    pd.DataFrame({"GSE": ["A", "B", "C"]}).to_csv(store / "gse_index.csv", index=False)
    pd.DataFrame({"probe_id": ["p1", "p2", "p3"]}).to_csv(
        store / "probe_index.csv", index=False
    )
    pd.DataFrame({"GSM": ["GSM1", "GSM2", "GSM3"]}).to_csv(
        store / "cel_manifest.csv", index=False
    )


def test_rebuild_replaces_matrix_and_writes_matching_provenance(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_minimal_store(data)
    sweep = tmp_path / "sweep"
    prepared = sweep / "prepared"
    prepared.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_index_python": [0, 1, 2],
            "sample_key": ["GEO:GSM1", "GEO:GSM2", "GEO:GSM3"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    result = rebuild_global_normalization(
        data_dir=data,
        sweep_root=sweep,
        replace=True,
        chunk_size=2,
    )
    rebuilt = np.load(data / "GEO_NUMPY_STORE" / "rma_global.npy")
    assert np.allclose(rebuilt[0], np.log2([1.0, 4.0, 2.0]))
    assert np.allclose(rebuilt[2], np.log2([1.0, 4.0, 2.0]))
    assert result["n_train_geo"] == 1
    assert (data / "GEO_NUMPY_STORE" / PROVENANCE_FILENAME).is_file()
    assert list((data / "GEO_NUMPY_STORE").glob("rma_global.all_samples_backup_*.npy"))


def test_prepared_global_loader_refuses_missing_provenance(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_minimal_store(data)
    prepared = tmp_path / "sweep" / "prepared"
    prepared.mkdir(parents=True)
    np.save(prepared / "geo_rows_global_rma.npy", np.arange(3))
    np.save(prepared / "geo_columns_global_rma.npy", np.arange(3))
    np.save(prepared / "supervised.npy", np.empty((0, 3), dtype=np.float32))
    pd.DataFrame(
        {
            "row_index_python": [0, 1, 2],
            "sample_key": ["GEO:GSM1", "GEO:GSM2", "GEO:GSM3"],
            "sample_id": ["GSM1", "GSM2", "GSM3"],
            "source_kind": ["geo", "geo", "geo"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    (prepared / "prepared.json").write_text(
        """{
          "format": 5,
          "n_samples": 3,
          "n_probes": 3,
          "sample_index": "sample_index.csv",
          "supplemental_matrices": {"Global RMA": "supervised.npy"},
          "method_geo_rows": {"Global RMA": "geo_rows_global_rma.npy"},
          "method_geo_columns": {"Global RMA": "geo_columns_global_rma.npy"}
        }"""
    )
    with pytest.raises(RuntimeError, match="no train-reference provenance"):
        load_prepared_pretraining_source(
            project_data_layout(data), METHOD_GLOBAL_RMA, prepared
        )
