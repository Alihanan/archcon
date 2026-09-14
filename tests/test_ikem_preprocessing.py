import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from archcon.data.defaults import project_data_layout
from archcon.data.geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from archcon.data.ikem_preprocessing import prepare_ikem_evaluation_sources
from archcon.data.training_sources import METHOD_PER_DATASET_STANDARDIZED


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_store(
    root: Path,
    raw: np.ndarray,
    samples: list[str],
    probes: list[str],
    split_path: Path,
) -> None:
    root.mkdir(parents=True)
    matrices = {
        "raw_original": raw,
        "rma_per_gse": raw + 100.0,
        "rma_global": raw + 200.0,
    }
    np.save(root / "expression.npy", matrices["rma_per_gse"].astype(np.float32))
    for name, matrix in matrices.items():
        np.save(root / f"{name}.npy", matrix.astype(np.float32))
    pd.DataFrame(
        {"GSM": samples, "row_index_python": np.arange(len(samples), dtype=np.int64)}
    ).to_csv(root / "sample_index.csv", index=False)
    pd.DataFrame({"probe_id": probes}).to_csv(root / "probe_index.csv", index=False)
    (root / "preprocessing_provenance.json").write_text(
        json.dumps(
            {
                "format": 2,
                "source": "GSE290167",
                "frozen_pretraining_split_sha256": _sha256(split_path),
                "methods": {
                    "raw_original": {
                        "uses_egfr": False,
                        "uses_egfr_cv_fold": False,
                        "transductive_across_egfr_folds": False,
                    },
                    "rma_per_gse": {
                        "uses_egfr": False,
                        "uses_egfr_cv_fold": False,
                        "transductive_across_egfr_folds": False,
                    },
                    "rma_global": {
                        "uses_egfr": False,
                        "uses_egfr_cv_fold": False,
                        "transductive_across_egfr_folds": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_ikem_inputs_are_preprocessed_for_each_encoder_arm(tmp_path: Path) -> None:
    probes = ["p1", "p2", "p3"]
    raw = np.asarray([[30.0, 10.0, 20.0], [5.0, 25.0, 15.0]])
    geo = tmp_path / "GEO_NUMPY_STORE"
    geo.mkdir()
    np.save(geo / "rma_global.npy", np.zeros((3, 3), dtype=np.float32))
    (geo / "rma_global_provenance.json").write_text("{}", encoding="utf-8")

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame(
        {
            "row_index_python": [0, 1, 2],
            "sample_key": ["GEO:GSM1", "GEO:GSM2", "GEO:GSM3"],
            "sample_id": ["GSM1", "GSM2", "GSM3"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    pd.DataFrame(
        {"probe_index_python": [0, 1, 2], "probe_id": probes}
    ).to_csv(prepared / "probe_index.csv", index=False)
    np.save(prepared / "train_rows.npy", np.asarray([0], dtype=np.int64))
    np.save(prepared / "validation_rows.npy", np.asarray([1], dtype=np.int64))
    np.save(prepared / "test_rows.npy", np.asarray([2], dtype=np.int64))
    for method, stem in (
        (METHOD_PER_DATASET_STANDARDIZED, "standardized"),
        (METHOD_PER_GSE_RMA, "per_dataset_rma"),
        (METHOD_GLOBAL_RMA, "global_rma"),
    ):
        np.save(prepared / f"geo_columns_{stem}.npy", np.arange(3, dtype=np.int64))
        np.save(prepared / f"geo_rows_{stem}.npy", np.arange(3, dtype=np.int64))
        np.save(prepared / f"supplemental_{stem}.npy", np.empty((0, 3), dtype=np.float32))
    np.save(
        prepared / "standardization_ikem_train_center.npy",
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )
    np.save(
        prepared / "standardization_ikem_train_scale.npy",
        np.asarray([2.0, 5.0, 10.0], dtype=np.float32),
    )
    (prepared / "prepared.json").write_text(
        json.dumps(
            {
                "format": 4,
                "n_samples": 3,
                "n_geo": 3,
                "n_supervised_no_egfr": 0,
                "n_probes": 3,
                "train_rows": "train_rows.npy",
                "validation_rows": "validation_rows.npy",
                "test_rows": "test_rows.npy",
                "sample_index": "sample_index.csv",
                "supplemental_matrices": {
                    METHOD_PER_DATASET_STANDARDIZED: "supplemental_standardized.npy",
                    METHOD_PER_GSE_RMA: "supplemental_per_dataset_rma.npy",
                    METHOD_GLOBAL_RMA: "supplemental_global_rma.npy",
                },
                "method_geo_rows": {
                    METHOD_PER_DATASET_STANDARDIZED: "geo_rows_standardized.npy",
                    METHOD_PER_GSE_RMA: "geo_rows_per_dataset_rma.npy",
                    METHOD_GLOBAL_RMA: "geo_rows_global_rma.npy",
                },
                "method_geo_columns": {
                    METHOD_PER_DATASET_STANDARDIZED: "geo_columns_standardized.npy",
                    METHOD_PER_GSE_RMA: "geo_columns_per_dataset_rma.npy",
                    METHOD_GLOBAL_RMA: "geo_columns_global_rma.npy",
                },
                "method_extra_files": {
                    METHOD_PER_DATASET_STANDARDIZED: {
                        "supplemental_center": "standardization_ikem_train_center.npy",
                        "supplemental_scale": "standardization_ikem_train_scale.npy",
                    }
                },
                "standardization": {
                    "n_ikem_reference_rows": 7,
                    "uses_egfr": False,
                    "uses_egfr_cv_fold": False,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_store(
        tmp_path / "IKEM_NUMPY_STORE",
        raw,
        ["IKEM1", "IKEM2"],
        probes,
        prepared / "sample_index.csv",
    )

    sources = prepare_ikem_evaluation_sources(
        project_data_layout(tmp_path), prepared, tmp_path / "downstream"
    )
    local = np.asarray(sources[METHOD_PER_GSE_RMA].source.matrix)
    np.testing.assert_array_equal(local, raw + 100.0)

    standardized = np.asarray(sources[METHOD_PER_DATASET_STANDARDIZED].source.matrix)
    np.testing.assert_allclose(standardized[0], [10.0, 0.0, 1.0])
    np.testing.assert_allclose(standardized[1], [-2.5, 3.0, 0.5])

    globally_normalized = np.asarray(sources[METHOD_GLOBAL_RMA].source.matrix)
    np.testing.assert_array_equal(globally_normalized, raw + 200.0)
    assert not sources[METHOD_GLOBAL_RMA].provenance["uses_egfr_cv_fold"]
    assert not sources[METHOD_PER_GSE_RMA].provenance[
        "transductive_across_egfr_folds"
    ]
    assert (tmp_path / "downstream" / "preprocessing_provenance.json").is_file()


def test_partial_per_gse_evaluation_does_not_require_global_reference(
    tmp_path: Path,
) -> None:
    probes = ["p1", "p2"]
    matrix = np.asarray([[3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame(
        {
            "row_index_python": [0],
            "sample_key": ["GEO:GSM1"],
            "sample_id": ["GSM1"],
            "split": ["train"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    pd.DataFrame(
        {"probe_index_python": [0, 1], "probe_id": probes}
    ).to_csv(prepared / "probe_index.csv", index=False)
    _write_store(
        tmp_path / "IKEM_NUMPY_STORE",
        matrix - 100.0,
        ["IKEM1", "IKEM2"],
        probes,
        prepared / "sample_index.csv",
    )
    sources = prepare_ikem_evaluation_sources(
        project_data_layout(tmp_path),
        prepared,
        tmp_path / "downstream",
        methods=[METHOD_PER_GSE_RMA],
    )

    assert set(sources) == {METHOD_PER_GSE_RMA}
    np.testing.assert_array_equal(
        np.asarray(sources[METHOD_PER_GSE_RMA].source.matrix), matrix
    )
