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


def _membership_sha256(namespace: str, sample_ids: list[str]) -> str:
    values = sorted(f"{namespace.upper()}:{value.upper()}" for value in sample_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


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
        {
            "GSM": samples,
            "row_index_python": np.arange(len(samples), dtype=np.int64),
            "donor_id": [value.rsplit("_", 1)[0] for value in samples],
            "training_role": [
                "pretrain_train_no_egfr",
                "pretrain_validation_no_egfr",
            ],
            "pretraining_split": ["train", "validation"],
        }
    ).to_csv(root / "sample_index.csv", index=False)
    pd.DataFrame({"probe_id": probes}).to_csv(root / "probe_index.csv", index=False)
    (root / "preprocessing_provenance.json").write_text(
        json.dumps(
            {
                "format": 4,
                "source": "private IKEM CEL collection",
                "outcome_gate_unit": "donor",
                "split_unit": "donor",
                "split_seed": 20260915,
                "validation_fraction": 0.20,
                "cohort_samples": 2,
                "no_measured_egfr_biopsies": 2,
                "donor_clean_pretraining_eligible_samples": 2,
                "pretraining_train_samples": 1,
                "pretraining_validation_samples": 1,
                "related_no_egfr_held_out_samples": 0,
                "held_out_measured_egfr_samples": 0,
                "pretraining_validation_donors": [samples[1].rsplit("_", 1)[0]],
                "ikem_train_sample_ids_sha256": _membership_sha256(
                    "SUPERVISED", [samples[0]]
                ),
                "ikem_validation_sample_ids_sha256": _membership_sha256(
                    "SUPERVISED", [samples[1]]
                ),
                "frozen_pretraining_split_sha256": _sha256(split_path),
                "methods": {
                    "raw_original": {
                        "uses_outcome_values_in_fit": False,
                        "uses_egfr_cv_fold": False,
                        "transductive_across_egfr_folds": False,
                    },
                    "rma_per_gse": {
                        "uses_outcome_values_in_fit": False,
                        "uses_egfr_cv_fold": False,
                        "transductive_across_egfr_folds": False,
                    },
                    "rma_global": {
                        "uses_outcome_values_in_fit": False,
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
            "row_index_python": [0, 1, 2, 3, 4],
            "sample_key": [
                "GEO:GSM1",
                "GEO:GSM2",
                "GEO:GSM3",
                "SUPERVISED:D001_L",
                "SUPERVISED:D002_L",
            ],
            "sample_id": ["GSM1", "GSM2", "GSM3", "D001_L", "D002_L"],
            "source_kind": ["geo", "geo", "geo", "ikem", "ikem"],
            "donor_id": [None, None, None, "D001", "D002"],
            "split": ["train", "validation", "test", "train", "validation"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    pd.DataFrame(
        {"probe_index_python": [0, 1, 2], "probe_id": probes}
    ).to_csv(prepared / "probe_index.csv", index=False)
    np.save(prepared / "train_rows.npy", np.asarray([0, 3], dtype=np.int64))
    np.save(prepared / "validation_rows.npy", np.asarray([1, 4], dtype=np.int64))
    np.save(prepared / "test_rows.npy", np.asarray([2], dtype=np.int64))
    for method, stem in (
        (METHOD_PER_DATASET_STANDARDIZED, "standardized"),
        (METHOD_PER_GSE_RMA, "per_dataset_rma"),
        (METHOD_GLOBAL_RMA, "global_rma"),
    ):
        np.save(prepared / f"geo_columns_{stem}.npy", np.arange(3, dtype=np.int64))
        np.save(prepared / f"geo_rows_{stem}.npy", np.arange(3, dtype=np.int64))
        np.save(prepared / f"supplemental_{stem}.npy", raw.astype(np.float32))
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
                "format": 5,
                "n_samples": 5,
                "n_geo": 3,
                "n_ikem_donor_clean_no_egfr": 2,
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
                    "n_ikem_reference_rows": 1,
                    "uses_outcome_values_in_fit": False,
                    "uses_egfr_cv_fold": False,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_store(
        tmp_path / "IKEM_CEL_NUMPY_STORE",
        raw,
        ["D001_L", "D002_L"],
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
            "row_index_python": [0, 1, 2],
            "sample_key": [
                "GEO:GSM1",
                "SUPERVISED:D001_L",
                "SUPERVISED:D002_L",
            ],
            "sample_id": ["GSM1", "D001_L", "D002_L"],
            "source_kind": ["geo", "ikem", "ikem"],
            "donor_id": [None, "D001", "D002"],
            "split": ["train", "train", "validation"],
        }
    ).to_csv(prepared / "sample_index.csv", index=False)
    pd.DataFrame(
        {"probe_index_python": [0, 1], "probe_id": probes}
    ).to_csv(prepared / "probe_index.csv", index=False)
    _write_store(
        tmp_path / "IKEM_CEL_NUMPY_STORE",
        matrix - 100.0,
        ["D001_L", "D002_L"],
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
