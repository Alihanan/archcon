import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archcon.data.defaults import project_data_layout
from archcon.data.geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from archcon.data.training_sources import (
    METHOD_PER_DATASET_STANDARDIZED,
    assert_probe_alignment,
    create_shared_preprocessing_split,
    load_ikem_source,
    load_prepared_pretraining_source,
    load_prepared_split_rows,
    load_pretraining_source,
    load_training_source,
    prepare_pretraining_assets,
    split_rows_for_source,
    validate_pretraining_split,
)


def _membership_sha256(namespace: str, sample_ids: list[str]) -> str:
    values = sorted(f"{namespace.upper()}:{value.upper()}" for value in sample_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _write_geo_rma_store(root: Path, samples: list[str], probes: list[str]) -> None:
    root.mkdir(parents=True)
    n = len(samples)
    p = len(probes)
    gses = [f"GSE{i+1}" for i in range(n)]
    sample_index = pd.DataFrame(
        {
            "GSM": samples,
            "GSE": gses,
            "source_GSE": gses,
            "global_row_python": range(n),
        }
    )
    sample_index.to_csv(root / "sample_index.csv", index=False)
    pd.DataFrame(
        {
            "GSM": samples,
            "source_GSE": gses,
            "canonical_GSE": gses,
            "occurrence_row_python": range(n),
        }
    ).to_csv(root / "source_sample_occurrences.csv", index=False)
    pd.DataFrame(
        {"GSE": gses, "n_samples": [1] * n, "start_row_python": list(range(n)), "stop_row_python": list(range(1, n + 1))}
    ).to_csv(root / "gse_index.csv", index=False)
    pd.DataFrame(
        {"source_GSE": gses, "n_samples": [1] * n, "start_row_python": list(range(n)), "stop_row_python": list(range(1, n + 1))}
    ).to_csv(root / "source_gse_occurrence_index.csv", index=False)
    pd.DataFrame({"probe_index_python": range(p), "probe_id": probes}).to_csv(
        root / "probe_index.csv", index=False
    )
    pd.DataFrame({"GSM": samples, "GSE": gses, "cel_source_GSE": gses}).to_csv(
        root / "cel_manifest.csv", index=False
    )
    matrix = np.arange(n * p, dtype=np.float32).reshape(n, p)
    np.save(root / "raw_original.npy", matrix)
    np.save(root / "rma_per_gse.npy", matrix + 10)
    np.save(root / "rma_global.npy", matrix + 20)
    np.save(root / "per_dataset_raw_original.npy", matrix)
    np.save(root / "per_dataset_rma_per_gse.npy", matrix + 10)


def _write_simple_store(root: Path, samples: list[str], probes: list[str], offset: float) -> None:
    root.mkdir(parents=True)
    matrix = np.arange(len(samples) * len(probes), dtype=np.float32).reshape(len(samples), len(probes))
    values = matrix + offset
    np.save(root / "expression.npy", values)
    np.save(root / "raw_original.npy", values)
    np.save(root / "rma_per_gse.npy", values)
    np.save(root / "rma_global.npy", values)
    pd.DataFrame({"GSM": samples, "row_index_python": range(len(samples))}).to_csv(
        root / "sample_index.csv", index=False
    )
    pd.DataFrame({"probe_id": probes}).to_csv(root / "probe_index.csv", index=False)


def _write_ikem_store(root: Path, probes: list[str]) -> None:
    samples = ["D001_L", "D002_L", "D003_L", "D004_L"]
    _write_simple_store(root, samples, probes, 200)
    roles = [
        "held_out_measured_egfr",
        "pretrain_train_no_egfr",
        "pretrain_train_no_egfr",
        "pretrain_validation_no_egfr",
    ]
    pd.DataFrame(
        {
            "GSM": samples,
            "row_index_python": range(len(samples)),
            "donor_id": ["D001", "D002", "D003", "D004"],
            "training_role": roles,
            "pretraining_split": [None, "train", "train", "validation"],
        }
    ).to_csv(root / "sample_index.csv", index=False)
    (root / "preprocessing_provenance.json").write_text(
        json.dumps(
            {
                "format": 4,
                "source": "private IKEM CEL collection",
                "outcome_gate_unit": "donor",
                "split_unit": "donor",
                "split_seed": 20260915,
                "validation_fraction": 0.20,
                "cohort_samples": 4,
                "no_measured_egfr_biopsies": 3,
                "donor_clean_pretraining_eligible_samples": 3,
                "pretraining_train_samples": 2,
                "pretraining_validation_samples": 1,
                "related_no_egfr_held_out_samples": 0,
                "held_out_measured_egfr_samples": 1,
                "pretraining_validation_donors": ["D004"],
                "ikem_train_sample_ids_sha256": _membership_sha256(
                    "SUPERVISED", ["D002_L", "D003_L"]
                ),
                "ikem_validation_sample_ids_sha256": _membership_sha256(
                    "SUPERVISED", ["D004_L"]
                ),
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


def _layout(tmp_path: Path):
    samples = [f"GSM{i}" for i in range(8)]
    probes = [f"probe_{i}" for i in range(5)]
    _write_geo_rma_store(tmp_path / "GEO_NUMPY_STORE", samples, probes)
    # Deliberately reverse row order: the fixed split must map by GSM, not row number.
    # Deliberately reverse both row and probe order: prepared sweeps must freeze
    # mappings by identity rather than requiring native stores to share order.
    _write_simple_store(
        tmp_path / "GEO_STADNIUK_STORE",
        list(reversed(samples)),
        list(reversed(probes)),
        100,
    )
    _write_ikem_store(tmp_path / "IKEM_CEL_NUMPY_STORE", probes)
    pd.DataFrame(
        {
            "patient": ["D001_L", "D002_L", "D003_L", "D004_L"],
            "egfr_7d": [50.0, None, None, None],
            "egfr_3m": [55.0, None, None, None],
        }
    ).to_excel(tmp_path / "egfr_data.xlsx", index=False)
    return project_data_layout(tmp_path), samples, probes


@pytest.mark.parametrize(
    ("method", "native_method"),
    [
        (METHOD_PER_DATASET_STANDARDIZED, "raw_original"),
        (METHOD_PER_GSE_RMA, "rma_per_gse"),
        (METHOD_GLOBAL_RMA, "rma_global"),
    ],
)
def test_ikem_method_source_rejects_transductive_provenance(
    tmp_path: Path,
    method: str,
    native_method: str,
) -> None:
    layout, _, _ = _layout(tmp_path)
    provenance_path = tmp_path / "IKEM_CEL_NUMPY_STORE" / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["methods"][native_method]["transductive_across_egfr_folds"] = True
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-transductive"):
        load_ikem_source(layout, method=method)


def test_ikem_method_source_rejects_mismatched_membership_hash(tmp_path: Path) -> None:
    layout, _, _ = _layout(tmp_path)
    provenance_path = tmp_path / "IKEM_CEL_NUMPY_STORE" / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["ikem_validation_sample_ids_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeError, match="membership hashes"):
        load_ikem_source(layout, method=METHOD_PER_GSE_RMA)


def test_shared_split_is_sampled_once_and_maps_by_gsm(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    first = create_shared_preprocessing_split(layout, seed=42, train_fraction=0.75)
    second = create_shared_preprocessing_split(layout, seed=42, train_fraction=0.75)
    assert first[["GSM", "split"]].equals(second[["GSM", "split"]])
    assert set(first.loc[first["dataset_role"] == "unsupervised data · GEO", "GSM"]) == set(samples)
    assert (first["dataset_role"] == "IKEM · donor-clean no eGFR").sum() == 3
    assert (first["split"] == "train").sum() >= 5
    assert (first["split"] == "validation").sum() >= 1
    assert (first["split"] == "test").sum() >= 1

    rma = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    standardized = load_pretraining_source(layout, METHOD_PER_DATASET_STANDARDIZED)
    rma_train, rma_val, rma_test = split_rows_for_source(first, rma, include_test=True)
    stad_train, stad_val, stad_test = split_rows_for_source(first, standardized, include_test=True)
    rma_ids = set(rma.sample_index.iloc[rma_train]["sample_key"])
    stad_ids = set(standardized.sample_index.iloc[stad_train]["sample_key"])
    assert rma_ids == stad_ids
    assert len(rma_val) == len(stad_val)
    assert len(rma_test) == len(stad_test)


def test_all_three_training_preprocessings_and_ikem_are_memory_mapped(tmp_path: Path) -> None:
    layout, _, _ = _layout(tmp_path)
    for method in (METHOD_PER_DATASET_STANDARDIZED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA):
        source = load_training_source(layout, method)
        assert isinstance(source.matrix, np.memmap)
        assert source.matrix.shape == (8, 5)
    ikem = load_ikem_source(layout)
    assert ikem is not None
    assert isinstance(ikem.matrix, np.memmap)
    assert ikem.matrix.shape == (4, 5)


def test_ikem_probe_mismatch_is_rejected(tmp_path: Path) -> None:
    layout, _, probes = _layout(tmp_path)
    pd.DataFrame({"probe_id": list(reversed(probes))}).to_csv(
        layout.ikem_store / "probe_index.csv", index=False
    )
    training = load_training_source(layout, METHOD_GLOBAL_RMA)
    ikem = load_ikem_source(layout)
    assert ikem is not None
    with pytest.raises(ValueError, match="Probe order differs"):
        assert_probe_alignment(training, ikem)


def test_comparison_pretraining_offers_all_three_preprocessings() -> None:
    from archcon.data.training_sources import TRAINING_PREPROCESSING_OPTIONS

    assert TRAINING_PREPROCESSING_OPTIONS == [
        METHOD_PER_DATASET_STANDARDIZED,
        METHOD_PER_GSE_RMA,
        METHOD_GLOBAL_RMA,
    ]


def test_pretraining_source_appends_only_supervised_samples_without_egfr(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    source = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    assert source.matrix.shape == (len(samples) + 3, 5)
    roles = source.sample_index["dataset_role"].value_counts().to_dict()
    assert roles["unsupervised data · GEO"] == len(samples)
    assert roles["IKEM · donor-clean no eGFR"] == 3
    assert "SUPERVISED:D001_L" not in set(source.sample_index["sample_key"])
    assert {"SUPERVISED:D002_L", "SUPERVISED:D003_L", "SUPERVISED:D004_L"}.issubset(
        set(source.sample_index["sample_key"])
    )


def test_shared_split_adds_supervised_no_egfr_rows_and_maps_to_combined_source(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    geo = split.loc[split["dataset_role"] == "unsupervised data · GEO"]
    supervised = split.loc[split["dataset_role"] == "IKEM · donor-clean no eGFR"]
    assert set(geo["GSM"].dropna()) == set(samples)
    assert len(supervised) == 3
    assert set(supervised["split"]) == {"train", "validation"}
    assert set(supervised.loc[supervised["split"].eq("validation"), "donor_id"]) == {
        "D004"
    }
    source = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    train, validation, test = split_rows_for_source(split, source, include_test=True)
    assert len(train) + len(validation) + len(test) == len(samples) + 3


def test_loaded_split_cannot_change_frozen_geo_assignments(tmp_path: Path) -> None:
    layout, _, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    changed = split.copy()
    geo_train = changed.index[
        changed["sample_key"].str.startswith("GEO:") & changed["split"].eq("train")
    ][0]
    geo_test = changed.index[
        changed["sample_key"].str.startswith("GEO:") & changed["split"].eq("test")
    ][0]
    changed.loc[geo_train, "split"] = "test"
    changed.loc[geo_test, "split"] = "train"
    path = tmp_path / "changed_split.csv"
    changed.to_csv(path, index=False)

    with pytest.raises(ValueError, match="frozen GEO/IKEM assignments"):
        validate_pretraining_split(layout, path)



def test_prepared_assets_align_extra_supervised_probes_and_freeze_rows(tmp_path: Path) -> None:
    layout, samples, probes = _layout(tmp_path)
    supervised_samples = ["D001_L", "D002_L", "D003_L", "D004_L"]
    extra_probes = ["unused_extra_probe", *probes]
    values = np.arange(
        len(supervised_samples) * len(extra_probes), dtype=np.float32
    ).reshape(len(supervised_samples), len(extra_probes))
    for filename in (
        "expression.npy",
        "raw_original.npy",
        "rma_per_gse.npy",
        "rma_global.npy",
    ):
        np.save(layout.ikem_store / filename, values)
    pd.DataFrame({"probe_id": extra_probes}).to_csv(
        layout.ikem_store / "probe_index.csv", index=False
    )

    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    prepared = prepare_pretraining_assets(
        layout,
        split,
        tmp_path / "prepared",
        methods=(METHOD_PER_DATASET_STANDARDIZED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA),
    )

    metadata = json.loads((prepared / "prepared.json").read_text())
    assert metadata["format"] == 5
    # D001_L has eGFR, so frozen supplemental rows are D002_L..D004_L.
    for method, filename in metadata["supplemental_matrices"].items():
        supplemental = np.load(prepared / filename, mmap_mode="r")
        assert supplemental.shape == (3, len(probes)), method
        np.testing.assert_array_equal(supplemental, values[1:, 1:])

    train, validation, test = load_prepared_split_rows(prepared)
    all_rows = np.sort(np.concatenate([train, validation, test]))
    np.testing.assert_array_equal(all_rows, np.arange(len(samples) + 3))

    for method in (METHOD_PER_DATASET_STANDARDIZED, METHOD_PER_GSE_RMA):
        source = load_prepared_pretraining_source(layout, method, prepared)
        assert source.matrix.shape == (len(samples) + 3, len(probes))
        assert source.probe_ids == tuple(probes)
        assert source.sample_index["row_index_python"].tolist() == list(range(len(samples) + 3))
        # Fetch the three supplemental logical rows through the prepared read-only stack.
        observed = source.matrix[np.arange(len(samples), len(samples) + 3), :]
        if method == METHOD_PER_DATASET_STANDARDIZED:
            train_rows, _, _ = load_prepared_split_rows(prepared)
            train_supervised = train_rows[train_rows >= len(samples)] - len(samples)
            expected_center = values[1:, 1:][train_supervised].mean(axis=0)
            expected_scale = values[1:, 1:][train_supervised].std(axis=0)
            expected_scale = np.where(expected_scale > 1e-6, expected_scale, 1.0)
            np.testing.assert_allclose(
                observed, (values[1:, 1:] - expected_center) / expected_scale
            )
        else:
            np.testing.assert_array_equal(observed, values[1:, 1:])


def test_prepared_assets_reject_ikem_roles_changed_after_manifest_audit(
    tmp_path: Path,
) -> None:
    layout, _, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    changed = split.copy()
    for sample_id, partition, role in (
        ("D002_L", "validation", "pretrain_validation_no_egfr"),
        ("D004_L", "train", "pretrain_train_no_egfr"),
    ):
        mask = changed["sample_key"].eq(f"SUPERVISED:{sample_id}")
        changed.loc[mask, "split"] = partition
        changed.loc[mask, "training_role"] = role

    with pytest.raises(ValueError, match="audited CEL manifest"):
        prepare_pretraining_assets(
            layout,
            changed,
            tmp_path / "prepared_changed_roles",
            methods=(METHOD_PER_GSE_RMA,),
        )


def test_prepared_assets_freeze_different_geo_probe_orders(tmp_path: Path) -> None:
    layout, samples, probes = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    prepared = prepare_pretraining_assets(
        layout,
        split,
        tmp_path / "prepared_reordered_geo",
        methods=(METHOD_PER_DATASET_STANDARDIZED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA),
    )

    # Every synthetic GSE has one sample, so per-dataset standardization maps
    # every GEO probe to zero without borrowing information from another split.
    source = load_prepared_pretraining_source(
        layout, METHOD_PER_DATASET_STANDARDIZED, prepared
    )
    assert source.probe_ids == tuple(probes)
    gsm0_row = int(
        source.sample_index.index[source.sample_index["sample_key"].eq("GEO:GSM0")][0]
    )
    expected = np.zeros(len(probes), dtype=np.float32)
    np.testing.assert_array_equal(source.matrix[[gsm0_row], :][0], expected)

    metadata = __import__("json").loads((prepared / "prepared.json").read_text())
    assert metadata["format"] == 5
    assert set(metadata["method_geo_columns"]) == {
        METHOD_PER_DATASET_STANDARDIZED,
        METHOD_PER_GSE_RMA,
        METHOD_GLOBAL_RMA,
    }
    assert (prepared / "geo_columns_standardized.npy").is_file()
    np.testing.assert_array_equal(
        np.load(prepared / "geo_columns_standardized.npy"),
        np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
    )
    assert metadata["standardization"]["uses_outcome_values_in_fit"] is False
    assert metadata["standardization"]["uses_egfr_cv_fold"] is False


def test_formal_sweep_bundle_embeds_frozen_prepared_plan(tmp_path: Path) -> None:
    import json

    from archcon.batch import build_run_request, generate_sweep_bundle
    from archcon.data.training import TrainingConfig

    layout, _, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    base = build_run_request(
        method=METHOD_PER_GSE_RMA,
        split_seed=42,
        train_fraction=0.6,
        validation_fraction=0.2,
        training=TrainingConfig(hidden_widths=(8,), latent_dim=3, epochs=2),
    )
    bundle = generate_sweep_bundle(
        base_request=base,
        grid_text=json.dumps({"method": [METHOD_PER_GSE_RMA]}),
        destination_root=tmp_path / "sweeps",
        sweep_name="prepared",
        data_dir=str(layout.root),
        data_layout=layout,
        split_frame=split,
    )
    root = Path(bundle["root"])
    assert (root / "prepared" / "prepared.json").is_file()
    assert (root / "prepared" / "train_rows.npy").is_file()
    assert (root / "prepared" / "validation_rows.npy").is_file()
    assert (root / "prepared" / "test_rows.npy").is_file()
    request = json.loads((root / "configs" / "run_0001.json").read_text())
    assert request["prepared_dir"] == "../prepared"
    job = (root / "jobs" / "run_0001.py").read_text()
    assert "load_prepared_pretraining_source" in job
    assert "load_prepared_split_rows" in job
    assert "create_shared_preprocessing_split" not in job
    assert "load_pretraining_source" not in job
    assert "pd.read_csv" not in job
