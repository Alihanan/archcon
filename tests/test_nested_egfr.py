from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archcon.data.contrastive_egfr import (
    STADNIUK_CONTRASTIVE_COLUMNS,
    FineTuneControls,
    contrastive_candidate_grid,
)
from archcon.data.defaults import ProjectDataLayout
from archcon.data.downstream import ValidationCheckpoint
from archcon.data.nested_cv import NestedSplitConfig, build_nested_split_manifest
from archcon.data.nested_egfr import (
    EncoderArm,
    MethodExpression,
    MixedModelDesign,
    _parse_feature_table,
    _pca_candidates,
    _selection_row,
    _standardize_from_train,
    evaluate_nested_egfr,
)
from archcon.data.supervised import IKEM_ROLE_MEASURED_HELD_OUT, IKEM_ROLE_TRAIN


def _patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": ["D001_P", "D002_P", "D003_P", "D004_P"],
            "donor": ["D001", "D002", "D003", "D004"],
            "egfr_7d": [0.8, 0.9, 1.0, 1.1],
            "egfr_3m": [0.9, 1.0, 1.1, 1.2],
            "egfr_6m": [1.0, 1.1, 1.2, 1.3],
            "egfr_12m": [1.1, 1.2, 1.3, 1.4],
        }
    )


def test_fold_scaling_never_uses_held_out_rows() -> None:
    train = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=np.float32)
    held_out = np.asarray([[5.0, 7.0]], dtype=np.float32)
    shifted = held_out + 1_000_000.0
    train_a, test_a = _standardize_from_train(train, held_out)
    train_b, test_b = _standardize_from_train(train, shifted)
    np.testing.assert_allclose(train_a, train_b)
    assert not np.allclose(test_a, test_b)


def test_contrastive_features_are_loaded_in_requested_molecular_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clinical.xlsx"
    pd.DataFrame(
        {
            "Sample_ID": ["D002_P_(PrimeView).CEL", "D001_P.CEL"],
            "AKI": [1, 0],
            "ECD_1": [0, 1],
        }
    ).to_excel(path, index=False)
    loaded = _parse_feature_table(path, ["D001_P", "D002_P"], ("AKI", "ECD_1"))
    assert loaded["sample_id"].tolist() == ["D001_P", "D002_P"]
    np.testing.assert_array_equal(
        loaded[["AKI", "ECD_1"]].to_numpy(),
        np.asarray([[0, 1], [1, 0]]),
    )


def test_pca_fit_is_independent_of_validation_values() -> None:
    rng = np.random.default_rng(3)
    expression = rng.normal(size=(20, 30)).astype(np.float32)
    train = np.arange(15, dtype=np.int64)
    validation = np.arange(15, 20, dtype=np.int64)
    first = _pca_candidates(expression, train, validation, (3, 8), seed=11)
    changed = expression.copy()
    changed[validation] += rng.normal(10_000.0, 100.0, size=changed[validation].shape)
    second = _pca_candidates(changed, train, validation, (3, 8), seed=11)
    for dimension in (3, 8):
        np.testing.assert_allclose(first[dimension][0], second[dimension][0])
        assert not np.allclose(first[dimension][1], second[dimension][1])


def test_mixed_model_design_matches_r_contract(tmp_path: Path) -> None:
    patients = _patients()
    builder = MixedModelDesign(patients)
    builder.add_fit(
        fit_id="inner_r000_f000_candidate",
        stage="inner_selection",
        model_id="candidate",
        model_label="Candidate",
        candidate_id="encoder",
        repeat=0,
        fold=0,
        inner_fold=0,
        train_indices=np.asarray([0, 1, 2]),
        test_indices=np.asarray([3]),
        train_main=np.ones((3, 2), dtype=np.float32),
        test_main=np.ones((1, 2), dtype=np.float32),
        train_interactions=np.ones((3, 3), dtype=np.float32),
        test_interactions=np.ones((1, 3), dtype=np.float32),
        feature_description="z + clinical",
    )
    design_path, specs_path = builder.write(tmp_path, "test")
    design = pd.read_csv(design_path)
    specs = pd.read_csv(specs_path)
    assert {"fit_id", "partition", "patient", "donor", "time", "egfr"}.issubset(
        design.columns
    )
    assert {f"x{index}" for index in range(1, 6)}.issubset(design.columns)
    assert specs.loc[0, "n_features"] == 5
    assert specs.loc[0, "n_main_features"] == 2
    assert specs.loc[0, "n_time_interaction_features"] == 3


def test_inner_selection_uses_rmse_then_prespecified_candidate_order() -> None:
    metrics = pd.DataFrame(
        {"model_id": ["arm__none", "arm__soft"], "rmse": [0.4, 0.4]}
    )
    catalog = pd.DataFrame(
        {
            "model_id": ["arm__none", "arm__soft"],
            "encoder_id": ["arm", "arm"],
            "candidate_order": [0, 1],
        }
    )
    selected = _selection_row(
        metrics, catalog, group_column="encoder_id", group_value="arm"
    )
    assert selected["model_id"] == "arm__none"


def test_nested_orchestration_is_complete_and_resumable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    import archcon.data.nested_egfr as nested_module
    from archcon.data.training import TrainingConfig, build_autoencoder

    def fake_r_runner(
        design_path,
        specs_path,
        _r_script,
        output_root,
        *,
        rscript="Rscript",
        output_prefix="",
    ):
        del rscript
        design = pd.read_csv(design_path)
        specs = pd.read_csv(specs_path)
        metric_rows = []
        prediction_frames = []
        metadata_columns = (
            "fit_id",
            "stage",
            "model_id",
            "model_label",
            "candidate_id",
            "repeat",
            "fold",
            "inner_fold",
        )
        for spec in specs.to_dict("records"):
            block = design.loc[design["fit_id"].eq(spec["fit_id"])]
            train = block.loc[block["partition"].eq("train")]
            test = block.loc[block["partition"].eq("test")].copy()
            prediction = np.repeat(float(train["egfr"].mean()), len(test))
            error = test["egfr"].to_numpy(dtype=float) - prediction
            metadata = {column: spec[column] for column in metadata_columns}
            metric_rows.append(
                {
                    **metadata,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "singular": False,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        **{
                            column: [value] * len(test)
                            for column, value in metadata.items()
                        },
                        "patient": test["patient"].astype(str).to_numpy(),
                        "donor": test["donor"].astype(str).to_numpy(),
                        "time": test["time"].astype(str).to_numpy(),
                        "egfr": test["egfr"].to_numpy(dtype=float),
                        "prediction": prediction,
                    }
                )
            )
        root = Path(output_root)
        metrics_path = root / f"{output_prefix}fold_metrics.csv"
        predictions_path = root / f"{output_prefix}oof_predictions.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            predictions_path, index=False
        )
        return metrics_path, predictions_path

    monkeypatch.setattr(nested_module, "run_lme4_benchmark", fake_r_runner)
    rng = np.random.default_rng(4)
    data_root = tmp_path / "data"
    data_root.mkdir()
    measured_ids = [f"D{index:03d}_P" for index in range(8)]
    external_ids = ["N000_P", "N001_P"]
    clinical = pd.DataFrame({"Sample_ID": measured_ids + external_ids})
    for column_index, column in enumerate(STADNIUK_CONTRASTIVE_COLUMNS):
        clinical[column] = rng.normal(size=10) + column_index
    clinical.to_excel(data_root / "Klasifikator_20_3_24_v2.xlsx", index=False)

    patients = pd.DataFrame(
        {
            "patient": measured_ids,
            "donor": [f"D{index:03d}" for index in range(8)],
            "KDRI_8": rng.normal(size=8),
            "don_patient_age": rng.normal(50, 5, size=8),
            "Cold_ischemia_hours": rng.normal(10, 2, size=8),
            "egfr_7d": rng.normal(size=8),
            "egfr_3m": rng.normal(size=8),
            "egfr_6m": rng.normal(size=8),
            "egfr_12m": rng.normal(size=8),
        }
    )
    manifest = build_nested_split_manifest(
        patients,
        NestedSplitConfig(outer_splits=2, outer_repeats=1, inner_splits=2),
    )
    all_ids = measured_ids + external_ids
    samples = pd.DataFrame(
        {
            "sample_id": all_ids,
            "donor_id": [value.split("_", 1)[0] for value in all_ids],
            "training_role": [IKEM_ROLE_MEASURED_HELD_OUT] * 8
            + [IKEM_ROLE_TRAIN] * 2,
            "pretraining_split": ["held_out"] * 8 + ["train"] * 2,
            "split_unit": [value.split("_", 1)[0] for value in all_ids],
        }
    )
    method = MethodExpression(
        "test",
        rng.normal(size=(10, 6)).astype(np.float32),
        samples,
    )
    config = TrainingConfig(
        hidden_widths=(8,),
        latent_dim=3,
        epochs=1,
        batch_size=4,
        device="CPU",
        background_prefetch=False,
    )
    checkpoint_path = tmp_path / "best.pt"
    model = build_autoencoder(6, config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": 6,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    record = ValidationCheckpoint(
        run="run_0001",
        path=checkpoint_path,
        checkpoint=None,
        method="test",
        architecture=config.architecture_family,
        latent_dim=3,
        validation_mse=0.1,
        validation_r2=0.1,
        validation_objective=0.1,
        best_epoch=1,
        config=config,
        input_dim=6,
        molecular_selection_mse=0.1,
    )
    arm = EncoderArm("candidate_test_z", "Test encoder", record)
    candidates = contrastive_candidate_grid((0.05,), (0.1,))
    controls = FineTuneControls(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        projection_dim=4,
    )
    r_script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "archcon"
        / "assets"
        / "molecular_mixed_models.R"
    )
    arguments = (
        ProjectDataLayout.from_root(data_root),
        patients,
        manifest,
        (arm,),
        {"test": method},
        candidates,
        controls,
        (2,),
        tmp_path / "evaluation",
        r_script,
    )
    result = evaluate_nested_egfr(
        *arguments,
        device="cpu",
        expected_external_no_egfr=2,
    )
    summary = pd.read_csv(result.summary_path)
    assert len(summary) == 9
    assert summary["n_folds"].eq(2).all()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A completed fold was recomputed.")

    monkeypatch.setattr(nested_module, "run_lme4_benchmark", forbidden)
    monkeypatch.setattr(nested_module, "fine_tune_checkpoint", forbidden)
    resumed = evaluate_nested_egfr(
        *arguments,
        device="cpu",
        expected_external_no_egfr=2,
    )
    assert len(pd.read_csv(resumed.model_catalog_path)) == 9
