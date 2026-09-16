import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from archcon.data import downstream
from archcon.data.downstream import (
    EmbeddingResult,
    ValidationCheckpoint,
    egfr_long,
    finalize_nested_selection,
    prepare_mixed_model_design,
    prepare_nested_mixed_model_design,
    repeated_donor_folds,
    scan_validation_checkpoints,
    select_embedding_checkpoints,
    select_molecular_group_winners,
    summarize_fixed_encoder_results,
    summarize_mixed_model_results,
)
from archcon.data.training import TrainingConfig


def _record(run: str, architecture: str, mse: float, method: str = "Per-dataset RMA"):
    config = TrainingConfig(
        hidden_widths=(8,), latent_dim=2, architecture_family=architecture, epochs=1
    )
    return ValidationCheckpoint(
        run=run,
        path=Path(f"/{run}/best.pt"),
        checkpoint={"input_dim": 4},
        method=method,
        architecture=architecture,
        latent_dim=2,
        validation_mse=mse,
        validation_r2=0.5,
        validation_objective=mse,
        best_epoch=1,
        config=config,
    )


def _patients(n_donors: int = 10) -> pd.DataFrame:
    rows = []
    for donor in range(n_donors):
        for kidney in (1, 2):
            rows.append(
                {
                    "patient": f"{donor}_{kidney}",
                    "donor": str(donor),
                    "KDRI_8": 0.8 + donor * 0.1,
                    "don_patient_age": 35 + donor,
                    "Cold_ischemia_hours": 8 + donor * 0.25 + kidney * 0.1,
                    "egfr_7d": 40 + donor + kidney,
                    "egfr_3m": 45 + donor + kidney,
                    "egfr_6m": 48 + donor + kidney,
                    "egfr_12m": 50 + donor + kidney,
                }
            )
    return pd.DataFrame(rows)


def test_checkpoint_scan_retains_metadata_not_tensor_payload(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "run_0001" / "best.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    payload = {
        "input_dim": 42_917,
        "method": "Per-dataset standardization",
        "config": {
            "hidden_widths": [8],
            "latent_dim": 2,
            "architecture_family": "Stadniuk MLP",
            "epochs": 1,
        },
        "history": {
            "val_mse": [0.1],
            "val_r2": [0.5],
            "val_loss": [0.1],
            "val_epoch": [1],
            "selection_score": [0.2],
            "geo_mse": [0.1],
            "ikem_mse": [0.3],
            "ikem_mse_donor_sd": [0.04],
        },
        "validation_selection_policy": {
            "metric": "clean_reconstruction_mse",
            "geo_weight": 0.5,
            "ikem_weight": 0.5,
            "ikem_aggregation": "donor_balanced",
            "ikem_uncertainty": "sample_sd_across_donor_mean_mse",
            "uses_geo_test": False,
            "uses_egfr_values": False,
        },
        "best_selection_score": 0.2,
        "model_state": {"large_tensor": object()},
        "optimizer_state": {"large_tensor": object()},
    }
    monkeypatch.setattr(downstream, "load_checkpoint_metadata", lambda _path: payload)
    records, warnings = scan_validation_checkpoints(tmp_path)
    assert not warnings
    assert len(records) == 1
    assert records[0].checkpoint is None
    assert records[0].input_dim == 42_917


def test_checkpoint_scan_rejects_inconsistent_validation_composite(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "run_0001" / "best.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    payload = {
        "input_dim": 4,
        "method": "Per-dataset RMA",
        "config": {
            "hidden_widths": [8],
            "latent_dim": 2,
            "architecture_family": "Stadniuk MLP",
            "epochs": 1,
        },
        "history": {
            "val_mse": [0.1],
            "val_r2": [0.5],
            "val_loss": [0.1],
            "val_epoch": [1],
            "selection_score": [0.9],
            "geo_mse": [0.1],
            "ikem_mse": [0.3],
            "ikem_mse_donor_sd": [0.04],
        },
        "validation_selection_policy": {
            "metric": "clean_reconstruction_mse",
            "geo_weight": 0.5,
            "ikem_weight": 0.5,
            "ikem_aggregation": "donor_balanced",
            "ikem_uncertainty": "sample_sd_across_donor_mean_mse",
            "uses_geo_test": False,
            "uses_egfr_values": False,
        },
        "best_selection_score": 0.9,
    }
    monkeypatch.setattr(downstream, "load_checkpoint_metadata", lambda _path: payload)

    records, warnings = scan_validation_checkpoints(tmp_path)

    assert not records
    assert len(warnings) == 1
    assert "composite score" in warnings[0]


def test_model_loader_imports_torch_for_meta_construction(monkeypatch) -> None:
    class FakeDevice:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeModel:
        def __init__(self):
            self.assigned = False
            self.target = None
            self.evaluation = False

        def load_state_dict(self, state, *, strict, assign):
            assert state == {"weight": "mapped"}
            assert strict is True
            self.assigned = assign

        def to(self, target):
            self.target = target.name
            return self

        def eval(self):
            self.evaluation = True
            return self

    fake_torch = SimpleNamespace(device=FakeDevice)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_model = FakeModel()
    monkeypatch.setattr(downstream, "build_autoencoder", lambda _input_dim, _config: fake_model)

    record = _record("run_0001", "Stadniuk MLP", 0.1)
    record = record.__class__(
        **{
            **record.__dict__,
            "checkpoint": {"input_dim": 4, "model_state": {"weight": "mapped"}},
        }
    )
    model, input_dim = downstream._model_from_record(record, "cpu")

    assert input_dim == 4
    assert model is fake_model
    assert fake_model.assigned is True
    assert fake_model.target == "cpu"
    assert fake_model.evaluation is True


def test_checkpoint_comparison_keeps_other_architecture_when_resnet_wins() -> None:
    records = [
        _record("run_0001", "ResNet-LN", 0.10),
        _record("run_0002", "Stadniuk MLP", 0.11),
        _record("run_0003", "ResNet-LN", 0.12),
    ]
    selected = select_embedding_checkpoints(records)
    assert [item[0] for item in selected] == ["winner_z", "best_stadniuk_z"]


def test_molecular_selection_is_separate_per_preprocessing_and_architecture() -> None:
    records = []
    run = 0
    for method in ("Per-dataset standardization", "Per-dataset RMA", "Global RMA"):
        for architecture in ("Stadniuk MLP", "ResNet-LN"):
            first = _record(f"run_{run:04d}", architecture, 0.1, method)
            second = _record(f"run_{run + 1:04d}", architecture, 0.2, method)
            records.extend(
                [
                    first.__class__(
                        **{**first.__dict__, "molecular_selection_mse": 10.0 + run}
                    ),
                    second.__class__(
                        **{**second.__dict__, "molecular_selection_mse": 5.0 + run}
                    ),
                ]
            )
            run += 2
    selected = select_molecular_group_winners(records)
    assert len(selected) == 6
    assert all(record.run.endswith(("1", "3", "5", "7", "9", "11")) for _, _, record in selected)


def test_molecular_group_selection_never_uses_geo_test_metric() -> None:
    first = _record("run_0001", "Stadniuk MLP", 0.1)
    second = _record("run_0002", "Stadniuk MLP", 0.2)
    first = first.__class__(
        **{
            **first.__dict__,
            "molecular_selection_mse": 0.1,
            "molecular_test_mse": 100.0,
        }
    )
    second = second.__class__(
        **{
            **second.__dict__,
            "molecular_selection_mse": 0.2,
            "molecular_test_mse": 0.0,
        }
    )
    selected = select_molecular_group_winners([first, second], expected_groups=1)
    assert selected[0][2].run == "run_0001"


def test_egfr_long_uses_ordered_categorical_time() -> None:
    wide = _patients(1)
    long = egfr_long(wide)
    assert len(long) == 8
    assert list(long["time"].cat.categories) == ["7d", "3m", "6m", "12m"]
    assert set(long["patient"]) == {"0_1", "0_2"}


def test_repeated_folds_never_split_sibling_kidneys() -> None:
    patients = _patients()
    folds = repeated_donor_folds(patients, n_splits=5, n_repeats=3, seed=7)
    assert len(folds) == 15
    for _, _, train, test in folds:
        train_donors = set(patients.iloc[train]["donor"])
        test_donors = set(patients.iloc[test]["donor"])
        assert train_donors.isdisjoint(test_donors)


def test_mixed_model_design_has_fold_local_features(tmp_path: Path) -> None:
    patients = _patients()
    patients.loc[0, "don_patient_age"] = np.nan
    patients.loc[3, "Cold_ischemia_hours"] = np.nan
    samples = pd.DataFrame(
        {
            "source_row_index": np.arange(len(patients)),
            "sample_id": patients["patient"],
            "donor_id": patients["donor"],
            "has_egfr": True,
        }
    )
    record = _record("run_0001", "Stadniuk MLP", 0.1)
    z = np.column_stack(
        [np.arange(len(patients), dtype=np.float32), np.arange(len(patients), dtype=np.float32) ** 2]
    )
    result = EmbeddingResult("winner_z", "Winner z", record, z, samples, 0.01)
    expression = np.arange(len(patients) * 4, dtype=np.float32).reshape(len(patients), 4)
    design_path, specs_path, specs = prepare_mixed_model_design(
        patients,
        [result],
        expression,
        samples,
        tmp_path,
        n_splits=5,
        n_repeats=2,
        seed=3,
        stratify_column="KDRI_8",
        include_pca=True,
    )
    design = pd.read_csv(design_path)
    assert specs_path.is_file()
    assert set(specs["model_id"]) == {
        "time_only",
        "winner_z",
        "pca",
        "clinical_age",
        "clinical_kdri",
        "clinical_cold_ischemia",
        "clinical_full",
        "winner_z_kdri",
        "winner_z_clinical",
        "pca_kdri",
        "pca_clinical",
    }
    assert len(specs) == 110
    full_spec = specs.loc[specs["model_id"].eq("winner_z_clinical")].iloc[0]
    assert full_spec["n_main_features"] == 2
    assert full_spec["n_time_interaction_features"] == 3
    for (repeat, fold), block in design.loc[design["model_id"].eq("winner_z")].groupby(
        ["repeat", "fold"]
    ):
        train = block.loc[block["partition"].eq("train")].drop_duplicates("patient")
        assert np.allclose(train[["x1", "x2"]].mean(axis=0), 0.0, atol=1e-6), (repeat, fold)
    for (repeat, fold), block in design.loc[design["model_id"].eq("clinical_full")].groupby(
        ["repeat", "fold"]
    ):
        train = block.loc[block["partition"].eq("train")].drop_duplicates("patient")
        assert np.isfinite(train[["x1", "x2", "x3"]].to_numpy()).all()
        assert np.allclose(train[["x1", "x2", "x3"]].mean(axis=0), 0.0, atol=1e-6), (
            repeat,
            fold,
        )


def test_mixed_model_design_can_disable_clinical_models(tmp_path: Path) -> None:
    patients = _patients()
    samples = pd.DataFrame(
        {
            "source_row_index": np.arange(len(patients)),
            "sample_id": patients["patient"],
            "donor_id": patients["donor"],
            "has_egfr": True,
        }
    )
    record = _record("run_0001", "Stadniuk MLP", 0.1)
    z = np.ones((len(patients), 2), dtype=np.float32)
    result = EmbeddingResult("winner_z", "Winner z", record, z, samples, 0.01)
    expression = np.arange(len(patients) * 4, dtype=np.float32).reshape(len(patients), 4)
    _, _, specs = prepare_mixed_model_design(
        patients,
        [result],
        expression,
        samples,
        tmp_path,
        n_splits=5,
        n_repeats=1,
        include_pca=True,
        include_clinical=False,
    )
    assert set(specs["model_id"]) == {"time_only", "winner_z", "pca"}


def test_fixed_design_supports_multiple_frozen_encoders_without_a_winner(
    tmp_path: Path,
) -> None:
    patients = _patients()
    samples = pd.DataFrame(
        {
            "source_row_index": np.arange(len(patients)),
            "sample_id": patients["patient"],
            "donor_id": patients["donor"],
            "has_egfr": True,
        }
    )
    first = EmbeddingResult(
        "candidate_a",
        "Candidate A",
        _record("run_0001", "Stadniuk MLP", 0.1),
        np.ones((len(patients), 2), dtype=np.float32),
        samples,
        0.01,
    )
    second = EmbeddingResult(
        "candidate_b",
        "Candidate B",
        _record("run_0002", "ResNet-LN", 0.2),
        np.ones((len(patients), 3), dtype=np.float32),
        samples,
        0.02,
    )
    expression = np.arange(len(patients) * 4, dtype=np.float32).reshape(
        len(patients), 4
    )
    _, _, specs = prepare_mixed_model_design(
        patients,
        [first, second],
        expression,
        samples,
        tmp_path,
        n_splits=5,
        n_repeats=1,
        include_pca=True,
        include_clinical=False,
    )
    assert set(specs["model_id"]) == {
        "time_only",
        "candidate_a",
        "candidate_b",
        "pca_d2",
        "pca_d3",
    }


def test_nested_design_has_inner_selection_and_unique_outer_fits(tmp_path: Path) -> None:
    patients = _patients()
    samples = pd.DataFrame(
        {
            "source_row_index": np.arange(len(patients)),
            "sample_id": patients["patient"],
            "donor_id": patients["donor"],
            "has_egfr": True,
        }
    )
    embeddings = []
    for index in range(6):
        record = _record(
            f"run_{index:04d}",
            "Stadniuk MLP" if index % 2 == 0 else "ResNet-LN",
            0.1,
            f"method_{index // 2}",
        )
        z = np.column_stack(
            [
                np.arange(len(patients), dtype=np.float32) + index,
                np.arange(len(patients), dtype=np.float32) ** 2 + index,
            ]
        )
        embeddings.append(
            EmbeddingResult(f"candidate_{index}", f"Candidate {index}", record, z, samples, 0.01)
        )
    expression = np.arange(len(patients) * 4, dtype=np.float32).reshape(len(patients), 4)
    design_path, _, specs = prepare_nested_mixed_model_design(
        patients,
        embeddings,
        expression,
        samples,
        tmp_path,
        n_splits=5,
        n_repeats=1,
        inner_splits=4,
        include_pca=True,
        include_clinical=True,
    )
    design = pd.read_csv(design_path)
    assert specs["fit_id"].is_unique
    assert design["fit_id"].nunique() == len(specs)
    assert set(specs["stage"]) == {"inner_selection", "outer_evaluation"}
    inner = specs.loc[specs["stage"].eq("inner_selection")]
    assert len(inner) == 5 * 4 * 6
    assert set(inner["model_id"]) == {f"candidate_{index}" for index in range(6)}


def test_summary_uses_matched_fold_deltas(tmp_path: Path) -> None:
    fold_values = [
        (0, 10.0, 8.0, 9.0, 7.0, 8.0, 7.0, 7.5),
        (1, 12.0, 11.0, 10.0, 9.0, 9.0, 8.0, 8.5),
    ]
    metrics = pd.DataFrame(
        [
            {"model_id": model, "model_label": label, "repeat": repeat, "fold": 0, "rmse": rmse, "mae": rmse / 2}
            for repeat, baseline, winner, kdri, winner_kdri, full, winner_full, pca_full in fold_values
            for model, label, rmse in [
                ("time_only", "Time only", baseline),
                ("winner_z", "Winner z", winner),
                ("clinical_kdri", "KDRI × time", kdri),
                ("winner_z_kdri", "Winner z + KDRI × time", winner_kdri),
                ("clinical_full", "Full clinical", full),
                ("winner_z_clinical", "Winner z + full clinical", winner_full),
                ("pca_clinical", "PCA + full clinical", pca_full),
            ]
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "model_id": row.model_id,
                "model_label": row.model_label,
                "repeat": row.repeat,
                "fold": row.fold,
                "patient": f"p{row.repeat}",
                "donor": f"d{row.repeat}",
                "time": "3m",
                "egfr": row.rmse,
                "prediction": 0.0,
            }
            for row in metrics.itertuples()
        ]
    )
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary, pairwise, clinical_incremental = summarize_mixed_model_results(
        metrics_path, predictions_path, tmp_path
    )
    winner = summary.set_index("model_id").loc["winner_z"]
    assert winner["mean_delta_vs_time"] == 1.5
    time = pairwise.set_index("model_id").loc["time_only"]
    assert time["mean_winner_gain"] == 1.5
    beyond_kdri = clinical_incremental.set_index("comparison_label").loc[
        "Winner z beyond KDRI"
    ]
    assert beyond_kdri["mean_gain"] == 1.5
    beyond_full = clinical_incremental.set_index("comparison_label").loc[
        "Winner z beyond full clinical"
    ]
    assert beyond_full["mean_gain"] == 1.0
    assert (tmp_path / "clinical_incremental_summary.csv").is_file()


def test_fixed_encoder_summary_reports_each_encoder_without_selecting_one(
    tmp_path: Path,
) -> None:
    samples = pd.DataFrame({"sample_id": ["p1"]})
    embeddings = [
        EmbeddingResult(
            f"candidate_{index}",
            f"Candidate {index}",
            _record(f"run_{index:04d}", "Stadniuk MLP", 0.1 + index),
            np.empty((1, 2), dtype=np.float32),
            samples,
            0.0,
        )
        for index in range(2)
    ]
    metrics = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "model_label": label,
                "repeat": 0,
                "fold": fold,
                "rmse": rmse,
                "mae": rmse / 2,
            }
            for fold in range(2)
            for model_id, label, rmse in (
                ("time_only", "Time only", 3.0 + fold),
                ("candidate_0", "Candidate 0", 1.0 + fold),
                ("candidate_1", "Candidate 1", 2.0 + fold),
            )
        ]
    )
    predictions = metrics.assign(
        patient="p1",
        donor="d1",
        time="3m",
        egfr=lambda frame: frame["rmse"],
        prediction=0.0,
    )
    metrics_path = tmp_path / "fixed_metrics.csv"
    predictions_path = tmp_path / "fixed_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    _, fixed, comparisons = summarize_fixed_encoder_results(
        metrics_path, predictions_path, embeddings, tmp_path
    )

    assert fixed["model_id"].tolist() == ["candidate_0", "candidate_1"]
    assert set(comparisons["candidate_id"]) == {"candidate_0", "candidate_1"}
    assert (tmp_path / "fixed_encoder_cv_summary.csv").is_file()
    assert (tmp_path / "fixed_encoder_comparisons.csv").is_file()
    assert not (tmp_path / "candidate_cv_ranking.csv").exists()


def test_final_nested_outputs_retain_every_fixed_encoder_cv_result(tmp_path: Path) -> None:
    samples = pd.DataFrame({"sample_id": ["p1"]})
    embeddings = [
        EmbeddingResult(
            f"candidate_{index}",
            f"Candidate {index}",
            _record(f"run_{index:04d}", "Stadniuk MLP", 0.1, f"method_{index}"),
            np.empty((1, 2), dtype=np.float32),
            samples,
            0.0,
        )
        for index in range(2)
    ]
    metric_rows = []
    prediction_rows = []
    for fold in range(2):
        for inner_fold in range(2):
            for index in range(2):
                metric_rows.append(
                    {
                        "fit_id": f"inner_{fold}_{inner_fold}_{index}",
                        "stage": "inner_selection",
                        "model_id": f"candidate_{index}",
                        "model_label": f"Candidate {index}",
                        "candidate_id": f"candidate_{index}",
                        "repeat": 0,
                        "fold": fold,
                        "inner_fold": inner_fold,
                        "rmse": 1.0 + index,
                        "mae": 1.0 + index,
                    }
                )
        for model_id, label, rmse in (
            ("time_only", "Time only", 3.0),
            ("candidate_0", "Candidate 0", 1.0 + fold),
            ("candidate_1", "Candidate 1", 2.0 + fold),
        ):
            fit_id = f"outer_{fold}_{model_id}"
            metric_rows.append(
                {
                    "fit_id": fit_id,
                    "stage": "outer_evaluation",
                    "model_id": model_id,
                    "model_label": label,
                    "candidate_id": model_id if model_id.startswith("candidate_") else "",
                    "repeat": 0,
                    "fold": fold,
                    "inner_fold": -1,
                    "rmse": rmse,
                    "mae": rmse,
                }
            )
            prediction_rows.append(
                {
                    "fit_id": fit_id,
                    "stage": "outer_evaluation",
                    "model_id": model_id,
                    "model_label": label,
                    "candidate_id": model_id if model_id.startswith("candidate_") else "",
                    "repeat": 0,
                    "fold": fold,
                    "inner_fold": -1,
                    "patient": f"p{fold}",
                    "donor": f"d{fold}",
                    "time": "3m",
                    "egfr": rmse,
                    "prediction": 0.0,
                }
            )
    raw_metrics = tmp_path / "raw_metrics.csv"
    raw_predictions = tmp_path / "raw_predictions.csv"
    pd.DataFrame(metric_rows).to_csv(raw_metrics, index=False)
    pd.DataFrame(prediction_rows).to_csv(raw_predictions, index=False)

    metrics_path, predictions_path, _, ranking = finalize_nested_selection(
        raw_metrics, raw_predictions, embeddings, tmp_path
    )
    final_metrics = pd.read_csv(metrics_path)
    assert {"candidate_0", "candidate_1", "winner_z", "time_only"}.issubset(
        set(final_metrics["model_id"])
    )
    assert len(ranking) == 2
    assert (tmp_path / "all_fixed_encoder_fold_metrics.csv").is_file()
    summary, _, _ = summarize_mixed_model_results(metrics_path, predictions_path, tmp_path)
    assert set(
        summary.loc[
            summary["model_id"].astype(str).str.startswith("candidate_"), "model_id"
        ]
    ) == {"candidate_0", "candidate_1"}
    assert (tmp_path / "all_fixed_encoder_cv_summary.csv").is_file()
