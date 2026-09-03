import json
from pathlib import Path

from archcon.batch import (
    build_run_request,
    generate_sweep_bundle,
    generated_job_python,
    load_run_request,
)
from archcon.data.geo_rma import METHOD_GLOBAL_RMA
from archcon.data.pretraining import LOSS_MASKED, LOSS_MSE
from archcon.data.training import TrainingConfig


def test_run_request_roundtrip(tmp_path: Path) -> None:
    request = build_run_request(
        method=METHOD_GLOBAL_RMA,
        split_seed=42,
        train_fraction=0.9,
        training=TrainingConfig(hidden_widths=(8, 4), dropout=0.0),
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    loaded, config = load_run_request(path)
    assert loaded["method"] == METHOD_GLOBAL_RMA
    assert config.hidden_widths == (8, 4)
    assert config.dropout == 0.0


def test_sweep_bundle_uses_cartesian_product_and_pbs_array(tmp_path: Path) -> None:
    base = build_run_request(
        method=METHOD_GLOBAL_RMA,
        split_seed=42,
        train_fraction=0.9,
        training=TrainingConfig(dropout=0.0, loss_name=LOSS_MSE),
    )
    bundle = generate_sweep_bundle(
        base_request=base,
        grid_text=json.dumps(
            {
                "learning_rate": [1e-3, 3e-4],
                "seed": [41, 42],
                "loss_name": ["mse", "masked"],
            }
        ),
        destination_root=tmp_path,
        sweep_name="unit",
        project_dir="$HOME/ArchCon",
        data_dir="$HOME/ArchCon/data",
        python_executable="$HOME/ArchCon/.venv/bin/python",
    )
    assert bundle["count"] == 8
    root = Path(bundle["root"])
    configs = sorted((root / "configs").glob("run_*.json"))
    jobs = sorted((root / "jobs").glob("run_*.py"))
    assert len(configs) == 8
    assert len(jobs) == 8
    losses = {json.loads(path.read_text())["training"]["loss_name"] for path in configs}
    assert losses == {LOSS_MSE, LOSS_MASKED}

    run_script = (root / "run_array.pbs.sh").read_text(encoding="utf-8")
    submit_script = (root / "submit.sh").read_text(encoding="utf-8")
    assert "PBS_ARRAY_INDEX" in run_script
    assert 'RUN_NAME=$(printf "run_%04d"' in run_script
    assert 'RUN_DIR="$RESULT_ROOT/$RUN_NAME"' in run_script
    assert '"$PYTHON_BIN" "$JOB"' in run_script
    assert "--data-dir" in run_script
    assert "--output-root" in run_script
    assert '--run-directory "$RUN_DIR"' in run_script
    assert 'latest.pt' in run_script
    assert 'RESUME_ARGS' in run_script
    assert '.run_' not in run_script
    assert "qsub -J 1-8" in submit_script

    job_source = jobs[0].read_text(encoding="utf-8")
    assert "TRAINING_CONFIG = TrainingConfig(" in job_source
    assert "class ArchConAutoencoder(nn.Module):" in job_source
    assert "def job_objective(" in job_source
    assert "def build_optimizer(" in job_source
    assert "def build_scheduler(" in job_source
    assert "load_prepared_pretraining_source" in job_source
    assert "load_prepared_split_rows" in job_source
    assert "create_shared_preprocessing_split" not in job_source
    compile(job_source, str(jobs[0]), "exec")


def test_sweep_bundle_preserves_current_split_when_split_is_not_a_grid_axis(tmp_path: Path) -> None:
    import pandas as pd

    base = build_run_request(
        method=METHOD_GLOBAL_RMA,
        split_seed=42,
        train_fraction=0.9,
        training=TrainingConfig(dropout=0.0),
    )
    split = pd.DataFrame(
        {
            "GSM": ["GSM1", "GSM2"],
            "row_index_python": [0, 1],
            "split": ["train", "validation"],
        }
    )
    bundle = generate_sweep_bundle(
        base_request=base,
        grid_text='{"learning_rate": [0.001, 0.0003]}',
        destination_root=tmp_path,
        sweep_name="split",
        split_frame=split,
    )
    root = Path(bundle["root"])
    assert (root / "split.csv").is_file()
    request = json.loads((root / "configs" / "run_0001.json").read_text())
    assert request["split_file"] == "../split.csv"


def test_recommended_comparison_sweep_has_900_architecture_preprocessing_runs(tmp_path: Path) -> None:
    import pandas as pd

    from archcon.batch import recommended_comparison_grid_json
    from archcon.data.geo_rma import METHOD_PER_GSE_RMA

    base = build_run_request(
        method=METHOD_PER_GSE_RMA,
        split_seed=42,
        train_fraction=0.9,
        training=TrainingConfig(),
    )
    split = pd.DataFrame(
        {
            "GSM": ["GSM1", "GSM2", "GSM3"],
            "row_index_python": [0, 1, 2],
            "split": ["train", "validation", "test"],
            "seed": [42, 42, 42],
        }
    )
    bundle = generate_sweep_bundle(
        base_request=base,
        grid_text=recommended_comparison_grid_json(),
        destination_root=tmp_path,
        sweep_name="comparison",
        split_frame=split,
    )
    assert bundle["count"] == 900
    root = Path(bundle["root"])
    manifest = pd.read_csv(root / "manifest.csv")
    assert (manifest["branch"] == "stadniuk_mlp").sum() == 540
    assert (manifest["branch"] == "resnet_ln").sum() == 360
    stadniuk = manifest.loc[manifest["branch"] == "stadniuk_mlp"]
    assert set(stadniuk["stadniuk_batch_norm"].astype(str).str.lower()) == {"true", "false"}
    assert set(manifest["method"]) == {"Stadniuk rescaling", "Per-dataset RMA", "Global RMA"}
    assert set(manifest["latent_dim"]) == {3, 8, 16}
    assert set(manifest["hidden_widths"]) == {
        "[256]",
        "[256, 64]",
        "[256, 128, 64]",
        "[256, 192, 128, 64]",
        "[256, 224, 192, 128, 64]",
    }
    assert manifest["preprocessing_policy"].astype(str).str.len().min() > 10
    assert set(manifest["seed"]) == {42}
    assert set(manifest["learning_rate"]) == {0.001}
    assert set(manifest["lr_schedule"]) == {"Cosine annealing (deterministic)"}
    assert set(manifest["epochs"]) == {1000}
    assert set(manifest["lr_decay_epochs"]) == {500}
    assert set(manifest["convergence_window"]) == {5}
    assert (root / "split.csv").is_file()
    assert "qsub -J 1-900" in (root / "submit.sh").read_text(encoding="utf-8")
    run_script = (root / "run_array.pbs.sh").read_text(encoding="utf-8")
    assert "ngpus=" not in run_script
    assert "ncpus=1" in run_script
    assert 'export OMP_NUM_THREADS="1"' in run_script
    jobs = sorted((root / "jobs").glob("run_*.py"))
    assert len(jobs) == 900
    first_job = jobs[0].read_text(encoding="utf-8")
    assert "Seed: 42" in first_job
    assert "LambdaLR" in first_job
    assert "LR decay epochs: 500" in first_job
    assert "Max epochs: 1000" in first_job
    assert "convergence_tolerance=1e-05" in first_job
    assert "convergence_window=5" in first_job
    assert "load_prepared_pretraining_source" in first_job
    assert "load_prepared_split_rows" in first_job
    assert "create_shared_preprocessing_split" not in first_job
    assert "Test: held out from neural training/ranking and not evaluated by sweep jobs" in first_job
    assert "Supervised samples with eGFR: completely excluded from molecular pretraining" in first_job
    compile(first_job, str(jobs[0]), "exec")


def test_generated_python_job_model_is_checkpoint_compatible_with_canonical_builder() -> None:
    from archcon.data.pretraining import ARCH_RESNET_LN
    from archcon.data.training import DEVICE_CPU, build_autoencoder

    config = TrainingConfig(
        hidden_widths=(12, 6),
        latent_dim=3,
        architecture_family=ARCH_RESNET_LN,
        residual_blocks=2,
        residual_expansion=4,
        dropout=0.0,
        device=DEVICE_CPU,
    )
    request = build_run_request(
        method=METHOD_GLOBAL_RMA,
        split_seed=42,
        train_fraction=0.9,
        training=config,
    )
    source = generated_job_python(
        index=1,
        branch_name="resnet_ln",
        request=request,
        expected_input_dim=16,
    )
    namespace = {"__name__": "generated_job_test"}
    exec(compile(source, "<generated-job>", "exec"), namespace)
    generated = namespace["build_job_model"](16, namespace["TRAINING_CONFIG"])
    canonical = build_autoencoder(16, config)
    assert list(generated.state_dict()) == list(canonical.state_dict())
    assert [tuple(value.shape) for value in generated.state_dict().values()] == [
        tuple(value.shape) for value in canonical.state_dict().values()
    ]
