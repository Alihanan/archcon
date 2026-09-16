import archcon


def test_public_api() -> None:
    assert callable(archcon.build_app)
    assert callable(archcon.start)
    assert not hasattr(archcon, "read_text_file")
    assert isinstance(archcon.__version__, str)


def test_packaging_does_not_exclude_runtime_data_package() -> None:
    from pathlib import Path

    import tomllib

    project_root = Path(__file__).resolve().parents[1]
    gitignore = (project_root / ".gitignore").read_text().splitlines()
    assert "data/" not in gitignore
    assert "/data/" in gitignore

    config = tomllib.loads((project_root / "pyproject.toml").read_text())
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["ignore-vcs"] is True
    assert wheel["packages"] == ["src/archcon"]
    assert (project_root / "src" / "archcon" / "data" / "__init__.py").is_file()


def test_evaluation_console_script_is_packaged() -> None:
    from pathlib import Path

    import tomllib

    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["archcon-evaluate-egfr"] == (
        "archcon.evaluate_egfr:main"
    )

    from archcon.evaluate_egfr import main

    assert callable(main)


def test_global_normalization_console_script_is_packaged() -> None:
    from pathlib import Path

    import tomllib

    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["archcon-rebuild-global-normalization"] == (
        "archcon.rebuild_global_normalization:main"
    )

    from archcon.rebuild_global_normalization import main

    assert callable(main)


def test_mixed_model_r_script_quotes_reserved_repeat_column() -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = (
        project_root / "src" / "archcon" / "assets" / "molecular_mixed_models.R"
    ).read_text()
    assert "design$repeat" not in script
    assert "spec$repeat" not in script
    assert "    repeat =" not in script
    assert 'design[["repeat"]]' in script
    assert 'spec[["repeat"]]' in script
    assert "spec$n_main_features" in script
    assert "spec$n_time_interaction_features" in script
    assert 'paste0("x", interaction_indices, " * time")' in script
    assert script.count("check.names = FALSE") == 5
    assert "design$fit_id == spec$fit_id" in script
    assert "do.call(rbind" not in script
    assert "write.table(" in script


def test_scratch_launcher_templates_are_packaged() -> None:
    from importlib.resources import files

    assets = files("archcon.assets")
    run_script = assets.joinpath("scratch_run_array.pbs.sh.in").read_text()
    submit_script = assets.joinpath("scratch_submit.sh.in").read_text()
    assert 'STAGE_ROOT=$(mktemp -d "$SCRATCHDIR/' in run_script
    assert 'timeout --signal=TERM' in run_script
    assert 'shuf --output="$TASK_LIST"' in submit_script


def test_standalone_scripts_are_organized_under_scripts_directory() -> None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    assert not (project_root / "evaluate_molecular_egfr.py").exists()
    assert not (project_root / "run_array.pbs.sh").exists()
    assert not (project_root / "submit.sh").exists()
    assert (project_root / "scripts" / "evaluate_molecular_egfr.py").is_file()
    assert (project_root / "scripts" / "metacentrum_run_array.pbs.sh").is_file()
    assert (project_root / "scripts" / "metacentrum_submit.sh").is_file()
