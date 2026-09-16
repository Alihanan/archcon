from pathlib import Path

import archcon.data.defaults as defaults_module
from archcon.data.defaults import (
    data_directory_status,
    detected_default_paths,
    project_data_layout,
    resolve_data_directory,
)


def test_explicit_data_directory_wins(tmp_path: Path, monkeypatch) -> None:
    environment_dir = tmp_path / "environment"
    explicit_dir = tmp_path / "explicit"
    monkeypatch.setenv("ARCHCON_DATA_DIR", str(environment_dir))

    assert resolve_data_directory(explicit_dir) == explicit_dir.resolve()


def test_environment_data_directory(tmp_path: Path, monkeypatch) -> None:
    environment_dir = tmp_path / "environment"
    monkeypatch.setenv("ARCHCON_DATA_DIR", str(environment_dir))

    assert resolve_data_directory() == environment_dir.resolve()


def test_source_tree_data_is_found_when_launched_elsewhere(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    source_file = project_root / "src" / "archcon" / "data" / "defaults.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    project_data = project_root / "data"
    (project_data / "GEO_NUMPY_STORE").mkdir(parents=True)

    launch_dir = tmp_path / "elsewhere"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("ARCHCON_DATA_DIR", raising=False)
    monkeypatch.setattr(defaults_module, "__file__", str(source_file))

    assert resolve_data_directory() == project_data.resolve()


def test_launch_data_wins_when_it_is_recognizable(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    source_file = project_root / "src" / "archcon" / "data" / "defaults.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / "data" / "GEO_NUMPY_STORE").mkdir(parents=True)

    launch_dir = tmp_path / "launch"
    launch_data = launch_dir / "data"
    (launch_data / "GEO_NUMPY_STORE").mkdir(parents=True)
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("ARCHCON_DATA_DIR", raising=False)
    monkeypatch.setattr(defaults_module, "__file__", str(source_file))

    assert resolve_data_directory() == launch_data.resolve()


def test_detected_defaults_only_include_existing_resources(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "expression_matrix.csv").write_text("sample,probe\n", encoding="utf-8")
    (data_dir / "egfr_data.xlsx").touch()
    (data_dir / "CEL").mkdir()
    (data_dir / "GEO_NUMPY_STORE").mkdir()

    layout = project_data_layout(data_dir)
    defaults = detected_default_paths(layout)

    assert defaults["expression"] == str((data_dir / "expression_matrix.csv").resolve())
    assert defaults["reference"] == defaults["expression"]
    assert defaults["egfr"] == str((data_dir / "egfr_data.xlsx").resolve())
    assert defaults["clinical"] == ""
    assert defaults["geo"] == ""
    assert defaults["geo_rma"] == str((data_dir / "GEO_NUMPY_STORE").resolve())
    assert defaults["cel"] == str((data_dir / "CEL").resolve())


def test_status_lists_expected_resources(tmp_path: Path) -> None:
    layout = project_data_layout(tmp_path / "data")
    status = data_directory_status(layout)

    assert "expression_matrix.csv" in status
    assert "common_probes.pkl" in status
    assert "geo_expr_normalized_to_ikem.parquet" in status
    assert "GEO_NUMPY_STORE/" in status
    assert "ARCHCON_DATA_DIR" in status


def test_private_cel_ikem_store_is_preferred_over_historical_store(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    historical = data / "IKEM_NUMPY_STORE"
    historical.mkdir(parents=True)
    assert project_data_layout(data).ikem_store == historical.resolve()

    private_cel = data / "IKEM_CEL_NUMPY_STORE"
    private_cel.mkdir()
    layout = project_data_layout(data)
    assert layout.ikem_store == private_cel.resolve()
    assert layout.ikem_legacy_store == historical.resolve()
