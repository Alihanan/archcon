from archcon import build_app


def test_build_app() -> None:
    assert build_app() is not None


def test_reference_qc_helpers_import() -> None:
    from archcon.data.qc import plot_dataset_vs_reference, plot_sample_vs_reference

    assert callable(plot_dataset_vs_reference)
    assert callable(plot_sample_vs_reference)


def test_cel_rma_dropdown_helpers(tmp_path) -> None:
    from archcon.app import (
        CEL_SOURCE_CUSTOM,
        CEL_SOURCE_DETECTED,
        CEL_SOURCE_UPLOAD,
        RMA_OUTPUT_CUSTOM,
        RMA_OUTPUT_DATA,
        RMA_OUTPUT_TEMPORARY,
        _cel_source_choices,
        _default_cel_source,
        _resolve_rma_output_directory,
    )

    detected = str(tmp_path / "CEL")
    assert _cel_source_choices("") == [CEL_SOURCE_UPLOAD, CEL_SOURCE_CUSTOM]
    assert _cel_source_choices(detected) == [
        CEL_SOURCE_UPLOAD,
        CEL_SOURCE_DETECTED,
        CEL_SOURCE_CUSTOM,
    ]
    assert _default_cel_source("") == CEL_SOURCE_UPLOAD
    assert _default_cel_source(detected) == CEL_SOURCE_DETECTED

    assert _resolve_rma_output_directory(RMA_OUTPUT_TEMPORARY, str(tmp_path), "") is None
    assert _resolve_rma_output_directory(RMA_OUTPUT_DATA, str(tmp_path), "") == str(
        (tmp_path / "rma").resolve()
    )
    custom = tmp_path / "custom-rma"
    assert _resolve_rma_output_directory(RMA_OUTPUT_CUSTOM, str(tmp_path), str(custom)) == str(
        custom
    )


def test_model_output_root_uses_launch_directory(tmp_path, monkeypatch) -> None:
    from archcon.app import _model_output_root

    monkeypatch.chdir(tmp_path)
    assert _model_output_root() == (tmp_path / "models").resolve()




def test_gradio_temp_root_uses_launch_directory(tmp_path, monkeypatch) -> None:
    from archcon.app import _configure_gradio_storage

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)
    root = _configure_gradio_storage()
    assert root == (tmp_path / ".archcon-gradio").resolve()
    assert root.is_dir()


def test_checkpoint_download_html_serves_original_path_without_cache(tmp_path) -> None:
    from archcon.app import _checkpoint_download_html

    checkpoint = tmp_path / "models" / "geo_ae_test" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    html = _checkpoint_download_html(checkpoint, "Latest checkpoint")
    assert "/gradio_api/file=" in html
    assert str(checkpoint.resolve()) in html
    assert "/tmp/gradio" not in html

def test_geo_pretraining_pipeline_is_single_page() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    tabs = [component for component in components if component["type"] == "tabs"]
    assert tabs == []

    button_values = {
        str(component.get("props", {}).get("value", ""))
        for component in components
        if component["type"] == "button"
    }
    assert any("01 · Unsupervised data" in value for value in button_values)
    assert any("02 · Supervised data" in value for value in button_values)
    assert any("05 · Split" in value for value in button_values)
    assert any("06 · Molecular AE" in value for value in button_values)
    assert any("07 · Downstream" in value for value in button_values)


def test_pipeline_navigation_marks_selected_button_primary() -> None:
    from archcon.app import _stage_navigation_callback

    updates = _stage_navigation_callback("rma")
    panel_updates = updates[:7]
    button_updates = updates[7:]

    assert [item.get("visible") for item in panel_updates] == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [item.get("variant") for item in button_updates] == [
        "secondary",
        "secondary",
        "primary",
        "secondary",
        "secondary",
        "secondary",
        "secondary",
    ]


def test_geo_data_stage_has_no_normalization_aggregate_buttons() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    button_values = {
        str(component.get("props", {}).get("value", ""))
        for component in components
        if component["type"] == "button"
    }

    assert "All unique · Raw original" not in button_values
    assert "All unique · Per-dataset RMA" not in button_values
    assert "All unique · Global RMA" not in button_values
    assert any("① No RMA" in value for value in button_values)
    assert any("② Per-study RMA" in value for value in button_values)
    assert any("③ Global RMA" in value for value in button_values)


def test_old_text_log_viewer_is_not_in_web_ui() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    labels = {str(component.get("props", {}).get("label", "")) for component in components}
    values = {str(component.get("props", {}).get("value", "")) for component in components}
    assert "Text file" not in labels
    assert not any("text/log viewer" in value.lower() for value in values)


def test_latent_stage_contains_real_training_controls() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    button_values = {
        str(component.get("props", {}).get("value", ""))
        for component in components
        if component["type"] == "button"
    }
    assert "▶ Start training with current settings" in button_values
    assert "⏹ Stop after current batch" in button_values


def test_async_view_guard_prevents_job_piling():
    import threading

    from archcon import app as app_module

    release = threading.Event()

    def slow_callback():
        release.wait(timeout=2.0)
        return "finished"

    first = app_module._async_view_result(slow_callback, 1)
    first_loading = next(first)
    assert "archcon-spinner" in first_loading[0]

    second = app_module._async_view_result(lambda: "should-not-run", 1)
    second_loading = next(second)
    assert "archcon-spinner" in second_loading[0]
    try:
        next(second)
        raise AssertionError("a second view task should not be queued")
    except StopIteration:
        pass

    release.set()
    final = next(first)
    assert final[1] == "finished"
    assert "active" not in final[0]


def test_architecture_and_aa_controls_are_present():
    app = build_app()
    components = app.get_config_file()["components"]
    labels = {str(component.get("props", {}).get("label", "")) for component in components}
    assert "Architecture preset" in labels
    assert "Layer family" in labels
    assert "Compute precision" in labels
    assert "Future AA design" in labels


def test_latent_stage_contains_pytorch_audit_view() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    labels = {str(component.get("props", {}).get("label", "")) for component in components}
    assert "Generated PyTorch structural audit" in labels


def test_loss_specific_controls_follow_selected_loss() -> None:
    from archcon.app import _loss_specific_control_updates
    from archcon.data.pretraining import LOSS_DVIB, LOSS_HUBER, LOSS_MASKED, LOSS_MSE

    assert [update.get("visible") for update in _loss_specific_control_updates(LOSS_MSE)] == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert [update.get("visible") for update in _loss_specific_control_updates(LOSS_HUBER)] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert [update.get("visible") for update in _loss_specific_control_updates(LOSS_DVIB)] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert [update.get("visible") for update in _loss_specific_control_updates(LOSS_MASKED)] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_latent_stage_contains_metacentrum_sweep_export() -> None:
    app = build_app()
    components = app.get_config_file()["components"]
    labels = {str(component.get("props", {}).get("label", "")) for component in components}
    button_values = {
        str(component.get("props", {}).get("value", ""))
        for component in components
        if component["type"] == "button"
    }
    assert "Architecture + preprocessing hyperparameter grid · JSON · 900 runs" in labels
    assert "Generated run_array.pbs.sh preview" in labels
    assert "Generate Python jobs + JSON + PBS array" in button_values
