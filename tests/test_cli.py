from archcon.cli import build_parser


def test_cli_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.no_browser is False


def test_cli_headless_config_arguments() -> None:
    args = build_parser().parse_args([
        "--run-config", "run.json",
        "--data-dir", "/data",
        "--output-root", "/results",
    ])
    assert args.run_config == "run.json"
    assert args.data_dir == "/data"
    assert args.output_root == "/results"
