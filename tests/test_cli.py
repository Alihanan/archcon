from archcon.cli import build_parser


def test_cli_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.no_browser is False
