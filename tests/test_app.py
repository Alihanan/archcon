from archcon import build_app


def test_build_app() -> None:
    assert build_app() is not None
