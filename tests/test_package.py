import archcon


def test_public_api() -> None:
    assert callable(archcon.build_app)
    assert callable(archcon.read_text_file)
    assert callable(archcon.start)
    assert isinstance(archcon.__version__, str)
