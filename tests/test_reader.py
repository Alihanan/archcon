from pathlib import Path

from archcon import read_text_file
from archcon.app import read_uploaded_text


def test_reads_utf8_text(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello, světe! Привет!", encoding="utf-8")

    status, content = read_text_file(file_path)

    assert status.startswith("Loaded example.txt")
    assert content == "Hello, světe! Привет!"


def test_web_wrapper_uses_reader(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("content", encoding="utf-8")

    assert read_uploaded_text(str(file_path))[1] == "content"


def test_rejects_binary_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.bin"
    file_path.write_bytes(b"abc\x00def")

    status, content = read_text_file(file_path)

    assert "binary" in status.lower()
    assert content == ""


def test_rejects_large_file(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("too large", encoding="utf-8")

    status, content = read_text_file(file_path, max_size_bytes=1)

    assert "too large" in status.lower()
    assert content == ""
