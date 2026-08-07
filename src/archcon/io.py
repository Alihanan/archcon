"""File-loading utilities used by ArchCon and its web interface."""

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

TEXT_FILE_TYPES = [
    "text",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".r",
    ".rst",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
]


def format_size(size_bytes: int) -> str:
    """Return a compact human-readable byte count."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes / 1024**2:.1f} MiB"


def read_text_file(
    file_path: str | Path | None,
    *,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> tuple[str, str]:
    """Read a text file and return status text plus decoded contents.

    The original file is never modified. The function is independent of the
    web interface, so it can also be used directly from Python code.
    """
    if file_path is None:
        return "No file selected.", ""

    path = Path(file_path)

    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"Could not inspect the file: {exc}", ""

    if size > max_size_bytes:
        return (
            f"File is too large ({format_size(size)}). "
            f"The current limit is {format_size(max_size_bytes)}.",
            "",
        )

    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"Could not read the file: {exc}", ""

    if b"\x00" in data[:8192]:
        return "This appears to be a binary file, not a text file.", ""

    match = from_bytes(data).best()
    if match is None:
        return "The text encoding could not be detected.", ""

    encoding = match.encoding or "unknown"
    status = f"Loaded {path.name} · {format_size(size)} · encoding: {encoding}"
    return status, str(match)
