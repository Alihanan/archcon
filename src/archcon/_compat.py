"""Compatibility helpers for the oldest supported Python release."""

from __future__ import annotations

from itertools import zip_longest
from typing import Iterable, Iterator, Tuple, TypeVar


_T = TypeVar("_T")


def strict_zip(*iterables: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    """Python 3.9-compatible equivalent of ``zip(..., strict=True)``."""

    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values  # type: ignore[misc]
