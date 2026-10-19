"""Unit tests for the demo calculator."""
from __future__ import annotations

import pytest

from calculator import add, divide, subtract


def test_add() -> None:
    assert add(2, 3) == 5


def test_subtract() -> None:
    assert subtract(5, 2) == 3


def test_divide() -> None:
    assert divide(10, 2) == 5


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ValueError):
        divide(1, 0)
