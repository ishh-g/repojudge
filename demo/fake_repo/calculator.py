"""A few small arithmetic helpers for the demo."""
from __future__ import annotations


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


def divide(a: float, b: float) -> float:
    """Return a divided by b, raising ValueError when b is zero."""
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b
