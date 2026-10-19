"""Number Buddy: a tiny demo program.

This file is part of the RepoJudge demo repository. It is read as text only;
RepoJudge never runs it.
"""
from __future__ import annotations

import argparse
import sys

import requests

from calculator import add


def fetch_number(url: str) -> float:
    """Fetch a single number from a URL. Network problems are handled."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return float(response.text.strip())
    except (requests.RequestException, ValueError) as exc:
        print(f"Could not fetch a number: {exc}", file=sys.stderr)
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add two numbers.")
    parser.add_argument("a", type=float)
    parser.add_argument("b", type=float)
    args = parser.parse_args(argv)

    try:
        print(add(args.a, args.b))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
