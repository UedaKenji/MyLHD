"""Utility helpers for the Thomson GP module."""

from __future__ import annotations

import sys

import numpy as np


class CallbackCounter:
    """Simple callable that counts how many times it is invoked."""

    def __init__(self) -> None:
        self.iter = 0

    def __call__(self, _xk) -> None:
        self.iter += 1


class Logger:
    """Lightweight stdout logger that stores the printed history."""

    def __init__(self) -> None:
        self.terminal = sys.stdout
        self.log = ""

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log += message

    def flush(self) -> None:  # pragma: no cover - required for file-like API
        pass


def numerical_differentiation_matrix(x: np.ndarray) -> np.ndarray:
    """
    Construct a numerical differentiation matrix using forward/central/backward differences.

    Parameters
    ----------
    x:
        One-dimensional array of strictly increasing sample points.
    """
    n = len(x)
    diff_matrix = np.zeros((n, n), dtype=float)

    # central differences
    for i in range(1, n - 1):
        span = x[i + 1] - x[i - 1]
        diff_matrix[i, i - 1] = -1.0 / span
        diff_matrix[i, i + 1] = 1.0 / span

    # forward difference at the beginning
    first_span = x[1] - x[0]
    diff_matrix[0, 0] = -1.0 / first_span
    diff_matrix[0, 1] = 1.0 / first_span

    # backward difference at the end
    last_span = x[-1] - x[-2]
    diff_matrix[-1, -2] = -1.0 / last_span
    diff_matrix[-1, -1] = 1.0 / last_span

    return diff_matrix


callback_counter = CallbackCounter  # backward compatibility alias


__all__ = ["CallbackCounter", "callback_counter", "Logger", "numerical_differentiation_matrix"]
