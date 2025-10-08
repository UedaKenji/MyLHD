"""Kernel functions used by the Thomson GP model."""

from __future__ import annotations

import numpy as np


def KSE(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Squared-exponential kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    return np.exp(-0.5 * (X[0] - X[1]) ** 2 / length**2)


def KSE_dx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """First derivative of the squared-exponential kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    return (-(X[0] - X[1]) / length**2) * np.exp(-0.5 * (X[0] - X[1]) ** 2 / length**2)


def KSE_dxdx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Second derivative of the squared-exponential kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    return (-(X[0] - X[1]) ** 2 / length**4 + 1 / length**2) * np.exp(-0.5 * (X[0] - X[1]) ** 2 / length**2)


def KRQ(x0: np.ndarray, x1: np.ndarray, length: float, alpha: float) -> np.ndarray:
    """Rational quadratic kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    return (1 + 0.5 * (X[0] - X[1]) ** 2 / length**2 / alpha) ** (-alpha)


def KRQ_dx(x0: np.ndarray, x1: np.ndarray, length: float, alpha: float) -> np.ndarray:
    """First derivative of the rational quadratic kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    return -(1 + 0.5 * (X[0] - X[1]) ** 2 / length**2 / alpha) ** (-alpha - 1) * (X[0] - X[1]) / length**2


def Kmatern32(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Matérn 3/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    R = np.abs(X[0] - X[1])
    return (1 + np.sqrt(3) * R / length) * np.exp(-np.sqrt(3) * R / length)


def Kmatern32_dx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """First derivative of the Matérn 3/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    tau = X[0] - X[1]
    return -3 * tau / length**2 * np.exp(-np.sqrt(3) * np.abs(tau) / length)


def Kmatern32_dxdx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Second derivative of the Matérn 3/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    R = np.abs(X[0] - X[1])
    return 3 / length**2 * (1 - np.sqrt(3) * R / length) * np.exp(-np.sqrt(3) * R / length)


def Kmartner52(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Matérn 5/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    R = np.abs(X[0] - X[1]) / length
    return (1 + np.sqrt(5) * R + 5 / 3 * R**2) * np.exp(-np.sqrt(5) * R)


def Kmartner52_dx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """First derivative of the Matérn 5/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    tau = (X[0] - X[1]) / length
    R = np.abs(tau)
    return -5 / 3 * tau / length * (1 + np.sqrt(5) * R) * np.exp(-np.sqrt(5) * R)


def Kmartner52_dxdx(x0: np.ndarray, x1: np.ndarray, length: float) -> np.ndarray:
    """Second derivative of the Matérn 5/2 kernel."""
    X = np.meshgrid(x0, x1, indexing="ij")
    R = np.abs(X[0] - X[1]) / length
    return 5 / 3 / length**2 * (1 + np.sqrt(5) * R - 5 * R**2) * np.exp(-np.sqrt(5) * R)


__all__ = [
    "KSE",
    "KSE_dx",
    "KSE_dxdx",
    "KRQ",
    "KRQ_dx",
    "Kmatern32",
    "Kmatern32_dx",
    "Kmatern32_dxdx",
    "Kmartner52",
    "Kmartner52_dx",
    "Kmartner52_dxdx",
]
