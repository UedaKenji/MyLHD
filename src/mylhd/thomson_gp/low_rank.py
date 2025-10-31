"""Low-rank positive definite matrix utilities used by the Thomson GP model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LowRankPDM:
    """Low-rank approximation of a positive definite matrix."""

    matrix: np.ndarray
    rank: int | None = None
    cutoff: float = 1e-5
    stabilization: float = 1e-5
    identity: bool = False
    iprint: bool = False

    def __post_init__(self) -> None:
        self.N = self.matrix.shape[0]

        if self.identity:
            self.Lambda = np.ones(self.N)
            self.V = self.Lambda[:, np.newaxis]
            if self.iprint:
                print("identity")
            return

        eigvals, eigvecs = np.linalg.eigh(self.matrix)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        self.Lambda = eigvals
        self.V = eigvecs

        if self.rank is not None:
            self.Lambdal = eigvals[: self.rank]
            self.Vl = eigvecs[:, : self.rank]
            self.not_lowrank = False
        else:
            eigenval_rate = np.cumsum(eigvals[::-1])[::-1] / np.sum(eigvals)
            cutoff_index = eigenval_rate > self.cutoff

            if int(np.sum(cutoff_index)) > int(self.N / 2):
                self.not_lowrank = True
                self.Lambdal = self.Lambda
                self.Vl = self.V
            else:
                self.not_lowrank = False
                self.Lambdal = eigvals[cutoff_index]
                self.Vl = eigvecs[:, cutoff_index]

        if self.iprint:
            print(self.Vl.shape)

    def __call__(self) -> np.ndarray:
        if self.identity:
            return np.eye(self.N)
        if self.not_lowrank:
            return self.matrix
        return self.Vl @ (self.Lambdal[:, np.newaxis] * self.Vl.T) + np.eye(self.Vl.shape[0]) * self.stabilization

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        if self.identity:
            return other

        try:
            dim = len(other.shape)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise TypeError("other must be a numpy.ndarray") from exc

        if self.not_lowrank:
            return self.matrix @ other

        if dim == 1:
            return self.Vl @ (self.Lambdal * (self.Vl.T @ other)) + self.stabilization * other

        if dim == 2:
            return self.Vl @ (self.Lambdal[:, np.newaxis] * (self.Vl.T @ other)) + self.stabilization * other

        raise TypeError("other must be a 1D or 2D numpy array")

    def __lmatmul__(self, other: np.ndarray) -> np.ndarray:
        if self.identity:
            return other

        try:
            dim = len(other.shape)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise TypeError("other must be a numpy.ndarray") from exc

        if self.not_lowrank:
            return other @ self.matrix

        if dim == 1:
            return ((other @ self.Vl) * self.Lambdal) @ self.Vl.T

        if dim == 2:
            return ((other @ self.Vl) @ self.Lambdal) @ self.Vl.T

        raise TypeError("other must be a 1D or 2D numpy array")

    def inverse(self, M: np.ndarray, sig_sq: float = 0) -> np.ndarray:
        if self.identity:
            return M

        try:
            dim = len(M.shape)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise TypeError("M must be a numpy.ndarray") from exc

        lambda_inv = 1.0 / (self.Lambda + self.stabilization + sig_sq)

        if dim == 1:
            return self.V @ (lambda_inv * (self.V.T @ M))

        if dim == 2:
            return self.V @ (lambda_inv[:, np.newaxis] * (self.V.T @ M))

        raise TypeError("M must be a 1D or 2D numpy array")


__all__ = ["LowRankPDM"]
