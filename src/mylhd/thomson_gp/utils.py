"""Utility helpers for the Thomson GP module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import interpolate

if TYPE_CHECKING:
    from .core import ThomsonGPCore


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


class ThomsonGPUtils:
    """Utility functions for the Thomson GP module."""

    def export_nTprofile(
        self: ThomsonGPCore,
        dir_path: str,
        Zeff: np.ndarray | float = 1,
        N_points: int = 101,
    ) -> None:

        dir_path = Path(dir_path) / str(self.shotNo)
        dir_path.mkdir(parents=True, exist_ok=True)

        for time_i in range(self.time_inp.size):
            # print(f"time_i={time_i}, time={model.time_inp[time_i]:.3f} [s]")

            rho_inp_i = self.rho_inp[time_i, :]
            f = interpolate.interp1d(rho_inp_i, self.rho_vac)
            rho_max = min(-rho_inp_i.min(), rho_inp_i.max(), 1.15)

            x = np.linspace(-rho_max, rho_max, (N_points - 1) * 2 + 1)

            Te = self.local_post_mean_function(
                ValName="Te",
                function_kind="f",
                rho_vac_l=f(x),
                timel=[self.time_inp[time_i]],
            )
            Te = Te[0, :]

            Te = 0.5 * (np.exp(Te) + np.exp(Te[::-1]))
            Ne = self.local_post_mean_function(
                ValName="Ne",
                function_kind="f",
                rho_vac_l=f(x),
                timel=[self.time_inp[time_i]],
            )
            Ne = Ne[0, :]
            Ne = 0.5 * (np.exp(Ne) + np.exp(Ne[::-1]))
            Ne = Ne[N_points - 1 :]
            rho_a99 = x[N_points - 1 :]
            Te = Te[N_points - 1 :]

            zeff = Zeff * np.ones_like(rho_a99)
            t = self.time_inp[time_i]
            # print(f"shot={shotno}, time={t:.3f} [s]")

            write_profile(
                f"{dir_path}/nT_profile_{self.shotNo}t{t:.3f}.txt",
                rho_a99,
                Ne,
                Te,
                zeff,
            )


def write_profile(filename, reff, ne, Te, Zeff):
    """
    reff, ne, Te, Zeff の配列を指定フォーマットでファイル出力する関数

    Parameters
    ----------
    filename : str
        出力ファイル名
    reff, ne, Te, Zeff : np.ndarray
        同じ長さの1次元配列
        同じ長さの1次元配列
    """
    npoints = len(reff)
    if not (len(ne) == len(Te) == len(Zeff) == npoints):
        raise ValueError("全ての配列は同じ長さである必要があります．")

    with open(filename, "w") as f:
        # ヘッダ部分
        f.write("CC  reff/a     Ne[1/m^3]     Te[keV]     Zeff\n")
        f.write(f" Number_of_points   {npoints}\n")

        # データ本体
        for r, n, t, z in zip(reff, ne, Te, Zeff):
            f.write(f" {r:10.4E}  {n*1e19:10.4E}  {t:10.4E}  {z:10.4E}\n")


__all__ = [
    "CallbackCounter",
    "callback_counter",
    "Logger",
    "numerical_differentiation_matrix",
]
