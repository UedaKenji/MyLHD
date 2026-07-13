from __future__ import annotations

import random
import string
import time
from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


def randfilename(n: int, template: str = "tmp%s.dat") -> str:
    """Create a random filename from ``template``."""
    chars = [random.choice(string.ascii_letters + string.digits) for _ in range(n)]
    return template % "".join(chars)


def map_onoff(
    t_ref: npt.ArrayLike,
    state_ref: npt.ArrayLike,
    t_target: npt.ArrayLike,
    deadtime_rise: float = 0.0,
    deadtime_fall: float = 0.0,
) -> np.ndarray:
    """
    Map a boolean ON/OFF waveform from ``t_ref`` onto ``t_target``.

    Parameters
    ----------
    t_ref : (N,) array
        Monotonic reference timestamps.
    state_ref : (N,) bool array
        ON/OFF state at each ``t_ref`` sample.
    t_target : (M,) array
        Monotonic timestamps to map onto.
    deadtime_rise : float, default 0.0
        Delay applied to rising edges.
    deadtime_fall : float, default 0.0
        Delay applied to falling edges.
    """
    ref = np.asarray(t_ref)
    state = np.asarray(state_ref, dtype=bool)
    target = np.asarray(t_target)

    if ref.ndim != 1 or state.ndim != 1 or target.ndim != 1:
        raise ValueError("t_ref, state_ref, and t_target must be 1D arrays")
    if ref.size == 0:
        raise ValueError("t_ref must not be empty")
    if ref.size != state.size:
        raise ValueError("t_ref and state_ref must have the same length")
    if np.any(np.diff(ref) < 0):
        raise ValueError("t_ref must be monotonically increasing")
    if np.any(np.diff(target) < 0):
        raise ValueError("t_target must be monotonically increasing")

    diff = np.diff(state.astype(int))
    rise_idx = np.where(diff == 1)[0] + 1
    fall_idx = np.where(diff == -1)[0] + 1

    t_rise = ref[rise_idx] + deadtime_rise
    t_fall = ref[fall_idx] + deadtime_fall

    initial_state = state[0]
    n_rise = np.searchsorted(t_rise, target, side="right")
    n_fall = np.searchsorted(t_fall, target, side="right")
    return initial_state ^ ((n_rise - n_fall) % 2 == 1)


def wait_for_opendata(
    diag: str,
    shotno: int,
    subno: int = 1,
    retry_delay: int = 60,
    retrieve_func: Callable[..., T] | None = None,
) -> T:
    """
    Poll an open-data retrieval function until the dataset becomes available.
    """
    if retry_delay <= 0:
        raise ValueError("retry_delay must be a positive number of seconds.")

    if retrieve_func is None:
        from .anadata import KaisekiData

        retrieve_func = KaisekiData.retrieve_opendata

    elapsed = 0
    while True:
        try:
            return retrieve_func(diag=diag, shotno=shotno, subno=subno)
        except FileNotFoundError:
            elapsed += retry_delay
            print(
                f"No data for diag={diag}, shotno={shotno}, subno={subno}. Waiting {elapsed:>6d}s...",
                end="\r",
                flush=True,
            )
            time.sleep(retry_delay)


def detect_data(
    diag: str,
    shotno: int,
    retry_delay: int = 60,
    retrieve_func: Callable[..., object] | None = None,
    search_num: int = 10,
) -> int:
    """
    Wait for ``shotno`` and nearby future shots until open data becomes available.
    """
    if retry_delay <= 0:
        raise ValueError("retry_delay must be a positive number of seconds.")
    if search_num < 0:
        raise ValueError("search_num must be non-negative.")

    if retrieve_func is None:
        from .anadata import KaisekiData

        retrieve_func = KaisekiData.retrieve_opendata

    elapsed = 0
    while True:
        try:
            retrieve_func(diag=diag, shotno=shotno)
            return shotno
        except (FileNotFoundError, RuntimeError):
            started_at = time.time()
            for candidate in range(shotno + 1, shotno + search_num + 1):
                try:
                    retrieve_func(diag=diag, shotno=candidate)
                    print(f"Found data for diag={diag}, shotno={candidate}.")
                    return candidate
                except (FileNotFoundError, RuntimeError):
                    continue

            elapsed += retry_delay
            print(
                f"No data for diag={diag}, shotno={shotno} Waiting {elapsed:>6d}s...",
                end="\r",
                flush=True,
            )
            time.sleep(max(0.0, retry_delay - (time.time() - started_at)))


__all__ = [
    "detect_data",
    "map_onoff",
    "randfilename",
    "wait_for_opendata",
]
