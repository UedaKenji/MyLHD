"""Plotting utilities for the Thomson GP module."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def axs_cbar(ax: plt.Axes, im, cbar_title: Optional[str] = None, **kwargs) -> None:
    """Attach a colorbar to the right side of the provided axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="3%")
    plt.colorbar(im, cax=cax, orientation="vertical", label=cbar_title, **kwargs)


def make_alpha_colormap(color: str) -> LinearSegmentedColormap:
    """Create a colormap with a linear alpha ramp based on a single base color."""
    rgba = plt.cm.colors.to_rgba(color)
    cmap_dict = {
        "red": [[0, rgba[0], rgba[0]], [1, rgba[0], rgba[0]]],
        "green": [[0, rgba[1], rgba[1]], [1, rgba[1], rgba[1]]],
        "blue": [[0, rgba[2], rgba[2]], [1, rgba[2], rgba[2]]],
        "alpha": [[0, 0, 0], [1, 1, 1]],
    }
    return LinearSegmentedColormap("CustomMap", cmap_dict)


def plotGP(
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    x_inp: np.ndarray,
    f_inp: np.ndarray,
    K_inp: np.ndarray,
    ax: Optional[plt.Axes] = None,
    color: str = "darkblue",
    alpha_max: float = 1.0,
    n_sample: int = 0,
    **_: dict,
) -> None:
    """Visualize a Gaussian process posterior as a heatmap with optional samples."""
    import scipy.interpolate

    if ax is None:
        ax = plt.gca()

    mean_interp = scipy.interpolate.interp1d(x_inp, f_inp)
    posterior_mean = mean_interp(x_plot)[np.newaxis, :]

    sigma = np.sqrt(np.diag(K_inp))
    sigma_interp = scipy.interpolate.interp1d(x_inp, sigma)
    posterior_sigma = sigma_interp(x_plot)[np.newaxis, :]

    _, Y = np.meshgrid(x_plot, y_plot)
    alpha_cmap = make_alpha_colormap(color)
    density = 1 / posterior_sigma * np.exp(-0.5 * (posterior_mean - Y) ** 2 / posterior_sigma**2)
    density = density / density.max() * alpha_max

    dx = x_plot[1] - x_plot[0]
    dy = y_plot[1] - y_plot[0]
    extent = (x_plot.min() - 0.5 * dx, x_plot.max() + 0.5 * dx, y_plot.min() - 0.5 * dy, y_plot.max() + 0.5 * dy)
    ax.imshow(density, vmin=0, vmax=1, extent=extent, cmap=alpha_cmap, origin="lower", aspect="auto", zorder=0)

    if n_sample > 0:
        samples = np.random.multivariate_normal(f_inp, K_inp, n_sample)
        for idx, sample in enumerate(samples):
            ax.plot(x_inp, sample, color=color, alpha=0.5, lw=0.3, label="sample path" if idx == 0 else None)

    ax.plot(x_inp, f_inp + sigma, color=color, linestyle="dashed", lw=1.0, label=r"$\pm 1\sigma$")
    ax.plot(x_inp, f_inp - sigma, color=color, linestyle="dashed", lw=1.0)


def plotLogGP(
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    x_inp: np.ndarray,
    f_inp: np.ndarray,
    K_inp: np.ndarray,
    ax: Optional[plt.Axes] = None,
    color: str = "darkblue",
    alpha_max: float = 1.0,
    n_sample: int = 0,
    **_: dict,
) -> None:
    """Plot a log-scale Gaussian process posterior."""
    import scipy.interpolate

    if ax is None:
        ax = plt.gca()

    mean_interp = scipy.interpolate.interp1d(x_inp, f_inp)
    posterior_mean = mean_interp(x_plot)[np.newaxis, :]

    sigma = np.sqrt(np.diag(K_inp))
    sigma_interp = scipy.interpolate.interp1d(x_inp, sigma)
    posterior_sigma = sigma_interp(x_plot)[np.newaxis, :]

    y_plot = y_plot.copy()
    y_plot[y_plot < 0] = np.nan
    _, log_y = np.meshgrid(x_plot, np.log(y_plot))

    alpha_cmap = make_alpha_colormap(color)
    density = 1 / posterior_sigma * np.exp(-0.5 * (posterior_mean - log_y) ** 2 / posterior_sigma**2)
    density = density / density.max() * alpha_max

    dx = x_plot[1] - x_plot[0]
    dy = y_plot[1] - y_plot[0]
    extent = (x_plot.min() - 0.5 * dx, x_plot.max() + 0.5 * dx, y_plot.min() - 0.5 * dy, y_plot.max() + 0.5 * dy)
    ax.imshow(density, vmin=0, vmax=1, extent=extent, cmap=alpha_cmap, origin="lower", aspect="auto", zorder=0)

    if n_sample > 0:
        samples = np.random.multivariate_normal(f_inp, K_inp, n_sample)
        for idx, sample in enumerate(samples):
            values = np.exp(sample)
            ax.plot(x_inp, values, color=color, alpha=0.5, lw=0.3, label="sample path" if idx == 0 else None)

    ax.plot(x_inp, np.exp(f_inp + sigma), color=color, linestyle="dashed", lw=1.0, label=r"$\pm 1\sigma$")
    ax.plot(x_inp, np.exp(f_inp - sigma), color=color, linestyle="dashed", lw=1.0)


__all__ = ["axs_cbar", "make_alpha_colormap", "plotGP", "plotLogGP"]
