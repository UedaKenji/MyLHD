"""Visualization mixin for Thomson GP reconstructions."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .plotting import axs_cbar, make_alpha_colormap, plotGP, plotLogGP
from ..plot_utils import cmocean_balance

if TYPE_CHECKING:
    from .core import ThomsonGPCore


class ThomsonGPVisualizer:
    """Plotting helpers that operate on ThomsonGP model state."""

    def plot_im_ax(
        self: "ThomsonGPCore",
        kind: str = None,
        ax: plt.Axes = None,
        x_axis: str = "reff",
        rho_contour=True,
        vmax=None,
        vmin=None,
        cmap=None,
        colorbar=True,
    ) -> None:
        """
        Plot image of Te or Ne fitting on given axis.
        Parameters:
            kind: 'Te' or 'ne' or 'ne_fit' or"Te_fit" or "dlogTe/dr" or "dTe/dr" or "dlogTe/dt" or "dTe/dt" or "dlogne/dr" or "dne/dr" or "dlogne/dt" or "dne/dt"
            ax: axis to plot on
            x_axis: 'rho' or 'rho_vac' or 'R' to specify which x-axis to use
            rho_contour: whether to plot rho contours
            vmax: maximum value for color scale
            vmin: minimum value for color scale
            cmap: colormap to use
            colorbar: whether to show colorbar
        Example:
            >>> fig,ax = plt.subplots(1,1,figsize=(4,8))
            >>> thomson.plot_im_ax('Te_fit',x_axis='rho_vac',ax=ax)
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 8))

        kind = kind.lower()
        is_diff = False

        if kind == "te":
            f = self.Te_inp
            title = r"$T_e$_data" + r" $\mathrm{[keV]}$"
        elif kind == "ne":
            f = self.ne_inp
            title = r"$n_e$_data" + r" $\mathrm{[10^{19}m^{-3}]}$"
        elif kind == "te_fit":
            f = np.exp(self.logTe_fit)
            title = r"$T_e$_fit" + r" $\mathrm{[keV]}$"
        elif kind == "ne_fit":
            f = np.exp(self.logNe_fit)
            title = r"$n_e$_fit" + r" $\mathrm{[10^{19}m^{-3}]}$"
        elif kind == "dlogte/dr":
            f = self.dlogTe_dr
            title = r"$d\log(T_e)/d\rho_v$" + r" $[ ]$"
            is_diff = True
        elif kind == "dte/dr":
            f = self.dlogTe_dr * np.exp(self.logTe_fit)
            title = r"$dT_e/d\rho_v$" + r" $[\mathrm{keV}]$"
            is_diff = True
        elif kind == "dlogte/dt":
            f = self.dlogTe_dt
            title = r"$d\log(T_e)/dt$" + r" $[\mathrm{1/s}]$"
            is_diff = True
        elif kind == "dte/dt":
            f = self.dlogTe_dt * np.exp(self.logTe_fit)
            title = r"$dT_e/dt$" + r" $[\mathrm{keV/s}]$"
            is_diff = True
        elif kind == "dlogne/dr":
            f = self.dlogNe_dr
            title = r"$d\log{n_e}/d\rho$" + r" $[ ]$"
            is_diff = True
        elif kind == "dne/dr":
            f = self.dlogNe_dr * np.exp(self.logNe_fit)
            title = r"$d{n_e}/d\rho_v$" + r" $[\mathrm{10^{19}m^{-3}}]$"
            is_diff = True
        elif kind == "dlogne/dt":
            f = self.dlogNe_dt
            title = r"$d\log{n_e}/dt$" + r" $[\mathrm{1/s}]$"
            is_diff = True
        elif kind == "dne/dt":
            f = self.dlogNe_dt * np.exp(self.logNe_fit)
            title = r"$d(n_e)/dt$" + r" $[\mathrm{10^{19}m^{-3}/s}]$"
            is_diff = True
        else:
            raise NameError(
                'name must be "Te" or "ne" or "ne_fit" or"Te_fit" or "dlogTe/dr" or "dTe/dr" or "dlogTe/dt" or "dTe/dt" or "dlogne/dr" or "dne/dr" or "dlogne/dt" or "dne/dt"'
            )

        xlim_is_default = ax.get_xlim() == (0, 1.0)

        if x_axis in ["rho_vac", "reff/a99_vac"]:
            X = self.rho_vac
            x_title = r"${r_\mathrm{eff}/a_{99}}$ (vacuum)"
            if xlim_is_default:
                ax.set_xlim(-1.05, 1.05)
        elif x_axis in ["rho", "reff/a99"]:
            rho_contour = False
            X = self.rho_inp
            x_title = r"${r_\mathrm{eff}/a_{99}}$"
            if xlim_is_default:
                ax.set_xlim(-1.05, 1.05)
        elif x_axis in ["R", "real", "realR"]:
            X = self.realR_origin
            x_title = r"${R} $[m]"
        else:
            raise ValueError('x_axis must be "rho" or "rho_vac" or "R"')

        ax.set_xlabel(x_title)

        # pcolormesh 縺ｧ逋ｺ逕溘☆繧玖ｭｦ蜻翫ｒ荳譎ら噪縺ｫ辟｡隕悶☆繧・

        if (vmax is None) and (vmin is None):
            if is_diff:
                vmax = np.percentile(np.abs(f), 95) * 1.1
                vmin = -vmax
            else:
                vmax = np.percentile(f, 85) * 1.2
                vmin = 0
        elif vmin is None:
            if is_diff:
                vmin = -vmax
            else:
                vmin = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.pcolormesh(X, self.time_inp, f, vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(title)
        if colorbar:
            axs_cbar(ax, im)

        if rho_contour:
            ax.contour(
                X,
                self.time_inp,
                self.rho_inp,
                levels=np.linspace(-1.0, 1.0, 11),
                colors="black",
                linewidths=0.5,
            )
            ax.contour(
                X,
                self.time_inp,
                self.rho_inp,
                levels=[0],
                colors="black",
                linewidths=1.0,
            )

    def plot_im(
        self: "ThomsonGPCore",
        ValName: str = "Te",
        x_axis: str = "reff",
        log: bool = False,
        save_name: str = None,
        rho_contour=True,
    ) -> None:
        """
        Plot image of Te or Ne fitting.

        Parameters:
            ValName: 'Te' or 'ne',

            x_axis: 'rho' or 'rho_vac' or 'R' to specify which x-axis to use

            #is_const_reff: boolean value indicating whether to use constant reff

        """
        ValName = self._valname(ValName)

        if ValName == "Te":
            data = self.Te_inp
            f = self.logTe_fit
            f_dr = self.dlogTe_dr
            f_dt = self.dlogTe_dt
            f_line = self.logTe_vline
            if log:
                axs_titles = [
                    "Te_data [keV]",
                    "Te_fit",
                    "dlogTe/dr [/m] ",
                    "dlogTe/dt [/s]",
                ]
            else:
                axs_titles = [
                    "Te_data [keV]",
                    "Te_fit [keV]",
                    "dTe/dr [keV/m]",
                    "dTe/dt [keV/s]",
                ]

        elif ValName == "ne":
            data = self.ne_inp
            f = self.logNe_fit
            f_dr = self.dlogNe_dr
            f_dt = self.dlogNe_dt
            f_line = self.logNe_vline
            if log:
                axs_titles = [
                    "ne_data [10^19 m^-3]",
                    "ne_fit",
                    "dlog(ne)/dr [/m] ",
                    "dlog(ne)/dt [/s]",
                ]
            else:
                axs_titles = [
                    "ne_data [10^19 m^-3]",
                    "ne_fit [10^19 m^-3]",
                    "dne/dr [10^19/m]",
                    "dne/dt [10^19/s]",
                ]

        else:
            raise NameError('ValName must be "Ne" or "Te"')

        # Add your code here
        if x_axis in ["rho_vac", "reff/a99_vac"]:
            x_axis = "rho_vac"
            X = self.rho_vac
            x_title = r"${r_\mathrm{eff}/a_{99}}$ (vacuum)"
        elif x_axis in ["rho", "reff/a99"]:
            x_axis = "rho"
            X = self.rho_inp
            x_title = r"${r_\mathrm{eff}/a_{99}}$"
        elif x_axis == "R":
            X = self.realR_origin
            x_title = r"${R}$"
        else:
            raise ValueError('x_axis must be "rho" or "rho_vac" or "R"')

        # Python 縺ｮ繝舌・繧ｸ繝ｧ繝ｳ縺ｫ蠢懊§縺ｦ figsize 繧定ｪｿ謨ｴ
        if sys.version_info[1] > 6:
            figunit = 1.5
        else:
            figunit = 2.0

        times = self.time_inp

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["none", "gray"])

        fig, ax = plt.subplots(1, 4, figsize=(figunit * 10, figunit * 5), sharey=True)

        im = ax[0].pcolormesh(
            self.realR_origin,
            times,
            data,
            vmax=np.exp(f).max() * 1.1,
            vmin=0,
            cmap="turbo",
        )
        axs_cbar(ax[0], im)
        ax[0].contour(
            self.realR_origin,
            times,
            self.rho_inp,
            levels=np.linspace(-0.7, 0.7, 15),
            colors="black",
            linewidths=0.5,
        )
        ax[0].set_title(axs_titles[0])
        ax[0].grid()
        ax[0].set_xlabel("R [m]")

        mask = f < np.log(np.exp(np.percentile(f, 95)) * 0.05)  # + self.idx_outlier

        im = ax[1].pcolormesh(X, times, np.exp(f), vmax=np.exp(f).max() * 1.1, vmin=0, cmap="turbo")
        axs_cbar(ax[1], im)

        if log:
            im = ax[2].pcolormesh(X, times, f_dr, vmax=5, vmin=-5, cmap=cmocean_balance)
        else:
            im = ax[2].pcolormesh(X, times, np.exp(f) * f_dr, vmax=10, vmin=-10, cmap=cmocean_balance)
        axs_cbar(ax[2], im)

        ax[2].pcolormesh(X, times, mask, cmap=cmap)

        if log:
            im = ax[3].pcolormesh(X, times, f_dt, vmax=5, vmin=-5, cmap=cmocean_balance)
        else:
            im = ax[3].pcolormesh(X, times, np.exp(f) * f_dt, vmax=10, vmin=-10, cmap=cmocean_balance)
        axs_cbar(ax[3], im)

        ax[3].pcolormesh(X, times, mask, cmap=cmap)

        print(self.idx_reff_outlier.shape)

        if x_axis in ["rho_vac", "R"]:
            for i in range(1, 4):
                if rho_contour:
                    ax[i].contour(
                        X,
                        times,
                        self.rho_inp,
                        levels=np.linspace(-1.0, 1.0, 11),
                        colors="black",
                        linewidths=0.5,
                        alpha=0.5,
                    )

        for i in range(1, 4):
            ax_i = ax[i]
            ax_i.set_title(axs_titles[i])
            ax_i.grid()
            ax_i.set_xlabel(x_title)
            if x_axis == "R":
                pass
                # ax_i.set_xlim(3.0,4.5)
            else:
                ax_i.set_xlim(-1.05, 1.05)

        ax[0].set_ylabel("time [s]")
        if save_name is not None:
            # tight_layout 縺ｧ菫晏ｭ・
            plt.tight_layout()
            plt.savefig(save_name)
            plt.close()

    def plot_im_ax(
        self: "ThomsonGPCore",
        ValName: str = "Te",
        funckind: str = "f",
        x_axis: str = "rho",
        ax: plt.Axes = None,
        rho_contour=True,
        log: bool = False,
        vmax=None,
        vmin=None,
        cmap=None,
        colorbar=False,
    ) -> plt.Axes:
        
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 8))

        res = self.get_func_properties(ValName, funckind, log)

        if cmap in  ['cmocean_balance' ,'balance', 'cmo_balance', 'cmbalance', 'cm_balance']:
            cmap = cmocean_balance

        f = res["f"]    
        ax_title = res["title"]

        xres = self.get_x_axis_properties(x_axis, ax)

        X = xres["X"]
        x_title = xres["x_title"]
        rho_contour = True if rho_contour and xres["rho_contour"] else False 
        ax.set_xlabel(x_title)

        im = ax.pcolormesh(X, self.time_inp, f, vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(ax_title)
        if colorbar:
            axs_cbar(ax, im)
        if rho_contour:
            ax.contour(
                X,
                self.time_inp,
                self.rho_inp,
                levels=np.linspace(-1.0, 1.0, 11),
                colors="black",
                linewidths=0.5,
            )
        
        return ax
    
    def get_func_properties(self: "ThomsonGPCore",
            ValName: str,
            FuncKind: str,
            log: bool = False) -> dict:
        """
        Get function properties for plotting.
        Parameters:
            ValName: The name of the value to plot (e.g., "Te" or "ne").
            FuncKind: The kind of function to plot (e.g., "f", "dfdr", or "dfdt").
            log: Whether to use a logarithmic scale.
        Returns:
            f: The function to plot.
            title: The title for the plot.
        """
        ValName = self._valname(ValName)
        kind_f = ['f', 'func', 'function', None]
        kind_dfdr = ['dfdr',  'dr', 'df/dr']
        kind_dfdt = ['dfdt',  'dt', 'df/dt']
        if ValName == "Te":
            if FuncKind in kind_f:
                if log:
                    f = self.logTe_fit
                    title = r"$\log T_e$_fit"+ r" $[ ]$"
                else:
                    f = np.exp(self.logTe_fit)
                    title = r"$T_e$_fit" + r" $\mathrm{[keV]}$"
            elif FuncKind in kind_dfdr:
                if log:
                    f = self.dlogTe_dr
                    title = r"$d\log(T_e)/d\rho_v$" + r" $[ ]$"
                else:
                    f = self.dlogTe_dr * np.exp(self.logTe_fit)
                    title = r"$dT_e/d\rho_v$" + r" $[\mathrm{keV}]$"
            elif FuncKind in kind_dfdt:
                if log:
                    f = self.dlogTe_dt
                    title = r"$d\log(T_e)/dt$" + r" $[\mathrm{1/s}]$"
                else:
                    f = self.dlogTe_dt * np.exp(self.logTe_fit)
                    title = r"$dT_e/dt$" + r" $[\mathrm{keV/s}]$"
            else:
                raise NameError('FuncKind must be "f" or "dfdr" or "dfdt"')

        elif ValName == "ne":
            if FuncKind in kind_f:
                if log:
                    f =  self.logNe_fit
                    title = r"$\log n_e$_fit" + r" $[ ]$"
                else:
                    f =  np.exp(self.logNe_fit)
                    title = r"$n_e$_fit $\mathrm{[10^{19}m^{-3}]}$"
            elif FuncKind in kind_dfdr:
                if log:
                    f = self.dlogNe_dr
                    title = r"$d\log(n_e)/d\rho_v$" + r" $[ ]$"
                else:
                    f = self.dlogNe_dr * np.exp(self.logNe_fit)
                    title = r"$d n_e/d\rho_v$" + r" $[\mathrm{10^{19}m^{-3}}]$"
            elif FuncKind in kind_dfdt:
                if log:
                    f = self.dlogNe_dt
                    title = r"$d\log(n_e)/dt$" + r" $[\mathrm{1/s}]$"
                else:
                    f = self.dlogNe_dt * np.exp(self.logNe_fit)
                    title = r"$d n_e/dt$" + r" $[\mathrm{10^{19}m^{-3}/s}]$"
            else:
                raise NameError('FuncKind must be "f" or "dfdr" or "dfdt"')
        else:
            raise NameError('ValName must be "ne" or "Te"')
        return {
            "f": f,
            "title": title
        }
    
    def get_x_axis_properties(self: "ThomsonGPCore",
            x_axis: str,
            ax:plt.Axes = None) -> dict:
        """
        Get x-axis properties for plotting.
        Parameters:
            x_axis: The name of the x-axis variable (e.g., "rho", "rho_vac", "R").
            ax: The matplotlib axis to use for setting limits (optional).
        Returns:
            X: The x-axis data.
            x_title: The title for the x-axis.
            rho_contour: Whether to plot rho contours.
        """
        if ax is not None:
            xlim_is_default = ax.get_xlim() == (0, 1.0)
            ylim_is_default = ax.get_ylim() == (0, 1.0)
        
        if x_axis in ["rho_vac", "reff/a99_vac"]:
            X = self.rho_vac
            x_title = r"${r_\mathrm{eff}/a_{99}}$ (vacuum)"
            rho_contour = True
            if ax is not None and xlim_is_default:
                ax.set_xlim(-1.01, 1.01)
            if ax is not None and ylim_is_default:
                ax.set_ylim(-1.01, 1.01)

        elif x_axis in ["rho", "reff/a99"]:
            rho_contour = False
            X = self.rho_inp
            x_title = r"${r_\mathrm{eff}/a_{99}}$"
            if ax is not None and xlim_is_default:
                ax.set_xlim(-1.15, 1.15)
            if ax is not None and ylim_is_default:
                ax.set_ylim(-1.15, 1.15)
        elif x_axis in ["R", "real", "realR"]:
            X = self.realR_origin
            x_title = r"${R} $[m]"
            rho_contour = True
            if ax is not None and ylim_is_default:
                ax.set_ylim(self.realR_origin.min(), self.realR_origin.max())
        else:
            raise ValueError('x_axis must be "rho" or "rho_vac" or "R"')
        return {
            "X": X,
            "x_title": x_title,
            "rho_contour": rho_contour
        }

    def plotProfile(
        self: "ThomsonGPCore",
        ValName: str = None,
        time: float = None,
        i: int = None,
        ax=None,
        color: str = "red",
        sampling: int = 0,
        add_noise: bool = False,
    ) -> None:
        """
        Plot the profile of the parameter.

        Parameters:
            ValName: 'ne' or 'Te'
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot
            add_noise: add noise to the plot

        Returns:
            None
        """
        ValName = self._valname(ValName)
        if ValName == "Te":
            y = self.Te_inp
            dy = self.dTe_data
            sig_scale = self.sigma_scale_Te
            f = self.logTe_fit
            ValUnit = self.Te_unit
            reff_scale = self.rho_scale_Te

        elif ValName == "ne":
            y = self.ne_inp
            sig_scale = self.sigma_scale_ne
            dy = abs(self.dne_data)
            f = self.logNe_fit
            ValUnit = self.Te_unit
            reff_scale = self.rho_scale_Ne
        else:
            raise NameError('ValName must be "ne" or "Te"')

        if time is not None:
            i = np.argmin(abs(self.time_inp - time))
        elif i is None:
            raise ValueError("time or i must be specified")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        timei = self.time_inp[i]

        idx_r = ~self.idx_reff_outlier

        rho_i = self.rho_origin[i, idx_r]
        rho_vac = self.rho_vac[idx_r]
        y = y[i, idx_r]
        dy = dy[i, idx_r]

        if ValName == "Te":
            idx_valid = ~self.is_outlier_Te[i, idx_r]
        elif ValName == "ne":
            idx_valid = ~self.is_outlier_ne[i, idx_r]

        if self.time_independence:
            Kpos_i = self.localPosterior_timeindependent(i, rho_vac, ValName=ValName)
        else:
            Kpos_i = self.localPosterior(np.array([timei]), rho_vac, ValName=ValName)
        sigma = np.sqrt(np.diag(Kpos_i))

        # self.Kpos_i = Kpos_i

        # valid data
        ax.errorbar(
            rho_i[idx_valid],
            y[idx_valid],
            yerr=dy[idx_valid],
            capsize=3,
            fmt="D",
            label="valid data",
            zorder=1,
        )

        ax.errorbar(
            rho_i[~idx_valid],
            y[~idx_valid],
            yerr=abs(dy[~idx_valid]),
            capsize=3,
            fmt="D",
            alpha=0.5,
            label="eliminated data",
            zorder=1,
        )

        f = f[i, idx_r]

        ax.plot(rho_i, np.exp(f), label="logGP-fit", zorder=2, color=color)

        noise = dy * sig_scale
        if add_noise:
            K0 = self.Kernel(rho_vac[idx_valid], rho_vac[idx_valid], reff_scale * 0.2)
            K1 = self.Kernel(rho_vac, rho_vac[idx_valid], reff_scale * 0.2)
            noise_fit = K1 @ np.linalg.solve(K0 + sig_scale**2 * np.eye(K0.shape[0]), noise[idx_valid])

            ax.fill_between(
                rho_i,
                np.exp(f) - noise_fit,
                np.exp(f) + noise_fit,
                alpha=0.1,
                label="noise range",
                zorder=0,
                color="grey",
            )
            ax.plot(
                rho_i,
                np.exp(f) + noise_fit,
                label=r"$\pm$noise",
                alpha=0.7,
                color="black",
                linestyle="--",
                linewidth=1.0,
                zorder=0,
            )
            ax.plot(
                rho_i,
                np.exp(f) - noise_fit,
                alpha=0.7,
                color="black",
                linestyle="--",
                linewidth=1.0,
                zorder=0,
            )

        x_plot = np.linspace(rho_i.min(), rho_i.max(), 200)
        y_min = 0

        y_max = np.exp(f + sigma).max() * 1.3
        y_plot = np.linspace(y_min + 0.01, y_max, 100)

        plotLogGP(
            x_plot=x_plot,
            y_plot=y_plot,
            x_inp=rho_i,
            f_inp=f,
            K_inp=Kpos_i,
            ax=ax,
            color=color,
            alpha_max=0.5,
            n_sample=sampling,
        )

        ax.axvline(0, linestyle="--", color="gray")

        ax.set_xlabel("reff")
        ax.set_ylabel(ValName + " [" + ValUnit + "]")

        # insert time and shotNo in upper left of the plot with white background without boxedge
        ax.text(
            0.05,
            0.95,
            "{:.3f}s ".format(self.time_inp[i]) + "#" + str(self.shotNo),
            transform=ax.transAxes,
            backgroundcolor="white",
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.7),
        )

        ax.set_ylim(y_min, y_max)
        ax.legend()

    def plotProfileDx(
        self: "ThomsonGPCore",
        ValName: str = None,
        DimName: str = None,
        time: float = None,
        i: int = None,
        ax=None,
        color: str = "red",
        sampling: int = 0,
    ):

        if ValName == "Te":
            if DimName == "reff":
                f = self.dlogTe_dr
                label = r"$\frac{\partial}{ \partial R }{\ln{T_{e}}}$ ・・・1m"
            elif DimName == "time":
                f = self.dlogTe_dt
                label = r"$\frac{\partial}{ \partial t }{\ln{T_{e}}}$ ・・・1s"
            else:
                raise NameError('DimName must be "reff" or "time"')
        elif ValName == "ne":
            if DimName == "reff":
                f = self.dlogNe_dr
                label = r"$\frac{\partial}{ \partial R }{\ln{N_{e}}}$ ・・・1m"
            elif DimName == "time":
                f = self.dlogNe_dt
                label = r"$\frac{\partial}{ \partial t }{\ln{N_{e}}}$ ・・・1s"
            else:
                raise NameError('DimName must be "reff" or "time"')
        else:
            raise NameError('ValName must be "Ne" or "Te"')

        if time is not None:
            i = np.argmin(abs(self.time_inp - time))
        elif i is None:
            raise ValueError("time or i must be specified")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        timei = self.time_inp[i]

        idx_r = ~self.idx_reff_outlier

        # reff = self.rho_origin[i,idx_r]
        reff = self.rho_vac[idx_r]
        f = f[i, idx_r]

        if DimName == "reff":
            Kpos_i = self.localPosteriorDr(np.array([timei]), reff, ValName=ValName)
        elif DimName == "time":
            Kpos_i = self.localPosteriorDt(np.array([timei]), reff, ValName=ValName)

        sigma = np.sqrt(np.diag(Kpos_i))

        ax.plot(reff, f, label=label, color=color, linewidth=1.5)

        x_plot = np.linspace(reff.min(), reff.max(), 200)
        y_min = -10
        y_max = 10
        y_plot = np.linspace(y_min, y_max, 100)

        plotGP(
            x_plot=x_plot,
            y_plot=y_plot,
            x_inp=reff,
            f_inp=f,
            K_inp=Kpos_i,
            ax=ax,
            color=color,
            alpha_max=0.5,
            n_sample=sampling,
        )

        ax.grid(True)
        ax.set_ylim(y_min, y_max)

        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

    def plotTrace(
        self: "ThomsonGPCore",
        ValName: str = None,
        reff: float = None,
        i: int = None,
        ax=None,
        color: str = "red",
        add_noise: bool = False,
        interpolation: bool = False,
        sampling: int = 0,
        is_label: bool = True,  
    ):
        ValName = self._valname(ValName)

        if ValName == "Te":
            y = self.Te_inp
            dy = self.dTe_data
            f = self.logTe_fit
            sig_scale = self.sigma_scale_Te
            ylabel = "Te [eV]"
        elif ValName == "ne":
            y = self.ne_inp
            dy = abs(self.dne_data)
            f = self.logNe_fit
            sig_scale = self.sigma_scale_ne
            ylabel = "Ne [10^19 m^-3]"
        else:
            raise NameError('ValName must be "Ne" or "Te"')

        if reff is not None:
            i = np.argmin(abs(self.rho_vac - reff))
        elif i is None:
            raise ValueError("reff or i must be specified")

        reffi = self.rho_vac[i]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        y = y[:, i]
        dy = dy[:, i]

        idx_valid = ~self.is_outlier_Te[:, i]

        Kpos_i = self.localPosterior(self.time_inp, reffi, ValName=ValName)
        sigma = np.sqrt(np.diag(Kpos_i))

        # valid data
        ax.errorbar(
            self.time_inp[idx_valid],
            y[idx_valid],
            yerr=dy[idx_valid],
            capsize=3,
            fmt="D",
            label="valid data" if is_label else None,
            zorder=1,
            markersize=4,
            color="C0",
        )

        # eliminated data
        ax.errorbar(
            self.time_inp[~idx_valid],
            y[~idx_valid],
            yerr=dy[~idx_valid],
            capsize=3,
            fmt="D",
            alpha=0.5,
            label="eliminated data" if is_label else None,
            zorder=1,
            markersize=4,
            color="C1",
        )

        f = f[:, i]

        ax.plot(self.time_inp, np.exp(f), 
                label="logGP-fit" if is_label else None
                , zorder=2, color=color)

        noise = dy * sig_scale

        if add_noise:
            K0 = self.Kernel(
                self.time_inp[idx_valid],
                self.time_inp[idx_valid],
                self.time_scale_Te * 0.3,
            )
            K1 = self.Kernel(self.time_inp, self.time_inp[idx_valid], self.time_scale_Te * 0.3)
            noise_fit = K1 @ np.linalg.solve(K0 + sig_scale**2 * np.eye(K0.shape[0]), noise[idx_valid])

            ax.fill_between(
                self.time_inp,
                np.exp(f) - noise_fit,
                np.exp(f) + noise_fit,
                alpha=0.1,
                label="noise range",
                zorder=0,
                color="gray",
            )
            ax.plot(
                self.time_inp,
                np.exp(f) + noise_fit,
                label=r"$\pm$noise",
                alpha=0.7,
                color="black",
                linestyle="--",
                linewidth=1.0,
                zorder=0,
            )
            ax.plot(
                self.time_inp,
                np.exp(f) - noise_fit,
                alpha=0.7,
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=0,
            )

        x_plot = np.linspace(self.time_inp.min(), self.time_inp.max(), 200)
        y_min = 0
        y_max = np.exp(f + sigma).max() * 1.3
        y_plot = np.linspace(y_min + 0.01, y_max, 100)

        plotLogGP(
            x_plot=x_plot,
            y_plot=y_plot,
            x_inp=self.time_inp,
            f_inp=f,
            K_inp=Kpos_i,
            ax=ax,
            color=color,
            alpha_max=0.5,
            n_sample=sampling,
            is_label=is_label,
        )

        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)

        ax.set_title("reff = {:.3f} m".format(reffi))
        ax.set_ylim(y_min, y_max)
        ax.legend()

    def plotEvolutionDx(
        self: "ThomsonGPCore",
        ValName: str = None,
        DimName: str = None,
        reff: float = None,
        i: int = None,
        ax=None,
        color: str = "red",
        sampling: int = 0,
    ):
        
        ValName = self._valname(ValName)
        if ValName == "Te":
            if DimName == "reff":
                f = self.dlogTe_dr
                label = r"$\frac{\partial}{ \partial R }{\ln{T_{e}}}$ ・・・1m"
            elif DimName == "time":
                f = self.dlogTe_dt
                label = r"$\frac{\partial}{ \partial t }{\ln{T_{e}}}$ ・・・1s"
            else:
                raise NameError('DimName must be "reff" or "time"')
        elif ValName == "ne":
            if DimName == "reff":
                f = self.dlogNe_dr
                label = r"$\frac{\partial}{ \partial R }{\ln{N_{e}}}$ ・・・1m"
            elif DimName == "time":
                f = self.dlogNe_dt
                label = r"$\frac{\partial}{ \partial t }{\ln{N_{e}}}$ ・・・1s"
            else:
                raise NameError('DimName must be "reff" or "time"')
        else:
            raise NameError('ValName must be "Ne" or "Te"')

        if reff is not None:
            i = np.argmin(abs(self.rho_vac - reff))
        elif i is None:
            raise ValueError("reff or i must be specified")

        reffi = self.rho_vac[i]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        y = f[:, i]

        if DimName == "reff":
            Kpos_i = self.localPosteriorDr(self.time_inp, reffi, ValName=ValName)
        elif DimName == "time":
            Kpos_i = self.localPosteriorDt(self.time_inp, reffi, ValName=ValName)

        sigma = np.sqrt(np.diag(Kpos_i))

        ax.plot(self.time_inp, y, label=label, color=color, linewidth=1.5)

        x_plot = np.linspace(self.time_inp.min(), self.time_inp.max(), 200)
        y_min = -10
        y_max = 10
        y_plot = np.linspace(y_min, y_max, 100)

        plotGP(
            x_plot=x_plot,
            y_plot=y_plot,
            x_inp=self.time_inp,
            f_inp=y,
            K_inp=Kpos_i,
            ax=ax,
            color=color,
            alpha_max=0.5,
            n_sample=sampling,
        )

        ax.axhline(0, linestyle="--", color="gray")

    # wrap  plotProfile in the case of Ne
    def plotNeProfile(
        self: "ThomsonGPCore",
        time: float = None,
        time_idx: int = None,
        ax=None,
        color="dodgerblue",
        add_noise: bool = False,
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None

        Examples:
            >>> fig,ax = plt.subplots()
            >>> plotNeProfile(time=0.1,ax=ax)
        """
        self.plotProfile(
            ValName="ne",
            time=time,
            i=time_idx,
            ax=ax,
            color=color,
            sampling=sampling,
            add_noise=add_noise,
        )

    def plotTeProfile(
        self: "ThomsonGPCore",
        time: float = None,
        time_idx: int = None,
        ax=None,
        color="red",
        add_noise: bool = False,
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot
            add_noise: add noise to the plot

        Returns:
            None
        """
        self.plotProfile(
            ValName="Te",
            time=time,
            i=time_idx,
            ax=ax,
            color=color,
            sampling=sampling,
            add_noise=add_noise,
        )

    def plotNeProfileDr(
        self: "ThomsonGPCore",
        time: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne with respect to t.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotProfileDx(
            ValName="ne",
            DimName="reff",
            time=time,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotTeProfileDr(
        self: "ThomsonGPCore",
        time: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne with respect to t.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotProfileDx(
            ValName="Te",
            DimName="reff",
            time=time,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotNeProfileDt(
        self: "ThomsonGPCore",
        time: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne with respect to t.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotProfileDx(
            ValName="ne",
            DimName="time",
            time=time,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotTeProfileDt(
        self: "ThomsonGPCore",
        time: float = None,
        time_idx: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the profile of Ne with respect to t.

        Parameters:
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotProfileDx(
            ValName="Te",
            DimName="time",
            time=time,
            i=time_idx,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotNeEvolution(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        add_noise: bool = False,
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Ne.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolution(
            ValName="ne",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
            add_noise=add_noise,
        )

    def plotTeEvolution(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="red",
        add_noise: bool = False,
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Te.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolution(
            ValName="Te",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
            add_noise=add_noise,
        )

    def plotNeEvolutionDr(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Ne with respect to reff.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolutionDx(
            ValName="ne",
            DimName="reff",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotTeEvolutionDr(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Ne with respect to reff.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolutionDx(
            ValName="Te",
            DimName="reff",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotNeEvolutionDt(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Ne with respect to time.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolutionDx(
            ValName="ne",
            DimName="time",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )

    def plotTeEvolutionDt(
        self: "ThomsonGPCore",
        reff: float = None,
        i: int = None,
        ax=None,
        color="dodgerblue",
        sampling: int = 0,
    ):
        """
        Plot the time evolution of Ne with respect to time.

        Parameters:
            reff: reff to plot
            i: index of reff to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot

        Returns:
            None
        """
        self.plotEvolutionDx(
            ValName="Te",
            DimName="time",
            reff=reff,
            i=i,
            ax=ax,
            color=color,
            sampling=sampling,
        )


__all__ = ["ThomsonGPVisualizer"]
