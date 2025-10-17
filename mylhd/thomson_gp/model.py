"""High-level Thomson GP interface combining core computations and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from .core import ThomsonGPCore
from .utils import ThomsonGPUtils
from .visualize import ThomsonGPVisualizer


class ThomsonGP(ThomsonGPCore, ThomsonGPVisualizer, ThomsonGPUtils):
    """Facade class exposing both computational and visualization capabilities."""

    pass
@dataclass
class Phase(ThomsonGP):
    """Class to hold phase information for MultiPhase processing."""

    time_start: float
    time_end: float
    time_independence: bool = False

    @classmethod
    def generate_for_MultiPhase(cls, base_obj: ThomsonGPCore, 
                                time_start: float, time_end: float, time_independence: bool = False) -> Phase:
        """Generate a new instance for MultiPhase processing."""

        cls_obj = cls(time_end=time_end, time_start=time_start, time_independence=time_independence)

        cls_obj.time_independence = base_obj.time_independence
        cls_obj.optimized = base_obj.optimized

        cls_obj.shotNo = base_obj.shotNo
        cls_obj.rho_origin = base_obj.rho_origin
        cls_obj.realR_origin = base_obj.realR_origin
        cls_obj.rho_vac = base_obj.rho_vac
        cls_obj.Te_origin = base_obj.Te_origin
        cls_obj.dTe_origin = base_obj.dTe_origin
        cls_obj.ne_origin = base_obj.ne_origin
        cls_obj.dne_origin = base_obj.dne_origin
        cls_obj.origin_shape = base_obj.origin_shape
        cls_obj.time_origin = base_obj.time_origin
        cls_obj.dt = base_obj.dt
        cls_obj.idx_reff_outlier = base_obj.idx_reff_outlier
        cls_obj.start_indx_base = base_obj.start_indx_base
        cls_obj.end_indx_base = base_obj.end_indx_base
        cls_obj.sigma_ne = base_obj.sigma_ne
        cls_obj.sigma_Te = base_obj.sigma_Te

        cls_obj.is_outlier_ne_all = base_obj.is_outlier_ne_all
        cls_obj.is_outlier_Te_all = base_obj.is_outlier_Te_all

        #cls_obj.set_time(time_start=time_start, time_end=time_end,  plot=False)
        cls_obj.set_time()
        cls_obj.set_kernel_type("Matern52")

        #del cls_obj.rho_origin
        #del cls_obj.realR_origin
        #del cls_obj.rho_vac
        del cls_obj.Te_origin
        del cls_obj.dTe_origin
        del cls_obj.ne_origin
        del cls_obj.dne_origin
        del cls_obj.origin_shape
        #del cls_obj.time_origin
        #del cls_obj.dt
        #del cls_obj.idx_reff_outlier
        del cls_obj.start_indx_base
        del cls_obj.end_indx_base

        return cls_obj

    def set_time(self,
            iprint=True):
        
        if self.time_start is None:
            self.time_start = self.time_origin[self.start_indx_base]
        if self.time_end is None:
            self.time_end = self.time_origin[self.end_indx_base-1] +1e-3 

        index = np.argmin(abs(self.time_origin - self.time_start)) 
        if self.time_origin[index]+1e-3 < self.time_start:
            self.itime_start = index + 1
        else:
            self.itime_start = index

        index = np.argmin(abs(self.time_origin - self.time_end))
        if self.time_origin[index]-1e-3 > self.time_end:
            self.itime_end = index
        else:
            self.itime_end = index + 1



        self.is_outlier_ne = self.is_outlier_ne_all[self.itime_start:self.itime_end, :]
        self.is_outlier_Te = self.is_outlier_Te_all[self.itime_start:self.itime_end, :]
        self.time_inp = self.time_origin[self.itime_start:self.itime_end]
        self.rho_inp = self.rho_origin[self.itime_start:self.itime_end, :]

        self.inp_shape = self.rho_inp.shape

        self.Te_inp = self.Te_origin[self.itime_start:self.itime_end, :]
        self.dTe_data = self.dTe_origin[self.itime_start:self.itime_end, :]
        self.ne_inp = self.ne_origin[self.itime_start:self.itime_end, :]
        self.dne_data = self.dne_origin[self.itime_start:self.itime_end, :]

        self.sigma_ne = self.sigma_ne[self.itime_start-self.start_indx_base:self.itime_end-self.start_indx_base, :]
        self.sigma_Te = self.sigma_Te[self.itime_start-self.start_indx_base:self.itime_end-self.start_indx_base, :]

        temp = self.Te_inp[~self.is_outlier_Te]
        self.out_scale_Te = np.log(temp[temp > 1e-5]).std()
        temp = self.ne_inp[~self.is_outlier_ne]
        self.out_scale_ne = np.log(temp[temp > 1e-5]).std()

    def plot_time_series(self, iprint=True):
        """Plot time series data."""
        pass


class ThomsonGPMultiPhase:
    """Class to handle multi-phase Thomson GP analysis."""

    def __init__(self, shotoNo: int) -> None:
        self.shotoNo = shotoNo
        self.base: ThomsonGP = ThomsonGP(shotoNo)
        self.Phases: list[Phase] = []

    def add_phase(self, time_start: float, time_end: float) -> None:

        phase = Phase.generate_for_MultiPhase(base_obj=self.base, time_start=time_start, time_end=time_end)
        self.Phases.append(phase)

    def plot_phases(self, )-> None:
        """Plot the evolution of a specified value across all phases."""


        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        vmax = np.percentile(self.base.Te_inp, 90) * 1.5
        axs[0].pcolormesh(self.base.realR_origin, self.base.time_origin, self.base.Te_origin, vmax=vmax, vmin=0)
        mask = np.ones_like(self.base.Te_origin)
        mask[~self.base.is_outlier_Te_all] = np.nan

        axs[0].pcolormesh(
            self.base.realR_origin,
            self.base.time_origin,
            mask,
            vmax=1,
            vmin=0,
            cmap="gray_r",
            alpha=0.45,
        )
        ax = axs[0]

        ax.set_xlabel("R [m]")
        ax.set_ylabel("time [s]")
        ax.set_title("Te")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2_index = [i for i in range(0, len(self.base.realR_origin), 20)]
        ax2.set_xticks(self.base.realR_origin[ax2_index])
        ax2.set_xticklabels([str(i) for i in ax2_index], fontsize=8)
        ax2.set_xlabel("index")

        ax3 = ax.twinx()
        ax3.set_ylim(ax.get_ylim()) 
        ax3_index = [i for i in range(0, len(self.base.time_origin), 50)]
        ax3.set_yticks(self.base.time_origin[ax3_index])
        ax3.set_yticklabels([str(i) for i in ax3_index], fontsize=8)
        ax3.set_ylabel("index")

        vmax = np.percentile(self.base.ne_origin, 90) * 1.5
        axs[1].pcolormesh(self.base.realR_origin, self.base.time_origin, self.base.ne_origin, vmax=vmax, vmin=0)
        mask = np.ones_like(self.base.ne_origin)
        mask[~self.base.is_outlier_ne_all] = np.nan
        axs[1].pcolormesh(
            self.base.realR_origin,
            self.base.time_origin,
            mask,
            vmax=1,
            vmin=0,
            cmap="gray_r",
            alpha=0.45,
        )

        for phase in self.Phases:
            phase.add_timeline(ax=axs[0], color="white", x=2.6)
            phase.add_timeline(ax=axs[1], color="white", x=2.6)

    def plot_im_ax(self,
            ax: plt.Axes = None,
            ValName: str = "Te",
            funckind: str = 'f',
            cmap = None,
            x_axis: str = 'rho',
            vmin: float = None,
            vmax: float = None) -> Tuple[plt.Figure, plt.Axes]:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 8))
        else:
            fig = ax.get_figure()


        ax_kwargs = dict(ValName=ValName, 
                         funckind=funckind, 
                         x_axis=x_axis, 
                         ax=ax, 
                         vmin=vmin, vmax=vmax, 
                         cmap=cmap)
        
        for phase in self.Phases:
            phase.plot_im_ax(**ax_kwargs, colorbar=True if phase==self.Phases[0] else False)

        y_max = max([phase.time_end for phase in self.Phases])    
        y_min = min([phase.time_start for phase in self.Phases])
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('time [s]')
        ax.grid(True)
        return fig,ax
    
    def plotProfile(self,
            ValName: str = None,
            time: float = None,
            ax=None,
            color: str = "red",
            sampling: int = 0,
            add_noise: bool = False,
            ) -> None:
        """Plot the evolution of a specified value across all phases."""

        if time is None:
            raise ValueError("time must be specified")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()        

        for phase in self.Phases:
            if phase.time_start < time <= phase.time_end:
                phase.plotProfile(ValName=ValName, time=time, ax=ax, sampling=sampling, add_noise=add_noise, color=color)
                break
        return fig,ax
    
    def plotTrace(self,
            ValName: str = None,
            reff: float = None,
            ax=None,
            color: str = "red",
            add_noise: bool = False,
            interpolation: bool = False,
            sampling: int = 0,
            is_label: bool = True,  
        ) -> None:
        """Plot the evolution of a specified value across all phases."""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()        

        kwargs = dict(ValName=ValName,reff = reff,ax=ax , sampling=sampling, add_noise=add_noise, color=color,
                      interpolation=interpolation)

        for phase in self.Phases:
            is_label_i = is_label if phase == self.Phases[0] else False
            phase.plotTrace(**kwargs, is_label=is_label_i)

        return fig, ax
    
    def localPostMeanFunction(
        self,
        ValName: str = None,
        function_kind: str = "f",
        rho_vac_local: np.ndarray = None,
        time_local: np.ndarray = None,
    ):
        if type(time_local) is not np.ndarray:
            time_local = np.array([time_local])
        if type(rho_vac_local) is not np.ndarray:
            rho_vac_local = np.array([rho_vac_local])

        func = np.zeros((time_local.size, rho_vac_local.size))
        for phase in self.Phases:
            index = (time_local >= phase.time_start) & (time_local <= phase.time_end)

            func[index, :] = phase.local_post_mean_function(
                ValName=ValName,
                function_kind=function_kind,
                rho_vac_l=rho_vac_local,
                timel=time_local[index],
            )
        return func
    
    
    def export_nTprofile(
        self,
        dir_path: str,
        Zeff: np.ndarray | float = 1,
        N_points: int = 101,
    ) -> None:
        
        for phase in self.Phases:
            phase.export_nTprofile(dir_path=dir_path, Zeff=Zeff, N_points=N_points)




        

            


__all__ = ["ThomsonGP", "ThomsonGPMultiPhase", "Phase"]
