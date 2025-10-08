#from nifs.voltdata import *
#from nifs.rawdata import *
#from nifs.timedata import *

#from nifs.anadata import *
from anadata import *
import matplotlib.pyplot as plt #グラフ描画ライブラリの読み込み
import matplotlib.gridspec as gridspec
import matplotlib.style as mplstyle
import numpy as np
import matplotlib as mpl
import scipy.signal as spsg
import sys,time,gc
from scipy.signal import medfilt
from scipy.signal import find_peaks
from typing import Callable
import warnings


import mpl_toolkits.axes_grid1

from scipy.optimize import Bounds,minimize

def axs_cbar(
    ax:plt.Axes,
    im,
    cbar_title =None,
    **kwargs
    ):

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical',label=cbar_title)


# 計測のデータの情報をプロットする関数
def ShowAnadataInfo(anadata):
    print('Diagnostics:',anadata.getDiagnostics())
    print('SubNo:',anadata.getSubNo())
    print('ShotNo:',anadata.getShotNo())
    print('DimNo:',anadata.getDimNo())

    for i in range(anadata.getDimNo()):
        print('DimName:',i,anadata.getDimName(i), anadata.getDimUnit(i), anadata.getDimSize(i))
    print('ValNo:',anadata.getValNo())
    for i in range(anadata.getValNo()):
        print('ValName:',i,anadata.getValName(i), anadata.getValUnit(i))
    print('')

class callback_counter:
    def __init__(self):
        self.iter = 0
    def __call__(self, xk):
        self.iter += 1




# low rank positibe define matrix
from typing import Any


class LowRankPDM:
    def __init__(self, matrix:np.ndarray, rank:int=None,cutoff=1e-5,stabilization=1e-5,iprint=False,identity=False):
        """
        与えられた正定値行列を低ランク近似するクラス。
        Parameters:
            matrix (ndarray): 正定値行列。
            rank (int): 保持する固有値の数。
            cutoff (flaot): 固有値のカットオフ値。固有値の総和にしてcutoff未満になる場合、カットする。
            stabilization: 行列を安定化するためのパラメータ
            identity: 行列が単位行列かどうか
        """
        self.rank = rank
        self.stabilization = stabilization
        self.identity = identity
        self.N = matrix.shape[0]
        self.cutoff = cutoff

        if self.identity:

            self.Lambda = np.ones(self.N)  
            self.V      = self.Lambda[:,np.newaxis] 

            if iprint: print('identity')
            return

        # 固有値分解
        eigvals, eigvecs = np.linalg.eigh(matrix)

        # 固有値を降順に並べ替え


        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        self.Lambda = eigvals
        self.V = eigvecs

        # 指定したランクまでの固有値・固有ベクトルを選択
        if rank is not None:
            self.Lambdal = eigvals[:rank]
            self.Vl = eigvecs[:,:rank]
            self.not_lowrank = False
        
        else:
            eignval_rate = np.cumsum(eigvals[::-1])[::-1] / np.sum(eigvals)
            cutoff_index = eignval_rate > cutoff 

            if np.sum(cutoff_index) > np.int(self.N/2):
                self.not_lowrank = True
                self.Lambdal = self.Lambda
                self.Vl = self.V
                self.matrix = matrix

            else:
                self.Lambdal = eigvals[cutoff_index]
                self.Vl = eigvecs[:, cutoff_index]
                self.not_lowrank = False

        if iprint: print(self.Vl.shape)

    # 低ランク近似された行列を返す
    def __call__(self, ) -> np.ndarray:
        """
        低ランク近似された行列を返します。
        """ 
        if self.identity:
            return np.eye(self.N)
        elif self.not_lowrank:
            return self.matrix
        else:
            return self.Vl @ ( self.Lambdal[:,np.newaxis] * self.Vl.T ) + np.eye(self.Vl.shape[0]) *self.stabilization



    def __matmul__(self, other)->np.ndarray:
        """
        他のベクトルや行列との右からの積演算を行います。
        単位行列ならそのままの配列を返す。

        Parameters:
            other (ndarray): 乗算対象のベクトルや行列。

        Returns:
            ndarray: 積の結果。
        """
        # V @ Λ @ V.T @ other
        # otherの次元数を取得し，それ以外ならエラーを出す
        if self.identity: return other
        try: 
            dim = len(other.shape)
        except: 
            return print('other must be ndarray') 
        if self.not_lowrank:
            return self.matrix @ other
        else:
            if dim == 1:
                return self.Vl @ (self.Lambdal * (self.Vl.T @ other)) + self.stabilization*other
            elif dim == 2:
                return self.Vl @ (self.Lambdal[:,np.newaxis] * (self.Vl.T @ other))  + self.stabilization*other

    def __lmatmul__(self, other):
        """
        他のベクトルや行列との左からの積演算を行います。

        Parameters:
            other (ndarray): 乗算対象のベクトルや行列。

        Returns:
            ndarray: 積の結果。
        """
        # otherの次元数を取得し，それ以外ならエラーを出す
        if self.identity: return other

        try: 
            dim = len(other.shape)
        except: 
            return print('other must be ndarray') 

        # other @ V @ Λ @ V.T
        if self.not_lowrank:
            return other @ self.matrix
        else:
            if dim == 1:
                return ((other @ self.Vl)* self.Lambdal) @ self.Vl.T
            elif dim == 2:
                return ((other @ self.Vl) @ self.Lambdal) @ self.Vl.T
            
    def inverse(self,
            M:np.ndarray,
            sig_sq: float=0,
            )->np.ndarray:
        """
        逆行列を計算する.
        単位行列ならそのままの配列を返す。

        Parameters:
    
        Return:
          self^-1 @ M
        """
        if self.identity: return M

        try: 
            dim = len(M.shape)
        except: 
            return print('M must be ndarray') 
        
        Lambda_inv = 1/(self.Lambda + self.stabilization + sig_sq)
        
        if dim == 1:
            return self.V @ (Lambda_inv * (self.V.T @ M))
        elif dim == 2:
            return self.V @ (Lambda_inv[:,np.newaxis] * (self.V.T @ M))    
        
class Logger(object):
    def __init__(self):
        import sys
        self.terminal = sys.stdout
        self.log = ""
    def write(self, message):
        self.terminal.write(message)
        # 一つの文字列として保管する
        self.log += message

    def flush(self):
        pass

     



class ThomsonGP:
    # jsonファイルからパラメータを読み込むクラスメソッド
    @classmethod
    def load_from_json(cls, filename: str):
        import json
        with open (filename, 'r') as f:
            params = json.load(f)

        cls_obj = cls(shotNo = params['shotNo'])
        cls_obj.optimized = params['optimized']

        # クラスオブジェクトの変数をparamから代入
        for key, value in params.items():
            setattr(cls_obj, key, value)

        cls_obj.mainNe(time_scale=cls_obj.time_scale_Ne,reff_scale=cls_obj.rho_scale_Ne,sigma_scale=cls_obj.sigma_scale_ne,iprint=False)
        cls_obj.mainTe(time_scale=cls_obj.time_scale_Te,reff_scale=cls_obj.rho_scale_Te,sigma_scale=cls_obj.sigma_scale_Te,iprint=False)

        cls_obj.filename = filename
        return cls_obj


    def __init__(self, 
                 shotNo: int=190600,filename :str = None, from_opendata:bool = False,iprint:bool = True):
    
        
        self.shotNo = shotNo
                
        import socket
        hostname = socket.gethostname()
        hostname[:6] == "egcalc"

        try:
            if hostname[:6] == "egcalc":    
                #thomsom    = KaisekiData.retrieve(diag='thomson',shotno=shotNo)

                if from_opendata:
                    self.data = KaisekiData.retrieve_opendata(diag='tsmap_calib',shotno=shotNo)
                else: 
                    self.data = KaisekiData.retrieve(diag='tsmap_calib',shotno=shotNo)

            else:
                #thomsom    = KaisekiData.retrieve_opendata(diag='thomson',shotno=shotNo)
                self.data = KaisekiData.retrieve_opendata(diag='tsmap_calib',shotno=shotNo)
        except:
            self.stop()

        self.data  = self.data

        #Cycleの情報を取得
        for i in self.data.comment.split('\n'):
            if 'Cycle' in i:
                #'Cycle = 25.0000'という文字列から25の整数を取り出す
                temp = i.split('=')[1]
                temp = temp.split('\r')[0]
                self.cycle = int(float(temp))

        #date = self.data.date
        #date = date.split(' ')[0]
        #date = date.split('/')
        #date = date[2]+date[0]+date[1]

        #self.date = date
        
        self.time_independence :bool = False
        self.optimized :bool = False
        self.Te_namelist = ['te', 'temperature', 'electron temperature', 'temperature of electron']
        self.ne_namelist = ['ne', 'density', 'electron density', 'density of electron']

        #self.data2 = tsmap_calib
        #ShowAnadataInfo(thomsom)
        #ShowAnadataInfo(tsmap_reff)


        self.rho_origin = self.data.get_val_data('reff/a99')
        self.realR_origin = self.data.get_dim_data('R')
        rho_vac  = self.rho_origin[0,:]
        # reff のハズレ値となる場合のインデックス
        self.idx_reff_outlier = (abs(rho_vac) > 1.1)
        self.rho_vac = rho_vac


        self.Te_origin  = self.data.get_val_data('Te')
        self.dTe_origin = abs(self.data.get_val_data('dTe'))
        self.ne_origin  = self.data.get_val_data('ne_calFIR')
        self.dne_origin = abs(self.data.get_val_data('dne_calFIR'))

        end_indx = self.check_data()
        
        self.origin_shape = self.Te_origin.shape

        if iprint: print('original data shape is' ,self.Te_origin.shape)

        self.Te_unit:str = 'keV'
        self.Ne_unit:str = '10^19 m^-3'

        self.time_origin = self.data.get_dim_data('Time')
        self.dt = np.diff(self.time_origin[1:]).mean()
        # tsmap_calibでは Time[0]のときはt = 0 と設定されているため、連続性を保つためにTime[0]をTime[1]-dtにする
        self.time_origin[0] = self.time_origin[1] - self.dt

        self.check_data(nt_limit=200)

        self.set_inp_data(start_idx=self.start_indx,end_indx=self.end_indx,iprint=iprint)
        self.set_kernel_type('Matern52',iprint=iprint)

    def stop(self,):
        raise Exception('shot ' + str(self.shotNo) + ' does not have data')
    
    def check_data(self,ne_th:float=0.2 , Te_th:float=0.4,nt_limit = None 
                   ):  
        
        ne_temp =self.ne_origin.copy()
        ne_temp[self.dne_origin>2] = 0
        Te_temp =self.Te_origin.copy()
        Te_temp[self.dTe_origin>2] = 0

        temp =  (np.median(ne_temp, axis=1)> ne_th ) + (np.median(Te_temp, axis=1)> Te_th )

        
        if np.sum(temp)< 5:
            self.stop()

        for i in range(0, len(temp)):
            if temp[i] == False:
                temp[i] = temp[i-1] * np.any(temp[i:i+10]) 
        #if temp[5]:
        #    temp[:5] = True

        self.plasma_exist = temp

        self.start_indx = 0
        self.end_indx = len(temp)
        alreadystart = False
        for i in range(0, len(temp)):

            if temp[i] == True and not alreadystart:
                self.start_indx = i
                alreadystart = True

            if temp[i] == False and alreadystart:
                self.end_indx = i
                
                break 
        
        if nt_limit is not None:
            if (self.end_indx - self.start_indx) > nt_limit:
                self.end_indx = self.start_indx + nt_limit
        
    
        

    def set_kernel_type(self,KernelType:str='KSE',iprint=True):
        self.KernelType = None

        KernelType = KernelType.lower() # 大文字小文字の区別をなくす

        if KernelType in  ['kse' , 'squared exponential' , 'gaussian' , 'k_se' , 'rbf' , 'se'] :
            self.Kernel     : Callable[[np.ndarray, np.ndarray, float], np.ndarray] =  KSE
            self.Kernel_dx  : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = KSE_dx
            self.Kernel_dxdx : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = KSE_dxdx
            self.KernelType = 'Squared Exponential'
        
        elif KernelType in ['krq' , 'rational quadratic', 'k_rq']:
            self.Kernel     : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = KRQ
            self.Kernel_dx  : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = KRQ_dx
            #self.Kernel_dxdx:Callable = KRQ_dxdx
            self.KernelType = 'Rational Quadratic'
        
        elif KernelType in ['matern3/2', 'k_matern32', 'matern32', 'matern 3/2' ]: 
            self.Kernel      : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmatern32
            self.Kernel_dx   : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmatern32_dx
            self.Kernel_dxdx : Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmatern32_dxdx
            self.KernelType = 'Matern 3/2'
            
        elif KernelType in ['matern5/2' , 'k_matern52', 'matern52', 'matern 5/2']:
            self.Kernel      :Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmartner52
            self.Kernel_dx   :Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmartner52_dx
            self.Kernel_dxdx :Callable[[np.ndarray, np.ndarray, float], np.ndarray] = Kmartner52_dxdx
            self.KernelType = 'Matern 5/2'
        else :
            raise ValueError('KernelType must be "KSE" or "KRQ" or "Matern32" or "Matern52"')
            
        
        if iprint:
            print('KernelType is set to',self.KernelType)
        
    def set_kernel(self,ValName:str=None, KernelType:str = None, reff_scale:float = 0.1,time_scale:float = 0.1, time_cutoff:float = 1e-8, reff_cutoff:float = 1e-5):

        ValName = ValName.lower()

        if KernelType is not None:
            self.set_kernel_type(KernelType)

        Ktt_low = LowRankPDM(self.Kernel(self.time_inp,self.time_inp,len=time_scale)          ,cutoff=time_cutoff,iprint=False)
        Krr_low = LowRankPDM(self.Kernel(self.rho_vac,self.rho_vac,len=reff_scale),cutoff=reff_cutoff,iprint=False)
        
        if ValName in self.Te_namelist:
            self.time_scale_Te = time_scale
            self.rho_scale_Te = reff_scale
            self.Ktt_low_Te = Ktt_low
            self.Krr_low_Te = Krr_low

        elif ValName in self.ne_namelist:
            self.time_scale_Ne = time_scale
            self.rho_scale_Ne = reff_scale
            self.Ktt_low_Ne = Ktt_low
            self.Krr_low_Ne = Krr_low

        else:
            raise NameError('ValName must be '+str(self.Te_namelist)+' or '+str(self.ne_namelist))
        
        
    def _valname(self,ValName:str)->str:
        """
        ValName が 'Te' か 'ne' かを判定する関数


        Parameters:
            ValName: 'Te' か 'ne' の文字列

        Returns:
            ValName: 'Te' か 'ne' の文字列
        """

        ValName = ValName.lower()
        if ValName in self.Te_namelist:
            ValName = 'Te'
        elif ValName in self.ne_namelist:
            ValName = 'ne' 
        else:
            raise NameError('ValName must be '+str(self.Te_namelist)+' or '+str(self.ne_namelist))
        
        return ValName

    def set_inp_data(self,start_idx: int= None,end_indx: int = None,iprint=True):
        """
        Set the input data for the fitting.

        Parameters:
            None

        Returns:
            None
        """
        time_inp = self.time_origin.copy()[start_idx:end_indx]
        Te_inp   = self.Te_origin.copy()[  start_idx:end_indx]
        ne_inp   = self.ne_origin.copy()[  start_idx:end_indx]
        sigma_Te = self.dTe_origin.copy()[ start_idx:end_indx]
        sigma_ne = self.dne_origin.copy()[ start_idx:end_indx]

        Te_median = medfilt(Te_inp, kernel_size=3)
        ne_median = medfilt(ne_inp, kernel_size=3)

        nt, nr = Te_inp.shape
    
        # ハズレ値となるデータのインデックスを定義
        is_outlier_Te = (sigma_Te > 3*Te_inp) + (abs(Te_median-Te_inp) > 2.5*sigma_Te)
        is_outlier_Te[:,self.idx_reff_outlier] = True

        #self.idx_outlier が行ごとに8割以上Trueの場合はすべてTrueにする
        is_outtime_Te    = is_outlier_Te.sum(axis=1) > 0.8*nr
        is_outlier_Te[is_outtime_Te,:] = True
        is_outlier_Te[sigma_Te < 0] = True

        is_outlier_ne = (sigma_ne > 3*ne_inp) + (abs(ne_median-ne_inp) > 3.5*sigma_ne)
        is_outlier_ne[:,self.idx_reff_outlier] = True
        
        is_outtime_ne = is_outlier_ne.sum(axis=1) > 0.6*nr
        is_outlier_ne[is_outtime_ne,:] = True
        is_outlier_ne[sigma_ne < 0] = True

        
        if (start_idx is None) or (start_idx == 0):
            is_outtime_Te[0] = True
            is_outtime_ne[0] = True
            is_outlier_Te[0,:] =True # t=0のときは必ず外れ値とする
            is_outlier_ne[0,:] =True

        
        # sigma の設定
        sigma_Te[is_outlier_Te ]  = 1000
        sigma_Te[sigma_Te<0.01 ]  = 0.01
        sigma_ne[is_outtime_Te,:] = 50
        

        sigma_ne[is_outlier_ne]   = 1000
        sigma_ne[sigma_ne<0.01]   = 0.01
        sigma_ne[is_outtime_ne,:] = 50
        
        Te_inp[is_outtime_Te,:] = 0
        ne_inp[is_outtime_ne,:] = 0


        temp = Te_median[~is_outlier_Te]
        self.out_scale_Te = np.log(temp[temp>1e-5]).std()

        temp = ne_median[~is_outlier_ne]
        self.out_scale_ne = np.log(temp[temp>1e-5]).std()

        if iprint: print('out_scale_Te:',str(self.out_scale_Te)[:5],'out_scale_Ne:',str(self.out_scale_ne)[:5])
        
        #self.Te_median = Te_median
        self.Te_inp    = Te_inp
        self.dTe_data  = self.dTe_origin[ start_idx:end_indx]

        #self.ne_median = ne_median
        self.ne_inp    = ne_inp
        self.dne_data  = self.dTe_origin[ start_idx:end_indx]

        self.sigma_Te = sigma_Te
        self.sigma_ne = sigma_ne

        self.inp_shape = Te_inp.shape
        self.time_inp = time_inp

        self.is_outlier_Te = is_outlier_Te
        self.is_outlier_ne = is_outlier_ne

        self.is_outlier_Te_all = np.ones(self.origin_shape,dtype=bool)
        self.is_outlier_Te_all[start_idx:end_indx,:] = is_outlier_Te[:,:]
        self.is_outlier_ne_all = np.ones(self.origin_shape,dtype=bool)
        self.is_outlier_ne_all[start_idx:end_indx,:] = is_outlier_ne[:,:]

        self.rho_inp = self.rho_origin[ start_idx:end_indx]

    def set_time(self,time_start=None, time_end=None,include_end = True, time_separate:list=[],time_independence:bool=False):
        """
        Set the time range for the fitting.

        Parameters:
            time_start: start time of the fitting
            time_end: end time of the fitting
            time_separate: list of time points to separate the fitting
            time_dependence: whether the fitting is time-dependent or not

        Returns:
            None
        """
        
        if time_independence:
            print('Time_dependence is enabled.')
            self.time_independence = True

        if time_start is None:
            #self.itime_start = 0
            self.itime_start = self.start_indx
        else: 
            self.itime_start = np.argmin(abs(self.time_origin-time_start))

        if time_end is None:
            #self.itime_end = len(self.time_origin)-1
            self.itime_end = self.end_indx - 1
        else:
            self.itime_end = np.argmin(abs(self.time_origin-time_end))

        time_start = self.time_origin[self.itime_start]
        time_end = self.time_origin[self.itime_end]


        self.set_inp_data(start_idx=self.itime_start,end_indx=self.itime_end+1)

        self.i_time_sep = []
        for time in time_separate:
            #time が time startとtime endの間にない場合はエラーを出す
            if (time < time_start) or (time > time_end):
                #print('separate time = '+str(time) +'s must be between time_start and time_end')
                raise ValueError('separate time = '+str(time) +'s must be between time_start and time_end')

            itime = np.argmin(abs(self.time_inp-time))
            print(itime)
            self.i_time_sep.append(itime)

        fig,axs = plt.subplots(1,2,figsize=(8,10))
        vmax = np.percentile(self.Te_inp,90)*1.5
        axs[0].pcolormesh(self.realR_origin,self.time_origin,self.Te_origin,vmax=vmax,vmin=0)
        mask = np.ones_like(self.Te_origin)
        mask[~self.is_outlier_Te_all] = np.nan
        
        axs[0].pcolormesh(self.realR_origin,self.time_origin,mask,vmax=1,vmin=0,cmap='gray_r',alpha=0.45)
        ax = axs[0]

        ax.set_xlabel('R [m]')
        ax.set_ylabel('time [s]')
        ax.set_title('Te')

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2_index =  [i for i in range(0,len(self.realR_origin),20)]
        ax2.set_xticks(self.realR_origin[ax2_index])
        ax2.set_xticklabels([str(i) for i in ax2_index],fontsize=8)
        ax2.set_xlabel('index')

        
        ax3 = ax.twinx()
        ax3.set_ylim(ax.get_ylim())
        ax3_index =  [i for i in range(0,len(self.time_inp),50) ]
        ax3.set_yticks(self.time_inp[ax3_index])
        ax3.set_yticklabels([str(i) for i in ax3_index],fontsize=8)
        ax3.set_ylabel('index')


        vmax = np.percentile(self.ne_inp,90)*1.5
        axs[1].pcolormesh(self.realR_origin,self.time_origin,self.ne_origin,vmax=vmax,vmin=0)
        mask = np.ones_like(self.ne_origin)
        mask[~self.is_outlier_ne_all] = np.nan
        axs[1].pcolormesh(self.realR_origin,self.time_origin,mask,vmax=1,vmin=0,cmap='gray_r',alpha=0.45)

        self.add_timeline(axs[0],x=2.6,color='red')
        self.add_timeline(axs[1],x=2.6)
    

    def add_timeline(self,ax:plt.Axes,
                     x=0.5,color='white',
                     istext: bool = True,
                     fontsize= 12):
        
        y = self.time_origin[self.itime_start]-0.5*self.dt
        ax.axhline(y,color=color,lw=1)
        if istext:

            #小数点以下4桁まで表示
            text = 'start: '+str(self.time_origin[self.itime_start])[:6]+'s'+', i=' +str(self.itime_start)
            ax.text(x,y+0.05,text,color=color,fontsize=fontsize)
        #上向きの矢印を追加
        ax.annotate('', xy=(x-0.1, y), xytext=(x-0.1, y+0.2), arrowprops=dict(arrowstyle='<-', color=color, lw=1.5))
        
        y = self.time_origin[self.itime_end]+0.5*self.dt
        ax.axhline(y,color=color,lw=1)
        if istext:
            text = 'end  : '+str(self.time_origin[self.itime_end])[:6]+'s'+', i=' +str(self.itime_end)
            ax.text(x,y-0.11,text,color=color,fontsize=fontsize)
        
        ax.annotate('', xy=(x-0.1, y), xytext=(x-0.1, y-0.2), arrowprops=dict(arrowstyle='<-', color=color, lw=1.5))

        for i,itime in enumerate(self.i_time_sep):
            y= self.time_inp[itime]-0.5*self.dt
            ax.axhline(y,color=color,lw=1)
            if istext:
                text = 'time'+str(i+1)+': '+str(self.time_inp[itime])[:6]+'s'+', i=' +str(itime)
                ax.text(x,y+0.05,text,color=color,fontsize=fontsize)
            ax.annotate('', xy=(x-0.1, y), xytext=(x-0.1, y+0.2), arrowprops=dict(arrowstyle='<-', color=color, lw=1.5))



        


    def mainTe(self,
            time_scale: float = 0.15,
            reff_scale: float = 0.15,
            sigma_scale: float = 1,
            mu: float = -2,
            reff_cutoff: float = 1e-5,
            time_cutoff: float = 1e-8,
            plot_im: bool = True,
            iprint: bool = True,
            a_init: np.ndarray = None,
            omega_line : float = 0.0,
            ):
        """
        main function of Te fitting.

        Parameters:
            time_scale: scale legnth of time
            reff_scale: scale length of reff
            sigma_scale:
            mu: means of posterior
            reff_cutoff:
            time_cutoff:
            plot_im: plot image or not

        """
        self.main_core(ValName='Te',
                          time_scale=time_scale,
                          rho_scale=reff_scale,
                          sigma_scale=sigma_scale,
                          mu=mu,
                          reff_cutoff=reff_cutoff,
                          time_cutoff=time_cutoff,
                          plot_im=plot_im,
                          iprint=iprint,
                          a_init=a_init,
                          omega=omega_line,
                          )
        
    def mainNe(self,
            time_scale: float = 0.15,
            reff_scale: float = 0.15,
            sigma_scale: float = 1,
            mu: float = -2,
            reff_cutoff: float = 1e-5,
            time_cutoff: float = 1e-8,
            plot_im: bool = True,
            iprint: bool = True,
            a_init: np.ndarray = None,
            omega_line : float = 0.,
            ):
        """
        main function of Te fitting.

        Parameters:
            time_scale: scale legnth of time
            reff_scale: scale length of reff
            sigma_scale:
            mu: means of posterior
            reff_cutoff:
            time_cutoff:
            plot_im: plot image or not
            omega_line: wight of the line

        """
        self.main_core(ValName='ne',
                          time_scale=time_scale,
                          rho_scale=reff_scale,
                          sigma_scale=sigma_scale,
                          mu=mu,
                          reff_cutoff=reff_cutoff,
                          time_cutoff=time_cutoff,
                          plot_im=plot_im,
                          iprint=iprint,
                          a_init=a_init,
                          omega=omega_line,
                          )


    def main_core(self,
            ValName: str = 'Te',
            time_scale: float = 0.15,
            rho_scale: float = 0.1,
            sigma_scale: float = 1,
            mu: float = -2,
            reff_cutoff: float = 1e-5,
            time_cutoff: float = 1e-8,
            plot_im: bool = True,
            iprint: bool = True,
            a_init: np.ndarray = None,
            optimize: bool | float = False,
            omega : float = 0.01,
            ):
        """
        main function of Te or Ne fitting.

        Parameters:
            ValName: 'ne' or 'Te'
            time_scale: scale legnth of time
            reff_scale: scale length of reff
            sigma_scale: 
            mu: means of posterior
            reff_cutoff: 
            time_cutoff: 
            plot_im: plot image or not

        """

        ValName = self._valname(ValName)


        if self.KernelType == 'Matern 3/2':
            time_scale = np.sqrt(3)*time_scale
            rho_scale = np.sqrt(3)*rho_scale
        elif self.KernelType == 'Matern 5/2':
            time_scale = np.sqrt(5/3)*time_scale
            rho_scale = np.sqrt(5/3)*rho_scale

        if ValName == 'Te':    
            self.time_scale_Te  = time_scale
            self.rho_scale_Te  = rho_scale
            self.sigma_scale_Te = sigma_scale
            self.mu_Te = mu
            out_scale=self.out_scale_Te
            y = self.Te_inp
            sigma = self.sigma_Te
            omega = self.out_scale_Te*omega

        elif ValName == 'ne':
            self.time_scale_Ne  = time_scale
            self.rho_scale_Ne  = rho_scale
            self.sigma_scale_ne = sigma_scale
            self.mu_Ne = mu
            out_scale=self.out_scale_ne
            y = self.ne_inp
            sigma = self.sigma_ne
            omega = self.out_scale_ne*omega

        # optimize が True　あるは数値が代入されている場合は最適化を行う
        if not optimize == False: 
            if type(optimize) == float:
                reff_cutoff = optimize
                time_cutoff = optimize
            else:
                reff_cutoff = 1e-5
                time_cutoff = 1e-5

        rho = self.rho_vac
        shape = self.inp_shape   
        times = self.time_inp

        Krr    = self.Kernel(rho,rho,rho_scale)
        Krr_dr = self.Kernel_dx(rho,rho,rho_scale)

        Krr_low = LowRankPDM(Krr,cutoff=reff_cutoff) 
        sigma_inv = 1 / (sigma*sigma_scale)
        sigma2_inv = sigma_inv**2
        nt = times.size
        
        if self.time_independence:
            Ktt_dt  = numerical_differentiation_matrix(times)
            Ktt_low = LowRankPDM(np.eye(nt), identity=True)
        else:
            Ktt     = self.Kernel(times,times,time_scale)
            Ktt_dt  = self.Kernel_dx(times,times,time_scale)
            Ktt_low = LowRankPDM(Ktt,cutoff=time_cutoff)


        from scipy.sparse.linalg import LinearOperator,cg
        from scipy.optimize import minimize


        if a_init is not None:
            if type(a_init) != np.ndarray:
                a = a_init*np.ones(shape)
            else:
                a = a_init
        else:
            a = 0*np.ones(shape)

        for i in range(100):
            #f = K a + muf 
            f_line = omega**2 * np.broadcast_to(a.sum(axis=0),(a.shape[0],a.shape[1]))
            f = out_scale**2*(Krr_low @(Ktt_low @ a).T).T + mu + f_line

            f[f>50] = 50
            f[f<-50] = -50

            expf = np.exp(f)
            W    = expf * sigma2_inv * (2*expf - y)

            W = np.abs(W)
            W[W<1e-10] = 1e-10 

            W_sqrt = np.sqrt(W) 
            W_sqrt_inv = 1/W_sqrt
            b = W_sqrt**2 *(f-mu) - expf * sigma2_inv* (expf-y)

            def B_kroneker(x):
                x =  x.reshape(*shape)
                A_line = np.broadcast_to((W_sqrt*x).sum(axis=0),(x.shape[0],x.shape[1]))
                A = x+  W_sqrt*(out_scale**2 *(Krr_low @(Ktt_low @(W_sqrt*x) ).T).T + omega**2 * A_line)
                return A.ravel()
            
            B = LinearOperator((y.size,y.size),matvec=B_kroneker)
            
            counter = callback_counter() 

            z,exit_code = cg(A=B,b=(W_sqrt_inv*b).flatten(),x0=W_sqrt_inv.flatten()*a.flatten(),callback=counter,maxiter=10000)

            if counter.iter == 10000:
                print(' failed greedy iteration '+str(ValName),end=' ')   
                raise RuntimeError('cg did not converge') 

            delta_a =   W_sqrt*z.reshape(*shape) - a 


            def Psi(xi):
                a_proposed = a + xi*delta_a
                temp = out_scale**2*(Krr_low @(Ktt_low @ a_proposed).T).T
                temp_line = omega**2 * np.broadcast_to(a_proposed.sum(axis=0),(a_proposed.shape[0],a_proposed.shape[1]))
                temp = temp + temp_line 
                f = temp+ mu
                first =  (a_proposed* temp ).sum()
                second = ((np.exp(f)-y)**2 * sigma2_inv).sum()
                return 0.5*first + 0.5*second

            
            res = minimize(Psi,x0=0.5,method="Nelder-Mead",options={'disp':False})

            xi = res.x[0]

            a += xi *delta_a 
            #print(i,xi,counter.iter,res.nit,delta_a.std())


            if iprint :print(i,xi,counter.iter,res.nit,delta_a.std())
            if delta_a.std() < 1e-4:
                #print(ValName+' converged')
                break

        f    = out_scale**2*(Krr_low @(Ktt_low @ a).T).T + mu

        f_line = omega**2 * np.broadcast_to(a.sum(axis=0),(a.shape[0],a.shape[1]))

        if not optimize == False:
            if self.time_independence == False: 
                V_very_low = np.kron(Ktt_low.Vl,Krr_low.Vl)
                Lambda_tr_very_low = np.kron(Ktt_low.Lambdal,Krr_low.Lambdal)

                if ValName == 'Te':
                    self.V_very_low_Te = V_very_low
                    self.Lambda_tr_very_low_Te = Lambda_tr_very_low
                elif ValName == 'ne':
                    self.V_very_low_ne = V_very_low
                    self.Lambda_tr_very_low_ne = Lambda_tr_very_low
            

            if ValName == 'Te':
                self.logTe_fit = f
                self.logTe_vline = f_line
                self.a_Te = a
                self.W_Te = W
            
            elif ValName == 'ne':
                self.logNe_fit = f
                self.logNe_vline = f_line
                self.a_ne = a
                self.W_ne = W

            return 


        f_dr = out_scale**2*(Krr_dr  @(Ktt_low @ a).T).T 
        f_dt = out_scale**2*(Krr_low @(Ktt_dt  @ a).T).T    

        #####
    
        
        if ValName == 'Te':
            self.Krr_low_Te = Krr_low
            self.Ktt_low_Te = Ktt_low

            self.logTe_fit = f
            self.logTe_vline = f_line
            self.dlogTe_dr = f_dr
            self.dlogTe_dt = f_dt

            self.W_Te = W
            self.a_Te = a


        elif ValName == 'ne':
            self.Krr_low_Ne = Krr_low
            self.Ktt_low_Ne = Ktt_low

            self.logNe_fit = f
            self.logNe_vline = f_line
            self.dlogNe_dr = f_dr
            self.dlogNe_dt = f_dt

            self.W_ne = W
            self.a_ne = a
            
        
        if plot_im:
            self.plot_im(parameter=ValName,is_const_reff=True,log=True)
        
        self.postProcess(ValName=ValName)

    def SigScale(self,
            valName:str=None):
        
        valName = self._valname(valName) 

        if valName == 'Te':
            temp = (self.Te_inp - np.exp(self.logTe_fit+self.logTe_vline))/self.sigma_Te
            return  temp[~self.is_outlier_Te].std()
        
        elif valName == 'ne':
            temp = (self.ne_inp - np.exp(self.logNe_fit+self.logNe_vline))/self.sigma_ne
            return  temp[~self.is_outlier_ne].std()
    
    def plot_im_ax(self,
            name :str= None,
            ax: plt.Axes=None,
            x_axis:str = 'reff',
            rho_contour= True,
            vmax= None,
            vmin= None,
            cmap = None,
            colorbar = True,
            ):
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(4,8))

        name = name.lower()
        is_diff = False
        
        if name == 'te':
            f = self.Te_inp
            title = r'$T_e$_data' + r' $\mathrm{[keV]}$'
        elif name == 'ne':
            f = self.ne_inp
            title = r'$n_e$_data' + r' $\mathrm{[10^{19}m^{-3}]}$'
        elif name == 'te_fit':
            f = np.exp(self.logTe_fit)
            title = r'$T_e$_fit' + r' $\mathrm{[keV]}$'
        elif name == 'ne_fit': 
            f = np.exp(self.logNe_fit)
            title = r'$n_e$_fit' + r' $\mathrm{[10^{19}m^{-3}]}$'
        elif name == 'dlogte/dr':
            f = self.dlogTe_dr
            title = r'$d\log(T_e)/d\rho_v$'+ r' $[ ]$'
            is_diff = True
        elif name == 'dte/dr':
            f = self.dlogTe_dr * np.exp(self.logTe_fit)
            title = r'$dT_e/d\rho_v$'+ r' $[\mathrm{keV}]$'
            is_diff = True
        elif name == 'dlogte/dt':
            f = self.dlogTe_dt
            title = r'$d\log(T_e)/dt$' + r' $[\mathrm{1/s}]$'
            is_diff = True
        elif name == 'dte/dt':
            f = self.dlogTe_dt * np.exp(self.logTe_fit)
            title = r'$dT_e/dt$'+ r' $[\mathrm{keV/s}]$'
            is_diff = True
        elif name == 'dlogne/dr':
            f = self.dlogNe_dr
            title = r'$d\log{n_e}/d\rho$' + r' $[ ]$' 
            is_diff = True
        elif name == 'dne/dr':
            f = self.dlogNe_dr * np.exp(self.logNe_fit)
            title = r'$d{n_e}/d\rho_v$'+ r' $[\mathrm{10^{19}m^{-3}}]$'
            is_diff = True
        elif name == 'dlogne/dt':
            f = self.dlogNe_dt
            title = r'$d\log{n_e}/dt$' + r' $[\mathrm{1/s}]$'
            is_diff = True
        elif name == 'dne/dt':
            f = self.dlogNe_dt * np.exp(self.logNe_fit)
            title = r'$d(n_e)/dt$' + r' $[\mathrm{10^{19}m^{-3}/s}]$'
            is_diff = True
        else:
            raise NameError('name must be "Te" or "ne" or "ne_fit" or"Te_fit" or "dlogTe/dr" or "dTe/dr" or "dlogTe/dt" or "dTe/dt" or "dlogne/dr" or "dne/dr" or "dlogne/dt" or "dne/dt"')
    
        
        xlim_is_default =  (ax.get_xlim() == (0,1.0))

        if x_axis in ['rho_vac', 'reff/a99_vac']:
            X = self.rho_vac
            x_title = r'${r_\mathrm{eff}/a_{99}}$ (vacuum)'
            if xlim_is_default:
                ax.set_xlim(-1.05,1.05)
        elif x_axis in ['rho', 'reff/a99']:
            rho_contour =False 
            X = self.rho_inp
            x_title = r'${r_\mathrm{eff}/a_{99}}$'
            if xlim_is_default:
                ax.set_xlim(-1.05,1.05)        
        elif x_axis in ['R', 'real', 'realR']:
            X = self.realR_origin
            x_title = r'${R} $[m]'
        else:
            raise ValueError('x_axis must be "rho" or "rho_vac" or "R"')

        ax.set_xlabel(x_title)

        #pcolormesh で発生する警告を一時的に無視する

        if (vmax is None) and (vmin is None):
            if is_diff:
                vmax = np.percentile(np.abs(f),95)*1.1
                vmin = -vmax
            else:
                vmax = np.percentile(f,85)*1.2
                vmin = 0
        elif vmin is None:
            if is_diff:
                vmin = -vmax
            else:
                vmin = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.pcolormesh(X,self.time_inp,f,vmax=vmax,vmin=vmin,cmap=cmap)    
        ax.set_title(title)
        if colorbar: axs_cbar(ax,im)

        if rho_contour:
            ax.contour(X,self.time_inp,self.rho_inp,levels=np.linspace(-1.0,1.0,11),colors='black',linewidths=0.5)
            ax.contour(X,self.time_inp,self.rho_inp,levels=[0],colors='black',linewidths=1.)

        
    

        





    def plot_im(self,
                parameter: str='Te',
                x_axis:str = 'reff',
                log :bool =False,
                save_name: str = None,
                rho_contour= True,
                ):
        """
        Plot image of Te or Ne fitting.

        Parameters:
            parameter: 'Te' or 'ne',

            x_axis: 'rho' or 'rho_vac' or 'R' to specify which x-axis to use

            #is_const_reff: boolean value indicating whether to use constant reff

        """
        parameter = self._valname(parameter)

        if parameter == 'Te':
            data = self.Te_inp
            f = self.logTe_fit
            f_dr = self.dlogTe_dr
            f_dt = self.dlogTe_dt
            f_line = self.logTe_vline
            if log :
                axs_titles =[ 'Te_data [keV]','Te_fit'      ,'dlogTe/dr [/m] ', 'dlogTe/dt [/s]' ]
            else:
                axs_titles =[ 'Te_data [keV]','Te_fit [keV]','dTe/dr [keV/m]',       'dTe/dt [keV/s]' ]


        elif parameter == 'ne':
            data = self.ne_inp
            f = self.logNe_fit
            f_dr = self.dlogNe_dr
            f_dt = self.dlogNe_dt
            f_line = self.logNe_vline
            if log :
                axs_titles =[ 'ne_data [10^19 m^-3]','ne_fit'      ,'dlog(ne)/dr [/m] ', 'dlog(ne)/dt [/s]' ]
            else:
                axs_titles =[ 'ne_data [10^19 m^-3]','ne_fit [10^19 m^-3]','dne/dr [10^19/m]',       'dne/dt [10^19/s]' ]

        else:
            raise NameError('parameter must be "Ne" or "Te"')

        # Add your code here
        if x_axis in ['rho_vac', 'reff/a99_vac']:
            x_axis = 'rho_vac'
            X = self.rho_vac
            x_title = r'${r_\mathrm{eff}/a_{99}}$ (vacuum)'
        elif x_axis in ['rho', 'reff/a99']:
            x_axis = 'rho'
            X = self.rho_inp
            x_title = r'${r_\mathrm{eff}/a_{99}}$'
        elif x_axis == 'R':
            X = self.realR_origin
            x_title = r'${R}$'
        else:
            raise ValueError('x_axis must be "rho" or "rho_vac" or "R"')

        # pythonのバージョンに依存してfigsize を変更する
        if sys.version_info[1] > 6:
            figunit = 1.5
        else:
            figunit = 2.0
             

        times = self.time_inp
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['none', 'gray']) 
        
        fig,ax=plt.subplots(1,4,figsize=(figunit*10,figunit*5),sharey=True)

        im = ax[0].pcolormesh(self.realR_origin,times,data,vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
        axs_cbar(ax[0],im)
        ax[0].contour(self.realR_origin,times,self.rho_inp,levels=np.linspace(-0.7,0.7,15),colors='black',linewidths=0.5)
        ax[0].set_title(axs_titles[0])
        ax[0].grid()
        ax[0].set_xlabel('R [m]')    

        

        mask = (f < np.log(np.exp(np.percentile(f,95))*0.05 ) ) #+ self.idx_outlier



        im = ax[1].pcolormesh(X,times,np.exp(f),vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
        axs_cbar(ax[1],im)

        if log:
            im = ax[2].pcolormesh(X,times,f_dr,vmax=10,vmin=-10,cmap='seismic')
        else:
            im = ax[2].pcolormesh(X,times,np.exp(f)*f_dr,vmax=15,vmin=-15,cmap='seismic')
        axs_cbar(ax[2],im)

        ax[2].pcolormesh(X,times,mask,cmap=cmap)

        if log:
            im = ax[3].pcolormesh(X,times,f_dt,vmax=10,vmin=-10,cmap='seismic')
        else:
            im = ax[3].pcolormesh(X,times,np.exp(f)*f_dt,vmax=15,vmin=-15,cmap='seismic')
        axs_cbar(ax[3],im)

        ax[3].pcolormesh(X,times,mask,cmap=cmap)

        if x_axis in ['rho_vac','R']:
            for i in range(1,4):
                if rho_contour: ax[i].contour(X[:,self.idx_reff_outlier],times,self.rho_inp[:,self.idx_reff_outlier],levels=np.linspace(-1.0,1.0,11),colors='black',linewidths=0.5,alpha=0.5)

        for i in range(1,4):
            ax_i = ax[i]
            ax_i.set_title(axs_titles[i])
            ax_i.grid()
            ax_i.set_xlabel(x_title)
            if x_axis == 'R':
                pass
                #ax_i.set_xlim(3.0,4.5)
            else:
                ax_i.set_xlim(-1.05,1.05)

        


        ax[0].set_ylabel('time [s]')   
        if save_name is not None:
            # tight lauoutで保存する
            plt.tight_layout()
            plt.savefig(save_name)
            plt.close()
            


    def postProcess(self, ValName: str = None):
        """
        Calculate the posterior distribution.

        Parameters:
            parameter: 'ne', 'Te' to specify which parameter(s) to calculate the posterior for.

        Returns:
            None
        """
        import time 
        ValName = self._valname(ValName)

        if ValName == 'ne':
            self.set_kernel(ValName=ValName,time_scale=self.time_scale_Ne,reff_scale=self.rho_scale_Ne, time_cutoff=1e-5, reff_cutoff=1e-5)
            Ktt_low = self.Ktt_low_Ne
            Krr_low = self.Krr_low_Ne
            W2 = self.W_ne.copy()

            start = time.time()
            self.X_sqrtinv_Ne = self.calc_X_sqrtinv(Ktt_low, Krr_low, W2)
            print('Ne time:',time.time()-start)

        elif ValName == 'Te':    
            self.set_kernel(ValName=ValName,time_scale=self.time_scale_Te,reff_scale=self.rho_scale_Te, time_cutoff=1e-5, reff_cutoff=1e-5)
            Ktt_low = self.Ktt_low_Te
            Krr_low = self.Krr_low_Te
            W2 = self.W_Te.copy()

            start = time.time()
            self.X_sqrtinv_Te = self.calc_X_sqrtinv(Ktt_low, Krr_low, W2)
            print('Te time:',time.time()-start)

        """
        事後分布を計算する
        一般的に,
        K_pos = V @ (Lam - Lam^0.5 @(Lam + sig_sq * I )^-1 @Lam^0.5 ) @ V.T
        が成り立ち、転じて
        K_pos = V @ (Lam - Lam^0.5 @(Lam + (V.T @ W @V)^-1  )^-1 @Lam^0.5 ) @ V.T
        """

        """
                    
        sigmasq_pos = self.out_scale_Te**2*(1 - ((X_sqrt_inv @ lam_V)**2).sum(axis=0)).reshape(*shape)
        self.sigma_pos = np.sqrt(sigmasq_pos)

        
        reff = self.reff_inp
        times = self.time_inp
        fig,ax=plt.subplots(1,4,figsize=(20,10))


        f = np.exp(self.Te_fit)*self.sigma_pos

        vmax = self.sigma_pos[~self.is_outlier_Te].max()
        im = ax[0].pcolormesh(reff,times,f,vmax=f.max(),vmin=0,cmap='jet')
        axs_cbar(ax[0],im,cbar_title='sigma_dTe [eV]')
        ax[0].grid()
        ax[0].set_xlim(-0.65,0.65)

        f = self.sigma_pos
    
        im = ax[1].pcolormesh(reff,times,f,vmax=f.max(),vmin=0,cmap='jet')
        axs_cbar(ax[1],im,cbar_title='sigma_dTe [eV]')
        ax[1].grid()
        ax[1].set_xlim(-0.65,0.65)
        """

    def calc_X_sqrtinv(self, Ktt_low: LowRankPDM, Krr_low:LowRankPDM, W2):
        """
        Calculate the square root of the inverse of the matrix X.

        X = Λ + (V.T @ W @V)^-1 

        Parameters:
            Ktt_low: LowRankPDM object for time
            Krr_low: LowRankPDM object for reff
            W2: W^2

        Returns:
            X_sqrt_inv: The square root of the inverse of the matrix X
        """


        W2[W2 <1e-10] = 1e-10

        W2_sqrt = np.sqrt(W2)

        if self.time_independence:
            #A = W2_sqrt @ Krr_low.Vl
            A = np.einsum('ij,jk->ijk',W2_sqrt,Krr_low.Vl)

            print('hoge')

            X_sqrt_inv = np.empty((A.shape[0],A.shape[2],A.shape[2]))

            for i in range(A.shape[0]):
                X0_i = A[i].T @A[i] + 1e-5*np.eye(A.shape[2])
                X_i = np.linalg.inv(X0_i)+np.diag(Krr_low.Lambdal)

                lam_i, V_i = np.linalg.eigh(X_i)
                X_sqrt_inv[i] = np.sqrt(1/lam_i)[:,np.newaxis] * V_i.T

            return X_sqrt_inv  

        else:     
            #shape = self.inp_shape   
            V_low = np.kron(Ktt_low.Vl,Krr_low.Vl)
            Lambda_tr_low = np.kron(Ktt_low.Lambdal,Krr_low.Lambdal)

            A = (W2_sqrt.flatten()[:,np.newaxis] * V_low)

            #lam_V = Lambda_tr_low[:,np.newaxis] * V_low.T

            X0 = A.T @ A+1e-5*np.eye(A.shape[1])

            X = np.linalg.inv(X0)+np.diag(Lambda_tr_low)

            lam, V = np.linalg.eigh(X)

            X_sqrt_inv = np.sqrt(1/lam)[:,np.newaxis] * V.T

            return  X_sqrt_inv

    def marginalLogLikelihood(self, ValName: str = None):
        if ValName == 'Te':
            out_scale = self.out_scale_Te
            W2 = self.W_Te.copy()
            alpha = self.a_Te
            f = self.logTe_fit
            res_f = f - self.mu_Te

            sigma = self.sigma_Te
            sigma_scale = self.sigma_scale_Te
            res_y = np.exp(f)-self.Te_inp

            V = self.V_very_low_Te
            Lambda_tr = self.Lambda_tr_very_low_Te
    

        elif ValName == 'ne':
            out_scale = self.out_scale_ne
            W2 = self.W_ne.copy()
            alpha = self.a_ne
            f = self.logNe_fit

            sigma = self.sigma_ne
            sigma_scale = self.sigma_scale_ne
            res_f = f - self.mu_Ne
            res_y = np.exp(f)-self.ne_inp

            V = self.V_very_low_Ne
            Lambda_tr = self.Lambda_tr_very_low_Ne


        else:
            raise ValueError('parameter must be "Ne" or "Te"')
         
        from scipy.linalg import cholesky


        N = Lambda_tr.size

        #print('N:',N,Lambda_tr.shape)   


        W2[W2 <1e-10] = 1e-10
        W2_sqrt = np.sqrt(W2)
        A = (W2_sqrt.flatten()[:,np.newaxis] * V)


        X0 = A.T @ A+1e-5*np.eye(N)

        X1 = np.eye(N) + out_scale**2*(Lambda_tr[:,np.newaxis] * X0)
        self.X1 = X1
        X1 = (X1.T + X1)*0.5
        self.X2 = X1
        self.X0 = X0

        try:
            L_diag = cholesky(X1).diagonal()
            log_det_X1 = np.log(L_diag).sum()*2
            plt.plot(lam)

        except:
            lam = np.linalg.eigvalsh(X1)
            lam[lam<1e-10] = 1e-10
            log_det_X1 = np.log(lam).sum()


        log_f_Kinv_f = 1/out_scale**2 * (alpha*res_f).sum()

        log_p = -0.5/sigma_scale**2 * ((res_y/sigma)**2).sum() - (np.log(sigma*sigma_scale)).sum()

        mll = -0.5*log_det_X1 - 0.5*log_f_Kinv_f + log_p

        return mll
    
    
    def marginalLogLikelihood_rough(self,
            reff_cutoff=1e-4,
            time_cutoff=1e-4,
            exec_main = False,
            ValName: str = None):
        
        if ValName == 'Te':
            out_scale = self.out_scale_Te
            W2 = self.W_Te
            alpha = self.a_Te
            f = self.logTe_fit
            res_f = f - self.mu_Te

            sigma = self.sigma_Te
            sigma_scale = self.sigma_scale_Te
            res_y = np.exp(f)-self.Te_inp

            #V = self.V_very_low_Te
            Lambda_tr = self.Lambda_tr_very_low_Te
    
        elif ValName == 'ne':
            out_scale = self.out_scale_ne
            W2 = self.W_ne
            alpha = self.a_ne
            f = self.logNe_fit

            sigma = self.sigma_ne
            sigma_scale = self.sigma_scale_ne
            res_f = f - self.mu_Ne
            res_y = np.exp(f)-self.ne_inp

            #V = self.V_very_low_Ne
            Lambda_tr = self.Lambda_tr_very_low_Ne


        else:
            raise NameError('parameter must be "Ne" or "Te"')
         
        #from scipy.linalg import cholesky


        N = Lambda_tr.size

        #print('N:',N,Lambda_tr.shape)   

        W_sort = np.sort(W2.flatten())[::-1]

        W_sort = W_sort[:Lambda_tr.size]
        Lambda = np.sort(Lambda_tr)[::-1]

        log_det_X1 = np.log(1+Lambda*W_sort).sum()

        log_f_Kinv_f = 1/out_scale**2 * (alpha*res_f).sum()

        log_p = -0.5/sigma_scale**2 * ((res_y/sigma)**2).sum() - (np.log(sigma*sigma_scale)).sum()

        mll = -0.5*log_det_X1 - 0.5*log_f_Kinv_f + log_p

        return mll


    def local_post_mean_function(self, ValName: str = None,function_type:str='f', rho_vac_l: np.ndarray = None, timel: np.ndarray = None):
        """
        Calculate the local posterior mean.

        Parameters:
            parameter: 'ne', 'Te' to specify which parameter(s) to calculate the posterior for.
            function_type: 'f' or 'df/dr' or 'df/dt' to specify which function to calculate
            reffl: 1D array of reff values
            timel: 1D array of time values

        Returns:
            Kpos_l: local posterior mean
        """
        ValName = self._valname(ValName)

        if ValName == 'Te':
            out_scale = self.out_scale_Te
            alpha = self.a_Te
            rho_scale = self.rho_scale_Te
            time_scale = self.time_scale_Te
            mu = self.mu_Te

        elif ValName == 'ne':
            out_scale = self.out_scale_ne
            alpha = self.a_ne
            rho_scale = self.rho_scale_Ne
            time_scale = self.time_scale_Ne
            
            mu = self.mu_Ne

        if function_type == 'f':
            Kr_la = self.Kernel(rho_vac_l, self.rho_vac, rho_scale)
            Kt_la = self.Kernel(timel, self.time_inp, time_scale) 
            mu_fac = True
        elif function_type == 'df/dr':
            Kr_la = self.Kernel_dx(rho_vac_l, self.rho_vac, rho_scale)
            Kt_la = self.Kernel(timel, self.time_inp, time_scale)
            mu_fac = False
        elif function_type == 'df/dt':
            Kr_la = self.Kernel(rho_vac_l, self.rho_vac, rho_scale)
            Kt_la = self.Kernel_dx(timel, self.time_inp, time_scale)
            mu_fac = False


        func_pos = out_scale**2*(Kr_la @ (Kt_la @ alpha).T).T + mu*mu_fac
        

        return func_pos
                

        

        


    def localPosterior(self,
                       timel:np.ndarray,
                       reffl:np.ndarray,
                       ValName: str = 'Te'):
        
        if ValName == 'Te':
            X_sqrt_inv = self.X_sqrtinv_Te
            out_scale = self.out_scale_Te
            Ktt_low = self.Ktt_low_Te
            Krr_low = self.Krr_low_Te
            reff_scale = self.rho_scale_Te
            time_scale = self.time_scale_Te
        elif ValName == 'ne':
            X_sqrt_inv = self.X_sqrtinv_Ne
            out_scale = self.out_scale_ne
            Ktt_low = self.Ktt_low_Ne
            Krr_low = self.Krr_low_Ne
            reff_scale = self.rho_scale_Ne
            time_scale = self.time_scale_Ne
        else:
            raise NameError('parameter must be "Ne" or "Te"')
    
        # l means local
        # a means all
 
        Krr_la = self.Kernel(reffl, self.rho_vac,reff_scale)
        Ktt_la = self.Kernel(timel, self.time_inp,time_scale) 
        Krr_la_Vr = Krr_la @ Krr_low.Vl
        Ktt_la_Vt = Ktt_la @ Ktt_low.Vl
        kron_all  = np.kron(Ktt_la_Vt,Krr_la_Vr)
        
        Ktt_ll = self.Kernel(timel,timel,time_scale) 
        Krr_ll = self.Kernel(reffl,reffl,reff_scale)

        Ktt_rr_ll = np.kron(Ktt_ll,Krr_ll)
        
        A = X_sqrt_inv @ kron_all.T
        Kpos_l = out_scale**2 *(Ktt_rr_ll - A.T @ A) 

        return Kpos_l       
    
    def localPosterior_timeindependent(self,
            time_idx:int,
            reffl:np.ndarray,
            ValName: str):
        
        if ValName == 'Te':
            X_sqrt_inv = self.X_sqrtinv_Te
            out_scale = self.out_scale_Te
            Krr_low = self.Krr_low_Te
            reff_scale = self.rho_scale_Te
        elif ValName == 'ne':
            X_sqrt_inv = self.X_sqrtinv_Ne
            out_scale = self.out_scale_ne
            Krr_low = self.Krr_low_Ne
            reff_scale = self.rho_scale_Ne
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        pass
    
        Krr_la = self.Kernel(reffl, self.rho_vac,reff_scale)
        Krr_la_Vr = Krr_la @ Krr_low.Vl
        Krr_ll = self.Kernel(reffl,reffl,reff_scale)

        A = X_sqrt_inv[time_idx,:,:] @ Krr_la_Vr.T
        Kpos_l = out_scale**2 *(Krr_ll - A.T @ A) 
        return Kpos_l
                                       


    def localPosteriorDr(self,
                       timel:np.ndarray,
                       reffl:np.ndarray,
                          parameter: str):
        
        if parameter == 'Te':
            X_sqrt_inv = self.X_sqrtinv_Te
            out_scale = self.out_scale_Te
            Ktt_low = self.Ktt_low_Te
            Krr_low = self.Krr_low_Te
            reff_scale = self.rho_scale_Te
            time_scale = self.time_scale_Te
        elif parameter == 'ne':
            X_sqrt_inv = self.X_sqrtinv_Ne
            out_scale = self.out_scale_ne
            Ktt_low = self.Ktt_low_Ne
            Krr_low = self.Krr_low_Ne
            reff_scale = self.rho_scale_Ne
            time_scale = self.time_scale_Ne
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        
        
        # l means local
        # a means all
        # dr means derivative with respect to reff

        Krr_la = self.Kernel_dx(reffl, self.rho_vac,reff_scale)
        Ktt_la = self.Kernel(timel, self.time_inp,time_scale) 
        Krr_la_Vr = Krr_la @ Krr_low.Vl
        Ktt_la_Vt = Ktt_la @ Ktt_low.Vl
        kron_all  = np.kron(Ktt_la_Vt,Krr_la_Vr)
        
        Ktt_ll = self.Kernel(timel,timel,time_scale)
        Krr_ll = self.Kernel_dxdx(reffl,reffl,reff_scale)

        Ktt_rr_ll = np.kron(Ktt_ll,Krr_ll)
        
        A = X_sqrt_inv @ kron_all.T
        Kpos_l = out_scale**2 *(Ktt_rr_ll - A.T @ A) 

        self.K_pos_temp = Kpos_l

        return Kpos_l    
    
    
    def localPosteriorDt(self,
                       timel:np.ndarray,
                       reffl:np.ndarray,
                          parameter: str = 'Te'):   
        
        if parameter == 'Te':
            X_sqrt_inv = self.X_sqrtinv_Te
            out_scale = self.out_scale_Te
            Ktt_low = self.Ktt_low_Te
            Krr_low = self.Krr_low_Te
            reff_scale = self.rho_scale_Te
            time_scale = self.time_scale_Te
        elif parameter == 'ne':
            X_sqrt_inv = self.X_sqrtinv_Ne
            out_scale = self.out_scale_ne
            Ktt_low = self.Ktt_low_Ne
            Krr_low = self.Krr_low_Ne
            reff_scale = self.rho_scale_Ne
            time_scale = self.time_scale_Ne
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        
        
        # l means local
        # a means all
        # dt means derivative with respect to time

        Krr_la = self.Kernel(reffl, self.rho_vac,reff_scale)
        Ktt_la = self.Kernel_dx(timel, self.time_inp,time_scale)
        Krr_la_Vr = Krr_la @ Krr_low.Vl
        Ktt_la_Vt = Ktt_la @ Ktt_low.Vl
        kron_all  = np.kron(Ktt_la_Vt,Krr_la_Vr)

        Ktt_ll = self.Kernel_dxdx(timel,timel,time_scale)
        Krr_ll = self.Kernel(reffl,reffl,reff_scale)

        Ktt_rr_ll = np.kron(Ktt_ll,Krr_ll)

        A = X_sqrt_inv @ kron_all.T
        Kpos_l = out_scale**2 *(Ktt_rr_ll - A.T @ A)


        #Krr_la = KSE(reffl, self.reff_inp,self.reff_scale_Te)
        #Ktt_la = KSE_dx(timel, self.time_inp,self.time_scale_Te) 
        #Krr_la_Vr = Krr_la @ self.Krr_low_Te.Vl
        #Ktt_la_Vt = Ktt_la @ self.Ktt_low_Te.Vl
        #kron_all  = np.kron(Ktt_la_Vt,Krr_la_Vr)
        #
        #Ktt_ll = KSE_dxdx(timel,timel,self.time_scale_Te) 
        #Krr_ll = KSE(reffl,reffl,self.reff_scale_Te)
#
        #Ktt_rr_ll = np.kron(Ktt_ll,Krr_ll)
        #
        #A = self.X_sqrt_inv @ kron_all.T
        #Kpos_l = self.out_scale_Te**2 *(Ktt_rr_ll - A.T @ A) 
#
        return Kpos_l    

    def plotProfile(self,
            ValName: str = None,
            time: float = None,
            i: int = None,
            ax = None,
            color: str = 'red',
            sampling: int = 0,
            add_noise: bool = False,
            ):
        """
        Plot the profile of the parameter.

        Parameters:
            parameter: 'ne' or 'Te'
            time: time to plot
            i: index of time to plot
            ax: axis to plot on
            color: color of the plot
            sampling: number of samples to plot
            add_noise: add noise to the plot

        Returns:
            None
        """
        if ValName == 'Te':
            y = self.Te_inp
            dy = self.dTe_data
            sig_scale = self.sigma_scale_Te
            f = self.logTe_fit
            ValUnit = self.Te_unit
            reff_scale = self.rho_scale_Te
            
        elif ValName == 'ne':
            y = self.ne_inp
            sig_scale = self.sigma_scale_ne
            dy = abs(self.dne_data)
            f = self.logNe_fit
            ValUnit = self.Te_unit
            reff_scale = self.rho_scale_Ne
        else:
            raise NameError('parameter must be "Ne" or "Te"')
                        
        if time is not None:
            i = np.argmin(abs(self.time_inp-time))
        elif i is None:
            raise ValueError('time or i must be specified')
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,5))
        else:
            fig = ax.get_figure()

        timei = self.time_inp[i]

        idx_r = ~self.idx_reff_outlier

        rho_i   = self.rho_origin[i,idx_r]
        rho_vac = self.rho_vac[idx_r]
        y = y[i,idx_r]
        dy = dy[i,idx_r]

        if ValName == 'Te':
            idx_valid =  ~self.is_outlier_Te[i,idx_r]
        elif ValName == 'ne':
            idx_valid =  ~self.is_outlier_ne[i,idx_r]


        if self.time_independence:
            Kpos_i = self.localPosterior_timeindependent(i,rho_vac,ValName=ValName)
        else:
            Kpos_i = self.localPosterior(np.array([timei]),rho_vac,ValName=ValName)
        sigma = np.sqrt(np.diag(Kpos_i))

        #self.Kpos_i = Kpos_i

        # valid data
        ax.errorbar(rho_i[idx_valid], y[idx_valid],yerr=dy[idx_valid],capsize=3,fmt='D',label='valid data',zorder=1)

        ax.errorbar(rho_i[~idx_valid], y[~idx_valid],yerr=abs(dy[~idx_valid]),capsize=3,fmt='D',alpha=0.5,label='eliminated data',zorder=1)


        f = f[i,idx_r]

        ax.plot(rho_i,np.exp(f),label='logGP-fit',zorder=2,color=color)

        noise = dy*sig_scale
        if add_noise:
            K0 = self.Kernel(rho_vac[idx_valid],rho_vac[idx_valid],reff_scale*0.2)
            K1 = self.Kernel(rho_vac,           rho_vac[idx_valid],reff_scale*0.2)
            noise_fit = K1@ np.linalg.solve(K0+sig_scale**2*np.eye(K0.shape[0]),noise[idx_valid])

            ax.fill_between(rho_i,np.exp(f)-noise_fit,np.exp(f)+noise_fit,alpha=0.1,label='noise range',zorder=0,color='grey')
            ax.plot(rho_i,np.exp(f)+noise_fit,label=r'$\pm$noise',alpha=0.7,color='black',linestyle='--',linewidth=1.,zorder=0)
            ax.plot(rho_i,np.exp(f)-noise_fit,alpha=0.7,color='black',linestyle='--',linewidth=1.,zorder=0)


        x_plot = np.linspace(rho_i.min(),rho_i.max(),200)
        y_min = 0

        y_max = np.exp(f+sigma).max()*1.3
        y_plot = np.linspace(y_min+0.01,y_max,100)

        plotLogGP(x_plot=x_plot,y_plot=y_plot,
                  x_inp=rho_i   ,f_inp=f,K_inp=Kpos_i,
                  ax=ax,color=color,alpha_max=0.5,n_sample=sampling)


        ax.axvline(0,linestyle='--',color='gray')

        ax.set_xlabel('reff')
        ax.set_ylabel(ValName+' ['+ValUnit+']')
        
        # insert time and shotNo in upper left of the plot with white background without boxedge
        ax.text(0.05,0.95,'{:.3f}s '.format(self.time_inp[i])+'#'+str(self.shotNo),
                transform=ax.transAxes,backgroundcolor='white'
                ,bbox=dict(facecolor='white', edgecolor='white',alpha=0.7))

        ax.set_ylim(y_min,y_max)
        ax.legend()


    def plotProfileDx(self,
            ValName: str = None,
            DimName: str = None,
            time: float = None,
            i: int = None,
            ax = None,
            color: str = 'red',
            sampling: int = 0):
        
        if ValName == 'Te':
            if DimName == 'reff':
                f = self.dlogTe_dr
                label = r'$\frac{\partial}{ \partial R }{\ln{T_{e}}}$ × 1m'
            elif DimName == 'time':
                f = self.dlogTe_dt
                label = r'$\frac{\partial}{ \partial t }{\ln{T_{e}}}$ × 1s'
            else:
                raise NameError('DimName must be "reff" or "time"')
        elif ValName == 'ne':
            if DimName == 'reff':
                f = self.dlogNe_dr
                label =  r'$\frac{\partial}{ \partial R }{\ln{N_{e}}}$ × 1m'
            elif DimName == 'time':
                f = self.dlogNe_dt
                label = r'$\frac{\partial}{ \partial t }{\ln{N_{e}}}$ × 1s'
            else:
                raise NameError('DimName must be "reff" or "time"')
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        
        if time is not None:
            i = np.argmin(abs(self.time_inp-time))
        elif i is None:
            raise ValueError('time or i must be specified')
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,5))
        else:
            fig = ax.get_figure()

        timei = self.time_inp[i]

        idx_r = ~self.idx_reff_outlier

        #reff = self.rho_origin[i,idx_r]
        reff = self.rho_vac[idx_r]
        f = f[i,idx_r]

        if DimName == 'reff':
            Kpos_i = self.localPosteriorDr(np.array([timei]),reff,parameter=ValName)
        elif DimName == 'time':
            Kpos_i = self.localPosteriorDt(np.array([timei]),reff,parameter=ValName)

        sigma = np.sqrt(np.diag(Kpos_i))


        ax.plot(reff,f,label=label,color=color,linewidth=1.5)

        x_plot = np.linspace(reff.min(),reff.max(),200)
        y_min = -10
        y_max = 10
        y_plot = np.linspace(y_min,y_max,100)

        plotGP(x_plot=x_plot,y_plot=y_plot,
                x_inp=reff,f_inp=f,K_inp=Kpos_i,
                ax=ax,color=color,alpha_max=0.5,n_sample=sampling)

        ax.grid(True)
        ax.set_ylim(y_min,y_max)
        
        ax.axhline(0,linestyle='--',color='gray')
        ax.axvline(0,linestyle='--',color='gray')
    
    def plotEvolution(self,
            ValName: str = None,
            reff: float = None,
            i: int = None,
            ax = None,
            color: str = 'red',
            add_noise: bool = False,
            interpolation: bool=False,
            sampling: int = 0):
        
        if ValName == 'Te':
            y = self.Te_inp
            dy = self.dTe_data
            f = self.logTe_fit
            sig_scale = self.sigma_scale_Te
            ylabel = 'Te [eV]'
        elif ValName == 'ne':
            y = self.ne_inp
            dy = abs(self.dne_data)
            f = self.logNe_fit
            sig_scale = self.sigma_scale_ne
            ylabel = 'Ne [10^19 m^-3]'
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        
        if reff is not None:
            i = np.argmin(abs(self.rho_vac-reff))
        elif i is None:
            raise ValueError('reff or i must be specified')
        
        reffi = self.rho_vac[i]
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,5))
        else:
            fig = ax.get_figure()

        y = y[:,i]
        dy = dy[:,i]

        idx_valid =  ~self.is_outlier_Te[:,i]

        Kpos_i = self.localPosterior(self.time_inp,reffi,ValName=ValName)
        sigma = np.sqrt(np.diag(Kpos_i))

        # valid data
        ax.errorbar(self.time_inp[idx_valid], y[idx_valid],yerr=dy[idx_valid],capsize=3,fmt='D',label='valid data',zorder=1,markersize=4)

        # eliminated data
        ax.errorbar(self.time_inp[~idx_valid], y[~idx_valid],yerr=dy[~idx_valid],capsize=3,fmt='D',alpha=0.5,label='eliminated data',zorder=1,markersize=4)

        f = f[:,i]

        ax.plot(self.time_inp,np.exp(f),label='logGP-fit',zorder=2,color=color)

        noise = dy*sig_scale


        if add_noise:
            K0 = KSE(self.time_inp[idx_valid],self.time_inp[idx_valid],self.time_scale_Te*0.3)
            K1 = KSE(self.time_inp,self.time_inp[idx_valid],self.time_scale_Te*0.3)
            noise_fit = K1@ np.linalg.solve(K0+sig_scale**2*np.eye(K0.shape[0]),noise[idx_valid])

            ax.fill_between(self.time_inp,np.exp(f)-noise_fit,np.exp(f)+noise_fit,alpha=0.1,label='noise range',zorder=0,color='gray')
            ax.plot(self.time_inp,np.exp(f)+noise_fit,label=r'$\pm$noise',alpha=0.7,color='black',linestyle='--',linewidth=1.,zorder=0)
            ax.plot(self.time_inp,np.exp(f)-noise_fit,alpha=0.7,color='black',linestyle='--',linewidth=1,zorder=0)


            

        x_plot = np.linspace(self.time_inp.min(),self.time_inp.max(),200)
        y_min = 0
        y_max = np.exp(f+sigma).max()*1.3
        y_plot = np.linspace(y_min+0.01,y_max,100)
        
        plotLogGP(x_plot=x_plot,y_plot=y_plot,
                    x_inp=self.time_inp,f_inp=f,K_inp=Kpos_i,
                    ax=ax,color=color,alpha_max=0.5,n_sample=sampling)
        
        ax.set_xlabel('time [s]')
        ax.set_ylabel(ylabel)

        ax.set_title('reff = {:.3f} m'.format(reffi))
        ax.set_ylim(y_min,y_max)
        ax.legend()

    def plotEvolutionDx(self,
            ValName: str = None,
            DimName: str = None,
            reff: float = None, 
            i: int = None,
            ax = None,
            color: str = 'red',
            sampling: int = 0):

        if ValName == 'Te':
            if DimName == 'reff':
                f = self.dlogTe_dr
                label = r'$\frac{\partial}{ \partial R }{\ln{T_{e}}}$ × 1m'
            elif DimName == 'time':
                f = self.dlogTe_dt
                label = r'$\frac{\partial}{ \partial t }{\ln{T_{e}}}$ × 1s'
            else:
                raise NameError('DimName must be "reff" or "time"')
        elif ValName == 'ne':
            if DimName == 'reff':
                f = self.dlogNe_dr
                label =  r'$\frac{\partial}{ \partial R }{\ln{N_{e}}}$ × 1m'
            elif DimName == 'time':
                f = self.dlogNe_dt
                label = r'$\frac{\partial}{ \partial t }{\ln{N_{e}}}$ × 1s'
            else:
                raise NameError('DimName must be "reff" or "time"')
        else:
            raise NameError('parameter must be "Ne" or "Te"')
        
        if reff is not None:
            i = np.argmin(abs(self.rho_vac-reff))
        elif i is None:
            raise ValueError('reff or i must be specified')
        
        reffi = self.rho_vac[i]

        if ax is None:
            fig,ax = plt.subplots(figsize=(10,5))
        else:
            fig = ax.get_figure()

        y = f[:,i]

        if DimName == 'reff':
            Kpos_i = self.localPosteriorDr(self.time_inp,reffi,parameter=ValName)
        elif DimName == 'time':
            Kpos_i = self.localPosteriorDt(self.time_inp,reffi,parameter=ValName)

        sigma = np.sqrt(np.diag(Kpos_i))

        ax.plot(self.time_inp,y,label=label,color=color,linewidth=1.5)

        x_plot = np.linspace(self.time_inp.min(),self.time_inp.max(),200)
        y_min = -10
        y_max = 10
        y_plot = np.linspace(y_min,y_max,100)

        plotGP(x_plot=x_plot,y_plot=y_plot,
                x_inp=self.time_inp,f_inp=y,K_inp=Kpos_i,
                ax=ax,color=color,alpha_max=0.5,n_sample=sampling)


        ax.axhline(0,linestyle='--',color='gray')


        
    # wrap  plotProfile in the case of Ne 
    def plotNeProfile(self,
            time: float = None,
            time_idx: int = None,
            ax = None,
            color = 'dodgerblue',
            add_noise: bool = False,
            sampling: int = 0):
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
        self.plotProfile(ValName='ne',time=time,i=time_idx,ax=ax,color=color,sampling=sampling,add_noise=add_noise)
    
    def plotTeProfile(self,
            time: float = None,
            time_idx: int = None,
            ax = None,
            color = 'red',
            add_noise: bool = False,
            sampling: int = 0):
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
        self.plotProfile(ValName='Te',time=time,i=time_idx,ax=ax,color=color,sampling=sampling,add_noise=add_noise)

        
    def plotNeProfileDr(self,
            time: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotProfileDx(ValName='ne',DimName='reff',
                           time=time,i=i,ax=ax,color=color,sampling=sampling)
        
    def plotTeProfileDr(self,
            time: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotProfileDx(ValName='Te',DimName='reff',
                           time=time,i=i,ax=ax,color=color,sampling=sampling)

        
    def plotNeProfileDt(self,
            time: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotProfileDx(ValName='ne',DimName='time',
                           time=time,i=i,ax=ax,color=color,sampling=sampling)

    def plotTeProfileDt(self,
            time: float = None,
            time_idx: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotProfileDx(ValName='Te',DimName='time',
                           time=time,i=time_idx,ax=ax,color=color,sampling=sampling)

    def plotNeEvolution(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            add_noise: bool = False,
            sampling: int = 0):
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
        self.plotEvolution(ValName='ne',reff=reff,i=i,ax=ax,color=color,sampling=sampling,add_noise=add_noise)

    def plotTeEvolution(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'red',
            add_noise: bool = False,
            sampling: int = 0):
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
        self.plotEvolution(ValName='Te',reff=reff,i=i,ax=ax,color=color,sampling=sampling,add_noise=add_noise)
    
    def plotNeEvolutionDr(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotEvolutionDx(ValName='ne',DimName='reff',
                           reff=reff,i=i,ax=ax,color=color,sampling=sampling)
        
    def plotTeEvolutionDr(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotEvolutionDx(ValName='Te',DimName='reff',
                           reff=reff,i=i,ax=ax,color=color,sampling=sampling)
        
    def plotNeEvolutionDt(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotEvolutionDx(ValName='ne',DimName='time',
                           reff=reff,i=i,ax=ax,color=color,sampling=sampling)
        
    def plotTeEvolutionDt(self,
            reff: float = None,
            i: int = None,
            ax = None,
            color = 'dodgerblue',
            sampling: int = 0):
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
        self.plotEvolutionDx(ValName='Te',DimName='time',
                           reff=reff,i=i,ax=ax,color=color,sampling=sampling)
        
    def sigma_optimize2(self,
            valName:str,
            time_scale:float,
            reff_scale:float,
            mu :float = -2,
            sigscale_init:float = 1,
            convergence:float = 1e-2,
            optimize_only:bool = False,
            a_init:float = 0,
            omega:float = 0,
            iprint = False):
        
        valName = self._valname(valName)

        self.main_core(ValName=valName,
                    time_scale=time_scale, rho_scale=reff_scale, sigma_scale=sigscale_init,
                    time_cutoff=1e-5, reff_cutoff=1e-5, mu=mu, 
                    plot_im=False, iprint =False, a_init=a_init, optimize=True,omega=omega)
        
        
        sigscale = self.SigScale(valName=valName)
        if iprint :print('initial sigma'+valName+' scale:',sigscale_init,'optimized sigma scale:',sigscale, 'cuttoff: 1e-5')
    
        while abs(sigscale-sigscale_init) > convergence:
            sigscale_init = sigscale

            if valName == 'ne':
                a = self.a_ne
            elif valName == 'Te':
                a = self.a_Te

            self.main_core(ValName=valName, time_scale=time_scale,rho_scale=reff_scale,
                    sigma_scale=sigscale_init, time_cutoff=1e-8, reff_cutoff=1e-8,
                    iprint=False, mu=mu, a_init=a,plot_im=False,optimize=True,omega=omega)
            sigscale = self.SigScale(valName=valName)
            if iprint: print('optimized sigma scale:',sigscale,)

            
        if valName == 'ne':
            a = self.a_ne
        elif valName == 'Te':
            a = self.a_Te
            
        self.main_core(ValName=valName,
            time_scale=time_scale,rho_scale=reff_scale,
            sigma_scale=sigscale,time_cutoff=1e-8, reff_cutoff=1e-8,
            a_init=a, plot_im=False,
            iprint=False,mu=-2,optimize=optimize_only,omega=omega)

        sigscale = self.SigScale(valName=valName)
        if iprint: print('final optimized sigma scale:',sigscale)

        if valName == 'ne':
            self.sigma_scale_ne = sigscale
        elif valName == 'Te':
            self.sigma_scale_Te = sigscale


    
    def sigma_optimize(self,
            valName:str,
            time_scale:float,
            reff_scale:float,
            mu :float = -2,
            sigscale_init:float = 1,
            is_convergence:bool = False,
            convergence:float = 1e-2,
            optimize_only:bool = False,
            a_init:float = 0,
            omega:float = 0,
            iprint = False):
        
        #print(valName,np.array(a_init).mean())
        self.main_core(ValName=valName,
                    time_scale=time_scale,
                    rho_scale=reff_scale,
                    sigma_scale=sigscale_init,
                    mu=mu,
                    plot_im=False,
                    iprint =False,
                    a_init=a_init,
                    optimize=True,
                    omega=omega)    
        
        sigscale = self.SigScale(valName=valName)
        if iprint :print('initial sigma'+valName+' scale:',sigscale_init,'optimized sigma scale:',sigscale)

        if is_convergence:
            while abs(sigscale-sigscale_init) > convergence:
                sigscale_init = sigscale

                if valName == 'ne':
                    a = self.a_ne
                elif valName == 'Te':
                    a = self.a_Te

                self.main_core(ValName=valName, time_scale=time_scale,rho_scale=reff_scale,
                       sigma_scale=sigscale_init,
                       iprint=False, mu=mu, a_init=a,plot_im=False,
                       optimize=True,omega=omega)
                sigscale = self.SigScale(valName=valName)
                if iprint: print('optimized sigma scale:',sigscale)
                #print('optimized sigma scale:',sigscale,valName)

        
        if valName == 'ne':
            a = self.a_ne
        elif valName == 'Te':
            a = self.a_Te
        
        self.main_core(ValName=valName,
            time_scale=time_scale,rho_scale=reff_scale,
            sigma_scale=sigscale,
            a_init=a, plot_im=False,
            iprint=False,mu=-2,optimize=optimize_only,omega=omega)

        sigscale = self.SigScale(valName=valName)
        if iprint: print('final optimized sigma scale:',sigscale)

        if valName == 'ne':
            self.sigma_scale_ne = sigscale
        elif valName == 'Te':
            self.sigma_scale_Te = sigscale

    def mll_optimaize_autoSigma(self,
                valName:str = 'Both',
                param_init = [0.15,0.15],
                bounds:Bounds =  Bounds(lb=[0.05,0.05],ub=[0.3,0.3])) -> None:
        """
        Optimize the hyperparameters of the model.

        Parameters:
            param_init: initial values of the hyperparameters
            bounds: bounds of the hyperparameters

        Returns:
            None
        """



        self.logger = Logger()
        self.logger.write('\n## Start optimization of hyperparameters: [time_scale, reff_scale] to maximize the marginal loglikehood of '+valName+ '\n') 
        self.logger.write('## Initial values: '+str(param_init) )
        self.logger.write(' and Bounds: '+str(bounds.lb)+' to '+str(bounds.ub)+' ') 
        self.logger.write('. Sigma scale is optimized automatically\n')
        
        self.z_optimal = None

        self.a_Ne_temp = 0
        self.a_Te_temp = 0

        def objective(params):
            time = params[0]
            length = params[1]
                
            param_dict = {'time':time,'length':length}
            # 列数は一定になるように調整, 進捗表示用の文字列を作成. paramは 0.xxx の形式で表示 param.key = param.value となるように表示
            param_str = '['+', '.join([f'{key}={value:.4f}' for key,value in param_dict.items()])+']'
            
            print(param_str,end=' ')

            Te_failed = False
            Ne_failed = False

            if valName == 'Both':
                
                try:
                    self.sigma_optimize('ne',time_scale = time, reff_scale=length, optimize_only=True,is_convergence=True,iprint=False,a_init=self.a_Ne_temp)
                    Ne_failed = False
                except:
                    Ne_failed = True
                    self.sigma_optimize('ne',time_scale = time, reff_scale=length, optimize_only=True,is_convergence=True,iprint=False,a_init=0)
                    self.logger.write('\r')

                try:
                    self.sigma_optimize('Te',time_scale = time, reff_scale=length, optimize_only=True,is_convergence=True,iprint=False,a_init=self.a_Te_temp)
                    Te_failed = False
                except:
                    Te_failed = True
                    self.sigma_optimize('Te',time_scale = time, reff_scale=length, optimize_only=True,is_convergence=True,iprint=False,a_init=0)
                    self.logger.write('\r')

                z = -self.marginalLogLikelihood_rough(ValName='ne')-self.marginalLogLikelihood_rough(ValName='Te')
            
            
            else:
                raise NameError('parameter must be "Ne" or "Te"')

                    
            sigTe = self.SigScale('Te')
            sigNe = self.SigScale('ne')

            param_sig_dict = {'sigTe':sigTe,'sigNe':sigNe}
            param_sig_str  = '->['+', '.join([f'{key}={value:.4f}' for key,value in param_sig_dict.items()])+']'

            self.logger.write('\r')
            self.logger.write(param_str+param_sig_str)
                        
            if self.z_optimal is None:
                self.z_optimal = z
                z_optimal_bk = z
            elif z < self.z_optimal:
                z_optimal_bk = self.z_optimal
                self.a_Te_temp = self.a_Te
                self.a_Ne_temp = self.a_ne
                self.z_optimal = z

            else :
                z_optimal_bk = self.z_optimal
            
            
            mll_str = '-> mll='+str(z)[:10]+' mll_diff='+str(z-z_optimal_bk)[:8]
            if Ne_failed or Te_failed:
                #print(param_str,end=' ')
                
                self.logger.write(mll_str)
                self.logger.write(str(Ne_failed)+str(Te_failed)+'\n')
            else:
                self.logger.write(mll_str+'\n')


            return z
        
        #self.logger.write('## Iteration starts ##\n\n')
        res = minimize(objective,x0=param_init,bounds=bounds,method='Nelder-Mead',options={'xatol':1e-2,'fatol':10,'disp':True})
        self.opt_res = res 
        self.logger.write(str(res))
        self.optimized = True

        self.time_scale_Ne = res.x[0]
        self.time_scale_Te = res.x[0]

        self.rho_scale_Ne = res.x[1]
        self.rho_scale_Te = res.x[1]

        self.mainNe(time_scale=self.time_scale_Ne,reff_scale=self.rho_scale_Ne,sigma_scale=self.SigScale('ne'),mu=-2)
        self.mainTe(time_scale=self.time_scale_Te,reff_scale=self.rho_scale_Te,sigma_scale=self.SigScale('Te'),mu=-2)




        del self.a_Ne_temp, self.a_Te_temp, self.z_optimal
    
    # このクラスのいくつかの変数をjsonファイルに保存するメソッド
    def save(self,path:str='result/', filename:str=None,timestamp:bool = True,save_fig:bool = True):
        """
        Save the variables of this class to a json file.

        Parameters:
            filename: name of the file to save

        Returns:
            None
        """
        import json
        import datetime
        if filename is not None:
            filename = path+filename+'.json'
        else:
            if timestamp:
                filename = path+str(self.shotNo)+ '_'+datetime.datetime.now().strftime('%Y%m%d%H%M')+'.json'
            else:
                filename = path+str(self.shotNo)+ '.json'

        #変数はすべてではなく、指定する変数のみ保存する
        save_dict = {}

        save_dict['shotNo'] = self.shotNo

        save_dict['out_scale_Ne'] = self.out_scale_ne
        save_dict['time_scale_Ne'] = self.time_scale_Ne
        save_dict['reff_scale_Ne'] = self.rho_scale_Ne
        save_dict['sigma_scale_Ne'] = self.sigma_scale_ne

        save_dict['out_scale_Te'] = self.out_scale_Te
        save_dict['time_scale_Te'] = self.time_scale_Te
        save_dict['reff_scale_Te'] = self.rho_scale_Te
        save_dict['sigma_scale_Te'] = self.sigma_scale_Te

        save_dict['optimized'] = self.optimized
        if self.optimized:            
            opt_res = {}
            opt_res['message'] = self.opt_res.message
            opt_res['success'] = self.opt_res.success
            opt_res['fun'] = self.opt_res.fun
            opt_res['x'] = self.opt_res.x.tolist()
            opt_res['nit'] = self.opt_res.nit
            opt_res['nfev'] = self.opt_res.nfev
            opt_res['status'] = self.opt_res.status
            save_dict['optimize_res'] = opt_res
            save_dict['optimize_log'] = self.logger.log

        
        # jsonファイルに保存
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=4)
            
        print('saved as',filename)

        if save_fig:

                
            reff = self.rho_vac
            x_title = r'$\overline{reff}$'

            # pythonのバージョンに依存してfigsize を変更する
            if sys.version_info[1] > 6:
                figunit = 1.5
            else:
                figunit = 2.0
                

            times = self.time_inp
            
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(['none', 'gray']) 
            
            fig,axs=plt.subplots(2,4,figsize=(figunit*10,figunit*10),sharey=True)
            ax = axs[0,:]

            axs_titles = ['Te_data','Te_fit','dTe/dr','dTe/dt']    
            f = self.logTe_fit
            f_dr = self.dlogTe_dr*np.exp(f)
            f_dt = self.dlogTe_dt*np.exp(f)
            data = self.Te_inp

            im = ax[0].pcolormesh(self.realR_origin,times,data,vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
            axs_cbar(ax[0],im)

            ax[0].set_title(axs_titles[0])
            ax[0].grid()
            ax[0].set_xlabel('R [m]')    

            mask = (f < np.log(np.exp(f.max())*0.05 ) ) #+ self.idx_outlier

            im = ax[1].pcolormesh(reff,times,np.exp(f),vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
            vmax = int(5*np.exp(f).max())
            
            axs_cbar(ax[1],im)


            im = ax[2].pcolormesh(reff,times,f_dr,vmax=vmax,vmin=-vmax,cmap='seismic')
            axs_cbar(ax[2],im)

            ax[2].pcolormesh(reff,times,mask,cmap=cmap)
            im = ax[3].pcolormesh(reff,times,f_dt,vmax=vmax,vmin=-vmax,cmap='seismic')
            axs_cbar(ax[3],im)

            ax[3].pcolormesh(reff,times,mask,cmap=cmap)

            for i in range(1,4):
                ax_i = ax[i]
                ax_i.set_title(axs_titles[i])
                ax_i.grid()
                ax_i.set_xlabel(x_title)
                ax_i.set_xlim(-0.7,0.7)

            ax = axs[1,:]

            axs_titles = ['Ne_data','Ne_fit','dNe/dr','dNe/dt']    
            data = self.ne_inp
            f = self.logNe_fit
            f_dr = self.dlogNe_dr*np.exp(f)
            f_dt = self.dlogNe_dt*np.exp(f)

            im = ax[0].pcolormesh(self.realR_origin,times,data,vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
            axs_cbar(ax[0],im)

            ax[0].set_title(axs_titles[0])
            ax[0].grid()
            ax[0].set_xlabel('R [m]')    

            mask = (f < np.log(np.exp(f.max())*0.05 ) ) #+ self.idx_outlier

            im = ax[1].pcolormesh(reff,times,np.exp(f),vmax=np.exp(f).max()*1.1,vmin=0,cmap='jet')
            vmax = int(5*np.exp(f).max())
            
            axs_cbar(ax[1],im)


            im = ax[2].pcolormesh(reff,times,f_dr,vmax=vmax,vmin=-vmax,cmap='seismic')
            axs_cbar(ax[2],im)

            ax[2].pcolormesh(reff,times,mask,cmap=cmap)
            im = ax[3].pcolormesh(reff,times,f_dt,vmax=vmax,vmin=-vmax,cmap='seismic')
            axs_cbar(ax[3],im)

            ax[3].pcolormesh(reff,times,mask,cmap=cmap)

            for i in range(1,4):
                ax_i = ax[i]
                ax_i.set_title(axs_titles[i])
                ax_i.grid()
                ax_i.set_xlabel(x_title)
                ax_i.set_xlim(-0.7,0.7)

            ax[0].set_ylabel('time [s]')   
            
            # タイトルにshotNo,time_scale,reff_scale, out_cale を表示、有効数字は小数点以下３桁まで
            title = 'shotNo:'+str(self.shotNo)
            title+= '\nout_scale_Te:'+str(self.out_scale_Te)[:4]+', time_scale_Te:'+str(self.time_scale_Te)[:4]+', reff_scale_Te:'+str(self.rho_scale_Te)[:4]+', sigma_scale_Te:'+str(self.sigma_scale_Te)[:4] 
            title+= '\nout_scale_Ne:'+str(self.out_scale_ne)[:4]+', time_scale_Ne:'+str(self.time_scale_Ne)[:4]+', reff_scale_Ne:'+str(self.rho_scale_Ne)[:4]+', sigma_scale_Ne:'+str(self.sigma_scale_ne)[:4]
            fig.suptitle(title)
            # tight lauoutで保存する
            plt.tight_layout()

            # self.filenameが存在するか？
            fig.savefig(filename.replace('.json','.png'))
            plt.close()
            






            
            
                        

def make_alpha_colormap(color):
    from matplotlib.colors import LinearSegmentedColormap
    c = plt.cm.colors.to_rgba(color)
    cmap_dict = {
        'red':   [[0, c[0], c[0]], [1, c[0], c[0]]],
        'green': [[0, c[1], c[1]], [1, c[1], c[1]]],
        'blue':  [[0, c[2], c[2]], [1, c[2], c[2]]],
        'alpha': [[0, 0, 0], [1, 1, 1]]  # アルファ値は0から1まで線形に変化
    }
    return LinearSegmentedColormap('CustomMap', cmap_dict)

def plotGP(
        x_plot,y_plot,
        x_inp,f_inp,K_inp,
        ax:plt.Axes = None,
        color = 'darkblue',alpha_max=1,n_sample=0,**sample_kwargs):

    if ax is None:
        ax = plt.gca()

    import scipy.interpolate

    func = scipy.interpolate.interp1d(x_inp,f_inp)

    F = func(x_plot)[np.newaxis,:]
    sig = np.sqrt(np.diag(K_inp))
    func = scipy.interpolate.interp1d(x_inp,sig)

    Sig = func(x_plot)[np.newaxis,:]

    X,Y = np.meshgrid(x_plot,y_plot)

    # このカラーマップを使用
    alpha_cmap = make_alpha_colormap(color)

    C =  1/Sig * np.exp(-0.5*(F-Y)**2/Sig**2)
    C =  C/ C.max() * alpha_max


    dx = x_plot[1]-x_plot[0]
    dy = y_plot[1]-y_plot[0]

    extent = (x_plot.min()-0.5*dx,x_plot.max()+0.5*dx, y_plot.min()-0.5*dy, y_plot.max() + 0.5*dy)
    ax.imshow(C, vmin=0, vmax=1,extent=extent, cmap= alpha_cmap,origin='lower' ,aspect='auto',zorder=0)

    # n=0 のとき、標本路をプロットしない。
    if n_sample > 0:
        f_sample = np.random.multivariate_normal(f_inp,K_inp,n_sample)
        for i in range(n_sample):
            if i == 0:
                
                ax.plot(x_inp,f_sample[i,:], color=color,alpha=0.5,lw=0.3,label='sample path')
            else:
                ax.plot(x_inp,f_sample[i,:], color=color,alpha=0.5,lw=0.3)
    
    ax.plot(x_inp,f_inp+sig,color=color,linestyle='dashed',lw=1.,label=r'$\pm 1\sigma$')
    ax.plot(x_inp,f_inp-sig,color=color,linestyle='dashed',lw=1.)


def plotLogGP(
        x_plot,y_plot,
        x_inp,f_inp,K_inp,
        ax:plt.Axes = None,
        color = 'darkblue',alpha_max=1,n_sample=0,**sample_kwargs):
    import scipy.interpolate
    
    if ax is None:
        ax = plt.gca() # get current axis


    func = scipy.interpolate.interp1d(x_inp,f_inp)

    F = func(x_plot)[np.newaxis,:]
    sig = np.sqrt(np.diag(K_inp))
    func = scipy.interpolate.interp1d(x_inp,sig)

    Sig = func(x_plot)[np.newaxis,:]

    y_plot[y_plot<0] = np.nan
    X,log_Y = np.meshgrid(x_plot,np.log(y_plot))

    # このカラーマップを使用
    alpha_cmap = make_alpha_colormap(color)

    C =  1/Sig * np.exp(-0.5*(F-log_Y)**2/Sig**2)
    C =  C/ C.max() * alpha_max


    dx = x_plot[1]-x_plot[0]
    dy = y_plot[1]-y_plot[0]
    #print(dy)

    extent = (x_plot.min()-0.5*dx,x_plot.max()+0.5*dx, y_plot.min()-0.5*dy, y_plot.max() + 0.5*dy)
    #print(extent)
    ax.imshow(C, vmin=0, vmax=1,extent=extent, cmap= alpha_cmap,origin='lower' ,aspect='auto',zorder=0)

    # n=0 のとき、標本路をプロットしない。
    if n_sample > 0:
        f_sample = np.random.multivariate_normal(f_inp,K_inp,n_sample)
        for i in range(n_sample):
            if i == 0:
                
                ax.plot(x_inp,np.exp(f_sample[i]), color=color,alpha=0.5,lw=0.3,label='sample path')
            else:
                ax.plot(x_inp,np.exp(f_sample[i]), color=color,alpha=0.5,lw=0.3)
    
    ax.plot(x_inp,np.exp(f_inp+sig),color=color,linestyle='dashed',lw=1.,label=r'$\pm 1\sigma$')
    ax.plot(x_inp,np.exp(f_inp-sig),color=color,linestyle='dashed',lw=1.)
    #plt.xlim(x_plot.min(),x_plot.max())
    #plt.ylim(y_plot.min(),y_plot.max())

        


def KSE(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    return np.exp(-0.5*(X[0]-X[1])**2/len**2)

def KSE_dx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    return (-(X[0]-X[1])/len**2)*np.exp(-0.5*(X[0]-X[1])**2/len**2)

def KSE_dxdx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    return (-(X[0]-X[1])**2/len**4+1/len**2)*np.exp(-0.5*(X[0]-X[1])**2/len**2)

# rational quadoratic
def KRQ(x0:np.ndarray,x1:np.ndarray,len:float,alpha:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    return (1+0.5*(X[0]-X[1])**2/len**2 /alpha) **(-alpha)


# rational quadoratic
def KRQ_dx(x0:np.ndarray,x1:np.ndarray,len:float,alpha:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')    
    return -(1+0.5*(X[0]-X[1])**2/len**2 /alpha) **(-alpha-1)*(X[0]-X[1]) / len**2

def Kmatern32(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    R = abs(X[0]-X[1])
    return (1+np.sqrt(3)*R/len)*np.exp(-np.sqrt(3)*R/len )   

def Kmatern32_dx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    Tau = X[0]-X[1]
    return -3*Tau/len**2 *np.exp(-np.sqrt(3)*abs(Tau)/len )

def Kmatern32_dxdx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    R = abs(X[0]-X[1])
    return 3/len**2*(1 - np.sqrt(3)*R/len)*np.exp(-np.sqrt(3)*R/len )


def Kmartner52(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    R = abs(X[0]-X[1]) /len
    return (1+np.sqrt(5)*R + 5/3*R**2)*np.exp(-np.sqrt(5)*R )

def Kmartner52_dx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    Tau = (X[0]-X[1])/len
    R = abs(Tau)
    return -5/3*Tau/len*(1+np.sqrt(5)*R)*np.exp(-np.sqrt(5)*R )

def Kmartner52_dxdx(x0:np.ndarray,x1:np.ndarray,len:float)->np.ndarray:
    X = np.meshgrid(x0,x1,indexing='ij')
    R = abs(X[0]-X[1])/len
    return 5/3/len**2*(1 +np.sqrt(5)* R - 5 *R**2) *np.exp(-np.sqrt(5)*R )

    #return -5/3*Tau/len**2*(1+np.sqrt(5)*R/len)*np.exp(-np.sqrt(5)*R/len )




def numerical_differentiation_matrix(x:np.ndarray):
    """
    数値微分行列を計算する関数です。与えられた点の配列に対して、中心差分、前進差分、後退差分を用いて数値微分行列を計算します。

    Parameters:
    x (np.ndarray): 微分を行う点の配列

    Returns:
    np.ndarray: 数値微分行列


    """

    n = len(x)
    D = np.zeros((n, n))
    
    # 中心差分
    for i in range(1, n-1):
        D[i, i-1] = -1 / (x[i+1] - x[i-1])
        D[i, i+1] = 1 / (x[i+1] - x[i-1])
    
    # 前進差分 (端点)
    D[0, 0] = -1 / (x[1] - x[0])
    D[0, 1] = 1 / (x[1] - x[0])
    
    # 後退差分 (端点)
    D[n-1, n-2] = -1 / (x[n-1] - x[n-2])
    D[n-1, n-1] = 1 / (x[n-1] - x[n-2])

    return D

