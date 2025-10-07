import matplotlib.pyplot as plt
from ..labcom_retrieve import LHDRetriever,LHDData
# improt anadata
from .. import anadata
from ..anadata import KaisekiData
import numpy as np
import scipy.signal as spsg
import sys,time,gc
from scipy.signal import medfilt
from scipy.signal import find_peaks
import os 
import sys


 
plt.rcParams["font.size"] = 12 # 全体のフォントサイズが変更されます。
plt.rcParams["xtick.direction"] = "in"      # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["ytick.direction"] = "in" 

mwscat_freq_wrong =  [None,None,None,None,None,None,None,None,500 ,900 ,1300,1500,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2825,2875,2925,2975,3025,3075,3125,3175]
mwscat2_freq_wrong = [3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4500,4700,5100,5500,None,None,1000,1300,None,None,None,None,None,None,None,None,None,None,None,None]


mwscat_freq       =  [None,None,None,None,None,None,None,None,500 ,900 ,1300,1500,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,3200,3300,3400,3500,3600,3700,3800,3900]
mwscat2_freq       = [4000,4100,4200,4300,4500,4700,5100,5500,2825,2875,2925,2975,3025,3075,3125,3175,None,None,1000,1300,None,None,None,None,None,None,None,None,None,None,None,None]

freq_list = {}
freq_list['mwscat']  = mwscat_freq
freq_list['mwscat2'] = mwscat2_freq

def get_latest_shot_num(shotno_start):
    """
    Get the latest shot number.
    
    Parameters
    ----------
    shot_num_start : int
        Start number of the search.
        
    Returns
    -------
    int
        The latest shot number.
    """
    shot_num = shotno_start
    retriever = LHDRetriever()
    while True:
        try:
            #print('checking shot_num:',shot_num)
            retriever.retrieve_data('mwscat',shot_num,1,1)
            shot_num += 1
        except:
            break
    return shot_num - 1

#最新のshot_num までmwscatのデータをプロットして保存する。
#プロット現在が最新に到達したら10秒待って再度チェックする。
def plot_mwscat_all_latest(shotno_start:int,shotno_stop:int=None,save:bool=True,
                           start_time:float=None,end_time:float=None,save_dir:str=''):  
    """
    Plot all channels of the latest mwscat data.
    
    Parameters
    ----------
    shot_num_start : int
        Start number of the search.
    save : bool, optional
        Save the plot if True.
        
    """
    shot_num_now = shotno_start
    shot_num_latest = get_latest_shot_num(shotno_start)
    if shotno_stop is None:
        shotno_stop = shot_num_latest 

    while shot_num_now < shotno_stop:
        if shot_num_now > shot_num_latest:
            time.sleep(10)
            shot_num_latest = get_latest_shot_num(shot_num_now)

        elif shot_num_now <= shot_num_latest:
            print('plotting shot_num:',shot_num_now)
            plot_mwscat_all(shot_num_now,'mwscat',save=save,plot_show=False,start_time=start_time,end_time=end_time, save_dir=save_dir)
            plot_mwscat_all(shot_num_now,'mwscat2',save=save,plot_show=False, start_time=start_time,end_time=end_time, save_dir=save_dir)
            shot_num_now += 1 



def plot_mwscat_all(
        shotno:int, 
        digi:str, # 'mwscat' or 'mwscat2'
        start_time:float=None,
        end_time:float=None,    
        plot_show:bool =True,
        save:bool=False,
        save_dir:str='',
        axs = None,  
        ):
    """
    Plot all channels of mwscat or mwscat2 data.
    
    Parameters
    ----------
    shot_num : int
        Shot number.
    digi_name : str
        'mwscat' or 'mwscat2'
    start_time : float, optional
        Start time of the plot.
    end_time : float, optional
        End time of the plot.
    save : bool, optional
        Save the plot if True.
        
    """
    retriever = LHDRetriever()
    if plot_show == False:
        plt.ioff()
    else:
        plt.ion()

    if axs is None:
        fig,axs = plt.subplots(8,4,figsize=(25,15),sharex=True)
    else:
        axs = axs
        fig = axs[0,0].get_figure()

    for i in range(32):
        icol, irow = int(i / 4), int( i % 4 )
        ax: plt.Axes = axs[icol,irow]

        ch_i = i+1

        data =  retriever.retrieve_data(diag=digi, 
                                          shotno=shotno, 
                                          subshot=1, 
                                          channel=ch_i,time_axis=True)
        val:np.ndarray  = data.get_val()
        timedata0:np.ndarray = data.time


        if start_time is not None:
            start_index = np.argmin(np.abs(timedata0 - start_time))
            xmin = timedata0[start_index]
        else:
            start_index = None
            xmin = timedata0[0]

        if end_time is not None:
            end_index = np.argmin(np.abs(timedata0 - end_time))
            xmax = timedata0[end_index]
        else:
            end_index = None
            xmax = timedata0[-1]

        sig_voltdata = val

        n = int(timedata0[start_index:end_index].size / 2000)
 
        sl = slice(start_index,end_index,n)
        ax.plot(timedata0[sl],sig_voltdata[sl])
        ax.set_title('ch'+str(i+1)+': ' +str(freq_list[digi][i])[:4]+' MHz')
        ax.set_xlim(xmin,xmax)

        if icol == 7:
            ax.set_xlabel('time [sec]')
        if irow == 0:
            ax.set_ylabel('Signal [V]')

        # 縦のグリッド線を追加
        ax.grid(axis='x', linestyle='--', alpha=1,linewidth=1)

        
    figname =str(shotno)+'_'+digi
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.suptitle(figname,fontsize=25)
    if save:
        fig.savefig(save_dir+figname+'.png')
    
    if plot_show == False:
        plt.close(fig)



def plot_mwscat_all2(
        shot_num:int, 
        digi_name:str, # 'mwscat' or 'mwscat2'
        start_time:float=None,
        end_time:float=None,
        figsize= (25,15),
        ):
    """
    Plot all channels of mwscat or mwscat2 data.
    
    Parameters
    ----------
    shot_num : int
        Shot number.
    digi_name : str
        'mwscat' or 'mwscat2'
    start_time : float, optional
        Start time of the plot.
    end_time : float, optional
        End time of the plot.
    save : bool, optional
        Save the plot if True.
        
    """
    retriever = LHDRetriever()

    fig,axs = plt.subplots(8,4,figsize=figsize,sharex=True)

    for i in range(32):
        icol, irow = int(i / 4), int( i % 4 )
        ax: plt.Axes = axs[icol,irow]

        ch_i = i+1

        data =  retriever.retrieve_data(diag=digi_name, 
                                          shotno=shot_num, 
                                          subshot=1, 
                                          channel=ch_i,time_axis=True)
        val:np.ndarray  = data.get_val()
        timedata0:np.ndarray = data.time


        if start_time is not None:
            start_index = np.where(timedata0 > start_time)[0][0]
            xmin = start_time
        else:
            start_index = None
            xmin = timedata0[0]

        if end_time is not None:
            end_index = np.where(timedata0 > end_time)[0][0]
            xmax = end_time
        else:
            end_index = None
            xmax = timedata0[-1]

        sig_voltdata = val

        n = int(timedata0[start_index:end_index].size / 2000)
 
        sl = slice(start_index,end_index,n)
        ax.plot(timedata0[sl],sig_voltdata[sl])
        ax.set_title('ch'+str(i+1)+': ' +str(freq_list[digi_name][i])[:4]+' MHz')
        ax.set_xlim(xmin,xmax)

        if icol == 7:
            ax.set_xlabel('time [sec]')
        if irow == 0:
            ax.set_ylabel('Signal [V]')

        
    figname =str(shot_num)+'_'+digi_name
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.suptitle(figname,fontsize=25)
    return fig,axs

def plot_mwscat_map(shot_num: int, 
                    is_print: bool = True,
                    heating_on: bool = False,
    ):
    """Plot a map of mwscat or mwscat2 data for a given shot number.
    Parameters
    ----------
    shot_num : int
        Shot number to retrieve data for.
    savepath : str, optional
        Directory path to save the plot. If not specified, the plot will not be saved.
    """
    def _print(*args, **kwargs):
        if is_print:
            print(*args, **kwargs)

    _print(f"Starting plot_map for shot_num: {shot_num}")
    retriever = LHDRetriever()
    try:
        _print("Retrieving mwscat channels...")
        mwscat = retriever.retrieve_multiple_channels(diag_name='mwscat', shot=shot_num, channels=np.arange(1, 33))
        _print("Retrieving mwscat2 channels...")
        mwscat2 = retriever.retrieve_multiple_channels(diag_name='mwscat2', shot=shot_num, channels=np.arange(1, 33))
    except Exception as e:
        _print(f"Error retrieving channels: {e}")
        return

    mwscat_all: list[LHDData] = []
    freqs = []
    for i, freq in enumerate(mwscat_freq):
        if freq is not None:
            try:
                mwscat_all.append(mwscat[i + 1])
                freqs.append(freq)
            except Exception as e:
                _print(f"Error appending mwscat channel {i+1}: {e}")

    for i, freq in enumerate(mwscat2_freq):
        if freq is not None:
            try:
                mwscat_all.append(mwscat2[i + 1])
                freqs.append(freq)
            except Exception as e:
                _print(f"Error appending mwscat2 channel {i+1}: {e}")
        if freq == 5500:
            break

    freqs = np.array(freqs)
    _print(f"Total channels used: {len(mwscat_all)}")

    try:
        _print("Retrieving ECH power and heating flag data...")
        ech_pw = anadata.KaisekiData.retrieve_opendata(diag='echpw', shotno=shot_num)
        heating = anadata.KaisekiData.retrieve_opendata(diag='heating_flg', shotno=shot_num)
    except Exception as e:
        _print(f"Error retrieving ECH/heating data: {e}")
        return

    try:
        heating_time = heating['time'][heating['Heating_on'] > 0]
        max_time = heating_time.max()
        min_time = heating_time.min()
        
        if not heating_on: min_time = 2.5
    except Exception as e:
        _print(f"Error processing heating time: {e}")
        return

    try:
        num_of_modulation = np.abs(np.diff(ech_pw['77G_2Our'] > 0)).sum() // 2
        time = mwscat_all[0].time
    except Exception as e:
        _print(f"Error processing modulation/time: {e}")
        return
    
    try:
        num_of_modulation2 = np.abs(np.diff(ech_pw.get_val_data(0) > 0)).sum() // 2
        time = mwscat_all[0].time
    except Exception as e:
        _print(f"Error processing modulation2/time: {e}")
        return

    before_time = time < (min_time - 0.5)
    after_time = time > (max_time + 0.5)
    dead_time_idx = np.zeros(time.size, dtype=bool)

    if num_of_modulation > 5:
        _print("Detected modulation(77GHz 2Our), masking dead time...")
        try:
            modulation = (ech_pw['77G_2Our'] > 0)
            modulation = np.diff(modulation)
            index = np.where(modulation == True)[0]
            for i in index:
                time_i = ech_pw['Time'][i]
                dead_time_idx += (time > (time_i - 0.0002)) & (time < (time_i + 0.005))
        except Exception as e:
            _print(f"Error masking dead time: {e}")

    elif num_of_modulation2 > 5:
        _print("Detected modulation(77GHz 1.5UO), masking dead time...")
        try:
            modulation = (ech_pw['77G_1.5UO'] > 0)
            modulation = np.diff(modulation)
            index = np.where(modulation == True)[0]
            for i in index:
                time_i = ech_pw['Time'][i]
                dead_time_idx += (time > (time_i - 0.0002)) & (time < (time_i + 0.005))
        except Exception as e:
            _print(f"Error masking dead time: {e}")

    try:
        index_ = np.where(~(before_time + after_time))[0]
        i_min = index_.min()
        i_max = index_.max()
        n_time = i_max - i_min + 1
    except Exception as e:
        _print(f"Error finding time indices: {e}")
        return

    x_pixel_max = 1000
    reduce = max(1, n_time // x_pixel_max)
    time_r = time[i_min:i_max + 1:reduce]

    ece_map = np.zeros((len(mwscat_all), time_r.size,))
    means = np.zeros((len(mwscat_all),))

    _print("Processing channel data for map...")
    for i, mwscat in enumerate(mwscat_all):
        try:
            offset = mwscat_all[i].val[before_time + after_time].mean()
            val2 = -(mwscat_all[i].val - offset)
            val2[dead_time_idx] = np.nan
            val_r = val2[i_min:i_max + 1:reduce]
            ece_map[i, :] = val_r[:]
            means[i] = np.nan_to_num(val_r).mean()
        except Exception as e:
            _print(f"Error processing channel {i}: {e}")

    d_freqs = np.zeros(freqs.size)
    if freqs.size > 1:
        d_freqs[1:] = np.diff(freqs) / 2
        d_freqs[:-1] += np.diff(freqs) / 2

    _print("Plotting...")
    try:
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=(1, 3, 0.2), wspace=0.1)
        ax_main = fig.add_subplot(gs[0, 1], sharex=None, sharey=None)
        im = ax_main.pcolormesh(time_r, freqs / 1e3, ece_map / means[:, np.newaxis], cmap='inferno', vmin=0, vmax=2)
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('Frequency (GHz)')
        ax_main.grid(True)

        ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
        ax_left.barh(freqs / 1e3, means, height=d_freqs / 1e3 * 0.9, alpha=0.5)
        ax_left.invert_xaxis()
        ax_left.spines['top'].set_visible(True)
        ax_left.spines['right'].set_visible(True)
        ax_left.spines['bottom'].set_visible(True)
        ax_left.spines['left'].set_visible(False)
        ax_left.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
        ax_left.grid(True)

        ax_cbar = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
        cbar.set_label('Value')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        fig.suptitle(f"mwscat map for Shot {shot_num}", fontsize=16)

        axes ={
            'main': ax_main,
            'left': ax_left,
            'cbar': ax_cbar
        }
        return fig, axes

    except Exception as e:
        _print(f"Error during plotting: {e}")


def take_mwscat_all(shotno:int) -> KaisekiData:
    flg = True
    kaisekidata  = KaisekiData()
    kaisekidata.dimunits = ['s','MHz']
    kaisekidata.dimnames = ['Time','Freq'] 
    ech = kaisekidata.retrieve_opendata(diag='ech', shotno=shotno)
    kaisekidata.date = ech.date
    kaisekidata.dimno = 2

    data = {}
    data['Time'] = None
    data['Freq'] = []
    kaisekidata.valunits = ['V'] 
    kaisekidata.valnames = ['mwscat']
    kaisekidata.valno = 1
    kaisekidata.name = 'mwscat_all'
    kaisekidata.shotno = shotno
    kaisekidata.subno = 1

    retriever = LHDRetriever()  

    for i,key in enumerate(mwscat_freq):
        if key is not None:
            if flg:
                try:
                    temp = retriever.retrieve_data(diag='mwscat', shotno=shotno, subshot=1, channel=i+1, time_axis=True)
                    data['Time'] = temp.time
                    flg = False        
                    data['Freq'].append(key)
                    data[key] = temp.val
                except Exception as e:
                    print(f"Error retrieving data for channel {i+1}: {e}")
                    continue

            else:
                try:
                    temp = retriever.retrieve_data(diag='mwscat', shotno=shotno, subshot=1, channel=i+1, time_axis=False)
                    data['Freq'].append(key)
                    data[key] = temp.val
                except Exception as e:
                    print(f"Error retrieving data for channel {i+1}: {e}")
                    continue

                
    for i,key in enumerate(mwscat2_freq):
        if key is not None:
            try:
                temp = retriever.retrieve_data(diag='mwscat2', shotno=shotno, subshot=1, channel=i+1, time_axis=False)
                data['Freq'].append(key)
                data[key] = temp.val
            except:
                continue

            if key == 5500:
                break



    kaisekidata.data = np.zeros((len(data['Time']), len(data['Freq'])))
    for i, freq in enumerate(data['Freq']):
        kaisekidata.data[:,i] = data[freq]

    kaisekidata._time = data['Time']
    kaisekidata.freq = np.array(data['Freq'])

    del data

    
    return kaisekidata


if __name__ == '__main__':
        
    shotnum = 183644
    #shotnum = input('input shot number:')
    plot_mwscat_all(shotno=shotnum,digi='mwscat',start_time=2.5 , save=True,plot_show=False)
    plot_mwscat_all(shotno=shotnum,digi='mwscat2',start_time=2.5, save=True,plot_show=False)

    
