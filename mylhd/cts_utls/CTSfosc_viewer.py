from ..labcom_retrieve import LHDRetriever
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fft import fft, fftfreq
import argparse
import time

retriever = LHDRetriever()

def average_spectrum(data, n,fs):
    N = len(data)
    segment_length = N // n
    segments = [data[i:i + segment_length] for i in range(0, N, segment_length)]
    spectrum = np.array([fft(segment) for segment in segments])
    freq = fftfreq(segment_length, d=1/fs)  # サンプリング周波数は12.5GHz
    avg_spectrum = np.mean(np.abs(spectrum), axis=0)
    return freq[:segment_length // 2], avg_spectrum[:segment_length // 2]

def plot_with_marginals(
    data2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    y_for_proj_x: np.ndarray = None,
    x_for_proj_y: np.ndarray = None,
    width_ratios: tuple = (1, 4, 0.2),
    height_ratios: tuple = (4, 1),
    wspace: float = 0.05,
    hspace: float = 0.05,
    figsize: tuple = (6, 6),
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
):
    """
    2Dカラーマップとその左右・下のマージナルプロットを描画する関数。

    Parameters
    ----------
    data2d : 2D numpy.ndarray
        カラーマップ用の2Dデータ
    proj_x : 1D numpy.ndarray
        左側にプロットする X方向積算データ（長さ = data2d.shape[0]）
    proj_y : 1D numpy.ndarray
        下側にプロットする Y方向積算データ（長さ = data2d.shape[1]）
    x : 1D numpy.ndarray
        data2d の列方向インデックス（長さ = data2d.shape[1]）
    y : 1D numpy.ndarray
        data2d の行方向インデックス（長さ = data2d.shape[0]）
    width_ratios : tuple, optional
        GridSpec の width_ratios (左, メイン, カラーバー)
    height_ratios : tuple, optional
        GridSpec の height_ratios (メイン, 下)
    wspace : float, optional
        GridSpec の wspace
    hspace : float, optional
        GridSpec の hspace
    figsize : tuple, optional
        図のサイズ
    cmap : str, optional
        カラーマップ

    Returns
    -------
    fig : matplotlib.figure.Figure
        図オブジェクト
    axes : dict of matplotlib.axes.Axes
        'left', 'main', 'bottom', 'cbar' の各Axes
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace
    )

    # 左側プロット: 枠は左と下のみ、目盛りラベル無し

    # メインマップ
    ax_main = fig.add_subplot(gs[0, 1], sharex=None, sharey=None)
    im = ax_main.pcolormesh(x, y, data2d, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.grid(True)

    
    if y_for_proj_x is None:
        y_for_proj_x = y
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_left.plot(proj_x, y_for_proj_x)
    ax_left.invert_xaxis()
    # スパイン設定
    ax_left.spines['top'].set_visible(True)
    ax_left.spines['right'].set_visible(True)
    ax_left.spines['bottom'].set_visible(True)
    ax_left.spines['left'].set_visible(False)
    # 目盛りとラベルを非表示
    ax_left.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    ax_left.set_xlabel('Σ over X')
    ax_left.grid(True)

    # カラーバー
    ax_cbar = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label('Value')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    # 下側プロット
    if x_for_proj_y is None:
        x_for_proj_y = x
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_bottom.plot(x_for_proj_y, proj_y)
    #ax_bottom.set_xlabel('X')
    ax_bottom.tick_params(left=False, bottom=False, labelleft=True, labelbottom=False)
    ax_bottom.set_ylabel('Σ over Y')
    # スパイン設定
    ax_bottom.spines['top'].set_visible(True)
    ax_bottom.spines['right'].set_visible(True)
    ax_bottom.spines['bottom'].set_visible(False)
    ax_bottom.spines['left'].set_visible(True)
    ax_bottom.grid(True)

    # 余白エリアをオフ
    for idx in [(1, 0), (1, 2)]:
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')

    # すべての軸を共有
    ax_main.set_xlim(x.min(), x.max())
    ax_main.set_ylim(y.min(), y.max())
    ax_left.set_ylim(y.min(), y.max())
    ax_bottom.set_xlim(x.min(), x.max())


    axes = {
        'left': ax_left,
        'main': ax_main,
        'bottom': ax_bottom,
        'cbar': ax_cbar
    }
    return fig, axes


def plot_all(shot_num:int = None, diag:str='CTSfosc1', subshot:int=1, channel:int=1, 
              t0:float = 0, f0:float = 0):
    """
    Save the spectrogram plot for a given shot number, diagnostic, subshot, and channel.
    Parameters
    ----------
    shot_num : int
        The shot number to retrieve data for.
    diag : str
        The diagnostic name to retrieve data from.
    subshot : int
        The subshot number to retrieve data for.
    channel : int
        The channel number to retrieve data for.
    t0 : float, optional
        The time offset to apply to the x-axis of the plot.
    f0 : float, optional
        The frequency offset to apply to the y-axis of the plot.
    """
    #時間経過を表示
    tmp_time = time.time()

    print(f"Retrieving data for shot {shot_num}, diag {diag}, subshot {subshot}, channel {channel}...")
    data = retriever.retrieve_data(
        diag=diag,
        shotno=shot_num,
        subshot=subshot, 
        channel=channel,
        time_axis=False,
        dtype=np.int8,
    )
    print(f"Data retrieval completed in {time.time() - tmp_time:.2f} seconds.")
    
    print(f"STFT processing data...")

    tmp_time = time.time()


    fs = float(data.metadata['MIN_SAMPLE_RATE'] )
    f, t, Zxx = stft(
        data.val,
        fs=fs, 
        window="hann",
        nperseg=2**16,
        noverlap=None,
        boundary=None,
        padded=False,
    )
    print(f"STFT processing completed in {time.time() - tmp_time:.2f} seconds.")
    n_segments = 10000
    freq, spectrum = average_spectrum(data.val, n_segments, fs)

    max_pixels = (1500,2000) 
    ny, nx = Zxx.shape
    y_reduce = ny // max_pixels[0]
    x_reduce = nx // max_pixels[1]
    Zxx_r = Zxx[::y_reduce, ::x_reduce]
    f_r = f[::y_reduce]
    t_r = t[::x_reduce]

    n_sparse = 100000
    val_sparse = data.val[::n_sparse]  # 10kごとに間引き
    time_sparse = np.arange(len(val_sparse)) / fs  * n_sparse# サンプリング周波数は12.5GHz
    
    Zxx_db =  20*(np.log10(abs(Zxx_r)+1e-12))
    vmax = np.percentile(Zxx_db, 99.9)
    vmin = np.percentile(Zxx_db, 30)

        

    fig, axes = plot_with_marginals(
        data2d= Zxx_db,  # dB スケール 
        x=t_r + t0,       # [s] 表示
        y=(f_r+f0) * 1e-9 ,  # [GHz] 表示
        proj_x=20*(np.log10(abs(spectrum)+1e-12))[1:],  # dB スケール
        proj_y=val_sparse,
        y_for_proj_x=(freq[1:]+f0) * 1e-9,  # [GHz] 表示
        x_for_proj_y=time_sparse + t0,  # [s] 表示\
        vmax= vmax,  # dB スケール
        vmin= vmin,  # dB スケール
        cmap='inferno',
        width_ratios=(1, 4, 0.1),
        height_ratios=(4, 1.5),
        figsize=(12, 8),
    )
        
    axes["main"].set_xlabel("Time [s]", fontsize=14)
    axes["main"].set_ylabel("Frequency [GHz]", fontsize=14)
    axes["cbar"].set_ylabel("Magnitude [dB]")
    axes["bottom"].set_ylabel("Sig [V]")
    axes["left"].set_xlabel("Spectral Magnitude [dB]")

    fig.suptitle(f"# {shot_num}, diag {diag}, subshot {subshot}, channel {channel}", fontsize=16)

    return fig, axes


from scipy.signal import resample

def plot_all2(shot_num:int = None, diag:str='CTSfosc1', subshot:int=1, channel:int=1, 
              t0:float = 0, f0:float = 0):
    
    data = retriever.retrieve_data(
        diag=diag,
        shotno=shot_num,
        subshot=subshot, 
        channel=channel,
        time_axis=False,
        dtype=np.int8,  # ← int8ではなくfloatで
    )
    ...
    fs = float(data.metadata['MIN_SAMPLE_RATE'])
    f, t, Zxx = stft(
        data.val,
        fs=fs, 
        window="hann",
        nperseg=2**16,
        noverlap=None,
        boundary=None,
        padded=False,
    )

    # ==== 縞が出ないように平均リサンプリング ====
    max_pixels = (1500, 2000)  # (freq方向, time方向)
    ny, nx = Zxx.shape
    if ny > max_pixels[0]:
        Zxx_r = resample(Zxx, max_pixels[0], axis=0)
        f_r = np.linspace(f[0], f[-1], max_pixels[0])
    else:
        Zxx_r, f_r = Zxx, f

    if nx > max_pixels[1]:
        Zxx_r = resample(Zxx_r, max_pixels[1], axis=1)
        t_r = np.linspace(t[0], t[-1], max_pixels[1])
    else:
        t_r = t
    # ===============================================

    n_segments = 10000
    freq, spectrum = average_spectrum(data.val, n_segments, fs)

    n_sparse = 100000
    val_sparse = data.val[::n_sparse]
    time_sparse = np.arange(len(val_sparse)) / fs * n_sparse

    Zxx_db = 20 * np.log10(np.abs(Zxx_r) + 1e-12)
    vmax = np.percentile(Zxx_db, 99.9)
    vmin = np.percentile(Zxx_db, 30)

    fig, axes = plot_with_marginals(
        data2d=Zxx_db,
        x=t_r + t0,
        y=(f_r + f0) * 1e-9,
        proj_x=20*np.log10(np.abs(spectrum)+1e-12)[1:],
        proj_y=val_sparse,
        y_for_proj_x=(freq[1:] + f0) * 1e-9,
        x_for_proj_y=time_sparse + t0,
        vmax=vmax,
        vmin=vmin,
        cmap='inferno',
        width_ratios=(1, 4, 0.1),
        height_ratios=(4, 1.5),
        figsize=(12, 8),
    )
    axes["main"].set_xlabel("Time [s]", fontsize=14)
    axes["main"].set_ylabel("Frequency [GHz]", fontsize=14)
    axes["cbar"].set_ylabel("Magnitude [dB]")
    axes["bottom"].set_ylabel("Sig [V]")
    axes["left"].set_xlabel("Spectral Magnitude [dB]")

    fig.suptitle(f"# {shot_num}, diag {diag}, subshot {subshot}, channel {channel}", fontsize=16)

    return fig, axes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save spectrogram plot for a given shot number.")
    parser.add_argument("--shot", type=int, required=True, help="Shot number")
    parser.add_argument("--diag", type=str, default='CTSfosc1', help="Diagnostic name")
    parser.add_argument("--subshot", type=int, default=1, help="Subshot number")
    parser.add_argument("--channel", type=int, default=1, help="Channel number")
    parser.add_argument("--t0", type=float, default=0, help="Time offset for x-axis")
    parser.add_argument("--f0", type=float, default=0, help="Frequency offset for y-axis")
    ## plt.show() を自動で呼び出すための引数
    parser.add_argument("--show", action='store_true', help="Show the plot after saving")

    args = parser.parse_args()
    
    fig, axes = save_plot(shot_num=args.shot, diag=args.diag, subshot=args.subshot, channel=args.channel)
    fig.savefig(f"CTSfosc_{args.shot}.png", dpi=300)
    print(f"Plot saved as CTSfosc_{args.shot}.png")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
