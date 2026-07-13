# labcom_retrieve 利用ガイド

`mylhd.labcom_retrieve` は、LABCOM の `Retrieve.exe` を Python から実行し、生成されたバイナリデータとパラメータファイルを `LHDData` に変換します。

- サンプル: [labcom_retrieve_sample.ipynb](../examples/labcom_retrieve_sample.ipynb)
- パッケージ概要: [README](../README.md)

## 1. 前提条件

- **LHD の解析サーバーへログイン済みであること**
- Windows、または Windows 側の実行ファイルへアクセスできる WSL
- 利用可能な `Retrieve.exe`
- 対象データへアクセスできる LABCOM 環境

`Retrieve.exe` がローカルに存在するだけでは取得できません。LHD の解析サーバー上、または同サーバーへログインして LABCOM データへアクセスできるセッションで実行することを前提としています。

環境を確認するには次を実行します。

```python
from mylhd.labcom_retrieve import check_windows_environment

environment = check_windows_environment()
print(environment)
```

自動探索の主な候補は次の通りです。

- `C:\LABCOM\Retrieve\bin\Retrieve.exe`
- `C:\LHD\Retrieve\Retrieve.exe`
- WSL 上の `/mnt/c/LABCOM/Retrieve/bin/Retrieve.exe`

## 2. Retriever を初期化する

自動探索できる場合:

```python
from mylhd import LHDRetriever

retriever = LHDRetriever()
```

明示する場合、現在の API では `retrieve_path` に **`Retrieve.exe` を含むディレクトリ** を渡します。

```python
retriever = LHDRetriever(
    retrieve_path=r"C:\LABCOM\Retrieve\bin",
)
```

一時ファイルの生成場所を変える場合は `working_dir` も指定できます。通常は `Retrieve.exe` のディレクトリが使用され、取得終了時に一時ファイルが削除されます。

## 3. 単一チャンネルを取得する

```python
waveform = retriever.retrieve_data(
    diag="Mag",
    shotno=139400,
    subshot=1,
    channel=32,
    time_axis=True,
)
```

主な引数:

| 引数 | 内容 |
| --- | --- |
| `diag` | LABCOM の診断名 |
| `shotno` | ショット番号 |
| `subshot` | サブショット番号。通常は `1` |
| `channel` | チャンネル番号 |
| `time_axis` | `True` の場合は `Retrieve.exe` に `-T` を渡す |
| `frame_number` | 特定フレームを取得する場合の番号 |
| `dtype` | 自動判定を上書きする NumPy dtype。例: `"int8"` |

戻り値の `LHDData` は、生データ、時間軸、メタデータを保持します。

```python
print(waveform.data.dtype)
print(waveform.time.shape)
print(waveform.metadata)

voltage = waveform.val
frame = waveform.to_pandas()
waveform.save_csv("outputs/mag_ch32.csv")
```

`val`（別名 `voltage`）は `VResolution` / `VOffset`、`VCoefficient1` / `VCoefficient0`、または整数型の `RangeLow` / `RangeHigh` メタデータを使って物理値へ変換します。必要なメタデータがない場合は `ValueError` になります。

## 4. 複数チャンネルを取得する

```python
channels = retriever.retrieve_multiple_channels(
    diag_name="Mag",
    shot=139400,
    subshot=1,
    channels=[1, 2, 3, 4],
    time_axis=True,
)

for channel, data in channels.items():
    print(channel, data.data.shape)
```

戻り値は `{channel_number: LHDData}` の辞書です。取得に失敗したチャンネルは warning を出してスキップされるため、要求数と辞書の要素数が一致するとは限りません。

## 5. 可視化する

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(waveform.time, waveform.val)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal [V]")
ax.grid(True)
```

簡易表示なら `waveform.plot()` も利用できます。

## 6. エラーの見方

- `Retrieve.exe not found`: 自動探索に失敗しています。実行ファイルを含むディレクトリを `retrieve_path` に指定します。
- `Retrieve.exe failed`: 診断名、ショット、サブショット、チャンネル、アクセス権、および表示されたコマンドを確認します。
- `Retrieve.exe timeout after 5 minutes`: 1 回の取得は 300 秒でタイムアウトします。LABCOM 側の状態や対象データを確認します。
- 空の複数チャンネル結果: warning を確認し、まず単一チャンネル取得で条件を検証します。
- 時間軸が不自然: `time_axis=True` を指定し、取得メタデータの sampling rate を確認します。

## 7. 運用上の注意

- 大量チャンネルを繰り返し取得する前に、単一チャンネルで診断名と dtype を確認してください。
- `metadata` は診断ごとに異なります。電圧変換や sampling rate に使われたキーを保存しておくと追跡しやすくなります。
- 一時ファイルは通常自動削除されます。取得中のプロセス強制終了時には作業ディレクトリに残る場合があります。
