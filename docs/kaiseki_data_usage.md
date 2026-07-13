# KaisekiData 利用ガイド

`KaisekiData` は、LHD の解析データを「次元軸」「物理量」「単位」「ショット情報」とともに扱うためのコンテナです。オープンデータからの取得だけでなく、手元で生成した解析結果の格納と保存にも利用できます。

- サンプル: [kaiseki_data_sample.ipynb](../examples/kaiseki_data_sample.ipynb)
- パッケージ概要: [README](../README.md)

## 1. オープンデータから取得する

```python
from mylhd import KaisekiData

data = KaisekiData.retrieve_opendata(
    diag="tsmap_calib",
    shotno=194042,
    subno=1,
)
```

取得にはネットワーク接続が必要です。データが存在しない場合は `FileNotFoundError`、通信や応答の問題は `RuntimeError` として通知されます。

計測直後など、データ公開まで待ちたい場合は `wait_for_opendata` を使います。

```python
from mylhd import wait_for_opendata

data = wait_for_opendata(
    diag="tsmap_calib",
    shotno=194042,
    subno=1,
    retry_delay=60,
)
```

この関数は成功するまで待機するため、終了条件が必要な処理では呼び出し側でタイムアウトを設けてください。

## 2. 内容を確認する

```python
data.show()  # 名前、shot、shape、軸・値・単位
data.info()  # show() の内容とコメント

print(data.dimnames)
print(data.valnames)
print(data.data.shape)
```

軸と値は名前またはインデックスで取得できます。

```python
time = data.get_dim_data("Time")
te = data.get_val_data("Te")

time_unit = data.get_dim_unit("Time")
te_unit = data.get_val_unit("Te")

# 短縮記法
assert data["Te"] is not None
first_value = data[0]
```

`data.time` は `Time`、`time`、`TIME` など、対応する時間軸が存在する場合に使えます。軸名や値名は診断ごとに異なるため、最初に `show()` で確認してください。

## 3. ローカルの配列から作る

`from_payload` では、軸を `dimdata`、物理量を `valdata` として分けて渡せます。以下はネットワークなしで実行できる例です。

```python
import numpy as np
from mylhd import KaisekiData

time = np.linspace(0.0, 1.0, 101)
radius = np.linspace(-1.0, 1.0, 41)
te = 2.0 * np.exp(-(radius[None, :] / 0.6) ** 2) * (1.0 + 0.1 * np.sin(2 * np.pi * time[:, None]))

payload = {
    "schema_version": 1,
    "name": "synthetic_profile",
    "shotno": 0,
    "subno": 1,
    "dimnames": ["Time", "rho"],
    "dimunits": ["s", "1"],
    "valnames": ["Te"],
    "valunits": ["keV"],
    "dimsizes": [time.size, radius.size],
    "dimdata": [time, radius],
    "valdata": [te],
    "comment": "Documentation example",
    "metadata": {"source": "synthetic"},
}

local_data = KaisekiData.from_payload(payload)
```

重要な制約:

- `dimnames`、`dimunits`、`dimsizes` の要素数を一致させる。
- `valnames` と `valunits` の要素数を一致させる。
- `valdata` の各配列は `dimsizes` と同じ shape、またはそこへブロードキャスト可能な shape にする。
- `dimdata` と `valdata` の並び順を、それぞれの名前リストと一致させる。

不整合は `KaisekiDataValidationError` になります。

## 4. 可視化する

`KaisekiData` 自体は汎用コンテナなので、NumPy 配列を取り出して Matplotlib で描画します。

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
mesh = ax.pcolormesh(
    local_data.get_dim_data("rho"),
    local_data.time,
    local_data["Te"],
    shading="auto",
)
ax.set_xlabel("rho")
ax.set_ylabel("Time [s]")
fig.colorbar(mesh, ax=ax, label="Te [keV]")
```

## 5. 保存と復元

```python
path = local_data.export_local(
    "outputs/synthetic_profile.pkl",
    overwrite=True,
)

restored = KaisekiData.from_local_file(path)
```

`metadata` には解析条件、作成コードのバージョン、入力データの由来などを記録すると再現性を高められます。

> [!WARNING]
> ローカル保存形式は Python の pickle を使用します。信頼できない `.pkl` ファイルは読み込まないでください。

## 6. どの取得方法を使うか

| 方法 | 用途 | 追加要件 |
| --- | --- | --- |
| `KaisekiData.retrieve_opendata(...)` | 公開済み LHD オープンデータ | ネットワーク |
| `KaisekiData.retrieve(...)` | `igetfile` を使う環境内取得 | LHD 内部環境と `igetfile` |
| `KaisekiData.from_payload(...)` | 手元の NumPy 配列を共通形式にする | なし |
| `KaisekiData.from_local_file(...)` | `export_local` の結果を復元する | 信頼できる pickle ファイル |

## 7. よくある問題

- `Time Key Not Found`: 診断に時間軸がないか、軸名が既知の表記ではありません。`dimnames` を確認し、`get_dim_data(軸名)` を使います。
- `Index Range Error`: 指定した軸名・値名が存在しません。`show()` で名前を確認します。
- 404 または `FileNotFoundError`: 診断名、ショット番号、サブショット番号、公開状況を確認します。
- shape のバリデーションエラー: `dimsizes` と `dimdata` / `valdata` の shape を確認します。
