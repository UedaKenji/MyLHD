# KaisekiData 利用ガイド

`KaisekiData` の基本的な使い方（オンラインのオープンデータから取得する場合）と、ローカルで解析した結果をインポート・保存する場合の双方について解説します。内部実装の詳細には踏み込まず、ユーザーが利用する公開インターフェースのみを整理しています。

---

## 1. オープンデータから取得する

### 1.1 即時取得
```python
from mylhd import KaisekiData

ts = KaisekiData.retrieve_opendata(diag="tsmap_calib", shotno=190600, subno=1)
```

- `diag` には診断名、`shotno` / `subno` には対象ショット番号とサブショット番号を指定します。
- 取得できない場合は `FileNotFoundError`、通信エラー時は `RuntimeError` が送出されます。

### 1.2 LABCOM サーバから取得
```python
ts = KaisekiData.retrieve(diag="cts", shotno=190600, subno=1)
```
内部で `igetfile` コマンドが呼ばれるため、対応環境でのみ利用可能です。

### 1.3 データ内容の確認
```python
ts.show()      # 名前や次元情報・単位を出力
ts.info()      # 上記に加えてコメント欄も表示

time = ts.time                # 代表的な時間軸（存在する場合）
rho = ts.get_dim_data("rho")  # 軸名やインデックスで取得可能
te = ts.get_val_data("Te")    # 値も名前または番号で取得
unit = ts.get_val_unit("Te")  # 単位確認
```

### 1.4 擬似的な辞書添字アクセス
```python
ts["Te"]      # == ts.get_val_data("Te")
ts["Time"]    # == ts.get_dim_data("Time")
ts[0]         # 最初の変数（ValName0）
```

### 1.5 シンプルな可視化例
```python
import matplotlib.pyplot as plt
plt.plot(ts.time, ts.get_val_data("Te")[:, 0])  # 例: 1 チャンネル目
plt.xlabel("Time [s]")
plt.ylabel("Te [keV]")
plt.show()
```

### 1.6 データのポーリング待ち
計測直後でまだオープンデータが準備されていない場合は `wait_for_opendata` を利用します。
```python
from mylhd.anadata import wait_for_opendata

ts = wait_for_opendata(diag="tsmap_calib", shotno=190600, subno=1, retry_delay=60)
```
指定秒数ごとにリトライし、利用可能になった段階で `KaisekiData` を返します。

---

## 2. 解析済みデータをローカルから生成する

### 2.1 事前に準備するもの
1. 軸名 (`dimnames`)、軸単位 (`dimunits`)  
   例: `["Time", "rho"]` / `["s", "1"]`
2. 変数名 (`valnames`)、変数単位 (`valunits`)  
   例: `["Te", "ne"]` / `["keV", "1e19 m^-3"]`
3. 軸ごとの長さ (`dimsizes`)  
   例: `[n, m]`
4. 実データ配列 (`data`)  
   - shape は `(n, m, len(dimnames) + len(valnames))`  
   - 最終次元を `[軸データ..., 値データ...]` の順に積む
   - 例: `(n, m, 4)` の最後の軸 `[Time, rho, Te, ne]`

### 2.2 Payload からの生成
```python
import numpy as np
from mylhd.anadata import KaisekiData

n, m = 100, 32
time = np.linspace(0, 1, n)
rho = np.linspace(-1, 1, m)
Te = np.random.rand(n, m)
ne = np.random.rand(n, m)

grid_time = np.repeat(time[:, None], m, axis=1)
grid_rho = np.repeat(rho[None, :], n, axis=0)
stacked = np.stack([grid_time, grid_rho, Te, ne], axis=-1)

payload = {
    "schema_version": 1,
    "name": "cts",
    "shotno": 123456,
    "subno": 1,
    "dimnames": ["Time", "rho"],
    "dimunits": ["s", "1"],
    "valnames": ["Te", "ne"],
    "valunits": ["keV", "1e19 m^-3"],
    "dimsizes": [n, m],
    "data": stacked,
    "comment": "local analysis result",
    "metadata": {"source": "custom pipeline"},
    "date": "2024-02-01",
}

kdata = KaisekiData.from_payload(payload)
```

これで `kdata` は通常の `KaisekiData` と同じインターフェースで利用できます。

### 2.3 `dimdata` / `valdata` を分けて指定する
`payload["data"]` を自前で組み立てる代わりに、軸と値を別々に渡すこともできます。

```python
payload = {
    "schema_version": 1,
    "name": "cts",
    "shotno": 123456,
    "subno": 1,
    "dimnames": ["Time", "rho"],
    "dimunits": ["s", "1"],
    "valnames": ["Te", "ne"],
    "valunits": ["keV", "1e19 m^-3"],
    "dimsizes": [n, m],
    "dimdata": [
        time,   # "Time" に対応。shape (n,) → 自動で (n, m) に拡張
        rho,    # "rho"  に対応。shape (m,) → 自動で (n, m) に拡張
    ],
    "valdata": [
        Te,     # "Te" に対応。shape (n, m)
        ne,     # "ne" に対応。shape (n, m)
    ],
    "comment": "local analysis result",
    "metadata": {"source": "custom pipeline"},
}

kdata = KaisekiData.from_payload(payload)
```

ポイント:

- `dimdata` / `valdata` は **リストまたはタプル** で渡し、要素順を `dimnames` / `valnames` と一致させます（辞書形式も使用できますが、キー名は新しいスペルに合わせてください）。
- 軸データは 1 次元配列でもかまいません（サイズが対応していれば自動的に `dimsizes` まで拡張されます）。
- 値データは `dimsizes` と同じ形、または NumPy のブロードキャスト規則で拡張可能な形状であれば渡せます。

この形式は、既存コードで軸ベクトルや物理量を個別に持っている場合に手軽です。

---

## 3. ローカル保存と復元

### 3.1 保存
```python
kdata.export_local("outputs/cts_local.pkl", overwrite=True)
```
または関数版:
```python
from mylhd.anadata_storage import export_kaiseki_data
export_kaiseki_data(kdata, "outputs/cts_local.pkl", overwrite=True)
```

### 3.2 復元
```python
restored = KaisekiData.from_local_file("outputs/cts_local.pkl")
```
または関数版:
```python
from mylhd.anadata_storage import import_kaiseki_data
restored = import_kaiseki_data("outputs/cts_local.pkl")
```

復元したインスタンスは元の `kdata` と同じ内容を保持します。

### 3.3 メタデータ活用
`metadata` に辞書を渡すと、解析条件や作成者などを保存できます。`restored.metadata` で取り出せるため、再解析やログ管理に有用です。

---

## 4. スナップショット API（必要に応じて）
- `snapshot = kdata.to_snapshot()`  
  出力された `snapshot` は `snapshot.to_payload()` で辞書化でき、独自のファイル形式に変換する際に便利です。
- 関数版 `mylhd.anadata_storage.KaisekiDataSnapshot.from_kaiseki(kdata)` も同等の役割を持ちますが、通常利用では不要です。

---

## 5. バリデーションとエラーハンドリング
- `from_payload` や `import_kaiseki_data` の内部で整合性チェックが実行され、shape や軸情報が一致しない場合は例外が送出されます。
- Pickle ファイルは `overwrite=False` が既定値のため、上書きする際は `overwrite=True` を指定してください。
- ネットワークアクセスが必要な API は `urllib3` を利用しているため、環境によってはプロキシ設定などが必要になる場合があります。

---

## 6. ヒントとベストプラクティス
- オープンデータ API のレスポンスが空の場合は、ショット番号や診断名を再確認するか、少し時間を置いて再取得してください。
- 自前データの作成では、軸データを直接 `data` 配列に埋め込む点に注意してください（別途 `dimdata` を渡す形式ではありません）。
- 長期的な保管や共有には、`metadata` に実験条件・解析パラメータ・コードバージョンなどを残しておくと再現性が高まります。

---

これらの API を組み合わせることで、オンライン／オフラインを問わず `KaisekiData` を柔軟に扱えます。困った点があれば issue などでフィードバックをお寄せください。
