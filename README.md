# MyLHD

MyLHD は、Large Helical Device（LHD）の計測データ取得と解析を支援する Python パッケージです。LHD オープンデータの読み込み、LABCOM の `Retrieve.exe` を使った波形取得、Thomson 散乱データの Gaussian Process（GP）再構成を、共通の Python 環境から扱えます。

> [!NOTE]
> CTS 関連機能（`mylhd.cts_utls`）は現在調整中です。API と解析手順が固まってから個別ドキュメントを追加します。

## 主な機能

| モジュール | 用途 | 実行環境 |
| --- | --- | --- |
| `mylhd.KaisekiData` | LHD オープンデータ、解析データ形式の読み込み・保存 | オープンデータ取得時はネットワーク接続が必要 |
| `mylhd.labcom_retrieve` | LABCOM `Retrieve.exe` による単一・複数チャンネル取得 | Windows または WSL と `Retrieve.exe` が必要 |
| `mylhd.thomson_gp` | Thomson 散乱 `tsmap_calib` の GP 再構成・可視化 | オープンデータ接続と計算時間が必要 |
| `mylhd.utils` | オープンデータ待機、ショット探索、ON/OFF 波形の時刻対応付け | 機能ごとに異なる |
| `mylhd.cts_utls` | CTS データ処理 | 調整中 |

## 必要環境

- Python 3.10 以上
- NumPy、pandas、Matplotlib、SciPy、urllib3
- LABCOM データ取得を行う場合は Windows または WSL、および利用可能な `Retrieve.exe`

## 導入方法

### uv を使う場合（推奨）

```powershell
git clone https://github.com/UedaKenji/MyLHD.git
cd MyLHD
uv sync
```

開発ツールと Jupyter kernel も導入する場合:

```powershell
uv sync --extra dev
```

コマンドや notebook からは、作成された環境を選ぶか `uv run` を使用します。

```powershell
uv run python -c "import mylhd; print(mylhd.__version__)"
uv run pytest
```

### pip を使う場合

```powershell
git clone https://github.com/UedaKenji/MyLHD.git
cd MyLHD
python -m pip install -e .
```

開発用依存関係も導入する場合:

```powershell
python -m pip install -e ".[dev]"
```

## クイックスタート

### LHD オープンデータを取得する

```python
from mylhd import KaisekiData

data = KaisekiData.retrieve_opendata(
    diag="tsmap_calib",
    shotno=194042,
    subno=1,
)

data.show()
time = data.get_dim_data("Time")
te = data.get_val_data("Te")
```

### LABCOM から波形を取得する

この機能は、LHD の解析サーバーへログイン済みで、LABCOM データへアクセスできる状態で使用します。

```python
from mylhd import LHDRetriever

retriever = LHDRetriever()
waveform = retriever.retrieve_data(
    diag="Mag",
    shotno=139400,
    subshot=1,
    channel=32,
    time_axis=True,
)

print(waveform.metadata)
waveform.plot()
```

`LHDRetriever` は `Retrieve.exe` を自動探索します。見つからない場合は、実行ファイルを含むディレクトリを `retrieve_path` に指定してください。

### Thomson 散乱データを GP 再構成する

```python
from mylhd import ThomsonGP

thomson = ThomsonGP(shotNo=194042)
thomson.pipeline_sigma_optimize(
    "Te",
    time_scale=0.2,
    reff_scale=0.5,
    omega=0.01,
    iprint=True,
)
thomson.plot_im(ValName="Te", x_axis="rho_vac")
```

GP の最適化には時間がかかります。まず対象ショットに `tsmap_calib` が存在することを確認してください。

## ドキュメントとサンプル

| 分野 | ガイド | サンプル notebook |
| --- | --- | --- |
| KaisekiData | [KaisekiData 利用ガイド](docs/kaiseki_data_usage.md) | [kaiseki_data_sample.ipynb](examples/kaiseki_data_sample.ipynb) |
| LABCOM Retrieve | [labcom_retrieve 利用ガイド](docs/labcom_retrieve.md) | [labcom_retrieve_sample.ipynb](examples/labcom_retrieve_sample.ipynb) |
| Thomson GP | [thomson_gp 利用ガイド](docs/thomson_gp.md) | [thomson_gp_sample.ipynb](examples/thomson_gp_sample.ipynb) |

サンプル notebook は、外部データ取得を既定で無効にしています。ショット番号、診断名、`Retrieve.exe` の場所を確認してから、notebook 内の実行フラグを有効にしてください。

## 開発時の確認

```powershell
uv run pytest
uv run black --check src tests
uv run isort --check-only src tests
uv build
```

## ライセンス

MIT License

## 作者

Kenji Ueda (`kenji.ueda@nifs.ac.jp`)
