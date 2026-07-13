# thomson_gp 利用ガイド

`mylhd.thomson_gp` は、LHD オープンデータの `tsmap_calib` を読み込み、電子温度 `Te` と電子密度 `ne` を Gaussian Process（GP）で時空間再構成する機能です。対数 GP を用いて、プロファイル、時間発展、空間・時間微分を可視化できます。

- サンプル: [thomson_gp_sample.ipynb](../examples/thomson_gp_sample.ipynb)
- パッケージ概要: [README](../README.md)

## 1. 前提条件

- 対象ショットに `tsmap_calib` オープンデータが存在すること
- オープンデータサーバーへのネットワーク接続
- 最適化と行列計算に十分なメモリと計算時間

最初に `KaisekiData` だけで対象データを確認すると、GP の計算を始める前に診断名やショットの問題を分離できます。

```python
from mylhd import KaisekiData

source = KaisekiData.retrieve_opendata(
    diag="tsmap_calib",
    shotno=194042,
)
source.show()
```

## 2. ThomsonGP を初期化する

```python
from mylhd import ThomsonGP

thomson = ThomsonGP(
    shotNo=194042,
    from_opendata=True,
    iprint=True,
)
```

初期化時に `tsmap_calib` を取得し、品質チェック、入力範囲の設定、既定 kernel（`Matern52`）の選択まで行います。データが不足しているショットでは例外になります。

## 3. GP を実行する

`Te` と `ne` は別々に再構成します。次は現在の基本的なワークフローです。

```python
thomson.pipeline_sigma_optimize(
    "Te",
    time_scale=0.2,
    reff_scale=0.5,
    omega=0.01,
    iprint=True,
)

thomson.pipeline_sigma_optimize(
    "ne",
    time_scale=0.2,
    reff_scale=0.5,
    omega=0.02,
    iprint=True,
)
```

主なパラメータ:

| 引数 | 内容 |
| --- | --- |
| `valName` | `"Te"` または `"ne"` |
| `time_scale` | 時間方向 kernel の代表スケール |
| `reff_scale` | 規格化小半径方向 kernel の代表スケール |
| `sigscale_init` | 観測不確かさスケールの初期値 |
| `convergence` | sigma scale 反復の収束判定 |
| `omega` | 最適化時の追加パラメータ |
| `iprint` | 進捗表示 |

適切なスケールはショットと解析目的に依存します。まず狭い時間範囲や既知のショットで挙動と計算時間を確認してください。

## 4. 結果を可視化する

### 時空間マップ

```python
thomson.plot_im(ValName="Te", x_axis="rho_vac")
thomson.plot_im(ValName="ne", x_axis="rho_vac")
```

`x_axis` は `"rho"`、`"rho_vac"`、`"R"` から選択します。

### 指定時刻の径方向プロファイル

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
thomson.plotProfile("Te", time=5.0, ax=axes[0])
thomson.plotProfile("ne", time=5.0, ax=axes[1], color="blue")
```

### 指定半径の時間発展

```python
thomson.plotTrace("Te", reff=0.4)
thomson.plotTrace("ne", reff=0.4)
```

プロファイルや trace は、対象時刻・半径に最も近い点を使います。解析範囲は `thomson.time_inp` と `thomson.rho_vac` で確認できます。

### 空間微分・時間微分

指定時刻における径方向分布は `plotProfileDx`、指定半径における時間発展は `plotEvolutionDx` で表示できます。

```python
# 指定時刻での dlog(Te)/dr と dlog(Te)/dt
thomson.plotProfileDx("Te", DimName="reff", time=5.0)
thomson.plotProfileDx("Te", DimName="time", time=5.0)

# 指定半径での微分量の時間発展
thomson.plotEvolutionDx("ne", DimName="reff", reff=0.4)
thomson.plotEvolutionDx("ne", DimName="time", reff=0.4)
```

`DimName="reff"` は規格化半径方向、`DimName="time"` は時間方向の対数微分を表します。Te・ne を並べたプロット例は [サンプル notebook](../examples/thomson_gp_sample.ipynb) にあります。

## 5. 最適化パラメータを保存する

```python
thomson.save(
    path="results/",
    filename="shot_194042_gp",
    timestamp=False,
    save_fig=False,
)
```

保存される JSON は、最適化パラメータとログを記録するためのものです。GP の全数値結果や元データを単体で保持する形式ではありません。

`ThomsonGP.load_from_json()` も実装されていますが、元ショットの再取得・再計算を伴い、現在は保存キーとの互換性を調整中です。現バージョンでは主要な復元経路として使わず、JSON を解析条件の記録として扱ってください。

## 6. 複数 phase の解析

`ThomsonGPMultiPhase` は、時間範囲を複数 phase に分けて扱うための高水準 API です。

```python
from mylhd import ThomsonGPMultiPhase

multi = ThomsonGPMultiPhase(194042)
multi.add_phase(time_start=3.0, time_end=4.0)
multi.add_phase(time_start=4.0, time_end=5.0)
multi.plot_phases()
```

この API は単一 `ThomsonGP` より利用実績が限られます。まず単一 phase の結果を確認してから使用してください。

## 7. よくある問題

- `shot ... does not have data`: `tsmap_calib` が存在しない、または品質チェックを満たすデータが不足しています。
- 初期化で通信エラー: `KaisekiData.retrieve_opendata` を単独実行し、ネットワークとデータ公開状況を確認します。
- 計算が長い: 入力時間点数、kernel scale、最適化反復が計算量に影響します。小さい範囲や既知のパラメータから始めます。
- `plot_im` で属性がない: 対象の `Te` / `ne` について GP パイプラインが完了しているか確認します。
- プロファイルの単位や軸が想定と違う: `ValName` と `x_axis` を明示し、元データの `show()` と `rho_vac` を確認します。
