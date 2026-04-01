# MambaAngleRegressor 簡明解釋

這份說明用簡單明瞭的方式解釋目前回歸模型 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py) 在做什麼。

## 一句話先講完

這個模型的工作流程可以理解成：

`CT 影像 -> 3 層特徵抽取 -> 每層用 Mamba 理解整體關係 -> 合併多尺度資訊 -> 輸出角度`

它不是 U-Net 那種會縮小再放大的分割網路，而是一個「只有編碼端的回歸網路」，最後直接輸出一個角度值。

## 1. 整體在做什麼

輸入是一個 3D CT volume，例如：

- `1 x 112 x 136 x 112`

模型會一路把影像縮小成越來越抽象的特徵，最後把不同尺度的資訊合在一起，輸出一個數值，也就是預測的塌陷角度。

你可以把它想成：

1. 先快速看整張 CT 的大方向
2. 再分三層抽特徵
3. 每層都保留重要資訊
4. 最後把三層資訊合起來判斷角度

## 2. Stem 是什麼

`Stem` 定義在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L105)。

它做三件事：

- `Conv3d(kernel=7, stride=4)`
- `GroupNorm`
- `GELU`

最重要的是 `stride=4`，它會先把空間尺寸大幅縮小。

例如輸入：

- `112 x 136 x 112`

經過 Stem 後大致變成：

- `32 x 28 x 34 x 28`

意思就是：先把超大的 3D CT 壓成比較小、比較容易處理的特徵圖。

## 3. 三個 Stage 在做什麼

模型有三個主要 stage，定義在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L116)。

- `Stage 1`
- `Stage 2`
- `Stage 3`

它們對應的大致尺寸是：

- Stem 後：`32 x 28 x 34 x 28`
- Stage 1 後：`32 x 28 x 34 x 28`
- Stage 2 後：`64 x 14 x 17 x 14`
- Stage 3 後：`128 x 7 x 9 x 7`

可以這樣理解：

- 前面的 stage 保留比較多空間細節
- 後面的 stage 空間變小，但語意更強
- 越後面越像在看「整體模式」而不是局部像素

## 4. 每個 Stage 裡面有什麼

每個 stage 本質上都是一個 `DownsampleStage`，定義在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L74)。

每個 stage 的流程是：

1. 先用一個 3D convolution 做尺寸與通道調整
2. 再接多個 `ResidualMambaBlock`

目前 config 設定是 `blocks=3`，所以每個 stage 都會有 3 個 Mamba block，這個設定來自 [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml#L3)。

也就是說，每個 stage 都是在做：

- 一次特徵縮放
- 三次更深入的特徵理解

## 5. ResidualMambaBlock 到底在做什麼

這是整個模型最核心的部分，定義在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L38)。

它的大致流程是：

1. `GroupNorm + 1x1x1 Conv`
2. 把 3D 特徵圖攤平成 token 序列
3. 用 `Mamba` 去建模 token 之間的關係
4. 再把 token 變回 3D 特徵圖
5. 跟原本輸入做 residual 相加

原本特徵圖形狀是：

- `[B, C, D, H, W]`

中間會被轉成：

- `[B, N, C]`

其中：

- `N = D x H x W`

意思是把整個 3D 空間中的每個位置，都當成一個 token。

簡單講：

- CNN 比較擅長看局部鄰近區域
- Mamba 比較擅長看較長距離的上下文關係

所以這個 block 的重點，就是讓模型不只看局部，還能理解整體結構之間的關係。

## 6. 為什麼要做 residual

在 `ResidualMambaBlock` 裡，最後會把原本輸入加回來，也就是：

- `x + residual`

這樣做的好處是：

- 比較容易訓練深層網路
- 原始資訊不容易被完全洗掉

你可以把它想成：不是完全推翻前面的特徵，而是在原本資訊上再修正一次。

## 7. 為什麼最後不是只用最深層

模型在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L152) 並不是只拿 `Stage 3` 的輸出。

它會把：

- `Stage 1`
- `Stage 2`
- `Stage 3`

都做一次 `AdaptiveAvgPool3d(1)`，把每層壓成一個向量，再把三個向量接起來。

所以最後的特徵向量維度是：

- `32 + 64 + 128 = 224`

這樣做的意義是：

- `Stage 1` 提供較細的局部資訊
- `Stage 2` 提供中尺度資訊
- `Stage 3` 提供更抽象的高階資訊

最後三者一起判斷，比只看最深層更穩。

## 8. 最後怎麼輸出角度

最後的 head 在 [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py#L123)。

它的結構是：

- `224 -> 128`
- `128 -> 64`
- `64 -> 1`

中間有：

- `GELU`
- `Dropout`

所以最後輸出的不是分類機率，而是一個實數。

例如：

- 輸入一個病人的 CT
- 模型最後可能輸出 `143.7`

這個值就是預測的塌陷角度。

## 9. 最簡單版本的理解方式

如果要用最簡單的方式記：

- 先用 3D CNN 把 CT 壓小
- 再用 Mamba block 理解整體關係
- 從三個不同尺度各取一份摘要
- 最後把三份摘要合起來回歸出角度

## 10. 簡短總結

這個網路本質上是：

- 用 3D CNN 做空間降維
- 用 Mamba 做長距離關係建模
- 用多尺度 pooling 保留不同層次資訊
- 最後用 MLP 輸出一個角度值
