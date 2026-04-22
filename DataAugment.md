最推薦先用

小角度旋轉
例如 ±5° ~ ±10°。模擬病人擺位差異，不要旋太大。

平移 / 縮放
平移 ±5%、scale 0.9 ~ 1.1。這通常很合理。

HU intensity jitter
對 CT 值做小幅度變化，例如 window shift、contrast jitter、gamma jitter。
這可以模擬不同 scanner / reconstruction 的差異。

Gaussian noise
加一點點噪聲，模擬低劑量或重建雜訊。

Gaussian blur / sharpen 輕微版
模擬不同 kernel，但要很小心，不要把 emphysema/air trapping 訊號洗掉。

可以考慮，但要小心
6. 左右翻轉
Chest CT 通常左右翻轉可能還能接受，但如果疾病分布、心臟位置、解剖方向有意義，就不要亂翻。保守起見我會先不開。

Elastic deformation
醫學影像常見，但對肺功能/GOLD 這種全肺結構任務可能會製造不真實 anatomy。除非很輕微，不然我不建議一開始用。

Random crop / zoom crop
可以，但要確保肺部沒有被裁掉。你的資料是全胸 CT，crop 太激進會傷 label。

不太建議
9. CutMix / Cutout
對自然影像常用，但 CT 裡遮掉一塊肺或混兩個病人的 CT，醫學語意比較怪。我不會先用。

MixUp
classification 可以試，但 GOLD 是病人級別疾病嚴重度，兩個病人的 volume 混在一起很難解釋。除非做實驗比較，不當預設方案。
我的建議策略是：

WeightedRandomSampler
+ 只對 training fold 做 augmentation
+ 少數類 GOLD 2/3/4 augmentation 機率較高
+ validation/test 完全不做 augmentation
第一版我會開這組：

random rotation: ±7°
random translation: ±5%
random scale: 0.95 ~ 1.05
HU/window jitter: 小幅
Gaussian noise: 小幅
先不要 flip、elastic、mixup。這樣比較穩，也比較容易在論文/報告裡說明：只是模擬掃描與擺位差異，不是假裝增加病人數。