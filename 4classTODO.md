通常這種分多類的回歸任務會用什麼論文指標去看
下午3:49



你這個任務嚴格來說比較不像「回歸」，比較像：

4-class classification
而且是 ordinal classification
因為 GOLD 1 < GOLD 2 < GOLD 3 < GOLD 4 有順序
所以論文通常不會只看一個指標。
對你這種任務，我會建議分成兩類看：

一般多分類一定會報的

Accuracy
Macro-F1
Balanced Accuracy
Confusion Matrix
per-class Precision / Recall / F1
這裡面如果類別不平衡，最重要通常不是 Accuracy，而是：

Macro-F1
Balanced Accuracy
因為它們比較不會被 GOLD 1 數量最多這件事騙掉。

因為你的類別有順序，最好再加的

Quadratic Weighted Kappa
MAE on class index
Within-1 accuracy
這幾個很適合 GOLD：

Quadratic Weighted Kappa
很常用在有等級、嚴重度分級的任務
錯一級比錯三級罰得輕
很符合醫學分級情境
MAE on class index
把 GOLD 1,2,3,4 當成 0,1,2,3
算預測級別和真實級別差多少
例如真實是 GOLD 4，你猜 GOLD 3，只錯 1
這比單純看對錯更有資訊
Within-1 accuracy
預測只要差一級以內就算 acceptable
例如真實 GOLD 3，猜 GOLD 2 或 GOLD 4 都算 within-1
在臨床分級任務裡常常很有說服力
如果是醫學影像論文，還常看到

one-vs-rest AUROC
one-vs-rest AUPRC
但你這個資料太小，我會覺得：

可以報
但不要當主指標
因為波動可能很大
如果是我幫你定 paper 主表，我會放這組

Accuracy
Macro-F1
Balanced Accuracy
Quadratic Weighted Kappa
MAE on class index
Confusion Matrix
主打指標

第一主指標：Macro-F1
第二主指標：Quadratic Weighted Kappa
輔助：Balanced Accuracy
因為你的任務同時有：

類別不平衡
類別有順序
只報 Accuracy 會很吃虧，也很容易被老師或 reviewer 問。

如果你要，我下一步可以直接幫你把目前 pipeline 再補：

Quadratic Weighted Kappa
class-index MAE
within-1 accuracy
讓 results.json 直接多出這幾個論文比較像樣的指標。