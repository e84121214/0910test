# 純地面氣象站 GRU 實驗

這是三個目標模型中的第二組 `GROUND`。訓練入口為 `train_ground_gru.py`，資料序列與空間聚合由 `create_ground_sequences.py` 建立；第一組衛星模型的 `train_gru.py` 未因本模型而修改。

## 模型輸入

模型不輸入任何衛星波段、波段差、氣象站能見度 `VS01`，也不輸入能見度儀當下的 `Visibility_km`。`Visibility_km` 僅用於未來霧標籤及 persistence 參考基準。

輸入包含：

- 核心氣象要素：氣溫 `TX01`、相對濕度 `RH01`、測站氣壓 `PS01`、風速 `WD01`、風向 `WD02`、降水量 `PP01` 與雨跡旗標。
- 多鄰站聚合：加權平均、最大值、有效鄰站數。風向先轉成正弦與餘弦；降水量跨站不加總。
- 資料品質：14 個聚合值的缺值遮罩、各要素有效站數、當小時有資料的鄰站數、氣象資料年齡。
- 空間關係：加權平均緯度差、經度差與海拔差、最近鄰站距離、配對鄰站總數。
- 共通輔助特徵：臺灣時間的小時與月份週期，以及能見度儀的經度、緯度與海拔。

總計 49 個特徵。聚合後缺值以訓練集平均值填補，並由缺值遮罩及有效站數告知模型資料可靠度。所有標準化統計量只由訓練資料計算。

## 空間權重

先排除品質表中 9 個「該年度暫不採用」氣象站。ACOS 26 個能見度站最後使用 158 個不重複氣象站、182 組 10 公里配對。

每個氣象要素在每個時刻，依有有效值的鄰站重新正規化：

```text
raw_weight(v, j) = exp(-distance_km(v, j) / 5)
weight(v, j, t, feature) = raw_weight(v, j) / sum(raw_weight of valid neighbors)
```

海拔不參與權重計算，而是作為相對海拔與能見度儀海拔輸入模型。如此可固定本次實驗為純指數距離衰減，日後只替換權重函式即可測試距離反比或相關係數複合權重。

## 時間與比較條件

逐時氣象資料以 UTC 小時對齊 10 分鐘能見度時間，只使用不晚於預測時點的當小時觀測；資料年齡為 0 至 50 分鐘。

為與第一組衛星模型公平比較，地面模型使用相同序列母體：衛星欄位只用來判斷該列能否進入共同比較樣本，不會成為地面模型輸入。因此四個切分的序列、標籤及測站均與第一組一致：

| 切分 | 序列數 | 霧標籤 | 測站數 |
|---|---:|---:|---:|
| Train | 140,290 | 1,992 | 25 |
| Validation | 37,515 | 489 | 25 |
| Temporal test | 38,744 | 437 | 25 |
| Spatial test C48 | 5,782 | 94 | 1 |

每筆序列仍使用過去 18 筆、每 10 分鐘一筆的資料，預測最後輸入後 60 分鐘是否起霧。驗證月份為 UTC 3、9 月，時間測試為 6、12 月，C48 僅作空間測試。

## 執行與輸出

```powershell
.\.venv\Scripts\python.exe 模型\02_地面氣象模型\create_ground_sequences.py
.\.venv\Scripts\python.exe 模型\02_地面氣象模型\train_ground_gru.py --epochs 30
```

預設輸出至 `模型/02_地面氣象模型/results/`：

- `best_ground_weather_gru_model.pt`：最佳模型權重。
- `ground_weather_gru_metadata.json`：特徵順序、標準化參數、門檻、空間權重與模型設定。
- `ground_weather_results.csv`：時間與空間測試指標。
- `training_validation_loss_ground_weather.png`：訓練與驗證損失。
- `training.log`：完整訓練紀錄。
- `ground_spatial_threshold_metrics.csv`：C48 在 0.00–1.00 共 101 個門檻下的完整指標。
- `ground_spatial_threshold_precision.png`、`ground_spatial_threshold_recall.png`、`ground_spatial_threshold_csi.png`：C48 門檻診斷曲線。

## 本次結果

模型使用單層 GRU 與 64 hidden units，和目前 `train_gru.py` 的程式設定一致。訓練最多 30 輪，於第 9 輪提前停止，最佳權重為第 4 輪；驗證集以 F1 選出的門檻為 0.70。

| 測試集 | TP | FP | FN | Precision | Recall | CSI |
|---|---:|---:|---:|---:|---:|---:|
| Temporal test | 256 | 358 | 181 | 41.69% | 58.58% | 32.20% |
| Spatial test C48 | 5 | 32 | 89 | 13.51% | 5.32% | 3.97% |

C48 仍只能偵測 5／94 筆霧，尚不能宣稱地面模型具有跨站泛化能力。以上門檻只由 validation 選擇，兩個 test 均未參與調整。

衛星、地面及混合模型的本次正式產物均使用 64 hidden units；完整的同架構比較見 `模型/三種GRU霧預測模型_實驗結果說明.md`。
